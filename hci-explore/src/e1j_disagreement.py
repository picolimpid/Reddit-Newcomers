"""E1j: does personalisation tell a coding team *which items they will disagree on*?  (Idea 1's use case: route an
item to discussion, as in CollabCoder's step 2, when the team is predicted to split.)

Uses the out-of-fold predictions of E1f (DICES-350, 104 raters) and E1i (DICES-990, 141 raters): new raters, new
conversations, each rater's own k anchors (random, JEV difficulties).  Teams of 3 raters from the same rater fold
(so their predictions come from the same fitted model).  A team "splits" on a conversation when its three labels
are not all equal.  Predicted split probability = 1 - prod(p) - prod(1 - p), p = each member's predicted P(unsafe).
  content only (k = 0)   every member gets JEV's content prediction: says which conversations are contested in general
  + per-person offsets   each member's offset from k = 5 or 20 anchors
DICES-350: fixed teams (every rater rated everything), scored on all 350 conversations.  DICES-990: per conversation,
random teams among the raters who rated it.  Conversation-cluster bootstrap (1,000 draws).

  python3 -I src/e1j_disagreement.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "e1j")
os.makedirs(OUT, exist_ok=True)
B, TEAMS_350, TEAMS_990, FLAG = 1000, 200, 30, .2


def split_prob(P):
    return 1 - P.prod(-1) - (1 - P).prod(-1)


def evaluate(rows, label):
    """rows: DataFrame with item, split (0/1), and one score column per predictor."""
    preds = [c for c in rows.columns if c not in ("item", "split")]
    items = rows.item.unique()
    ipos = rows.item.map({i: k for k, i in enumerate(items)}).values
    rng = np.random.default_rng(1)
    y = rows.split.values

    def metrics(w):
        out = {}
        for c in preds:
            s = rows[c].values
            order = np.argsort(-s, kind="stable")
            cw = np.cumsum(w[order])
            top = order[cw <= FLAG * w.sum()]
            out[c] = (roc_auc_score(y, s, sample_weight=w), (w[top] * y[top]).sum() / (w * y).sum())
        return out

    pt = metrics(np.ones(len(y)))
    bs = []
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, len(items), len(items)), minlength=len(items)).astype(float)
        bs.append(metrics(cnt[ipos]))
    tab = []
    base = preds[0]
    for c in preds:
        a_b = np.array([b[c][0] for b in bs])
        r_b = np.array([b[c][1] for b in bs])
        d_a = a_b - np.array([b[base][0] for b in bs])
        d_r = r_b - np.array([b[base][1] for b in bs])
        tab.append({"data": label, "predictor": c, "AUROC": f"{pt[c][0]:.3f} [{np.quantile(a_b, .025):.3f}, {np.quantile(a_b, .975):.3f}]",
                    f"splits caught, top {int(FLAG * 100)}% flagged": f"{pt[c][1]:.3f} [{np.quantile(r_b, .025):.3f}, {np.quantile(r_b, .975):.3f}]",
                    "AUROC vs content only": "" if c == base else f"{pt[c][0] - pt[base][0]:+.3f} [{np.quantile(d_a, .025):+.3f}, {np.quantile(d_a, .975):+.3f}]",
                    "caught vs content only": "" if c == base else f"{pt[c][1] - pt[base][1]:+.3f} [{np.quantile(d_r, .025):+.3f}, {np.quantile(d_r, .975):+.3f}]"})
    return pd.DataFrame(tab), float(y.mean())


def dices350():
    z = np.load(os.path.join(ROOT, "results", "e1f", "oof_104.npz"))
    Y = z["Y"]
    I, P = Y.shape
    rfold = np.empty(P, int)
    rfold[np.random.default_rng(2).permutation(P)] = np.arange(P) % 5  # E1f's rater folds
    S = {"content only (k = 0)": z["0|0|0"], "+ per-person offsets, k = 5": z["0|5|0"],
         "+ per-person offsets, k = 20": z["0|20|0"]}
    rng = np.random.default_rng(3)
    rows = []
    for g in range(5):
        members = np.flatnonzero(rfold == g)
        for _ in range(TEAMS_350 // 5):
            t = rng.choice(members, 3, replace=False)
            lab = Y[:, t]
            ok = ~np.isnan(lab).any(1)
            d = {"item": np.flatnonzero(ok), "split": (np.nanmax(lab[ok], 1) != np.nanmin(lab[ok], 1)).astype(float)}
            for name, M in S.items():
                d[name] = split_prob(expit(M[np.ix_(np.flatnonzero(ok), t)]))
            rows.append(pd.DataFrame(d))
    return pd.concat(rows, ignore_index=True)


def dices990():
    z = np.load(os.path.join(ROOT, "results", "e1i", "oof_filtered.npz"))
    Y, rfold = z["Y"], z["rfold"]
    S = {"content only (k = 0)": z["0|0|0"], "+ per-person offsets, k = 5": z["0|5|0"],
         "+ per-person offsets, k = 20": z["0|20|0"]}
    rng = np.random.default_rng(3)
    rows = []
    for i in range(Y.shape[0]):
        for g in range(5):
            members = np.flatnonzero((rfold == g) & ~np.isnan(Y[i]))
            if len(members) < 3:
                continue
            for _ in range(TEAMS_990 // 5):
                t = rng.choice(members, 3, replace=False)
                lab = Y[i, t]
                d = {"item": [i], "split": [float(lab.max() != lab.min())]}
                for name, M in S.items():
                    d[name] = [split_prob(expit(M[i, t]))]
                rows.append(d)
    return pd.DataFrame({k: np.concatenate([r[k] for r in rows]) for k in rows[0]})


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    tabs = []
    for label, fn in (("DICES-350, 104 raters, fixed teams of 3", dices350), ("DICES-990, 141 raters, teams of 3", dices990)):
        rows = fn()
        tab, rate = evaluate(rows, label)
        print(f"\n== {label}: {len(rows)} (team, conversation) cases, split rate {rate:.3f}\n" +
              tab.drop(columns="data").to_string(index=False))
        tabs.append(tab)
    pd.concat(tabs).to_csv(os.path.join(OUT, "summary.csv"), index=False)


if __name__ == "__main__":
    main()
