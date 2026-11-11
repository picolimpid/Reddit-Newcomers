"""E4d: Venire-style panel-review simulation (Koshy et al. 2024) on DICES, plus E1j's team-disagreement test with
shared parts learned from singly-labelled logs (E4c).  PLAN.md D5 (descriptive re-expression of E1j) and D2's
secondary analysis.

A case = (team of 3 raters, conversation).  Without review, one random team member decides.  A routed case goes to
the whole team (majority vote).  For a 2-1 split a single member agrees with the majority with probability 2/3, so the
expected share of final decisions that match the team majority is 1 - (1/3) * (splits not routed) / cases.
Routing: random, judge only (k = 0, "rater-blind" in Venire's terms), judge + each member's intercept (k = 5, 20
anchors, "rater-aware"), and an oracle that routes exactly the splits.  95% CIs: conversation-cluster bootstrap.

  python3 -I src/e4d_venire_sim.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, os.path.dirname(__file__))
import jev  # noqa: E402
import e1j_disagreement as e1j  # noqa: E402

OUT = os.path.join(jev.ROOT, "results", "e4d")
os.makedirs(OUT, exist_ok=True)
BUDGETS = (.1, .2, .3, .5)
B = 1000


def teams_from(npz, S_keys, seed=3, per_item=e1j.TEAMS_990):
    z = np.load(npz)
    Y, rfold = z["Y"], z["rfold"]
    S = {name: z[key] for name, key in S_keys.items()}
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(Y.shape[0]):
        for g in range(5):
            members = np.flatnonzero((rfold == g) & ~np.isnan(Y[i]))
            if len(members) < 3:
                continue
            for _ in range(per_item // 5):
                t = rng.choice(members, 3, replace=False)
                lab = Y[i, t]
                d = {"item": [i], "split": [float(lab.max() != lab.min())]}
                for name, M in S.items():
                    d[name] = [e1j.split_prob(expit(M[i, t]))]
                rows.append(d)
    return pd.DataFrame({k: np.concatenate([r[k] for r in rows]) for k in rows[0]})


def sim(rows, title):
    preds = [c for c in rows.columns if c not in ("item", "split")]
    items = rows.item.unique()
    ipos = rows.item.map({i: k for k, i in enumerate(items)}).values
    y = rows.split.values
    rng = np.random.default_rng(1)

    def metrics(w):
        out = {}
        N, S = w.sum(), (w * y).sum()
        for name in preds + ["random", "oracle"]:
            if name == "random":
                s = rng.random(len(y))
            elif name == "oracle":
                s = y + 1e-9 * rng.random(len(y))
            else:
                s = rows[name].values
            order = np.argsort(-s, kind="stable")
            cw = np.cumsum(w[order])
            for b in BUDGETS:
                top = order[cw <= b * N]
                caught = (w[top] * y[top]).sum()
                out[name, b] = (caught / S, 1 - (S - caught) / 3 / N)
        return out

    pt = metrics(np.ones(len(y)))
    bs = [metrics(np.bincount(rng.integers(0, len(items), len(items)), minlength=len(items)).astype(float)[ipos])
          for _ in range(B)]
    base_cons = 1 - y.mean() / 3
    tab = []
    for name in ["random"] + preds + ["oracle"]:
        for b in BUDGETS:
            s_ = np.array([x[name, b][0] for x in bs])
            c_ = np.array([x[name, b][1] for x in bs])
            tab.append(dict(routing=name, budget=f"{int(b * 100)}%", surfaced=f"{pt[name, b][0]:.3f} "
                            f"[{np.quantile(s_, .025):.3f}, {np.quantile(s_, .975):.3f}]",
                            consistency=f"{pt[name, b][1]:.4f}"))
    tab = pd.DataFrame(tab)
    print(f"\n== {title}: {len(y)} (team, conversation) cases, split rate {y.mean():.3f}; consistency with no review "
          f"{base_cons:.4f}, with universal review 1.0")
    print(tab.pivot(index="routing", columns="budget", values="surfaced").reindex(
        ["random"] + preds + ["oracle"]).to_string())
    print("expected agreement of the final decision with the team majority:")
    print(tab.pivot(index="routing", columns="budget", values="consistency").reindex(
        ["random"] + preds + ["oracle"]).to_string())
    # paired: rater-aware (k = 20) minus rater-blind at each budget
    aware = [p for p in preds if "20" in p]
    blind = [p for p in preds if "k = 0" in p]
    for a, bl in zip(aware, blind):
        for b in BUDGETS:
            d = np.array([x[a, b][0] - x[bl, b][0] for x in bs])
            print(f"  surfaced, {a} - {bl} at {int(b * 100)}%: {pt[a, b][0] - pt[bl, b][0]:+.3f} "
                  f"[{np.quantile(d, .025):+.3f}, {np.quantile(d, .975):+.3f}]")
    tab.insert(0, "dataset", title)
    return tab


def main():
    pd.set_option("display.width", 250)
    out = []
    out.append(sim(e1j.dices350(), "DICES-350 (104 raters, fixed teams)"))
    out.append(sim(e1j.dices990(), "DICES-990 (141 raters, E1i predictions)"))
    rows = teams_from(os.path.join(jev.ROOT, "results", "e4c", "oof.npz"),
                      {"judge only (k = 0), all labels": "all labels|0",
                       "+ intercepts k = 20, all labels": "all labels|20",
                       "judge only (k = 0), single-label logs": "m=1|0",
                       "+ intercepts k = 20, single-label logs": "m=1|20"})
    print("\n== D2 secondary: team-disagreement detection when the shared parts come from singly-labelled logs")
    t, rate = e1j.evaluate(rows, "DICES-990, shared parts from singly-labelled logs")
    print(f"split rate {rate:.3f}\n" + t.to_string(index=False))
    t.to_csv(os.path.join(OUT, "single_label_teams.csv"), index=False)
    pd.concat(out).to_csv(os.path.join(OUT, "venire_sim.csv"), index=False)


if __name__ == "__main__":
    main()
