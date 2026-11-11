"""E12 (2026-11-06, exploratory): do agents know *who* returns? r/science 2016 newcomers, 90-day return.

Baselines (logistic regression, 5-fold CV grouped by post, 5 repeats):
  counts     9 activity features (history size, activity per window, number of subreddits, comment length, entry delay,
             top-level)
  text       counts + bge-small embeddings of the first comment and of the mean history item (PCA to 32 dims inside the fold)
Agents: the panel model's P(Yes) in the no-sticky condition, alone and added to counts / counts + text.
AUC CIs: post-cluster bootstrap of the out-of-fold predictions (1,000 resamples).

    python3 who_returns.py <embedding dir with first.npy, hist.npy>   -> results/panel/who_returns.md"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import RESULTS
from panel_summary import SKIP, inputs

R = inputs("rscience")
y = R.return_90d.astype(int).values
g = pd.factorize(R.post_id)[0]
cnt = lambda c: R[c].map(lambda s: sum(json.loads(s).values()) if s else 0)  # noqa: E731
COUNTS = np.column_stack([np.log1p(R.n_hist), np.log1p(R.first_body.str.len()), np.log1p(R.entry_delay_min.clip(lower=0)),
                          R.top_level, *[np.log1p(cnt(c)) for c in ["subs_0_30", "subs_30_90", "subs_90_365"]],
                          np.log1p(R.subs_90_365.map(lambda s: len(json.loads(s)) if s else 0))])


def oof(blocks, idx, reps=5):
    """out-of-fold P averaged over repeats; blocks = list of (array, 'pca' or None)."""
    P = np.zeros(len(idx))
    for r in range(reps):
        rng = np.random.RandomState(r)
        perm = rng.permutation(g.max() + 1)
        for tr, te in GroupKFold(5).split(idx, y[idx], perm[g[idx]]):
            Xtr, Xte = [], []
            for B, kind in blocks:
                a, b = B[idx][tr], B[idx][te]
                if kind == "pca":
                    p = PCA(32, random_state=0).fit(a)
                    a, b = p.transform(a), p.transform(b)
                Xtr.append(a); Xte.append(b)
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))
            m.fit(np.hstack(Xtr), y[idx][tr])
            P[te] += m.predict_proba(np.hstack(Xte))[:, 1] / reps
    return P


def auc_ci(p, idx, B=1000):
    yy, gg = y[idx], g[idx]
    groups = pd.Series(np.arange(len(idx))).groupby(gg).indices
    keys = list(groups)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(B):
        ix = np.concatenate([groups[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        if yy[ix].min() != yy[ix].max():
            bs.append(roc_auc_score(yy[ix], p[ix]))
    return roc_auc_score(yy, p), *np.percentile(bs, [2.5, 97.5])


def main(emb):
    F, H = np.load(os.path.join(emb, "first.npy")), np.load(os.path.join(emb, "hist.npy"))
    idx_all = np.arange(len(R))
    rows = []
    base = {"counts": oof([(COUNTS, None)], idx_all), "counts + text": oof([(COUNTS, None), (F, "pca"), (H, "pca")], idx_all)}
    for k, p in base.items():
        rows.append({"predictor": k, **dict(zip(["AUC", "lo", "hi"], auc_ci(p, idx_all)))})
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
    for f in sorted(glob.glob(f"{RESULTS}/probes/rscience-*/scores.parquet")):
        name = os.path.basename(os.path.dirname(f))[len("rscience-"):]
        if any(x in name for x in SKIP):
            continue
        S = pd.read_parquet(f)
        if "off" not in set(S.condition):
            continue
        a = S[S.condition == "off"].set_index("idx").p.reindex(R.index)
        ok = a.notna().values
        A = np.zeros((len(R), 1)); A[ok, 0] = lg(a.values[ok])
        idx = idx_all[ok]
        rows.append({"predictor": f"agent {name} alone", **dict(zip(["AUC", "lo", "hi"], auc_ci(a.values[ok], idx)))})
        for k, blocks in [("counts", [(COUNTS, None)]), ("counts + text", [(COUNTS, None), (F, "pca"), (H, "pca")])]:
            p0, p1 = oof(blocks, idx), oof(blocks + [(A, None)], idx)
            r0, r1 = auc_ci(p0, idx), auc_ci(p1, idx)
            rows.append({"predictor": f"{k} + agent {name}", "AUC": r1[0], "lo": r1[1], "hi": r1[2], "Δ vs " + k: r1[0] - r0[0]})
    D = pd.DataFrame(rows)
    txt = ("# Who returns? r/science 2016 newcomers, 90-day return (exploratory; E12)\n\n"
           f"n = {len(R)}, base rate {y.mean():.3f}. 5-fold CV grouped by post, averaged over 5 repeats; AUC CI by post bootstrap.\n\n"
           + D.round(3).to_markdown(index=False) + "\n")
    open(os.path.join(RESULTS, "panel", "who_returns.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main(sys.argv[1])
