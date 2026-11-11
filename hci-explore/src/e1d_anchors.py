"""E1d (Idea 1, no API calls): which 20 anchors should a person code, and does using all six answers help?

E1b drew the k anchors at random.  Idea 1's design says they could be picked with JEV's own probabilities.  Fair test:
each repetition splits the 400 comments into a 200-item pool and a fixed 200-item test set; every strategy picks its
k anchors from the same pool and is scored on the same test set (picking hard items must not make the test easier).
  random        k items at random (E1b)
  uncertain     k items drawn from the 2k most uncertain (highest entropy of JEV's p)
  spread        one item drawn at random from each of k equal-size bins of JEV's p (covers the whole score range)
  half-half     k/2 'uncertain' + k/2 'spread'
Calibration = Platt on the soft label (E1b's weighting), MAP with the prior centred on "trust JEV" (slope 1, intercept
0, unit precision), so 10 anchors cannot flip the slope.  Second question, same splits: calibrate each code with all six
JEV answers (prior centred on its own answer) instead of only its own.

  python3 src/e1d_anchors.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import jev  # noqa: E402
from e1_perspectivist import CODES, OUT as E1OUT, ent  # noqa: E402
from e2b_learning_curve import fit_map, logit  # noqa: E402

OUT = os.path.join(jev.ROOT, "results", "e1d")
os.makedirs(OUT, exist_ok=True)
REPS, KS = 200, (10, 20, 40)
STRATS = ["random", "uncertain", "spread", "half-half"]


def pick(strategy, p, pool, k, rng):
    if strategy == "random":
        return rng.choice(pool, k, replace=False)
    if strategy == "uncertain":
        top = pool[np.argsort(-ent(p[pool]))[:2 * k]]
        return rng.choice(top, k, replace=False)
    if strategy == "spread":
        bins = np.array_split(pool[np.argsort(p[pool])], k)
        return np.array([rng.choice(b) for b in bins])
    a = pick("uncertain", p, pool, k // 2, rng)
    rest = np.setdiff1d(pool, a)
    return np.r_[a, pick("spread", p, rest, k - k // 2, rng)]


def platt_soft(X, h, w0, lam):
    """Fit on soft labels: every item once as 'yes' with weight h and once as 'no' with weight 1-h."""
    Xr, yr, wr = np.r_[X, X], np.r_[np.ones(len(h)), np.zeros(len(h))], np.r_[h, 1 - h]
    # fit_map has no sample weights: weights enter by scaling rows of the gradient -> use a small wrapper
    from scipy.optimize import minimize
    k = X.shape[1]
    lower = [0.] + [-np.inf] * (k - 1)

    def f(th):
        w, b = th[:k], th[k]
        m = Xr @ w + b
        r = (expit(m) - yr) * wr
        loss = (wr * (np.logaddexp(0, m) - yr * m)).sum() + (lam * (w - w0) ** 2).sum() / 2 + b * b / 2
        return loss, np.append(Xr.T @ r + lam * (w - w0), r.sum() + b)

    bounds = [(lo if np.isfinite(lo) else None, None) for lo in lower] + [(None, None)]
    th = minimize(f, np.append(w0, 0.), jac=True, method="L-BFGS-B", bounds=bounds).x
    return th[:k], th[k]


def score(p, h):
    maj, cont = (h > .5).astype(int), (h > 0) & (h < 1)
    return [((p > .5) == maj).mean(), np.mean((p - h) ** 2),
            roc_auc_score(cont, ent(p)) if cont.any() and (~cont).any() else np.nan]


def main():
    pd.set_option("display.width", 220)
    s = pd.read_parquet(os.path.join(E1OUT, "sample.parquet")).merge(pd.read_parquet(os.path.join(E1OUT, "jev.parquet")))
    Z = np.c_[tuple(logit(s["j_" + c]) for c in CODES)]
    rows = []
    for rep in range(REPS):
        rng = np.random.default_rng(rep)
        perm = rng.permutation(len(s))
        pool, test = perm[:200], perm[200:]
        for ci, c in enumerate(CODES):
            h, p = s["h_" + c].values, s["j_" + c].values
            rows.append(dict(rep=rep, code=c, k=0, method="JEV raw", **dict(zip(["acc", "brier", "auroc_contested"],
                                                                                  score(p[test], h[test])))))
            for k in KS:
                srng = np.random.default_rng([rep, ci, k])
                for st in STRATS:
                    a = pick(st, p, pool, k, srng)
                    w, b = platt_soft(Z[a, ci:ci + 1], h[a], np.array([1.]), 1.)
                    q = expit(Z[test, ci] * w[0] + b)
                    rows.append(dict(rep=rep, code=c, k=k, method=f"own answer, {st} anchors",
                                     **dict(zip(["acc", "brier", "auroc_contested"], score(q, h[test])))))
                    if st == "random":  # all six answers, prior centred on the code's own answer
                        w0 = np.eye(len(CODES))[ci]
                        lam = np.where(w0 == 1, 1., 4.)
                        w, b = platt_soft(np.c_[Z[a, ci:ci + 1], np.delete(Z[a], ci, 1)], h[a],
                                          np.r_[1., np.zeros(len(CODES) - 1)], np.r_[1., np.full(len(CODES) - 1, 4.)])
                        q = expit(np.c_[Z[test, ci:ci + 1], np.delete(Z[test], ci, 1)] @ w + b)
                        rows.append(dict(rep=rep, code=c, k=k, method="all six answers, random anchors",
                                         **dict(zip(["acc", "brier", "auroc_contested"], score(q, h[test])))))
    r = pd.DataFrame(rows)
    r.to_parquet(os.path.join(OUT, "reps.parquet"))
    mean_codes = r.groupby(["rep", "k", "method"])[["acc", "brier", "auroc_contested"]].mean().reset_index()
    tab = mean_codes.groupby(["k", "method"])[["acc", "brier", "auroc_contested"]].mean()
    tab.to_csv(os.path.join(OUT, "summary.csv"))
    print(f"mean over 6 codes and {REPS} pool/test splits (200/200); test set identical across methods\n" + tab.round(4).to_string())
    base = mean_codes[mean_codes.method == "own answer, random anchors"].set_index(["rep", "k"])
    drows = []
    for mth in mean_codes.method.unique():
        if mth in ("own answer, random anchors", "JEV raw"):
            continue
        o = mean_codes[mean_codes.method == mth].set_index(["rep", "k"])
        for k in KS:
            for met in ("acc", "brier", "auroc_contested"):
                d = (o.loc[(slice(None), k), met] - base.loc[(slice(None), k), met]).values
                better = (d > 0).mean() if met != "brier" else (d < 0).mean()
                drows.append(dict(method=mth, k=k, metric=met, diff=d.mean(), q025=np.quantile(d, .025),
                                  q975=np.quantile(d, .975), share_splits_better=better))
    dif = pd.DataFrame(drows)
    dif.to_csv(os.path.join(OUT, "diffs_vs_random.csv"), index=False)
    print("\nminus 'own answer, random anchors' (same split, same test set); q025/q975 = spread over splits\n" +
          dif.round(4).to_string(index=False))
    per_code = r[r.k == 20].groupby(["code", "method"]).acc.mean().unstack().round(3)
    print("\naccuracy per code, k = 20\n" + per_code.to_string())


if __name__ == "__main__":
    main()
