"""E4k: sensitivity of the per-person intercept to the prior variance tau2 (PLAN.md D12).  DICES-990, E4c's pipeline
with all training labels; the fitted tau2 is multiplied by 0.25, 0.5, 1, 2, 4 before new raters' anchors are used.
Criterion (descriptive): range of within-conversation AUROC at k = 20 across the five scales < 0.01.

  python3 -I src/e4k_prior_sensitivity.py
"""
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import jev  # noqa: E402
import e1f_onboarding as e1f  # noqa: E402
import e1i_dices990 as e1i  # noqa: E402
import e4c_single_label as e4c  # noqa: E402

OUT = os.path.join(jev.ROOT, "results", "e4k")
os.makedirs(OUT, exist_ok=True)
SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
_fit = e4c.fit_joint


def job(args):
    scale, f, g = args
    def scaled(X, Yc, iters=15):
        w, c, u, t2 = _fit(X, Yc, iters)
        return w, c, u, t2 * scale
    e4c.fit_joint = scaled
    out = e4c.cell_job((f, g, "all labels"))
    e4c.fit_joint = _fit
    return (scale,) + out


def main():
    pd.set_option("display.width", 200)
    G = e1i.build("filtered")
    I, P = G["I"], G["P"]
    cfold = np.empty(I, int)
    cfold[np.random.default_rng(0).permutation(I)] = np.arange(I) % 5
    rfold = np.empty(P, int)
    rfold[np.random.default_rng(2).permutation(P)] = np.arange(P) % 5
    e4c._G.update(G, cfold=cfold, rfold=rfold)
    jobs = [(s, f, g) for s in SCALES for f in range(5) for g in range(5)]
    with ProcessPoolExecutor(25, mp_context=mp.get_context("fork"), initializer=e1f._init_worker) as ex:
        out = list(ex.map(job, jobs))
    M = {}
    for scale, f, g, cond, test, teR, res, info in out:
        for rep in range(e1f.R):
            for ki, k in enumerate(e4c.KS):
                M.setdefault((scale, k, rep), np.full((I, P), np.nan))[np.ix_(test, teR)] = res[rep, ki]
    lab = ~np.isnan(G["Y"])
    brng = np.random.default_rng(1)
    C = np.stack([np.bincount(brng.integers(0, I, I), minlength=I) for _ in range(200)]).astype(float)
    W = np.stack([np.bincount(brng.integers(0, P, P), minlength=P) for _ in range(200)]).astype(float)
    rows = []
    for s in SCALES:
        for k in e4c.KS:
            reps = [e1f.boot_stats(np.where(lab, M[s, k, rep], 0.), G["Y"], C, W, rfold) for rep in range(e1f.R)]
            pt = np.mean([x[0] for x in reps], 0)
            rows.append(dict(tau2_scale=s, k=k, within_auroc=round(pt[3], 4), pooled_auroc=round(pt[2], 4),
                             logloss=round(pt[0], 4)))
    t = pd.DataFrame(rows)
    print(t.pivot(index="tau2_scale", columns="k", values="within_auroc").to_string())
    print(t.pivot(index="tau2_scale", columns="k", values="logloss").to_string())
    r = t[t.k == 20].within_auroc
    print(f"range of within-conversation AUROC at k = 20: {r.max() - r.min():.4f} -> {'MET' if r.max() - r.min() < .01 else 'NOT met'}")
    t.to_csv(os.path.join(OUT, "sensitivity.csv"), index=False)


if __name__ == "__main__":
    main()
