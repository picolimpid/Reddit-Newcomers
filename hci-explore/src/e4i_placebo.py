"""E4i: negative control for the per-person intercept (PLAN.md D10).  DICES-990, E4c's pipeline with all training
labels, except that each new rater's intercept is estimated from ANOTHER new rater's k anchors and labels (a random
derangement within the same rater fold).  If the gain is person-specific, within-conversation AUROC should fall back
to ~0.5.  Pre-registered: the 95% CI at k = 20 contains 0.5 (or its upper bound is below 0.52).

  python3 -I src/e4i_placebo.py
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

OUT = os.path.join(jev.ROOT, "results", "e4i")
os.makedirs(OUT, exist_ok=True)
KS = e4c.KS
_G = e4c._G


def cell_job(args):
    f, g, placebo = args
    Y, rated, Z, cfold, rfold = (_G[x] for x in ("Y", "rated", "Z", "cfold", "rfold"))
    pool, test = np.flatnonzero(cfold != f), np.flatnonzero(cfold == f)
    trR, teR = np.flatnonzero(rfold != g), np.flatnonzero(rfold == g)
    Yp = Y[np.ix_(pool, trR)]
    mz, sz = Z[pool].mean(0), Z[pool].std(0) + 1e-8
    Zs = (Z - mz) / sz
    w, c, u_tr, tau2 = e4c.fit_joint(Zs[pool], Yp)
    b_test = Zs[test] @ w + c
    inner = np.random.default_rng([f, g, 3]).permutation(len(pool)) % e1f.NIN
    b_jev = np.empty(len(pool))
    for h in range(e1f.NIN):
        wh, ch = e1f.fit_content(Zs[pool][inner != h], Yp[inner != h], u_tr)
        b_jev[inner == h] = Zs[pool][inner == h] @ wh + ch
    # derangement of the test raters: donor(ri) != ri
    rng = np.random.default_rng([f, g, 77])
    while True:
        perm = rng.permutation(len(teR))
        if not np.any(perm == np.arange(len(teR))):
            break
    res = np.full((e1f.R, len(KS), len(test), len(teR)), np.nan)
    for ri, r in enumerate(teR):
        donor = teR[perm[ri]] if placebo else r
        yr, cand = Y[pool, donor], rated[pool, donor]
        for rep in range(e1f.R):
            order = np.random.default_rng([f, g, int(donor), rep, 11]).permutation(np.flatnonzero(cand))
            for ki, k in enumerate(KS):
                a = [int(i) for i in order[:k] if not np.isnan(yr[i])]
                u, v = e1f.post_u(b_jev[a], yr[a], 0., tau2) if a else (0., tau2)
                res[rep, ki, :, ri] = (b_test + u) / np.sqrt(1 + np.pi * v / 8)
    return f, g, placebo, test, teR, res


def main():
    pd.set_option("display.width", 200)
    G = e1i.build("filtered")
    I, P = G["I"], G["P"]
    cfold = np.empty(I, int)
    cfold[np.random.default_rng(0).permutation(I)] = np.arange(I) % 5
    rfold = np.empty(P, int)
    rfold[np.random.default_rng(2).permutation(P)] = np.arange(P) % 5
    _G.update(G, cfold=cfold, rfold=rfold)
    jobs = [(f, g, pl) for pl in (False, True) for f in range(5) for g in range(5)]
    with ProcessPoolExecutor(25, mp_context=mp.get_context("fork"), initializer=e1f._init_worker) as ex:
        out = list(ex.map(cell_job, jobs))
    M = {}
    for f, g, pl, test, teR, res in out:
        for rep in range(e1f.R):
            for ki, k in enumerate(KS):
                M.setdefault((pl, k, rep), np.full((I, P), np.nan))[np.ix_(test, teR)] = res[rep, ki]
    lab = ~np.isnan(G["Y"])
    brng = np.random.default_rng(1)
    C = np.stack([np.bincount(brng.integers(0, I, I), minlength=I) for _ in range(e1f.B)]).astype(float)
    W = np.stack([np.bincount(brng.integers(0, P, P), minlength=P) for _ in range(e1f.B)]).astype(float)
    rows = []
    for pl in (False, True):
        for k in KS:
            reps = [e1f.boot_stats(np.where(lab, M[pl, k, rep], 0.), G["Y"], C, W, rfold) for rep in range(e1f.R)]
            pt, lb, wb = (np.mean([x[j] for x in reps], 0) for j in range(3))
            rows.append(dict(anchors="someone else's (placebo)" if pl else "own", k=k, within_auroc=round(pt[3], 4),
                             lo=round(np.quantile(wb, .025), 4), hi=round(np.quantile(wb, .975), 4),
                             logloss=round(pt[0], 4)))
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    r = t[(t.k == 20) & (t.anchors != "own")].iloc[0]
    print(f"pre-registered: placebo k = 20 CI [{r.lo}, {r.hi}] contains 0.5 or upper < 0.52 -> "
          f"{'MET' if (r.lo <= .5 <= r.hi) or r.hi < .52 else 'NOT met'}")
    t.to_csv(os.path.join(OUT, "placebo.csv"), index=False)


if __name__ == "__main__":
    main()
