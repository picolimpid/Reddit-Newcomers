"""E4j: replicate D2 (single-label logs), E1j (team disagreement) and D10 (negative control) on POPQUORN with the local
Qwen3-32B judge (PLAN.md D11, pre-registered).  Reuses E4h's data/judge and E4c's / E4i's cell jobs.

  python3 -I src/e4j_popquorn_more.py
"""
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import expit  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import jev  # noqa: E402
import e1f_onboarding as e1f  # noqa: E402
import e1j_disagreement as e1j  # noqa: E402
import e4c_single_label as e4c  # noqa: E402
import e4h_popquorn as e4h  # noqa: E402
import e4i_placebo as e4i  # noqa: E402

OUT = os.path.join(jev.ROOT, "results", "e4j")
os.makedirs(OUT, exist_ok=True)
KS = e4c.KS


def job(args):
    kind, f, g = args
    if kind == "placebo":
        f_, g_, pl, test, teR, res = e4i.cell_job((f, g, True))
        return kind, test, teR, res
    f_, g_, cond, test, teR, res, info = e4c.cell_job((f, g, kind))
    return kind, test, teR, res


def main():
    pd.set_option("display.width", 220)
    Y, Z, demo, items, raters = e4h.build()
    I, P = Y.shape
    lab = ~np.isnan(Y)
    cfold = np.empty(I, int)
    cfold[np.random.default_rng(0).permutation(I)] = np.arange(I) % 5
    rfold = np.empty(P, int)
    rfold[np.random.default_rng(2).permutation(P)] = np.arange(P) % 5
    e4c._G.update(Y=Y, rated=lab, Z=Z, cfold=cfold, rfold=rfold)
    kinds = ["all labels", "m=1", "placebo"]
    jobs = [(k, f, g) for k in kinds for f in range(5) for g in range(5)]
    with ProcessPoolExecutor(25, mp_context=mp.get_context("fork"), initializer=e1f._init_worker) as ex:
        out = list(ex.map(job, jobs))
    M = {}
    for kind, test, teR, res in out:
        for rep in range(e1f.R):
            for ki, k in enumerate(KS):
                M.setdefault((kind, k, rep), np.full((I, P), np.nan))[np.ix_(test, teR)] = res[rep, ki]
    brng = np.random.default_rng(1)
    C = np.stack([np.bincount(brng.integers(0, I, I), minlength=I) for _ in range(e1f.B)]).astype(float)
    W = np.stack([np.bincount(brng.integers(0, P, P), minlength=P) for _ in range(e1f.B)]).astype(float)
    st = {}
    for kind in kinds:
        for k in KS:
            reps = [e1f.boot_stats(np.where(lab, M[kind, k, rep], 0.), Y, C, W, rfold) for rep in range(e1f.R)]
            st[kind, k] = tuple(np.mean([x[j] for x in reps], 0) for j in range(3))
    rows = [dict(training=kind, k=k, within_auroc=round(pt[3], 4), lo=round(np.quantile(wb, .025), 4),
                 hi=round(np.quantile(wb, .975), 4), logloss=round(pt[0], 4))
            for (kind, k), (pt, lb, wb) in st.items()]
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    pa, la, wa = st["m=1", 20]
    pb, lb, wb = st["all labels", 20]
    d = wa - wb
    lo1 = np.quantile(d, .025)
    print(f"\n(1) single-label - all labels, k = 20: {pa[3] - pb[3]:+.4f} [{lo1:+.4f}, {np.quantile(d, .975):+.4f}] -> "
          f"{'MET' if lo1 > -.02 else 'NOT met'}")
    r = t[(t.training == "placebo") & (t.k == 20)].iloc[0]
    print(f"(3) placebo k = 20: {r.within_auroc} [{r.lo}, {r.hi}] -> "
          f"{'MET' if (r.lo <= .5 <= r.hi) or r.hi < .52 else 'NOT met'}")
    # (2) teams of 3 annotators who labelled the same comment (any folds; every prediction is out-of-fold)
    rng = np.random.default_rng(3)
    S = {"judge only (k = 0)": M["all labels", 0, 0], "+ per-person offsets, k = 5": M["all labels", 5, 0],
         "+ per-person offsets, k = 20": M["all labels", 20, 0]}
    recs = []
    for i in range(I):
        members = np.flatnonzero(lab[i])
        if len(members) < 3:
            continue
        for _ in range(30):
            tm = rng.choice(members, 3, replace=False)
            yy = Y[i, tm]
            rec = {"item": i, "split": float(yy.max() != yy.min())}
            for name, Mm in S.items():
                rec[name] = e1j.split_prob(expit(Mm[i, tm]))
            recs.append(rec)
    rows2 = pd.DataFrame(recs)
    tab, rate = e1j.evaluate(rows2, "POPQUORN")
    print(f"\n(2) team disagreement: {len(rows2)} (team, comment) cases, split rate {rate:.3f}\n" + tab.to_string(index=False))
    t.to_csv(os.path.join(OUT, "curves.csv"), index=False)
    tab.to_csv(os.path.join(OUT, "teams.csv"), index=False)


if __name__ == "__main__":
    main()
