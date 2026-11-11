"""E1g: use every answer a new rater gives on an anchor.  DICES raters answer five harm categories plus the overall
verdict for each conversation (overall = "yes" exactly when some category is "yes"), so an anchor carries six
answers, not one.  A person gets a six-dimensional offset (one per question) whose prior covariance is learned from
the training raters -- a meta-learned prior: how strictness on one question goes with strictness on another.  With
it, the five category answers on an anchor also sharpen the estimate of the person's overall offset.

Same design as E1f (src/e1f_onboarding.py): DICES-350, 5 x 5 two-way CV (conversations x raters), the same anchor
orders, pairs for the within-conversation AUROC formed inside a rater fold.  Target: the overall verdict on new
conversations.  Pre-registered in PLAN.md (A1b): within-conversation AUROC at k = 5, "random | jev | six answers"
minus "random | jev | overall only" >= +0.02 with a 95% CI excluding 0.

  python3 -I src/e1g_multianswer.py [--all-raters]
"""
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import expit  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import e1e_dices as e1e  # noqa: E402
import e1f_onboarding as e1f  # noqa: E402

OUT = os.path.join(e1e.jev.ROOT, "results", "e1g")
os.makedirs(OUT, exist_ok=True)
NF, NIN, B, R, KS = e1f.NF, e1f.NIN, e1f.B, e1f.R, e1f.KS
QS = list(e1e.QUESTIONS)  # harmful, bias, misinfo, political, policy, overall
OV = QS.index("overall")
VARIANTS = [  # (strategy, difficulties, answers used)
    ("random", "crowd", "overall"), ("random", "crowd", "six"),
    ("random", "jev", "overall"), ("random", "jev", "six"),
    ("CAT-crowd", "crowd", "overall"), ("CAT-crowd", "crowd", "six"),
    ("CAT-JEV", "jev", "overall"), ("CAT-JEV", "jev", "six"),
]
NAMES = [f"{s} | {d} difficulties | {'six answers' if a == 'six' else 'overall answer only'}" for s, d, a in VARIANTS]
_G = {}


def build_multi(raters):
    df, items = e1e.load()
    if raters == "104":
        df = df[~df.rater_id.isin(e1e.REMOVED)]
    ipos = {i: n for n, i in enumerate(items.index)}
    rids = sorted(df.rater_id.unique())
    rpos = {r: n for n, r in enumerate(rids)}
    Yq = np.full((len(QS), len(items), len(rids)), np.nan)
    ii, rr = df.item_id.map(ipos).values, df.rater_id.map(rpos).values
    for qi, q in enumerate(QS):
        Yq[qi, ii, rr] = df[e1e.TARGETS[q]].map({"Yes": 1., "No": 0.}).values
    return Yq


def post_U(Bd, Ya, Sinv):
    """Laplace posterior of a person's Q-dim offset, prior N(0, S), given anchors' difficulties Bd and answers Ya
    (n x Q, NaN = no information).  Damped Newton on a strictly convex objective."""
    m, y = ~np.isnan(Ya), np.nan_to_num(Ya)
    obj = lambda u: np.where(m, np.logaddexp(0, Bd + u) - y * (Bd + u), 0.).sum() + .5 * u @ Sinv @ u
    u = np.zeros(Bd.shape[1])
    f0 = obj(u)
    for _ in range(100):
        p = expit(Bd + u)
        g = np.where(m, p - y, 0.).sum(0) + Sinv @ u
        H = np.diag(np.where(m, p * (1 - p), 0.).sum(0)) + Sinv
        d = np.linalg.solve(H, g)
        t = 1.
        while True:
            f1 = obj(u - t * d)
            if f1 <= f0 - 1e-4 * t * (g @ d) or t < 1e-10:
                break
            t *= .5
        u, f0 = u - t * d, f1
        if np.abs(t * d).max() < 1e-10:
            break
    p = expit(Bd + u)
    H = np.diag(np.where(m, p * (1 - p), 0.).sum(0)) + Sinv
    return u, np.linalg.inv(H)


def cell_job(args):
    f, g = args
    Yq, Z, cfold, rfold = (_G[x] for x in ("Yq", "Z", "cfold", "rfold"))
    Q = len(QS)
    pool, test = np.flatnonzero(cfold != f), np.flatnonzero(cfold == f)
    trR, teR = np.flatnonzero(rfold != g), np.flatnonzero(rfold == g)
    mz, sz = Z[pool].mean(0), Z[pool].std(0) + 1e-8
    Zs = (Z - mz) / sz
    inner = np.random.default_rng([f, g, 3]).permutation(len(pool)) % NIN  # E1f's inner folds
    bc, bj, U, noise = np.zeros((len(pool), Q)), np.zeros((len(pool), Q)), np.zeros((len(trR), Q)), np.zeros(Q)
    for qi in range(Q):
        Yp = Yq[qi][np.ix_(pool, trR)]
        bc[:, qi], U[:, qi] = e1f.fit_rasch(Yp)
        pc = expit(bc[:, qi][:, None] + U[:, qi][None, :])
        noise[qi] = np.mean(1 / (np.where(~np.isnan(Yp), pc * (1 - pc), 0.).sum(0) + e1f.LAM_U))
        for h in range(NIN):
            wh, ch = e1f.fit_content(Zs[pool][inner != h], Yp[inner != h], U[:, qi])
            bj[inner == h, qi] = Zs[pool][inner == h] @ wh + ch
        if qi == OV:
            w, c = e1f.fit_content(Zs[pool], Yp, U[:, qi])
            b_test = Zs[test] @ w + c
    S = np.cov(U.T) - np.diag(noise)  # covariance of true offsets = observed minus estimation noise
    ev, V = np.linalg.eigh(S)
    S = (V * np.maximum(ev, .01)) @ V.T
    Sinv = np.linalg.inv(S)
    tau2 = S[OV, OV]
    s2 = float(np.var(bc[:, OV] - bj[:, OV]))
    res = {(vi, rep): np.full((len(KS), len(test), len(teR)), np.nan) for vi, (s, _, _) in enumerate(VARIANTS)
           for rep in range(R if s == "random" else 1)}
    for ri, r in enumerate(teR):
        Ya_all = Yq[:, pool, r].T  # pool x Q
        tie = np.random.default_rng([f, g, int(r), 5]).random(len(pool))
        for vi, (strat, dif, ans) in enumerate(VARIANTS):
            Bd = bc if dif == "crowd" else bj

            def estimate(asked):
                a = np.array(asked, int)
                if ans == "overall":
                    lab = a[~np.isnan(Ya_all[a, OV])]
                    return e1f.post_u(Bd[lab, OV], Ya_all[lab, OV], 0., tau2) if len(lab) else (0., tau2)
                if len(a) == 0:
                    return 0., tau2
                u, C = post_U(Bd[a], Ya_all[a], Sinv)
                return u[OV], C[OV, OV]

            for rep in range(R if strat == "random" else 1):
                if strat == "random":
                    order = np.random.default_rng([f, g, int(r), rep, 11]).permutation(len(pool))  # E1f's orders
                    ests = [estimate(order[:k]) for k in KS]
                else:  # adaptive: most informative remaining conversation for the overall verdict
                    asked, avail, ests = [], np.ones(len(pool), bool), []
                    u, v = 0., tau2
                    for k in range(max(KS) + 1):
                        if k in KS:
                            ests.append((u, v))
                        if k == max(KS):
                            break
                        p = expit(Bd[:, OV] + u)
                        i = int(np.argmax(np.where(avail, p * (1 - p), -1.) + 1e-12 * tie))
                        avail[i] = False
                        asked.append(i)
                        u, v = estimate(asked)
                for ki, (u, v) in enumerate(ests):
                    res[vi, rep][ki, :, ri] = (b_test + u) / np.sqrt(1 + np.pi * (s2 + v) / 8)
    corr = S / np.sqrt(np.outer(np.diag(S), np.diag(S)))
    return f, g, test, teR, res, dict(f=f, g=g, tau2=tau2, s2=s2, corr_overall=corr[OV].round(2).tolist())


def main(raters):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    G = e1e.build(raters)
    Yq = build_multi(raters)
    assert np.array_equal(np.isnan(Yq[OV]), np.isnan(G["Y"])) and np.allclose(np.nan_to_num(Yq[OV]), np.nan_to_num(G["Y"]))
    I, P = G["I"], G["P"]
    cfold = np.empty(I, int)
    cfold[np.random.default_rng(0).permutation(I)] = np.arange(I) % NF
    rfold = np.empty(P, int)
    rfold[np.random.default_rng(2).permutation(P)] = np.arange(P) % NF
    _G.update(Yq=Yq, Z=G["Z"], cfold=cfold, rfold=rfold)
    print(f"DICES-350, raters = {raters}: {P} x {I}; questions {QS}; yes rates "
          + ", ".join(f"{q} {np.nanmean(Yq[i]):.3f}" for i, q in enumerate(QS)))
    t0 = time.time()
    jobs = [(f, g) for f in range(NF) for g in range(NF)]
    with ProcessPoolExecutor(len(jobs), mp_context=mp.get_context("fork"), initializer=e1f._init_worker) as ex:
        out = list(ex.map(cell_job, jobs))
    print(f"{len(jobs)} cells in {time.time() - t0:.0f}s")
    M = {}
    infos = []
    for f, g, test, teR, res, info in out:
        for (vi, rep), arr in res.items():
            for ki, k in enumerate(KS):
                M.setdefault((vi, k, rep), np.full((I, P), np.nan))[np.ix_(test, teR)] = arr[ki]
        infos.append(info)
    cm = np.mean([i["corr_overall"] for i in infos], 0)
    print("learned prior: correlation of the overall offset with each question's offset (mean over cells): "
          + ", ".join(f"{q} {c:.2f}" for q, c in zip(QS, cm)))
    brng = np.random.default_rng(1)
    C = np.stack([np.bincount(brng.integers(0, I, I), minlength=I) for _ in range(B)]).astype(float)
    W = np.stack([np.bincount(brng.integers(0, P, P), minlength=P) for _ in range(B)]).astype(float)
    st = {}
    for vi, (s, _, _) in enumerate(VARIANTS):
        for k in KS:
            reps = [e1f.boot_stats(M[vi, k, rep], G["Y"], C, W, rfold) for rep in range(R if s == "random" else 1)]
            st[vi, k] = tuple(np.mean([x[j] for x in reps], 0) for j in range(3))
    rows = [dict(model=NAMES[vi], k=k, within_auroc=pt[3], wi_lo=np.quantile(wib, .025), wi_hi=np.quantile(wib, .975),
                 logloss=pt[0], auroc=pt[2]) for (vi, k), (pt, llb, wib) in st.items()]
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, f"curve_{raters}.csv"), index=False)
    piv = lambda col: tab.pivot(index="model", columns="k", values=col).reindex(NAMES)
    print("\n== within-conversation AUROC (pairs within a rater fold)\n" + piv("within_auroc").round(3).to_string())
    print("\n== log loss\n" + piv("logloss").round(4).to_string())
    pairs = [(3, 2), (1, 0), (5, 4), (7, 6), (5, 0), (7, 2), (3, 0)]  # first = pre-registered primary
    drows = []
    for a, b in pairs:
        for k in KS[1:]:
            pa, la, wa = st[a, k]
            pb, lb, wb = st[b, k]
            drows.append(dict(pair=f"{NAMES[a]}  -  {NAMES[b]}", k=k, d_within=pa[3] - pb[3],
                              wi_lo=np.quantile(wa - wb, .025), wi_hi=np.quantile(wa - wb, .975),
                              d_logloss=pa[0] - pb[0], ll_lo=np.quantile(la - lb, .025), ll_hi=np.quantile(la - lb, .975)))
    dif = pd.DataFrame(drows)
    dif.to_csv(os.path.join(OUT, f"diffs_{raters}.csv"), index=False)
    dif["within-AUROC diff [95% CI]"] = [f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]" for d, lo, hi in zip(dif.d_within, dif.wi_lo, dif.wi_hi)]
    dif["log loss diff [95% CI]"] = [f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]" for d, lo, hi in zip(dif.d_logloss, dif.ll_lo, dif.ll_hi)]
    print("\n== paired differences (two-way bootstrap; the first pair is the pre-registered primary comparison)\n" +
          dif[["pair", "k", "within-AUROC diff [95% CI]", "log loss diff [95% CI]"]].to_string(index=False))


if __name__ == "__main__":
    main("all" if "--all-raters" in sys.argv else "104")
