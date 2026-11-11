"""E1f: onboarding a new rater.  Which k calibration conversations should a new person label, and how should their
answers be used?  DICES-350 (Aroyo et al., NeurIPS D&B 2023), two-way cross-validation: 5 folds over conversations x
5 folds over raters (the people being onboarded are never seen in training).

For a cell (f, g): the 280 pool conversations (conversation fold != f) labelled by the training raters (rater fold
!= g) train everything shared -- JEV's calibration, each pool conversation's crowd difficulty (Rasch model), the
prior over people.  Each test rater (rater fold g) answers k pool conversations picked by a strategy ("Unsure" uses
up the question and carries no information); the model is scored on that rater's labels for the 70 test
conversations, which nobody labelled before (the deployment case of Ideas 1 and 2).

Model:  logit P(rater r calls conversation i unsafe) = beta_i + u_r,   u_r ~ N(mu_r, tau^2)
  beta_i of a pool conversation: the crowd estimate (Rasch fit on the training raters), or, when no crowd labels
  exist, JEV's prediction of it (content model on JEV's six answers, cross-fitted over pool conversations).
  beta_j of a new conversation: JEV's prediction.  Predictions integrate the uncertainty in beta_j and u_r (probit
  approximation), so log loss is meaningful.
Strategies:
  random          each person their own random k (E1e's design)
  random-common   the same random k for everyone in the cell
  disputed-crowd  the k pool conversations the training raters split most evenly on (static)
  CAT-crowd       adaptive testing: next = argmax p(1-p) at the person's current estimate, crowd difficulties
  borderline-JEV  the k pool conversations whose JEV-predicted difficulty is closest to a coin flip (static)
  CAT-JEV         adaptive testing with JEV-predicted difficulties only (no crowd labels needed anywhere)
"+ demographics" variants centre the prior on a ridge regression of training raters' offsets on demographics.

  python3 -I src/e1f_onboarding.py [--all-raters]
"""
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # workers are forked
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import brentq, minimize  # noqa: E402
from scipy.special import expit  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import e1e_dices as e1e  # noqa: E402

OUT = os.path.join(e1e.jev.ROOT, "results", "e1f")
os.makedirs(OUT, exist_ok=True)
NF, NIN, B, R = 5, 4, 1000, 5     # folds (conversations and raters), inner folds, bootstrap draws, random repeats
KS = (0, 1, 2, 3, 5, 10, 20, 50)
LAM_B, LAM_U, LAM_W, LAM_D = .05, .05, 1., 4.
VARIANTS = [  # (strategy, difficulties used to pick and to estimate, demographic prior)
    ("random", "jev", False), ("random", "crowd", False),
    ("random-common", "jev", False), ("random-common", "crowd", False),
    ("disputed-crowd", "crowd", False), ("CAT-crowd", "crowd", False),
    ("borderline-JEV", "jev", False), ("CAT-JEV", "jev", False),
    ("random", "jev", True), ("CAT-crowd", "crowd", True), ("CAT-JEV", "jev", True),
]
NAMES = [f"{s} | {e} difficulties" + (" + demographics" if d else "") for s, e, d in VARIANTS]
_G = {}


def fit_rasch(Yc):
    """Joint MAP of logit P = beta_i + u_r on an items x raters matrix with NaN; returns (beta, u) with mean(u) = 0."""
    mask, y = ~np.isnan(Yc), np.nan_to_num(Yc)
    I, P = Yc.shape

    def f(th):
        M = th[:I, None] + th[None, I:]
        G = np.where(mask, expit(M) - y, 0.)
        L = np.where(mask, np.logaddexp(0, M) - y * M, 0.).sum() + LAM_B / 2 * th[:I] @ th[:I] + LAM_U / 2 * th[I:] @ th[I:]
        return L, np.r_[G.sum(1) + LAM_B * th[:I], G.sum(0) + LAM_U * th[I:]]

    th = minimize(f, np.zeros(I + P), jac=True, method="L-BFGS-B", options={"maxiter": 5000}).x
    b, u = th[:I], th[I:]
    return b + u.mean(), u - u.mean()


def fit_content(X, Yc, u):
    """logit P(y_ri) = X_i w + c + u_r with the raters' offsets u fixed; returns (w, c)."""
    mask, y = ~np.isnan(Yc), np.nan_to_num(Yc)
    k = X.shape[1]

    def f(th):
        M = (X @ th[:k] + th[k])[:, None] + u[None, :]
        G = np.where(mask, expit(M) - y, 0.).sum(1)
        L = np.where(mask, np.logaddexp(0, M) - y * M, 0.).sum() + LAM_W / 2 * th[:k] @ th[:k]
        return L, np.r_[X.T @ G + LAM_W * th[:k], G.sum()]

    th = minimize(f, np.zeros(k + 1), jac=True, method="L-BFGS-B", options={"maxiter": 5000}).x
    return th[:k], th[k]


def post_u(b, y, mu, t2):
    """Laplace posterior (mean, variance) of a person's offset given anchors with difficulties b and labels y.
    The mode is the root of a strictly increasing gradient, bracketed by mu +- t2 * n (plain Newton diverged for
    raters who call almost everything unsafe; found in E1f's first run)."""
    grad = lambda u: (expit(b + u) - y).sum() + (u - mu) / t2
    w = t2 * len(b) + 1e-6
    u = brentq(grad, mu - w, mu + w, xtol=1e-10)
    p = expit(b + u)
    return u, 1 / ((p * (1 - p)).sum() + 1 / t2)


def run_person(order_or_adaptive, b_est, yr, mu, t2, tie):
    """Ask up to max(KS) questions; returns [(u, v, asked items)] at each k in KS."""
    out, asked = [], []
    u, v = mu, t2
    avail = np.ones(len(yr), bool)
    for k in range(max(KS) + 1):
        if k in KS:
            out.append((u, v, list(asked)))
        if k == max(KS):
            break
        if isinstance(order_or_adaptive, str):  # adaptive: most informative remaining question at the current u
            p = expit(b_est + u)
            score = np.where(avail, p * (1 - p), -1.) + 1e-12 * tie
            i = int(np.argmax(score))
        else:
            i = int(order_or_adaptive[k])
        avail[i] = False
        asked.append(i)
        lab = np.array([a for a in asked if not np.isnan(yr[a])], int)
        u, v = post_u(b_est[lab], yr[lab], mu, t2) if len(lab) else (mu, t2)
    return out


def cell_job(args):
    f, g = args
    Y, Z, D, cfold, rfold = (_G[x] for x in ("Y", "Z", "D", "cfold", "rfold"))
    pool, test = np.flatnonzero(cfold != f), np.flatnonzero(cfold == f)
    trR, teR = np.flatnonzero(rfold != g), np.flatnonzero(rfold == g)
    Yp = Y[np.ix_(pool, trR)]
    b_crowd, u_tr = fit_rasch(Yp)
    pc = expit(b_crowd[:, None] + u_tr[None, :])
    noise = 1 / (np.where(~np.isnan(Yp), pc * (1 - pc), 0.).sum(0) + LAM_U)
    tau2 = max(u_tr.var() - noise.mean(), .05)
    Dc = D[trR] - D[trR].mean(0)
    gam = np.linalg.solve(Dc.T @ Dc + LAM_D * np.eye(D.shape[1]), Dc.T @ u_tr)
    mu_demo = (D - D[trR].mean(0)) @ gam
    tau2_demo = max((u_tr - mu_demo[trR]).var() - noise.mean(), .05)
    mz, sz = Z[pool].mean(0), Z[pool].std(0) + 1e-8
    Zs = (Z - mz) / sz
    w, c = fit_content(Zs[pool], Yp, u_tr)
    b_test = Zs[test] @ w + c
    inner = np.random.default_rng([f, g, 3]).permutation(len(pool)) % NIN
    b_jev = np.empty(len(pool))
    for h in range(NIN):
        wh, ch = fit_content(Zs[pool][inner != h], Yp[inner != h], u_tr)
        b_jev[inner == h] = Zs[pool][inner == h] @ wh + ch
    s2 = float(np.var(b_crowd - b_jev))  # what JEV cannot see about a new conversation's difficulty
    p0c, p0j = expit(b_crowd), expit(b_jev)
    static = {"disputed-crowd": np.argsort(-(p0c * (1 - p0c)), kind="stable"),
              "borderline-JEV": np.argsort(-(p0j * (1 - p0j)), kind="stable")}
    res = {(vi, rep): np.full((len(KS), len(test), len(teR)), np.nan) for vi, (s, _, _) in enumerate(VARIANTS)
           for rep in range(R if s.startswith("random") else 1)}
    uv = {key: np.full((len(KS), len(teR), 2), np.nan) for key in res}  # the person's (offset, variance) estimates
    diag = []
    for ri, r in enumerate(teR):
        yr = Y[pool, r]
        tie = np.random.default_rng([f, g, int(r), 5]).random(len(pool))
        for vi, (strat, est, demo) in enumerate(VARIANTS):
            mu, t2 = (mu_demo[r], tau2_demo) if demo else (0., tau2)
            b_est = b_crowd if est == "crowd" else b_jev
            for rep in range(R if strat.startswith("random") else 1):
                if strat == "random":
                    sel = np.random.default_rng([f, g, int(r), rep, 11]).permutation(len(pool))
                elif strat == "random-common":
                    sel = np.random.default_rng([f, g, rep, 13]).permutation(len(pool))
                elif strat in static:
                    sel = static[strat]
                else:
                    sel = "adaptive"
                for ki, (u, v, asked) in enumerate(run_person(sel, b_est, yr, mu, t2, tie)):
                    res[vi, rep][ki, :, ri] = (b_test + u) / np.sqrt(1 + np.pi * (s2 + v) / 8)
                    uv[vi, rep][ki, ri] = u, v
                    if rep == 0 and KS[ki] in (5, 10, 20):
                        a = np.array(asked, int)
                        diag.append(dict(variant=NAMES[vi], k=KS[ki], unsure=float(np.isnan(yr[a]).mean()),
                                         crowd_info=float((p0c[a] * (1 - p0c[a])).mean()),
                                         crowd_yes=float(p0c[a].mean())))
    # reference: everything the person labelled in the pool, crowd difficulties
    orc = np.full((len(test), len(teR)), np.nan)
    u_orc = np.full(len(teR), np.nan)
    for ri, r in enumerate(teR):
        lab = np.flatnonzero(~np.isnan(Y[pool, r]))
        u, v = post_u(b_crowd[lab], Y[pool, r][lab], 0., tau2)
        orc[:, ri] = (b_test + u) / np.sqrt(1 + np.pi * (s2 + v) / 8)
        u_orc[ri] = u
    info = dict(f=f, g=g, tau2=tau2, tau2_demo=tau2_demo, s2=s2, n_train_raters=len(trR), uv=uv, u_orc=u_orc)
    return f, g, test, teR, res, orc, diag, info


def boot_stats(M, Y, C, W, rgroup):
    """e1e.boot_stats, except that within-conversation pairs are formed only between raters of the same rater fold
    (they share one fitted content model, so cross-fold differences in it cannot leak into the person ranking)."""
    mask = ~np.isnan(Y)
    y = np.nan_to_num(Y)
    L = np.where(mask, np.logaddexp(0, M) - y * M, 0.)
    Br = np.where(mask, (expit(M) - y) ** 2, 0.)
    I = Y.shape[0]
    num, den = np.zeros((len(C), I)), np.zeros((len(C), I))
    n0 = d0 = 0.
    for i in range(I):
        for g in np.unique(rgroup):
            sel = mask[i] & (rgroup == g)
            pos, neg = np.flatnonzero(sel & (y[i] == 1)), np.flatnonzero(sel & (y[i] == 0))
            if len(pos) == 0 or len(neg) == 0:
                continue
            d = M[i, pos][:, None] - M[i, neg][None, :]
            K = (d > 0) + .5 * (d == 0)
            n0, d0 = n0 + K.sum(), d0 + K.size
            Wp, Wn = W[:, pos], W[:, neg]
            num[:, i] += ((Wp @ K) * Wn).sum(1)
            den[:, i] += Wp.sum(1) * Wn.sum(1)
    point = [L.sum() / mask.sum(), Br.sum() / mask.sum(), e1e.roc_auc_score(y[mask], M[mask]), n0 / d0]
    ll_b = ((C @ L) * W).sum(1) / ((C @ mask.astype(float)) * W).sum(1)
    wi_b = (C * num).sum(1) / (C * den).sum(1)
    return np.array(point), ll_b, wi_b


def _init_worker():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def main(raters):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    G = e1e.build(raters)
    I, P = G["I"], G["P"]
    cfold = np.empty(I, int)
    cfold[np.random.default_rng(0).permutation(I)] = np.arange(I) % NF  # E1e's conversation folds
    rfold = np.empty(P, int)
    rfold[np.random.default_rng(2).permutation(P)] = np.arange(P) % NF
    _G.update(Y=G["Y"], Z=G["Z"], D=G["D"], cfold=cfold, rfold=rfold)
    print(f"DICES-350, raters = {raters}: {P} raters x {I} conversations; two-way CV {NF} x {NF}; "
          f"{len(VARIANTS)} strategies, k = {KS}, random strategies repeated {R} times")
    t0 = time.time()
    jobs = [(f, g) for f in range(NF) for g in range(NF)]
    with ProcessPoolExecutor(len(jobs), mp_context=mp.get_context("fork"), initializer=_init_worker) as ex:
        out = list(ex.map(cell_job, jobs))
    print(f"{len(jobs)} cells in {time.time() - t0:.0f}s")
    M = {}
    ORC = np.full((I, P), np.nan)
    diags, infos = [], []
    for f, g, test, teR, res, orc, diag, info in out:
        for (vi, rep), arr in res.items():
            for ki, k in enumerate(KS):
                M.setdefault((vi, k, rep), np.full((I, P), np.nan))[np.ix_(test, teR)] = arr[ki]
        ORC[np.ix_(test, teR)] = orc
        diags += diag
        infos.append({k: v for k, v in info.items() if k not in ("uv", "u_orc")})
    print(pd.DataFrame(infos)[["tau2", "tau2_demo", "s2"]].describe().loc[["mean", "min", "max"]].round(3).to_string())
    np.savez_compressed(os.path.join(OUT, f"oof_{raters}.npz"), Y=G["Y"], oracle=ORC,
                        **{f"{vi}|{k}|{rep}": v for (vi, k, rep), v in M.items()})
    brng = np.random.default_rng(1)
    C = np.stack([np.bincount(brng.integers(0, I, I), minlength=I) for _ in range(B)]).astype(float)
    W = np.stack([np.bincount(brng.integers(0, P, P), minlength=P) for _ in range(B)]).astype(float)
    t0 = time.time()
    st = {}
    for vi, (s, _, _) in enumerate(VARIANTS):
        for k in KS:
            reps = [boot_stats(M[vi, k, rep], G["Y"], C, W, rfold) for rep in range(R if s.startswith("random") else 1)]
            st[vi, k] = tuple(np.mean([x[j] for x in reps], 0) for j in range(3))
    st["oracle"] = boot_stats(ORC, G["Y"], C, W, rfold)
    print(f"bootstrap ({B} draws over conversations x raters) in {time.time() - t0:.0f}s")
    rows = []
    for (key, (pt, llb, wib)) in st.items():
        name, k = (NAMES[key[0]], key[1]) if key != "oracle" else ("reference: all ~280 pool labels, crowd", -1)
        rows.append(dict(model=name, k=k, within_auroc=pt[3], wi_lo=np.quantile(wib, .025), wi_hi=np.quantile(wib, .975),
                         logloss=pt[0], ll_lo=np.quantile(llb, .025), ll_hi=np.quantile(llb, .975), auroc=pt[2], brier=pt[1]))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, f"curve_{raters}.csv"), index=False)
    piv = lambda col: tab[tab.k >= 0].pivot(index="model", columns="k", values=col).reindex(NAMES)
    print("\n== within-conversation AUROC (which of the new raters call a new conversation unsafe; pairs within a rater fold)\n" +
          piv("within_auroc").round(3).to_string())
    print(f"reference (all pool labels): {tab[tab.k < 0].within_auroc.iloc[0]:.3f}")
    print("\n== log loss\n" + piv("logloss").round(4).to_string())
    print("\n== pooled AUROC\n" + piv("auroc").round(3).to_string())
    base_j, base_c = NAMES.index("random | jev difficulties"), NAMES.index("random | crowd difficulties")
    pairs = [(NAMES.index("CAT-JEV | jev difficulties"), base_j),             # primary 1 (pre-registered in PLAN.md)
             (NAMES.index("CAT-crowd | crowd difficulties"), base_j),         # primary 2
             (base_c, base_j), (NAMES.index("CAT-crowd | crowd difficulties"), base_c),
             (NAMES.index("disputed-crowd | crowd difficulties"), base_c),
             (NAMES.index("random-common | crowd difficulties"), base_c),
             (NAMES.index("random-common | jev difficulties"), base_j),
             (NAMES.index("borderline-JEV | jev difficulties"), base_j),
             (NAMES.index("random | jev difficulties + demographics"), base_j),
             (NAMES.index("CAT-crowd | crowd difficulties + demographics"), NAMES.index("CAT-crowd | crowd difficulties")),
             (NAMES.index("CAT-JEV | jev difficulties + demographics"), NAMES.index("CAT-JEV | jev difficulties"))]
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
    print("\n== paired differences (two-way bootstrap; first two pairs are the pre-registered primary comparisons)\n" +
          dif[["pair", "k", "within-AUROC diff [95% CI]", "log loss diff [95% CI]"]].to_string(index=False))
    dg = pd.DataFrame(diags).groupby(["variant", "k"]).mean().unstack("k")
    dg.to_csv(os.path.join(OUT, f"anchors_{raters}.csv"))
    print("\n== what the asked questions look like (share answered Unsure; crowd p(1-p); crowd yes-rate)\n" +
          dg.reindex(NAMES).round(3).to_string())


if __name__ == "__main__":
    main("all" if "--all-raters" in sys.argv else "104")
