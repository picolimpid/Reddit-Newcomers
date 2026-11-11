"""E4c: can the shared parts be learned from singly-labelled logs?  (Eshwar-lens check of Venire's limitation, "real
moderation log data is largely singly-labeled"; pre-registered in PLAN.md as D2.)

DICES-990, E1i's design (141 quality-filtered raters, 5 conversation folds x 5 rater folds, random anchors with JEV
difficulties).  For a cell, the training raters' labels on the pool conversations are thinned so that every pool
conversation keeps the labels of only m randomly chosen training raters (m = 1: a moderation log where each case was
decided by one person; m = 3; all).  The shared parts -- JEV's content calibration and the prior over people -- are
fitted by one joint estimator in every condition (logit P = Z w + c + u_r, u_r ~ N(0, tau2), alternating fits with an
EM update of tau2), so the conditions differ only in the training labels.  "E1i estimator" re-runs E1i's own
pipeline (Rasch-based offsets) on all labels as a check that this script reproduces E1i.
New raters answer k random anchors exactly as in E1i and are scored on the test conversations.

Primary (pre-registered): within-conversation AUROC at k = 20, m = 1 minus all labels (joint estimator); the
single-label logs are "enough" if the lower 95% bound (two-way bootstrap) is above -0.02.

  python3 -I src/e4c_single_label.py
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
import e1i_dices990 as e1i  # noqa: E402

OUT = os.path.join(jev.ROOT, "results", "e4c")
os.makedirs(OUT, exist_ok=True)
KS = (0, 5, 10, 20)
CONDS = ["m=1", "m=3", "all labels", "all labels, E1i estimator"]
_G = {}


def thin(Yp, m, seed):
    """Keep the labels of m randomly chosen raters per conversation (rows = conversations)."""
    if m is None:
        return Yp
    out = np.full_like(Yp, np.nan)
    rng = np.random.default_rng(seed)
    for i in range(Yp.shape[0]):
        lab = np.flatnonzero(~np.isnan(Yp[i]))
        keep = rng.choice(lab, min(m, len(lab)), replace=False) if len(lab) else lab
        out[i, keep] = Yp[i, keep]
    return out


def fit_joint(X, Yc, iters=15):
    """logit P(y_ri) = X_i w + c + u_r, u_r ~ N(0, tau2): alternate content fit (offsets fixed), per-rater MAP offsets,
    and an EM update of tau2.  Works with any labelling pattern, including one label per conversation."""
    P = Yc.shape[1]
    u, t2 = np.zeros(P), .5
    for _ in range(iters):
        w, c = e1f.fit_content(X, Yc, u)
        b = X @ w + c
        post = []
        for j in range(P):
            lab = np.flatnonzero(~np.isnan(Yc[:, j]))
            post.append(e1f.post_u(b[lab], Yc[lab, j], 0., t2) if len(lab) else (0., t2))
        u = np.array([p[0] for p in post])
        t2 = max(float(np.mean([p[0] ** 2 + p[1] for p in post])), .02)
    return w, c, u, t2


def cell_job(args):
    f, g, cond = args
    Y, rated, Z, cfold, rfold = (_G[x] for x in ("Y", "rated", "Z", "cfold", "rfold"))
    pool, test = np.flatnonzero(cfold != f), np.flatnonzero(cfold == f)
    trR, teR = np.flatnonzero(rfold != g), np.flatnonzero(rfold == g)
    m = {"m=1": 1, "m=3": 3}.get(cond)
    Yp = thin(Y[np.ix_(pool, trR)], m, [f, g, 99])
    mz, sz = Z[pool].mean(0), Z[pool].std(0) + 1e-8
    Zs = (Z - mz) / sz
    if cond == "all labels, E1i estimator":
        b_crowd, u_tr = e1f.fit_rasch(Yp)
        pc = expit(b_crowd[:, None] + u_tr[None, :])
        noise = 1 / (np.where(~np.isnan(Yp), pc * (1 - pc), 0.).sum(0) + e1f.LAM_U)
        tau2 = max(u_tr.var() - noise.mean(), .05)
        w, c = e1f.fit_content(Zs[pool], Yp, u_tr)
    else:
        w, c, u_tr, tau2 = fit_joint(Zs[pool], Yp)
    b_test = Zs[test] @ w + c
    inner = np.random.default_rng([f, g, 3]).permutation(len(pool)) % e1f.NIN
    b_jev = np.empty(len(pool))
    for h in range(e1f.NIN):
        wh, ch = e1f.fit_content(Zs[pool][inner != h], Yp[inner != h], u_tr)
        b_jev[inner == h] = Zs[pool][inner == h] @ wh + ch
    s2 = float(np.var(b_crowd - b_jev)) if cond == "all labels, E1i estimator" else 0.
    res = np.full((e1f.R, len(KS), len(test), len(teR)), np.nan)
    for ri, r in enumerate(teR):
        yr, cand = Y[pool, r], rated[pool, r]
        for rep in range(e1f.R):
            order = np.random.default_rng([f, g, int(r), rep, 11]).permutation(np.flatnonzero(cand))
            for ki, k in enumerate(KS):
                a = [int(i) for i in order[:k] if not np.isnan(yr[i])]
                u, v = e1f.post_u(b_jev[a], yr[a], 0., tau2) if a else (0., tau2)
                res[rep, ki, :, ri] = (b_test + u) / np.sqrt(1 + np.pi * (s2 + v) / 8)
    n_tr = int((~np.isnan(Yp)).sum())
    return f, g, cond, test, teR, res, dict(cond=cond, tau2=tau2, s2=s2, train_labels=n_tr,
                                            labels_per_train_rater=n_tr / len(trR))


def main():
    pd.set_option("display.width", 250)
    G = e1i.build("filtered")
    I, P = G["I"], G["P"]
    cfold = np.empty(I, int)
    cfold[np.random.default_rng(0).permutation(I)] = np.arange(I) % 5
    rfold = np.empty(P, int)
    rfold[np.random.default_rng(2).permutation(P)] = np.arange(P) % 5
    _G.update(G, cfold=cfold, rfold=rfold)
    jobs = [(f, g, c) for c in CONDS for f in range(5) for g in range(5)]
    with ProcessPoolExecutor(25, mp_context=mp.get_context("fork"), initializer=e1f._init_worker) as ex:
        out = list(ex.map(cell_job, jobs))
    info = pd.DataFrame([o[6] for o in out]).groupby("cond").mean().reindex(CONDS)
    print("fitted shared parts (means over the 25 cells)\n" + info.round(3).to_string())
    M = {}
    for f, g, cond, test, teR, res, _ in out:
        for rep in range(e1f.R):
            for ki, k in enumerate(KS):
                M.setdefault((cond, k, rep), np.full((I, P), np.nan))[np.ix_(test, teR)] = res[rep, ki]
    lab = ~np.isnan(G["Y"])
    for key in M:
        M[key] = np.where(lab, M[key], 0.)
    brng = np.random.default_rng(1)
    C = np.stack([np.bincount(brng.integers(0, I, I), minlength=I) for _ in range(e1f.B)]).astype(float)
    W = np.stack([np.bincount(brng.integers(0, P, P), minlength=P) for _ in range(e1f.B)]).astype(float)
    st = {}
    for cond in CONDS:
        for k in KS:
            reps = [e1f.boot_stats(M[cond, k, rep], G["Y"], C, W, rfold) for rep in range(e1f.R)]
            st[cond, k] = tuple(np.mean([x[j] for x in reps], 0) for j in range(3))
    tab = pd.DataFrame([dict(condition=c, k=k, within_auroc=pt[3], logloss=pt[0], auroc=pt[2])
                        for (c, k), (pt, lb, wb) in st.items()])
    tab.to_csv(os.path.join(OUT, "curve.csv"), index=False)
    for col in ("within_auroc", "logloss", "auroc"):
        print(f"\n== {col}\n" + tab.pivot(index="condition", columns="k", values=col).reindex(CONDS).round(4).to_string())
    rows = []
    for tag, a, b in [("primary", ("m=1", 20), ("all labels", 20)), ("", ("m=1", 5), ("all labels", 5)),
                      ("", ("m=3", 20), ("all labels", 20)), ("", ("m=1", 0), ("all labels", 0)),
                      ("check", ("all labels", 20), ("all labels, E1i estimator", 20)),
                      ("", ("m=1", 20), ("m=1", 0))]:
        pa, la, wa = st[a]
        pb, lb, wb = st[b]
        lo = np.quantile(wa - wb, .025)
        rows.append(dict(test=tag, pair=f"{a[0]} k={a[1]}  -  {b[0]} k={b[1]}",
                         within=f"{pa[3] - pb[3]:+.4f} [{lo:+.4f}, {np.quantile(wa - wb, .975):+.4f}]",
                         logloss=f"{pa[0] - pb[0]:+.4f} [{np.quantile(la - lb, .025):+.4f}, "
                                 f"{np.quantile(la - lb, .975):+.4f}]",
                         noninferior=(lo > -.02) if tag == "primary" else ""))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "tests.csv"), index=False)
    print("\n== paired differences (two-way bootstrap, 95% CI)\n" + d.to_string(index=False))
    np.savez_compressed(os.path.join(OUT, "oof.npz"), Y=G["Y"], rfold=rfold, cfold=cfold,
                        **{f"{c}|{k}": M[c, k, 0] for c in CONDS for k in KS})


if __name__ == "__main__":
    main()
