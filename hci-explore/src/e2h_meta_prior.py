"""E2h: a meta-learned per-user prior for the acceptance gate (RealHumanEval; the linear-Gaussian form of "learn how
users differ from many users, adapt to a new one from a few decisions").  Global gate = E2f's best,
"CDHF features + JEV, LR" (unseen programmers, E2e's folds).  A new programmer's first n decisions personalise it;
scored on the E2e evaluation set (suggestions with index >= 30 of programmers with >= 40).

  global          no personalisation
  intercept (CV)  E2f's per-user Bayesian intercept, prior precision chosen on training programmers by log loss
  intercept (EB)  the same, prior variance estimated from the training programmers' own offsets (empirical Bayes)
  EB slopes       intercept + slopes on the global margin and on JEV's two answers; prior mean and covariance
                  estimated from the training programmers (cross-fitted margins), so a new user starts at the
                  population's typical correction and moves along the directions in which users actually differ
Pre-registered in PLAN.md (B2): pooled AUROC at n = 10 or 20 above "intercept (CV)", 95% CI excluding 0.
(v2: the population prior is fitted by EM; v1's moment estimate collapsed, log kept as results/e2h/log_v1.txt.)

  python3 -I src/e2h_meta_prior.py
"""
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import expit  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import e2e_realhumaneval as e2e  # noqa: E402
import e2f_cdhf as e2f  # noqa: E402

OUT = os.path.join(e2e.jev.ROOT, "results", "e2h")
os.makedirs(OUT, exist_ok=True)
NF, NIN, B = 5, 4, 2000
NS = (5, 10, 20, 30)
BASE = "CDHF features + JEV, LR"
_G = {}


def design_rows(m, J):
    """Per-user design: [1, global margin, JEV accept logit, JEV useful logit]."""
    return np.c_[np.ones(len(m)), m, J]


def map_fit(X, m, y, mu, Pinv):
    """MAP of per-user coefficients th: logit = m + X th, prior N(mu, P).  Convex; L-BFGS."""
    def f(th):
        z = m + X @ th
        r = expit(z) - y
        d = th - mu
        return np.logaddexp(0, z).sum() - y @ z + .5 * d @ Pinv @ d, X.T @ r + Pinv @ d
    th = minimize(f, mu.copy(), jac=True, method="L-BFGS-B").x
    p = expit(m + X @ th)
    H = (X * (p * (1 - p))[:, None]).T @ X + Pinv
    return th, np.linalg.inv(H)


def em_prior(Ds, ms, ys, d, iters=40, fix_mean=False):
    """Population prior N(mu, S) of per-user coefficients by EM with Laplace E-steps (a Gaussian GLMM fitted on the
    training programmers).  The first version estimated it from weak-prior per-user fits, whose noise swamped the
    spread (the variance collapsed to its floor); this is the standard fix."""
    mu, S = np.zeros(d), np.eye(d) * .25
    for _ in range(iters):
        Sinv = np.linalg.inv(S)
        fits = [map_fit(D, m, y, mu, Sinv) for D, m, y in zip(Ds, ms, ys)]
        th = np.array([t for t, _ in fits])
        new_mu = np.zeros(d) if fix_mean else th.mean(0)
        S = np.mean([np.outer(t - new_mu, t - new_mu) + C for t, C in fits], 0)
        ev, V = np.linalg.eigh(S)
        S = (V * np.maximum(ev, 1e-4)) @ V.T
        mu = new_mu
    return mu, S


def fold_job(f):
    a, Xs_all, ufold = _G["a"], _G["Xs"], _G["ufold"]
    y = a.accepted.values.astype(float)
    uid, idx = a.programmer_id.values, a.idx.values
    J = _G["J"]
    tr, te = np.flatnonzero(ufold[uid] != f), np.flatnonzero(ufold[uid] == f)
    Xs = e2f.blocks(f, tr)
    m_te = e2f.fit_predict(BASE, Xs, y, tr, te)
    tr_users = np.unique(uid[tr])
    inner = {u: k % NIN for k, u in enumerate(np.random.default_rng([f, 7]).permutation(tr_users))}  # E2f's
    ifold = np.array([inner.get(u, -1) for u in uid])
    m_cf = np.full(len(y), np.nan)
    for g in range(NIN):
        ra, rb = tr[ifold[tr] != g], tr[ifold[tr] == g]
        m_cf[rb] = e2f.fit_predict(BASE, e2f.blocks(f, ra), y, ra, rb)
    full = np.full(len(y), np.nan)
    full[te], full[tr] = m_te, m_cf[tr]
    # --- learn the population prior from training programmers (all their suggestions, cross-fitted margins)
    D = design_rows(full, J)
    rows_u = [np.flatnonzero(uid == u) for u in tr_users]
    mu, S = em_prior([D[r] for r in rows_u], [full[r] for r in rows_u], [y[r] for r in rows_u], 4)
    _, Sa = em_prior([D[r][:, :1] for r in rows_u], [full[r] for r in rows_u], [y[r] for r in rows_u], 1, fix_mean=True)
    var_a = float(Sa[0, 0])
    # E2f's CV choice of the intercept-only prior precision (on training programmers)
    nuser = a.groupby("programmer_id").size()
    ev_tr = tr[(nuser[uid[tr]].values >= e2e.MIN_N) & (idx[tr] >= e2e.EVAL_FROM)]
    best = None
    for lam in e2e.LAMS:
        ll = []
        for u in np.unique(uid[ev_tr]):
            ru = np.flatnonzero(uid == u)
            eu = ru[idx[ru] >= e2e.EVAL_FROM]
            for n in NS:
                h = ru[idx[ru] < n]
                mm = m_cf[eu] + e2e.user_offset(m_cf[h], y[h], lam)
                ll.append(np.logaddexp(0, mm) - y[eu] * mm)
        ll = np.mean(np.concatenate(ll))
        if best is None or ll < best[0]:
            best = (ll, lam)
    lam = best[1]
    out = {"global": m_te}
    for n in NS:
        for name in ("intercept (CV)", "intercept (EB)", "EB slopes"):
            out[f"{name} | n={n}"] = np.full(len(te), np.nan)
    pos = {r: k for k, r in enumerate(te)}
    Pinv = np.linalg.inv(S)
    for u in np.unique(uid[te]):
        ru = np.flatnonzero(uid == u)
        for n in NS:
            h, later = ru[idx[ru] < n], ru[idx[ru] >= n]
            k = [pos[r] for r in later]
            out[f"intercept (CV) | n={n}"][k] = full[later] + e2e.user_offset(full[h], y[h], lam)
            out[f"intercept (EB) | n={n}"][k] = full[later] + e2e.user_offset(full[h], y[h], 1 / var_a)
            th, _ = map_fit(D[h], full[h], y[h], mu, Pinv) if len(h) else (mu, S)
            out[f"EB slopes | n={n}"][k] = full[later] + D[later] @ th
    info = dict(fold=f, lam_cv=lam, var_a_eb=var_a, prior_mean=mu.round(3).tolist(),
                prior_sd=np.sqrt(np.diag(S)).round(3).tolist())
    return f, te, out, info


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    a, T, tcols, J, E, U = e2e.design()
    F = e2f.cdhf_features(a)
    fcols = tcols + list(F.columns)
    s1 = np.array([not c.startswith(e2f.STAGE1_DROP) for c in fcols])
    um = a.groupby("programmer_id").model.first()  # E2e's folds
    order = np.concatenate([np.random.default_rng(0).permutation(g.index.values) for _, g in um.groupby(um)])
    ufold = np.full(a.programmer_id.max() + 1, -1)
    ufold[order] = np.arange(len(order)) % NF
    e2f._G.update(a=a, T=T, F=F.values.astype(float), J=J, E=E, s1=s1, ufold=ufold)
    _G.update(a=a, J=J, ufold=ufold, Xs=None)
    with ProcessPoolExecutor(NF, mp_context=mp.get_context("fork"), initializer=e2f._init_worker) as ex:
        res = list(ex.map(fold_job, range(NF)))
    print("per-fold prior: " + "; ".join(str(i) for *_, i in res))
    y = a.accepted.values.astype(float)
    uid, idx = a.programmer_id.values, a.idx.values
    names = list(res[0][2])
    pred = {n: np.full(len(y), np.nan) for n in names}
    for f, te, out, info in res:
        for n in names:
            pred[n][te] = out[n]
    nuser = a.groupby("programmer_id").size()
    ev = np.flatnonzero((nuser[uid].values >= e2e.MIN_N) & (idx >= e2e.EVAL_FROM))
    tab, pt, bt = e2f.boot_table(y[ev], uid[ev], {n: pred[n][ev] for n in names}, names, B, np.random.default_rng(1))
    print(f"\n== E2e evaluation set: {len(ev)} suggestions of {len(np.unique(uid[ev]))} programmers\n"
          + tab[["auroc", "logloss", "within_user_auroc"]].to_string())
    pairs = []
    for n in NS:
        pairs += [(f"EB slopes | n={n}", f"intercept (CV) | n={n}"), (f"intercept (EB) | n={n}", f"intercept (CV) | n={n}"),
                  (f"intercept (CV) | n={n}", "global"), (f"EB slopes | n={n}", "global")]
    d = e2f.diffs(pairs, pt, bt)
    print("\n== paired differences (programmer bootstrap)\n" + d[["pair", "auroc", "logloss", "within_user_auroc"]]
          .to_string(index=False))
    tab.to_csv(os.path.join(OUT, "summary.csv"))
    d.to_csv(os.path.join(OUT, "diffs.csv"), index=False)


if __name__ == "__main__":
    main()
