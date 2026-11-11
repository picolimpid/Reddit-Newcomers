"""E2g, part 2: CDHF-style operating points for a gate with and without telemetry (RealHumanEval, unseen programmers).
A threshold is chosen on the training programmers' cross-fitted scores so that >= p of the suggestions it would hide
were rejected (CDHF's rule, as in E2f), then applied to the held-out programmers.

  python3 -I src/e2g_operating.py
"""
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import e2e_realhumaneval as e2e  # noqa: E402
import e2f_cdhf as e2f  # noqa: E402
import e2g_when_jev_helps as e2g  # noqa: E402

NF, NIN, TNRS = 5, 4, (0.95, 0.97, 0.99)
MODELS = {"JEV only": ("lr", ()), "LR | text": ("lr", ("text",)), "LR | text + JEV": ("lrj", ("text",)),
          "LR | all": ("lr", ("text", "conf", "hist")), "LR | all + JEV": ("lrj", ("text", "conf", "hist")),
          "CDHF replica (GBDT | all)": ("gbdt", ("text", "conf", "hist"))}
_G = {}


def margins(name, X, J, y, grp, tr, te):
    kind, parts = MODELS[name]
    if not parts:
        return LogisticRegression(C=100., max_iter=5000).fit(J[tr], y[tr]).decision_function(J[te])
    cols = np.any([grp[p] for p in parts], 0)
    if kind == "gbdt":
        return e2f.hgb().fit(X[tr][:, cols], y[tr]).decision_function(X[te][:, cols])
    Xn = np.where(np.isnan(X), np.nanmedian(X[tr], 0), X)
    Xs = (Xn - Xn[tr].mean(0)) / (Xn[tr].std(0) + 1e-8)
    A = np.c_[Xs[:, cols], J] if kind == "lrj" else Xs[:, cols]
    return LogisticRegression(C=1., max_iter=5000).fit(A[tr], y[tr]).decision_function(A[te])


def fold_job(f):
    X, J, y, uid, ufold, grp = (_G[k] for k in ("X", "J", "y", "uid", "ufold", "grp"))
    tr, te = np.flatnonzero(ufold[uid] != f), np.flatnonzero(ufold[uid] == f)
    users = np.unique(uid[tr])
    inner = {u: k % NIN for k, u in enumerate(np.random.default_rng([f, 7]).permutation(users))}  # E2f's inner folds
    ifold = np.array([inner.get(u, -1) for u in uid])
    out = {}
    for n in MODELS:
        m_te = margins(n, X, J, y, grp, tr, te)
        m_cf = np.full(len(y), np.nan)
        for g in range(NIN):
            ra, rb = tr[ifold[tr] != g], tr[ifold[tr] == g]
            m_cf[rb] = margins(n, X, J, y, grp, ra, rb)
        m = m_cf[tr]
        o = np.argsort(m)
        rej = np.cumsum(1 - y[tr][o]) / np.arange(1, len(o) + 1)
        thr = {}
        for p in TNRS:
            ok = np.flatnonzero(rej >= p)
            thr[p] = m[o][ok.max()] if len(ok) else -np.inf
        out[n] = (m_te, thr)
    return f, te, out


def main():
    a, T, tcols, J, E, U = e2e.design()
    F = e2f.cdhf_features(a)
    cols = tcols + list(F.columns)
    X = np.c_[T, F.values.astype(float)]
    um = a.groupby("programmer_id").model.first()
    order = np.concatenate([np.random.default_rng(0).permutation(g.index.values) for _, g in um.groupby(um)])
    ufold = np.full(a.programmer_id.max() + 1, -1)
    ufold[order] = np.arange(len(order)) % NF
    y = a.accepted.values.astype(float)
    uid, idx = a.programmer_id.values, a.idx.values
    _G.update(X=X, J=J, y=y, uid=uid, ufold=ufold, grp=e2g.groups(cols))
    with ProcessPoolExecutor(NF, mp_context=mp.get_context("fork"), initializer=e2g._init_worker) as ex:
        res = list(ex.map(fold_job, range(NF)))
    rows = []
    for n in MODELS:
        for p in TNRS:
            for sub, sel in (("all", np.ones(len(y), bool)), ("first 10 per programmer", idx < 10)):
                hide = np.zeros(len(y), bool)
                for f, te, out in res:
                    m_te, thr = out[n]
                    hide[te] = m_te <= thr[p]
                h = hide & sel
                rows.append({"model": n, "target": p, "subset": sub, "hidden %": 100 * h.sum() / sel.sum(),
                             "rejected among hidden %": 100 * (1 - y[h].mean()) if h.any() else np.nan,
                             "accepted lost %": 100 * y[h].sum() / y[sel].sum()})
    op = pd.DataFrame(rows)
    out = os.path.join(e2g.OUT, "operating_points.csv")
    op.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    for sub in ("all", "first 10 per programmer"):
        print(f"\n== {sub}: hide when the score is below a threshold set on training programmers "
              "(>= target of hidden ones rejected there)\n" +
              op[op.subset == sub].drop(columns="subset").round(1).assign(target=op.target.map("{:.2f}".format))
              .to_string(index=False))


if __name__ == "__main__":
    main()
