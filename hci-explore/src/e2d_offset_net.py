"""E2d: does an external network that learns an offset on top of JEV's probability help the accept gate?

Calibration (Platt) is monotone, so it can fix probabilities but never the ranking (AUROC).  A correction that depends
on the offer itself can:   logit = Platt(JEV) + MLP(offer & activity embedding, JEV's three answers).
Compared, all on ProactiveBench's reward train set, leave-one-session-out (E2b's P3), with every label of the other
15 sessions (~1,500, the most this benchmark allows):
  JEV zero-shot | JEV + Platt | linear version of the same idea (E2b's "Emb + JEV") | JEV + MLP offset |
  the same MLP without JEV (offset = base rate)
Hyper-parameters (width, weight decay, dropout, epochs, or "no correction at all") are picked by 3-fold grouped CV
inside the 15 training sessions; the final model is a 5-seed ensemble.  Second table, E2c's within-session design:
do 10 earlier decisions of the same session, used as a per-session intercept on top of either global model, help?
No API calls.

  python3 src/e2d_offset_net.py
"""
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit

sys.path.insert(0, os.path.dirname(__file__))
import jev  # noqa: E402
import e2b_learning_curve as e2b  # noqa: E402
import e2c_incontext as e2c  # noqa: E402
from offset_mlp import OffsetMLP, _logloss  # noqa: E402

OUT = os.path.join(jev.ROOT, "results", "e2d")
os.makedirs(OUT, exist_ok=True)
GRID = [dict(hidden=(32,), wd=1e-2, dropout=.2), dict(hidden=(32,), wd=1e-1, dropout=.2),
        dict(hidden=(128,), wd=1e-2, dropout=.5), dict(hidden=(128,), wd=1e-1, dropout=.5),
        dict(hidden=(64, 32), wd=3e-2, dropout=.3)]
EPOCHS, BATCH, LR, SEEDS, INNER, B = 80, 64, 1e-3, 5, 3, 2000
_G = {}


def data():
    t, hist, ev = e2c.design()  # train proposal items = E2b's P3 pool; hist/ev = E2c's within-session split
    _, hist_all, _ = e2c.design(k=None)
    y = t.valid.values.astype(float)
    J = np.c_[e2b.logit(t.j_accept), e2b.logit(t.j_need), e2b.logit(t.j_disrupt)]
    z = np.load(e2b.EMB, allow_pickle=False)
    pos = {i: k for k, i in enumerate(z["ids"])}
    E = z["emb"][[pos[i] for i in t.id]].astype(float)
    return t, y, J, E, hist, hist_all, ev


def select(X, off, y, tr, sess):
    """Grouped inner CV over sessions: mean validation log loss per (config, epoch); epoch 0 = keep the offset."""
    folds = np.array_split(np.random.default_rng(0).permutation(np.unique(sess[tr])), INNER)
    curves, base = np.zeros((len(GRID), EPOCHS)), 0.0
    for f in folds:
        va, it = tr[np.isin(sess[tr], f)], tr[~np.isin(sess[tr], f)]
        base += _logloss(off[va], y[va]) / INNER
        for gi, cfg in enumerate(GRID):
            h = OffsetMLP(**cfg, epochs=EPOCHS, batch=BATCH, lr=LR, seed=0, patience=0).fit(
                X[it], off[it], y[it], val=(X[va], off[va], y[va]))
            curves[gi] += np.array(h) / INNER
    gi, ep = np.unravel_index(curves.argmin(), curves.shape)
    if curves[gi, ep] >= base:
        return GRID[0], 0, base, base
    return GRID[gi], int(ep) + 1, curves[gi, ep], base


def _init_worker():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def fold(s):
    t, y, J, E = _G["t"], _G["y"], _G["J"], _G["E"]
    sess = t.session.values
    tr = np.flatnonzero(sess != s)
    Es = (E - E[tr].mean(0)) / (E[tr].std(0) + 1e-8)
    a, b = e2b.fit_map(J[tr, :1], y[tr], np.array([1.]), 1., 1., [0.])
    o = a[0] * J[:, 0] + b
    d = E.shape[1]
    Xl = np.c_[J, Es / np.sqrt(d)]
    w, bl = e2b.fit_map(Xl[tr], y[tr], np.r_[1., 0, 0, np.zeros(d)], np.ones(d + 3), 1., [0.] + [-np.inf] * (d + 2))
    out = {"JEV + Platt": o, "JEV + linear offset (E2b Emb + JEV)": Xl @ w + bl}
    info = {}
    base = np.full(len(y), e2b.logit(np.array([y[tr].mean()]))[0])
    for name, off, X in [("JEV + MLP offset", o, np.c_[Es, J]), ("MLP alone (no JEV)", base, Es)]:
        cfg, ep, cv, cv0 = select(X, off, y, tr, sess)
        models = [OffsetMLP(**cfg, epochs=ep, batch=BATCH, lr=LR, seed=k, patience=0) for k in range(SEEDS)]
        for mdl in models:
            mdl.fit(X[tr], off[tr], y[tr])
        out[name] = np.mean([mdl.decision(X, off) for mdl in models], 0)
        info[name] = dict(session=s, hidden=str(cfg["hidden"]), wd=cfg["wd"], dropout=cfg["dropout"], epochs=ep,
                          inner_cv_logloss=cv, inner_cv_logloss_offset_only=cv0)
    return s, out, info


def session_intercept(m_hist, y_hist, lam=1.):
    """MAP intercept u for one session: logit = m + u, u ~ N(0, 1/lam)."""
    f = lambda u: (np.logaddexp(0, m_hist + u) - y_hist * (m_hist + u)).sum() + lam / 2 * u * u
    return minimize_scalar(f, bounds=(-6, 6), method="bounded").x


def boot_table(per, names, sessions, rng, pairs):
    """per: dict name -> (n_sessions x n_metrics) array. Means over sessions, session-bootstrap CIs, paired diffs."""
    picks = [rng.integers(0, len(sessions), len(sessions)) for _ in range(B)]
    rows, drows = [], []
    for n in names:
        bt = np.array([np.nanmean(per[n][p], 0) for p in picks])
        for j, k in enumerate(e2b.METRICS):
            rows.append(dict(method=n, metric=k, mean=np.nanmean(per[n][:, j]), lo=np.quantile(bt[:, j], .025),
                             hi=np.quantile(bt[:, j], .975)))
    for a, b in pairs:
        bt = np.array([np.nanmean(per[a][p] - per[b][p], 0) for p in picks])
        for j, k in enumerate(e2b.METRICS):
            drows.append(dict(pair=f"{a} - {b}", metric=k, mean=np.nanmean(per[a][:, j] - per[b][:, j]),
                              lo=np.quantile(bt[:, j], .025), hi=np.quantile(bt[:, j], .975)))
    return pd.DataFrame(rows), pd.DataFrame(drows)


def wide(tab):
    tab = tab.assign(cell=[f"{m:.3f} [{lo:.3f}, {hi:.3f}]" for m, lo, hi in zip(tab["mean"], tab.lo, tab.hi)])
    key = "method" if "method" in tab else "pair"
    return tab.pivot(index=key, columns="metric", values="cell").reindex(columns=e2b.METRICS)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    t, y, J, E, hist, hist_all, ev = data()
    _G.update(t=t, y=y, J=J, E=E)
    sessions = np.unique(t.session)
    t0 = time.time()
    with ProcessPoolExecutor(len(sessions), mp_context=mp.get_context("fork"), initializer=_init_worker) as ex:
        res = list(ex.map(fold, sessions))
    print(f"{len(t)} proposal items, {len(sessions)} sessions; {len(res)} outer folds in {time.time() - t0:.0f}s")
    info = pd.DataFrame([r for _, _, inf in res for r in inf.values()],
                        index=[n for _, _, inf in res for n in inf])
    info.to_csv(os.path.join(OUT, "chosen_configs.csv"))
    print("\nchosen by inner CV (epochs 0 = the correction did not beat the offset):\n" +
          info.groupby(level=0).agg(epochs_median=("epochs", "median"), folds_no_correction=("epochs", lambda v: (v == 0).sum()),
                                    cv_logloss=("inner_cv_logloss", "mean"),
                                    cv_logloss_offset=("inner_cv_logloss_offset_only", "mean")).round(4).to_string())

    # ---- table 1: all items of the held-out session
    names = ["JEV zero-shot", "JEV + Platt", "JEV + linear offset (E2b Emb + JEV)", "JEV + MLP offset", "MLP alone (no JEV)",
             "always show", "never show"]
    per = {n: [] for n in names}
    oof = {n: np.full(len(y), np.nan) for n in names}
    for s, out, _ in res:
        e = np.flatnonzero(t.session.values == s)
        pr = {"JEV zero-shot": t.j_accept.values[e], "always show": np.ones(len(e)), "never show": np.zeros(len(e)),
              **{k: expit(v[e]) for k, v in out.items()}}
        for n in names:
            per[n].append(e2b.metrics(y[e], pr[n]))
            oof[n][e] = pr[n]
    per = {n: np.array(v) for n, v in per.items()}
    pairs = [("JEV + MLP offset", "JEV + Platt"), ("JEV + MLP offset", "JEV + linear offset (E2b Emb + JEV)"),
             ("JEV + linear offset (E2b Emb + JEV)", "JEV + Platt"), ("JEV + MLP offset", "MLP alone (no JEV)")]
    tab, dif = boot_table(per, names, sessions, np.random.default_rng(0), pairs)
    tab.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    dif.to_csv(os.path.join(OUT, "diffs.csv"), index=False)
    pd.DataFrame({"id": t.id, "session": t.session, "y": y, **oof}).to_parquet(os.path.join(OUT, "oof.parquet"))
    print("\n== held-out session, all its items (mean over 16 sessions, session bootstrap 95% CI)\n" + wide(tab).to_string())
    print("\n== paired differences\n" + wide(dif).to_string())

    # ---- table 2: E2c's within-session design (eval = second half of each session)
    lam_grid = (1., 4.)
    names2 = ["JEV + Platt", "JEV + MLP offset"] + [f"{g} + session intercept ({h}, lam={lam:g})"
                                                    for g in ("JEV + Platt", "JEV + MLP offset")
                                                    for h in ("10 labels", "all first half") for lam in lam_grid]
    per2 = {n: [] for n in names2}
    idx = {i: k for k, i in enumerate(t.id)}
    for s, out, _ in res:
        e = np.array([idx[i] for i in ev.id[ev.session == s]])
        for g in ("JEV + Platt", "JEV + MLP offset"):
            per2[g].append(e2b.metrics(y[e], expit(out[g][e])))
            for h, H in (("10 labels", hist), ("all first half", hist_all)):
                hi = np.array([idx[i] for i in H[s].id])
                for lam in lam_grid:
                    u = session_intercept(out[g][hi], y[hi], lam)
                    per2[f"{g} + session intercept ({h}, lam={lam:g})"].append(e2b.metrics(y[e], expit(out[g][e] + u)))
    per2 = {n: np.array(v) for n, v in per2.items()}
    pairs2 = [(f"{g} + session intercept ({h}, lam=1)", g) for g in ("JEV + Platt", "JEV + MLP offset")
              for h in ("10 labels", "all first half")]
    tab2, dif2 = boot_table(per2, names2, sessions, np.random.default_rng(1), pairs2)
    tab2.to_csv(os.path.join(OUT, "within_session.csv"), index=False)
    dif2.to_csv(os.path.join(OUT, "within_session_diffs.csv"), index=False)
    print(f"\n== within-session design ({len(ev)} second-half items; history = 10 most recent first-half labels, or all "
          f"{sum(len(h) for h in hist_all.values())} first-half labels)\n" + wide(tab2).to_string())
    print("\n== paired differences\n" + wide(dif2).to_string())


if __name__ == "__main__":
    main()
