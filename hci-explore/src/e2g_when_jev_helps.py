"""E2g: when does JEV add something to an acceptance gate?  RealHumanEval, the same 4,765 unrequested suggestions,
83 programmers and programmer folds as E2e / E2f (unseen programmers at test time).

Telemetry ablation (what a proactive assistant may or may not log), each with and without JEV's two answers:
  text        what any assistant can compute from the code and the suggestion itself (lengths, keywords, mid-word,
              cursor line, which model)
  + conf      the suggesting model's token log-probabilities (needs a model that exposes them)
  + history   the user's own past accept / reject actions and timing (needs a logging client and time)
  all         CDHF-style features (= E2f's "CDHF features LR")
Learners: logistic regression (E2f's best) and histogram gradient boosting (E2f's CDHF replica).
Cold start: the same comparison stratified by how many suggestions the programmer had seen before (0-9, 10-29, 30+).
Pre-registered in PLAN.md (B1): text-only JEV gain >= +0.03 AUROC with CI excluding 0; gain shrinks as telemetry grows.

  python3 -I src/e2g_when_jev_helps.py
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
from sklearn.linear_model import LogisticRegression  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import e2e_realhumaneval as e2e  # noqa: E402
import e2f_cdhf as e2f  # noqa: E402

OUT = os.path.join(e2e.jev.ROOT, "results", "e2g")
os.makedirs(OUT, exist_ok=True)
NF, B = 5, 2000
HIST = ("prev", "accepts_in_last10", "event_index_all", "log_index", "log_seconds_since_last")
SCEN = {"text": ("text",), "text + conf": ("text", "conf"), "text + history": ("text", "hist"),
        "all (text + conf + history)": ("text", "conf", "hist")}
BUCKETS = {"suggestions 0-9": (0, 10), "suggestions 10-29": (10, 30), "suggestions 30+": (30, 10 ** 9)}
_G = {}


def groups(cols):
    g = np.array(["conf" if c.startswith("lp_") else "hist" if c.startswith(HIST) else "text" for c in cols])
    return {k: g == k for k in ("text", "conf", "hist")}


def fold_job(f):
    X, J, y, uid, ufold, grp = (_G[k] for k in ("X", "J", "y", "uid", "ufold", "grp"))
    tr, te = np.flatnonzero(ufold[uid] != f), np.flatnonzero(ufold[uid] == f)
    Xn = np.where(np.isnan(X), np.nanmedian(X[tr], 0), X)
    Xs = (Xn - Xn[tr].mean(0)) / (Xn[tr].std(0) + 1e-8)
    out = {"JEV only": LogisticRegression(C=100., max_iter=5000).fit(J[tr], y[tr]).decision_function(J[te])}
    for s, parts in SCEN.items():
        cols = np.any([grp[p] for p in parts], 0)
        for jv in (False, True):
            tag = s + (" + JEV" if jv else "")
            A, Araw = (np.c_[Xs[:, cols], J], np.c_[X[:, cols], J]) if jv else (Xs[:, cols], X[:, cols])
            out[f"LR | {tag}"] = LogisticRegression(C=1., max_iter=5000).fit(A[tr], y[tr]).decision_function(A[te])
            out[f"GBDT | {tag}"] = e2f.hgb().fit(Araw[tr], y[tr]).decision_function(Araw[te])
    return f, te, out


def _init_worker():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    a, T, tcols, J, E, U = e2e.design()
    F = e2f.cdhf_features(a)
    cols = tcols + list(F.columns)
    X = np.c_[T, F.values.astype(float)]
    grp = groups(cols)
    um = a.groupby("programmer_id").model.first()  # E2e's folds, reproduced exactly
    order = np.concatenate([np.random.default_rng(0).permutation(g.index.values) for _, g in um.groupby(um)])
    ufold = np.full(a.programmer_id.max() + 1, -1)
    ufold[order] = np.arange(len(order)) % NF
    y = a.accepted.values.astype(float)
    uid, idx = a.programmer_id.values, a.idx.values
    _G.update(X=X, J=J, y=y, uid=uid, ufold=ufold, grp=grp)
    print("feature groups: " + "; ".join(f"{k} ({m.sum()}): {', '.join(np.array(cols)[m])}" for k, m in grp.items()))
    t0 = time.time()
    with ProcessPoolExecutor(NF, mp_context=mp.get_context("fork"), initializer=_init_worker) as ex:
        res = list(ex.map(fold_job, range(NF)))
    print(f"{NF} folds over programmers in {time.time() - t0:.0f}s")
    names = list(res[0][2])
    pred = {n: np.full(len(y), np.nan) for n in names}
    for f, te, out in res:
        for n in names:
            pred[n][te] = out[n]
    pd.DataFrame({"row": a.row.values, "programmer_id": uid, "idx": idx, "accepted": y, **pred}).to_parquet(
        os.path.join(OUT, "oof.parquet"))
    pairs = [(f"{l} | {s} + JEV", f"{l} | {s}") for l in ("LR", "GBDT") for s in SCEN]
    pairs += [("LR | text + JEV", "JEV only"), ("LR | text", "JEV only"),
              ("LR | text + JEV", "GBDT | all (text + conf + history)"),   # text + JEV vs the CDHF replica
              ("LR | text + JEV", "LR | text + conf"), ("LR | text + JEV", "LR | text + history")]
    tabs, difs = [], []
    subsets = {"all suggestions": np.arange(len(y))}
    subsets.update({b: np.flatnonzero((idx >= lo) & (idx < hi)) for b, (lo, hi) in BUCKETS.items()})
    for sname, rows in subsets.items():
        tab, pt, bt = e2f.boot_table(y[rows], uid[rows], {n: pred[n][rows] for n in names}, names, B,
                                     np.random.default_rng(0))
        d = e2f.diffs(pairs, pt, bt)
        print(f"\n== {sname}: {len(rows)} suggestions, {len(np.unique(uid[rows]))} programmers, "
              f"accept rate {y[rows].mean():.3f}\n" + tab[["auroc", "logloss", "within_user_auroc"]].to_string())
        print("\n-- JEV's increment (paired programmer bootstrap)\n" + d[["pair", "auroc", "logloss", "within_user_auroc"]]
              .to_string(index=False))
        tabs.append(tab.assign(subset=sname))
        difs.append(d.assign(subset=sname))
    # does JEV's increment shrink as the programmer accumulates history?  same programmer draws for both buckets
    from sklearn.metrics import roc_auc_score
    users = np.unique(uid)
    uix = np.searchsorted(users, uid)
    rng9 = np.random.default_rng(9)
    CU = np.stack([np.bincount(rng9.integers(0, len(users), len(users)), minlength=len(users)) for _ in range(B)]).astype(float)
    early, late = idx < 10, idx >= 30
    rows_dd = []
    for l in ("LR", "GBDT"):
        for s_ in ("text", "all (text + conf + history)"):
            a_, b_ = f"{l} | {s_} + JEV", f"{l} | {s_}"
            gain = lambda sel, w: (roc_auc_score(y[sel], pred[a_][sel], sample_weight=w[sel]) -
                                   roc_auc_score(y[sel], pred[b_][sel], sample_weight=w[sel]))
            one = np.ones(len(y))
            pt = gain(early, one) - gain(late, one)
            bt = np.array([gain(early, cu[uix]) - gain(late, cu[uix]) for cu in CU])
            rows_dd.append(dict(model=f"{l} | {s_}", gain_first10=gain(early, one), gain_30plus=gain(late, one),
                                diff=pt, lo=np.quantile(bt, .025), hi=np.quantile(bt, .975)))
    dd = pd.DataFrame(rows_dd)
    dd.to_csv(os.path.join(OUT, "cold_start_diff.csv"), index=False)
    print("\n== JEV's AUROC increment on a programmer's first 10 suggestions minus on suggestions 30+ "
          "(same programmer bootstrap for both)\n" + dd.round(3).to_string(index=False))
    pd.concat(tabs).to_csv(os.path.join(OUT, "summary.csv"))
    pd.concat(difs).to_csv(os.path.join(OUT, "diffs.csv"), index=False)


if __name__ == "__main__":
    main()
