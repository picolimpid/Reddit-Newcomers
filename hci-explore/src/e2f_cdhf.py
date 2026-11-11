"""E2f: head-to-head with CDHF (Mozannar, Bansal, Fourney, Horvitz, AAAI 2024, arXiv 2306.04930) on RealHumanEval.

What CDHF does (read from the paper, sections 5-6 and appendix C): a gradient-boosted tree model (XGBoost, default
parameters) on telemetry features predicts whether a shown inline completion will be accepted; suggestions whose
predicted acceptance falls below a threshold are hidden, the threshold being set on validation data so that hidden
suggestions would have been rejected with probability >= p (their headline: hide 25.3% of shown suggestions at
p = 0.947).  Features: the suggesting model's confidence, suggestion length / lines / tokens, Python keyword flags,
prompt and document length, the previous five actions, the event index, codeBERTa embeddings of prompt and
suggestion (and a programmer one-hot only in their session split).  A stage-1 model without suggestion features
decides some cases before generating (AUC 0.631 there).

Here: same data, rows and programmer folds as E2e (4,765 unrequested suggestions, 83 programmers, 5 folds over
programmers).  xgboost is not installed, so the replica uses scikit-learn's histogram gradient boosting with its
default parameters (the same model family; CDHF did no tuning either).  Embeddings: the bge-small vectors E2e already
computed for "last 400 characters before the cursor + suggestion" (PCA 32, fitted per fold), standing in for
codeBERTa.  No API calls: JEV's answers are E2e's cached ones.

  python3 src/e2f_cdhf.py
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
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import e2e_realhumaneval as e2e  # noqa: E402

OUT = os.path.join(e2e.jev.ROOT, "results", "e2f")
os.makedirs(OUT, exist_ok=True)
NF, NIN, B = 5, 4, 2000
HIST_NS = (10, 20, 30)
TNRS = (0.95, 0.97, 0.99)
KW = ["def ", "return", "#", "import", "if ", "for ", "while ", "class ", "print", "try", "self.", "(", ":"]
_G = {}


def cdhf_features(a):
    """Telemetry features in the spirit of CDHF's figure 4, computed only from the past and the current suggestion."""
    full = pd.read_csv(os.path.join(e2e.D, "autocomplete_autocomplete_data.csv"))
    full["row"] = np.arange(len(full))  # same row numbering as e2e.load()
    full = full.sort_values(["programmer_id", "timestamp", "row"])
    hist = {}
    for _, g in full.groupby("programmer_id"):  # all shown events, requested ones included, as in CDHF's telemetry
        acc, rows = g.accepted.astype(float).values, g.row.values
        for k in range(len(g)):
            p5 = list(acc[max(0, k - 5):k][::-1])
            hist[rows[k]] = p5 + [np.nan] * (5 - len(p5)) + [acc[max(0, k - 10):k].sum() if k else np.nan, k]
    H = pd.DataFrame([hist[r] for r in a.row], columns=[f"prev{i}_accepted" for i in range(1, 6)] +
                     ["accepts_in_last10", "event_index_all"])
    s, pre, suf = a.suggestion, a.prefix_code, a.suffix_code
    cur = pre.str.split("\n").str[-1]
    X = {"sugg_tokens": s.str.split().str.len(),
         "sugg_single_nonalnum": ((s.str.strip().str.len() <= 1) & ~s.str.strip().str.isalnum()),
         "mid_word": pre.str[-1:].str.isalnum() & s.str[:1].str.isalnum(),
         "prefix_lines": pre.str.count("\n") + 1, "doc_len": pre.str.len() + suf.str.len(),
         "cur_line_len": cur.str.len(), "cur_line_blank": cur.str.strip().str.len() == 0,
         "prefix_ends_newline": pre.str.endswith("\n")}
    for kw in KW:
        X["kw_" + kw.strip()] = s.str.contains(kw, regex=False)
    X = pd.DataFrame(X).astype(float)
    return pd.concat([X.reset_index(drop=True), H], axis=1)


STAGE1_DROP = ("lp_", "log_len_suggestion", "log_lines_suggestion", "sugg_", "mid_word", "kw_")


def logit(p):
    return e2e.logit(p)


def hgb():
    return HistGradientBoostingClassifier(random_state=0)


def fit_predict(name, Xs, y, tr, te):
    """One global model: train rows -> margins (log-odds) on test rows."""
    kind, cols = MODELS[name]
    X = Xs[cols]
    if kind == "hgb":
        return hgb().fit(X[tr], y[tr]).decision_function(X[te])
    C = 100. if cols == "J" else 1.
    return LogisticRegression(C=C, max_iter=5000).fit(X[tr], y[tr]).decision_function(X[te])


MODELS = {  # name: (learner, feature block)
    "JEV two answers + Platt": ("lr", "J"),
    "telemetry LR (E2e)": ("lr", "T"),
    "telemetry + JEV LR (E2e)": ("lr", "TJ"),
    "CDHF stage 1 (no suggestion features)": ("hgb", "S1"),
    "CDHF replica (GBDT)": ("hgb", "F"),
    "CDHF replica + embeddings": ("hgb", "FE"),
    "CDHF replica + JEV": ("hgb", "FJ"),
    "CDHF replica + embeddings + JEV": ("hgb", "FEJ"),
    "CDHF features LR": ("lr", "Fs"),
    "CDHF features + JEV, LR": ("lr", "FsJ"),
}
OFFSET_BASES = ("CDHF replica (GBDT)", "CDHF replica + JEV", "CDHF features LR", "CDHF features + JEV, LR")


def blocks(f, tr):
    T, F, J, E, s1 = (_G[k] for k in ("T", "F", "J", "E", "s1"))
    Ts = (T - T[tr].mean(0)) / (T[tr].std(0) + 1e-8)
    Fn = np.where(np.isnan(F), np.nanmedian(F[tr], 0), F)  # LR needs no NaN; trees get the raw NaN
    Fs = (Fn - Fn[tr].mean(0)) / (Fn[tr].std(0) + 1e-8)
    Ec = E - E[tr].mean(0)
    _, _, Vt = np.linalg.svd(Ec[tr], full_matrices=False)
    Ep = Ec @ Vt[:32].T
    FT = np.c_[T, F]  # CDHF features = E2e telemetry (confidence, lengths, model, timing, index) + the extra ones
    return {"J": J, "T": Ts, "TJ": np.c_[Ts, J], "S1": FT[:, s1], "F": FT, "FE": np.c_[FT, Ep], "FJ": np.c_[FT, J],
            "FEJ": np.c_[FT, Ep, J], "Fs": np.c_[Ts, Fs], "FsJ": np.c_[Ts, Fs, J]}


def fold_job(f):
    a, ufold = _G["a"], _G["ufold"]
    y = a.accepted.values.astype(float)
    uid, idx = a.programmer_id.values, a.idx.values
    tr, te = np.flatnonzero(ufold[uid] != f), np.flatnonzero(ufold[uid] == f)
    Xs = blocks(f, tr)
    m_te = {n: fit_predict(n, Xs, y, tr, te) for n in MODELS}
    # cross-fitted margins on the training programmers: thresholds for the TNR guarantee and the offsets' prior
    tr_users = np.unique(uid[tr])
    inner = {u: k % NIN for k, u in enumerate(np.random.default_rng([f, 7]).permutation(tr_users))}
    ifold = np.array([inner.get(u, -1) for u in uid])
    m_cf = {n: np.full(len(y), np.nan) for n in MODELS}
    for g in range(NIN):
        ra, rb = tr[ifold[tr] != g], tr[ifold[tr] == g]
        Xg = blocks(f, ra)
        for n in MODELS:
            m_cf[n][rb] = fit_predict(n, Xg, y, ra, rb)
    thr = {}
    for n in MODELS:  # CDHF's post-hoc rule: largest threshold whose hidden set is >= p rejected (on training rows)
        m = m_cf[n][tr]
        o = np.argsort(m)
        rej = np.cumsum(1 - y[tr][o]) / np.arange(1, len(o) + 1)
        for p in TNRS:
            ok = np.flatnonzero(rej >= p)
            thr[(n, p)] = m[o][ok.max()] if len(ok) else -np.inf
    # per-programmer offset (Bayesian intercept): prior precision chosen on training programmers, as in E2e
    nuser = a.groupby("programmer_id").size()
    ev_tr = tr[(nuser[uid[tr]].values >= e2e.MIN_N) & (idx[tr] >= e2e.EVAL_FROM)]
    lam = {}
    for b in OFFSET_BASES:
        best = None
        for l_ in e2e.LAMS:
            ll = []
            for u in np.unique(uid[ev_tr]):
                ru = np.flatnonzero(uid == u)
                eu = ru[idx[ru] >= e2e.EVAL_FROM]
                for n in HIST_NS:
                    h = ru[idx[ru] < n]
                    mm = m_cf[b][eu] + e2e.user_offset(m_cf[b][h], y[h], l_)
                    ll.append(np.logaddexp(0, mm) - y[eu] * mm)
            ll = np.mean(np.concatenate(ll))
            if best is None or ll < best[0]:
                best = (ll, l_)
        lam[b] = best[1]
    M = np.full((len(y), len(MODELS)), np.nan)
    for k, n in enumerate(MODELS):
        M[te, k] = m_te[n]
    off = {}
    for b in OFFSET_BASES:
        for n in HIST_NS:
            v = np.full(len(y), np.nan)
            for u in np.unique(uid[te]):
                ru = np.flatnonzero(uid == u)
                h = ru[idx[ru] < n]
                later = ru[idx[ru] >= n]
                full_m = np.full(len(y), np.nan)
                full_m[te] = m_te[b]
                v[later] = full_m[later] + e2e.user_offset(full_m[h], y[h], lam[b])
            off[f"{b} + programmer offset | n={n}"] = v[te]
    return f, te, M[te], off, thr, lam


def _init_worker():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def boot_table(y, user, pred, names, B_, rng):
    users = np.unique(user)
    uix = np.searchsorted(users, user)
    tieb = np.random.default_rng(5).random(len(y))
    CU = np.stack([np.bincount(rng.integers(0, len(users), len(users)), minlength=len(users)) for _ in range(B_)]).astype(float)
    point, boot = {}, {}
    for n in names:
        m = pred[n]
        order, pairs = np.lexsort((tieb, m)), e2e.user_pairs(y, m, user, users)
        point[n] = e2e.metric_vec(y, m, np.ones(len(y)), order, np.ones(len(users)), pairs)
        boot[n] = np.array([e2e.metric_vec(y, m, cu[uix], order, cu, pairs) for cu in CU])
    tab = pd.DataFrame({n: [f"{point[n][j]:.3f} [{np.quantile(boot[n][:, j], .025):.3f}, {np.quantile(boot[n][:, j], .975):.3f}]"
                            for j in range(len(e2e.METRICS))] for n in names}, index=e2e.METRICS).T
    return tab, point, boot


def diffs(pairs, point, boot):
    rows = []
    for x, z in pairs:
        d, db = point[x] - point[z], boot[x] - boot[z]
        rows.append({"pair": f"{x}  -  {z}", **{m: f"{d[j]:+.3f} [{np.nanquantile(db[:, j], .025):+.3f}, "
                                                  f"{np.nanquantile(db[:, j], .975):+.3f}]" for j, m in enumerate(e2e.METRICS)}})
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    a, T, tcols, J, E, U = e2e.design()
    F = cdhf_features(a)
    fcols = tcols + list(F.columns)
    s1 = np.array([not c.startswith(STAGE1_DROP) for c in fcols])
    um = a.groupby("programmer_id").model.first()  # the E2e folds, reproduced exactly
    order = np.concatenate([np.random.default_rng(0).permutation(g.index.values) for _, g in um.groupby(um)])
    ufold = pd.Series(np.arange(len(order)) % NF, index=order)
    ufold_arr = np.full(a.programmer_id.max() + 1, -1)
    ufold_arr[ufold.index.values] = ufold.values
    _G.update(a=a, T=T, F=F.values.astype(float), J=J, E=E, s1=s1, ufold=ufold_arr)
    print(f"CDHF-style features: {len(fcols)} ({len(tcols)} from E2e + {F.shape[1]} new); stage 1 keeps {s1.sum()}")
    t0 = time.time()
    with ProcessPoolExecutor(NF, mp_context=mp.get_context("fork"), initializer=_init_worker) as ex:
        res = list(ex.map(fold_job, range(NF)))
    print(f"{NF} folds over programmers in {time.time() - t0:.0f}s; offset prior precision per fold: "
          + "; ".join(f"{f}: {lam}" for f, _, _, _, _, lam in res))
    y = a.accepted.values.astype(float)
    uid, idx = a.programmer_id.values, a.idx.values
    names = list(MODELS)
    pred = {n: np.full(len(y), np.nan) for n in names}
    offn = list(res[0][3])
    for n in offn:
        pred[n] = np.full(len(y), np.nan)
    for f, te, Mte, off, thr, lam in res:
        for k, n in enumerate(names):
            pred[n][te] = Mte[:, k]
        for n in offn:
            pred[n][te] = off[n]
    # ---- 1. all 4,765 suggestions, global models
    rng = np.random.default_rng(0)
    tab_all, p_all, b_all = boot_table(y, uid, pred, names, B, rng)
    print(f"\n== all {len(y)} unrequested suggestions, {len(np.unique(uid))} held-out programmers "
          f"(95% CI, programmer bootstrap {B})\n" + tab_all.to_string())
    d_all = diffs([("CDHF replica (GBDT)", "telemetry LR (E2e)"), ("CDHF replica + embeddings", "CDHF replica (GBDT)"),
                   ("CDHF replica + JEV", "CDHF replica (GBDT)"), ("CDHF replica + embeddings + JEV", "CDHF replica + embeddings"),
                   ("CDHF features + JEV, LR", "CDHF replica + JEV"), ("CDHF replica (GBDT)", "JEV two answers + Platt"),
                   ("CDHF features LR", "CDHF replica (GBDT)"), ("CDHF features + JEV, LR", "CDHF features LR"),
                   ("CDHF replica + JEV", "telemetry + JEV LR (E2e)")], p_all, b_all)
    print("\n== paired differences, all suggestions\n" + d_all.to_string(index=False))
    # CDHF's operating points: thresholds set on the training programmers, applied to the held-out ones
    op = []
    for n in names:
        for p in TNRS:
            hide = np.zeros(len(y), bool)
            for f, te, *_ in res:
                hide[te] = pred[n][te] <= res[f][4][(n, p)]
            hid = hide.sum()
            op.append({"model": n, "target TNR": p, "hidden %": 100 * hid / len(y),
                       "TNR on held-out %": 100 * (1 - y[hide].mean()) if hid else np.nan,
                       "accepted lost %": 100 * y[hide].sum() / y.sum()})
    op = pd.DataFrame(op)
    print("\n== CDHF operating points (hide when the score is below a threshold chosen on training programmers so that "
          ">= p of hidden ones were rejected)\n" + op.round(1).to_string(index=False))
    # ---- 2. the E2e evaluation set (index >= 30 of programmers with >= 40), with per-programmer offsets
    nuser = a.groupby("programmer_id").size()
    ev = np.flatnonzero((nuser[uid].values >= e2e.MIN_N) & (idx >= e2e.EVAL_FROM))
    names2 = names + offn
    tab_ev, p_ev, b_ev = boot_table(y[ev], uid[ev], {n: pred[n][ev] for n in names2}, names2, B, np.random.default_rng(1))
    print(f"\n== E2e evaluation set: {len(ev)} suggestions (index >= {e2e.EVAL_FROM}) of {len(np.unique(uid[ev]))} programmers\n"
          + tab_ev.to_string())
    pairs = [("CDHF replica + JEV", "CDHF replica (GBDT)"), ("CDHF features + JEV, LR", "CDHF features LR"),
             ("CDHF features LR", "CDHF replica (GBDT)")]
    for n in HIST_NS:
        pairs += [(f"CDHF replica (GBDT) + programmer offset | n={n}", "CDHF replica (GBDT)"),
                  (f"CDHF replica + JEV + programmer offset | n={n}", "CDHF replica + JEV"),
                  (f"CDHF features LR + programmer offset | n={n}", "CDHF features LR"),
                  (f"CDHF features + JEV, LR + programmer offset | n={n}", "CDHF features + JEV, LR")]
    pairs += [("CDHF replica + JEV + programmer offset | n=30", "CDHF replica (GBDT)"),
              ("CDHF features + JEV, LR + programmer offset | n=30", "CDHF features LR + programmer offset | n=30"),
              ("CDHF features + JEV, LR + programmer offset | n=30", "CDHF replica (GBDT)")]
    d_ev = diffs(pairs, p_ev, b_ev)
    print("\n== paired differences, E2e evaluation set\n" + d_ev.to_string(index=False))
    tab_all.to_csv(os.path.join(OUT, "summary_all.csv"))
    tab_ev.to_csv(os.path.join(OUT, "summary_eval.csv"))
    pd.concat([d_all.assign(set="all"), d_ev.assign(set="eval")]).to_csv(os.path.join(OUT, "diffs.csv"), index=False)
    op.to_csv(os.path.join(OUT, "operating_points.csv"), index=False)
    pd.DataFrame({"row": a.row.values, "programmer_id": uid, "idx": idx, "accepted": y, **pred}).to_parquet(
        os.path.join(OUT, "oof.parquet"))


if __name__ == "__main__":
    main()
