"""E2e: Idea 2 on real users.  RealHumanEval (Mozannar et al. 2024, arXiv 2404.02806; HF hsseinmz/realhumaneval):
83 programmers in the autocomplete conditions, 5,204 inline suggestions shown (4,765 unrequested), each logged with accepted / not, the
code before and after the cursor, the suggesting model's token-logprob summary, and the programmer id.  Unlike
ProactiveBench (no user ids, 16 sessions), this has per-user accept decisions, so the user's question "does an
external network that learns an offset on top of JEV help?" can be asked where offsets could actually exist.

Deployment setting: a global gate is trained on *other* programmers (5-fold over programmers); a new programmer's own
first n decisions (n = 0, 5, 10, 20, 30, in time order) personalise it; it is scored on that programmer's later
suggestions with index >= 30.  Only unrequested suggestions are
used (history and evaluation), the ones a proactive gate would decide to show or hide.

  python3 src/e2e_realhumaneval.py prep --dry   # cost estimate only
  python3 src/e2e_realhumaneval.py prep         # JEV on the 4,765 unrequested suggestions, embedding texts, local-Qwen prompts
  CUDA_VISIBLE_DEVICES="" $PY src/embed_local.py results/e2e/emb_in.jsonl results/e2e/emb_bge-small.npz
  python3 src/e2e_realhumaneval.py analyze
"""
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import jev  # noqa: E402

OUT = os.path.join(jev.ROOT, "results", "e2e")
os.makedirs(OUT, exist_ok=True)
D = os.path.join(jev.ROOT, "data", "ext", "realhumaneval")
PRE, SUF = 1500, 600  # characters of code kept before / after the cursor
QUESTIONS = {  # short on purpose: every question's text is billed on every request
    "accept": "An AI code assistant has just shown `suggestion` as an inline completion at the cursor, between "
              "`code_before_cursor` and `code_after_cursor`. Would the programmer accept it (insert it into their code)?",
    "useful": "Is `suggestion` a correct and useful completion at the cursor, between `code_before_cursor` and "
              "`code_after_cursor`, i.e. it fits the code and does what the code is evidently trying to do?",
}


def load():
    a = pd.read_csv(os.path.join(D, "autocomplete_autocomplete_data.csv"))
    a["row"] = np.arange(len(a))
    a = a[~a.requested]  # proactive (unrequested) suggestions only: the ones a gate would show or hide
    a = a.sort_values(["programmer_id", "timestamp", "row"]).reset_index(drop=True)
    for c in ("prefix_code", "suffix_code", "suggestion"):
        a[c] = a[c].fillna("")
    return a


def state(r):
    return {"code_before_cursor": r.prefix_code[-PRE:], "code_after_cursor": r.suffix_code[:SUF],
            "suggestion": r.suggestion}


def prep(dry=False):
    a = load()
    qs = {k: {"type": "noul", "instructions": q} for k, q in QUESTIONS.items()}
    reqs = [{"state": state(r), "questions": qs, "tag": "e2e-rhe"} for r in a.itertuples()]
    est = sum((len(json.dumps({"model": jev.MODEL, "state": q["state"], "questions": q["questions"]})) / 2.5 + 200)
              for q in reqs) * jev.USD_PER_TOKEN
    print(f"{len(reqs)} requests, conservative estimate ${est:.4f}; JEV spent so far ${jev.spent_usd():.4f}")
    if dry:
        return
    t0, before = time.time(), jev.spent_usd()
    outs = jev.ask_many(reqs, workers=8)
    pd.DataFrame([{"row": r, **{"j_" + k: jev.p_yes(o, k) for k in QUESTIONS}, "cached": o["_cached"]}
                  for r, o in zip(a.row, outs)]).to_parquet(os.path.join(OUT, "jev.parquet"))
    with open(os.path.join(OUT, "emb_in.jsonl"), "w") as f:
        for r in a.itertuples():
            f.write(json.dumps({"id": str(r.row), "text": f"{r.prefix_code[-400:]}\n# --- suggestion ---\n{r.suggestion}"}) + "\n")
    with open(os.path.join(OUT, "qwen_in.jsonl"), "w") as f:
        for r in a.itertuples():
            s = state(r)
            for k, q in QUESTIONS.items():
                user = (f"`code_before_cursor`:\n```python\n{s['code_before_cursor']}\n```\n\n`code_after_cursor`:\n```python\n"
                        f"{s['code_after_cursor']}\n```\n\n`suggestion`:\n```python\n{s['suggestion']}\n```\n\n"
                        f"Question: {q}\nAnswer with Yes or No only.")
                f.write(json.dumps({"id": f"{r.row}|{k}", "options": ["Yes", "No"], "user": user}) + "\n")
    print(f"done in {time.time() - t0:.0f}s, spent now ${jev.spent_usd() - before:.4f}; JEV total ${jev.spent_usd():.4f}")


# ----------------------------------------------------------------------------------------------------------- analysis
import ast  # noqa: E402
import multiprocessing as mp  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.special import expit  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from offset_mlp import OffsetMLP  # noqa: E402

NS = (0, 5, 10, 20, 30)    # history: the programmer's own first n decisions (time order)
EVAL_FROM, MIN_N = 30, 40  # scored on suggestions with index >= 30, programmers with >= 40 suggestions
NF, NIN, B = 5, 4, 2000
LAMS = (0.25, 0.5, 1., 2., 4., 8., 16.)  # prior precision of a programmer's offset
MLP_CFG = dict(hidden=(32,), wd=1e-2, lr=1e-3, batch=128, dropout=0.2)
MLP_MAX_EPOCHS, MLP_PATIENCE, MLP_SEEDS, NPCA = 60, 6, 5, 16
ORD = {"prog_experience": ["0 to 2", "3 to 5", "6 to 10", "11 to 15", "More than 16"],
       "python_experience": ["Beginner", "Intermediate", "Advanced"],
       "ai_experience": ["Never", "Rarely", "Sometimes", "Often", "Always"],
       "aiToolTypicalUsage": ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]}
_G = {}


def logit(p):
    p = np.clip(np.asarray(p, float), .005, .995)
    return np.log(p / (1 - p))


def design():
    a = load()
    j = pd.read_parquet(os.path.join(OUT, "jev.parquet")).set_index("row").loc[a.row]
    a["j_accept"], a["j_useful"] = j.j_accept.values, j.j_useful.values
    a["idx"] = a.groupby("programmer_id").cumcount()
    lp = pd.DataFrame([ast.literal_eval(x) for x in a.logprobs])  # logged as a Python dict literal; literal_eval only
    dt = a.groupby("programmer_id").timestamp.diff() / 1000.
    T = pd.DataFrame({"lp_mean": lp["mean"], "lp_min": lp["min"], "lp_median": lp["median"], "lp_std": lp["std"],
                      "lp_first": lp["firstElement"], "lp_last": lp["lastElement"],
                      "log_len_suggestion": np.log1p(a.suggestion.str.len()),
                      "log_lines_suggestion": np.log1p(a.suggestion.str.count("\n")),
                      "log_len_prefix": np.log1p(a.prefix_code.str.len()),
                      "has_suffix": (a.suffix_code.str.strip().str.len() > 0).astype(float),
                      "model_codellama7b": (a.model == "CodeLlama7b").astype(float),
                      "model_gpt35": (a.model == "GPT-3.5").astype(float),
                      "log_seconds_since_last": np.log1p(dt.fillna(dt.median()).clip(lower=0)),
                      "log_index": np.log1p(a.idx)})
    T = T.fillna(T.median())
    J = np.c_[logit(a.j_accept), logit(a.j_useful)]
    z = np.load(os.path.join(OUT, "emb_bge-small.npz"))
    pos = {i: k for k, i in enumerate(z["ids"])}
    E = z["emb"][[pos[str(r)] for r in a.row]].astype(float)
    st = pd.read_csv(os.path.join(D, "study_study_data.csv")).set_index("programmer_id").loc[a.programmer_id]
    U = []
    for c, levels in ORD.items():
        v = st[c].astype(str).map(lambda x: next((k for k, l in enumerate(levels) if x.startswith(l)), np.nan))
        U += [v.fillna(v.median()).values, v.isna().astype(float).values]
    return a, T.values.astype(float), list(T.columns), J, E, np.c_[tuple(U)]


def lr(X, y, C):
    return LogisticRegression(C=C, max_iter=5000).fit(X, y)


GLOBAL = {  # name: (feature builder, C)
    "JEV + Platt": (lambda T, J, U: J[:, :1], 100.),
    "JEV two answers + Platt": (lambda T, J, U: J, 100.),
    "telemetry LR (CDHF-style, no JEV)": (lambda T, J, U: T, 1.),
    "telemetry + JEV LR": (lambda T, J, U: np.c_[T, J], 1.),
    "telemetry + JEV + survey LR": (lambda T, J, U: np.c_[T, J, U], 1.),
}


def global_margins(tr_rows, te_rows, Ts, J, Us, y):
    out = {}
    for name, (fx, C) in GLOBAL.items():
        X = fx(Ts, J, Us)
        out[name] = lr(X[tr_rows], y[tr_rows], C).decision_function(X[te_rows])
    return out


def user_offset(m, y, lam):
    if len(m) == 0:
        return 0.
    f = lambda u: (np.logaddexp(0, m + u) - y * (m + u)).sum() + lam / 2 * u * u
    return minimize_scalar(f, bounds=(-8, 8), method="bounded").x


def hist_feats(m_hist, y_hist, n):
    return np.array([n / 30., (y_hist.sum() - expit(m_hist).sum()) / (n + 5.), y_hist.sum() / 5.])


def mlp_fit(X, o, y, val):
    h = OffsetMLP(**MLP_CFG, epochs=MLP_MAX_EPOCHS, patience=MLP_PATIENCE, seed=0).fit(
        X[~val], o[~val], y[~val], val=(X[val], o[val], y[val]))
    base = float(np.mean(np.logaddexp(0, o[val]) - y[val] * o[val]))
    ep = int(np.argmin(h)) + 1 if h and min(h) < base else 0
    models = [OffsetMLP(**MLP_CFG, epochs=ep, patience=0, seed=s) for s in range(MLP_SEEDS)] if ep else []
    for mdl in models:
        mdl.fit(X, o, y)
    return (lambda Xt: np.mean([mdl.residual(Xt) for mdl in models], 0) if models else 0.), ep


def fold_job(f):
    a, T, J, E, U, ufold = (_G[x] for x in ("a", "T", "J", "E", "U", "ufold"))
    y = a.accepted.values.astype(float)
    uid, idx = a.programmer_id.values, a.idx.values
    nuser = a.groupby("programmer_id").size()
    tr = np.flatnonzero(ufold[uid] != f)
    te = np.flatnonzero(ufold[uid] == f)
    mu, sd = T[tr].mean(0), T[tr].std(0) + 1e-8
    Ts = (T - mu) / sd
    Us = (U - U[tr].mean(0)) / (U[tr].std(0) + 1e-8)
    Ec = E - E[tr].mean(0)
    _, _, Vt = np.linalg.svd(Ec[tr], full_matrices=False)
    Ep = Ec @ Vt[:NPCA].T
    Ep = Ep / (Ep[tr].std(0) + 1e-8)
    # final global fits (train programmers -> test programmers) and cross-fitted margins on the train programmers
    m_te = global_margins(tr, te, Ts, J, Us, y)
    tr_users = np.unique(uid[tr])
    inner = {u: k % NIN for k, u in enumerate(np.random.default_rng([f, 3]).permutation(tr_users))}
    ifold = np.array([inner.get(u, -1) for u in uid])
    m_cf = {n: np.full(len(y), np.nan) for n in GLOBAL}
    for g in range(NIN):
        a_rows, b_rows = tr[ifold[tr] != g], tr[ifold[tr] == g]
        for name, v in global_margins(a_rows, b_rows, Ts, J, Us, y).items():
            m_cf[name][b_rows] = v
    M = {n: np.full(len(y), np.nan) for n in GLOBAL}
    for n in GLOBAL:
        M[n][te] = m_te[n]
        M[n][tr] = m_cf[n][tr]  # train rows: cross-fitted (used only to train the personalisation layers)
    # global MLP offsets (no history)
    val = np.isin(uid, [u for u in tr_users if inner[u] == 0])
    preds, chosen = {}, {}
    for name, base, X in [("telemetry + JEV + MLP offset", "telemetry + JEV LR", np.c_[Ts, J, Ep]),
                          ("MLP alone (no JEV)", "telemetry LR (CDHF-style, no JEV)", np.c_[Ts, Ep])]:
        res, ep = mlp_fit(X[tr], M[base][tr], y[tr], val[tr])
        preds[name] = M[base][te] + res(X[te])
        chosen[name] = ep
    # personalisation: per-programmer offset (Bayesian intercept) and the MLP that reads the history
    ev_tr = tr[(nuser[uid[tr]].values >= MIN_N) & (idx[tr] >= EVAL_FROM)]
    lam_best = {}
    for base in ("JEV two answers + Platt", "telemetry + JEV LR"):
        best = None
        for lam in LAMS:
            ll = []
            for u in np.unique(uid[ev_tr]):
                rows_u = np.flatnonzero(uid == u)
                ev_u = rows_u[idx[rows_u] >= EVAL_FROM]
                for n in NS[1:]:
                    h = rows_u[idx[rows_u] < n]
                    off = user_offset(M[base][h], y[h], lam)
                    mm = M[base][ev_u] + off
                    ll.append(np.logaddexp(0, mm) - y[ev_u] * mm)
            ll = np.mean(np.concatenate(ll))
            if best is None or ll < best[0]:
                best = (ll, lam)
        lam_best[base] = best[1]
    base = "telemetry + JEV LR"
    Xh, oh, yh, vh = [], [], [], []
    for u in np.unique(uid[ev_tr]):
        rows_u = np.flatnonzero(uid == u)
        ev_u = rows_u[idx[rows_u] >= EVAL_FROM]
        for n in NS:
            h = rows_u[idx[rows_u] < n]
            hf = hist_feats(M[base][h], y[h], n)
            Xh.append(np.c_[Ts[ev_u], J[ev_u], np.repeat(hf[None], len(ev_u), 0)])
            oh.append(M[base][ev_u])
            yh.append(y[ev_u])
            vh.append(np.full(len(ev_u), inner[u] == 0))
    res_h, chosen["telemetry + JEV + MLP reading the history"] = mlp_fit(
        np.concatenate(Xh), np.concatenate(oh), np.concatenate(yh), np.concatenate(vh))
    ev_te = te[(nuser[uid[te]].values >= MIN_N) & (idx[te] >= EVAL_FROM)]
    out = {"row_index": ev_te}
    pos_te = {r: k for k, r in enumerate(te)}
    sel = np.array([pos_te[r] for r in ev_te])
    for n in GLOBAL:
        out[n] = M[n][ev_te]
    for n in preds:
        out[n] = preds[n][sel]
    for n in NS:
        for b in ("JEV two answers + Platt", "telemetry + JEV LR"):
            out[f"{b} + programmer offset | n={n}"] = np.full(len(ev_te), np.nan)
        out[f"telemetry + JEV + MLP reading the history | n={n}"] = np.full(len(ev_te), np.nan)
    for u in np.unique(uid[ev_te]):
        rows_u = np.flatnonzero(uid == u)
        k_ev = np.flatnonzero(uid[ev_te] == u)
        ev_u = ev_te[k_ev]
        for n in NS:
            h = rows_u[idx[rows_u] < n]
            for b in ("JEV two answers + Platt", "telemetry + JEV LR"):
                out[f"{b} + programmer offset | n={n}"][k_ev] = M[b][ev_u] + user_offset(M[b][h], y[h], lam_best[b])
            hf = hist_feats(M[base][h], y[h], n)
            Xe = np.c_[Ts[ev_u], J[ev_u], np.repeat(hf[None], len(ev_u), 0)]
            out[f"telemetry + JEV + MLP reading the history | n={n}"][k_ev] = M[base][ev_u] + res_h(Xe)
    return f, out, dict(fold=f, **{f"lam {k}": v for k, v in lam_best.items()}, **{f"epochs {k}": v for k, v in chosen.items()})


def _init_worker():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


METRICS = ["auroc", "logloss", "brier", "within_user_auroc", "accepted_lost_hide50", "precision_hide25"]


def user_pairs(y, m, user, users):
    """Per programmer: concordant (accepted, rejected) pairs + half ties, and number of pairs."""
    out = np.zeros((len(users), 2))
    for k, u in enumerate(users):
        sel = user == u
        pos, neg = m[sel][y[sel] == 1], m[sel][y[sel] == 0]
        if len(pos) and len(neg):
            d = pos[:, None] - neg[None, :]
            out[k] = [(d > 0).sum() + .5 * (d == 0).sum(), d.size]
    return out


def metric_vec(y, m, w, order, cu, pairs):
    """Weighted metrics; w = row weights (programmer multiplicity in a bootstrap draw), order = rows sorted by score
    with a fixed random tie-break (JEV's probabilities come in steps of 0.01), cu = programmer weights."""
    W = w.sum()
    ll = (w * (np.logaddexp(0, m) - y * m)).sum() / W
    br = (w * (expit(m) - y) ** 2).sum() / W
    cw = np.cumsum(w[order])
    h50, h25 = order[cw <= .5 * W], order[cw <= .25 * W]
    return np.array([roc_auc_score(y, m, sample_weight=w), ll, br, (cu * pairs[:, 0]).sum() / (cu * pairs[:, 1]).sum(),
                     (w[h50] * y[h50]).sum() / (w * y).sum(), 1 - (w[h25] * y[h25]).sum() / w[h25].sum()])


def analyze():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    a, T, tcols, J, E, U = design()
    um = a.groupby("programmer_id").model.first()
    order = np.concatenate([np.random.default_rng(0).permutation(g.index.values) for _, g in um.groupby(um)])
    ufold = pd.Series(np.arange(len(order)) % NF, index=order)
    ufold_arr = np.full(a.programmer_id.max() + 1, -1)
    ufold_arr[ufold.index.values] = ufold.values
    _G.update(a=a, T=T, J=J, E=E, U=U, ufold=ufold_arr)
    t0 = time.time()
    with ProcessPoolExecutor(NF, mp_context=mp.get_context("fork"), initializer=_init_worker) as ex:
        res = list(ex.map(fold_job, range(NF)))
    print(f"{len(a)} unrequested suggestions, {a.programmer_id.nunique()} programmers, accept rate {a.accepted.mean():.3f}; "
          f"{NF} folds over programmers in {time.time() - t0:.0f}s")
    chosen = pd.DataFrame([c for _, _, c in res])
    chosen.to_csv(os.path.join(OUT, "chosen.csv"), index=False)
    print("chosen per fold:\n" + chosen.to_string(index=False))
    rows = np.concatenate([o["row_index"] for _, o, _ in res])
    names = [k for k in res[0][1] if k != "row_index"]
    pred = {n: np.concatenate([o[n] for _, o, _ in res]) for n in names}
    pred["JEV zero-shot"] = logit(a.j_accept.values[rows])
    names = ["JEV zero-shot"] + names
    y, user = a.accepted.values[rows].astype(float), a.programmer_id.values[rows]
    pd.DataFrame({"row": a.row.values[rows], "programmer_id": user, "accepted": y, **pred}).to_parquet(
        os.path.join(OUT, "oof.parquet"))
    print(f"evaluation: {len(rows)} suggestions (index >= {EVAL_FROM}) of {len(np.unique(user))} programmers, "
          f"accept rate {y.mean():.3f}")
    users = np.unique(user)
    uix = np.searchsorted(users, user)
    tieb = np.random.default_rng(5).random(len(y))
    rng = np.random.default_rng(0)
    CU = np.stack([np.bincount(rng.integers(0, len(users), len(users)), minlength=len(users)) for _ in range(B)]).astype(float)
    point, boot = {}, {}
    t0 = time.time()
    for n in names:
        m = pred[n]
        order, pairs = np.lexsort((tieb, m)), user_pairs(y, m, user, users)
        point[n] = metric_vec(y, m, np.ones(len(y)), order, np.ones(len(users)), pairs)
        boot[n] = np.array([metric_vec(y, m, cu[uix], order, cu, pairs) for cu in CU])
    print(f"programmer bootstrap ({B}) in {time.time() - t0:.0f}s")
    tab = pd.DataFrame({n: [f"{point[n][j]:.3f} [{np.quantile(boot[n][:, j], .025):.3f}, {np.quantile(boot[n][:, j], .975):.3f}]"
                            for j in range(len(METRICS))] for n in names}, index=METRICS).T
    tab.to_csv(os.path.join(OUT, "summary.csv"))
    print("\n== held-out programmers, suggestions from index 30 on (95% CI, programmer bootstrap)\n" + tab.to_string())
    pairs = [("JEV two answers + Platt", "JEV + Platt"), ("telemetry + JEV LR", "telemetry LR (CDHF-style, no JEV)"),
             ("telemetry + JEV LR", "JEV two answers + Platt"), ("telemetry + JEV + survey LR", "telemetry + JEV LR"),
             ("telemetry + JEV + MLP offset", "telemetry + JEV LR"), ("telemetry + JEV + MLP offset", "MLP alone (no JEV)")]
    for n in NS[1:]:
        pairs += [(f"JEV two answers + Platt + programmer offset | n={n}", "JEV two answers + Platt"),
                  (f"telemetry + JEV LR + programmer offset | n={n}", "telemetry + JEV LR"),
                  (f"telemetry + JEV + MLP reading the history | n={n}", f"telemetry + JEV LR + programmer offset | n={n}")]
    drows = []
    for x, z in pairs:
        d, db = point[x] - point[z], boot[x] - boot[z]
        drows.append({"pair": f"{x}  -  {z}", **{m: f"{d[j]:+.3f} [{np.nanquantile(db[:, j], .025):+.3f}, "
                                                   f"{np.nanquantile(db[:, j], .975):+.3f}]" for j, m in enumerate(METRICS)}})
    dif = pd.DataFrame(drows)
    dif.to_csv(os.path.join(OUT, "diffs.csv"), index=False)
    print("\n== paired differences (programmer bootstrap)\n" + dif.to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1:2] == ["prep"]:
        prep(dry="--dry" in sys.argv)
    elif sys.argv[1:] == ["analyze"]:
        analyze()
    else:
        print(__doc__)
