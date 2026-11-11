"""E2b (Idea 2, Study 1 / RQ2): how many accept/reject labels does a JEV acceptance gate need?

Data: ProactiveBench reward-model train set (thunlp/ProactiveAgent, dataset/reward_data/train_data.jsonl, 1,629
items).  What the data is (read off dataset/annotation/convert_annotations.py and the files themselves): 16 recorded
computer sessions, each cut at ~20 time points (313 "traces", each a prefix of its session); at each time point the
agent offered several candidate tasks and 2-3 real users ticked the ones they would accept; valid = majority (>= 2).
The official train/test split is item-level and random: 14 of the 16 sessions and all 96 test traces also occur in
train (with other candidate tasks), and the 60 test proposal items were drawn from unanimous votes first.
Protocols for "would the user accept this offer?" (proposal items only):
  P1 as published   fit on train proposal items, evaluate on the 60 test proposal items
  P2 unseen moments as P1, minus train items on test traces (same sessions, other time points)
  P3 unseen session leave-one-session-out over the 1,387 train proposal items (split votes included); metrics per
                    held-out session, then averaged
  P4 per-session    sessions as stand-in users: global model from the other 15 sessions, then Platt on the first k
                    labels of the held-out session (time order), evaluated on its second half
Learners (logistic regressions fitted by MAP; N labels drawn at random from the training pool):
  JEV zero-shot   JEV P(accept), no labels
  JEV + Platt     logit p' = a*logit(p) + b, Gaussian prior centred on (a, b) = (1, 0), i.e. on zero-shot JEV
  JEV 3Q + LR     accept / need / disrupt logits, prior centred on zero-shot JEV
  TF-IDF + LR     supervised from scratch on activity + proposal text (CDHF-style small-N baseline)
  Emb + LR        bge-small-en-v1.5 sentence embedding (open weights, local CPU), from scratch
  Emb + JEV       embedding + the 3 JEV logits, prior centred on zero-shot JEV
JEV-anchored calibrations are kept monotone (weight on the JEV accept logit >= 0), as a deployed gate would be.
Decision metrics: acc = accuracy of "show iff p > 0.5" (equals the gate's utility, up to a constant, when a rejected
offer costs as much as an accepted one gains); util@1/3 = utility per moment of "show iff p > 1/3" with +1 for an
accepted offer, -0.5 for a rejected one, 0 when nothing is shown.  "always show" / "never show" rows are the trivial
policies to beat.  Intervals: every repetition redraws the N labels and resamples sessions.

  python3 src/e2b_learning_curve.py prep     # JEV on the 1,629 train items (~$0.05); writes local-model inputs
  $PY src/embed_local.py results/e2b/emb_in.jsonl results/e2b/emb_bge-small.npz   # CPU, ~3 min
  python3 src/e2b_learning_curve.py analyze  # E2B_REPS (100), E2B_WORKERS (32)
"""
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import jev  # noqa: E402
import e2_proactive_gate as e2  # noqa: E402

OUT = os.path.join(jev.ROOT, "results", "e2b")
os.makedirs(OUT, exist_ok=True)
TRAIN = os.path.join(jev.ROOT, "data", "ext", "proactive_train", "train_data.jsonl")
EMB = os.path.join(OUT, "emb_bge-small.npz")
NS = [0, 10, 20, 50, 100, 200, 500, "all"]
KS = [0, 5, 10, 20]
REPS = int(os.environ.get("E2B_REPS", "100"))
WORKERS = int(os.environ.get("E2B_WORKERS", "32"))
METRICS = ["auroc", "acc", "util@1/3", "brier", "fa_block90"]
TRIVIAL = ("always show", "never show")


def trace_id(obs):
    return hashlib.sha1(json.dumps(obs, sort_keys=True).encode()).hexdigest()[:10]


def prep():
    tr, te = e2.load(TRAIN, prefix="t"), e2.load()
    print("JEV spent before: $%.4f" % jev.spent_usd())
    t0 = time.time()
    outs = jev.ask_many([e2.jev_item(r, "e2b-train") for r in tr]) + [jev.ask(**e2.jev_item(r)) for r in te]
    first = [json.dumps(r["obs"][0], sort_keys=True) for r in tr + te]
    sess = {k: i for i, k in enumerate(dict.fromkeys(first))}  # a session = all traces sharing the first event
    recs = []
    for split, r, o, f in zip(["train"] * len(tr) + ["test"] * len(te), tr + te, outs, first):
        t = r["pred_task"]
        recs.append({"id": r["id"], "split": split, "session": sess[f], "step": len(r["obs"]),
                     "trace": trace_id(r["obs"]), "cat": r["category"][-3:-1],
                     "help_needed": r["help_needed"], "valid": r["valid"], "has_task": bool(t),
                     "annotation": [int(a) for a in r["annotation"]],
                     "j_need": jev.p_yes(o, "need"), "j_disrupt": jev.p_yes(o, "disrupt"),
                     "j_accept": jev.p_yes(o, "accept") if t else np.nan,
                     "n_tok": o["usage"]["input_tokens"], "cached": o["_cached"],
                     "text": "\n".join(r["activity"]) + (f"\nProposed task: {t}" if t else "")})
    d = pd.DataFrame(recs)
    d.to_parquet(os.path.join(OUT, "jev_all.parquet"))
    # inputs for local open-weight models (never sent to an API)
    with open(os.path.join(OUT, "qwen_in.jsonl"), "w") as f:
        for r in tr:
            for line in e2.llm_lines(r):
                f.write(line + "\n")
    with open(os.path.join(OUT, "emb_in.jsonl"), "w") as f:
        for r in tr + te:
            t = r["pred_task"]
            head = f"Proposed task: {t}\n" if t else "No proposed task.\n"
            f.write(json.dumps({"id": r["id"], "text": head + "Recent activity (newest first):\n" +
                                "\n".join(reversed(r["activity"]))}) + "\n")
    led = pd.read_json(jev.LEDGER, lines=True).query("tag == 'e2b-train'")
    print(f"{len(tr)} train items in {time.time() - t0:.0f}s ({(~d.cached).sum()} uncached now); ledger e2b-train: "
          f"{len(led)} requests, {led.input_tokens.sum()} tokens, ${led.cost_usd.sum():.4f}, "
          f"latency median {led.latency_s.median():.3f}s p90 {led.latency_s.quantile(.9):.3f}s")
    print(f"sessions {d.session.nunique()} (train {d[d.split == 'train'].session.nunique()}, "
          f"test {d[d.split == 'test'].session.nunique()}); JEV spent after: ${jev.spent_usd():.4f}")


# ----------------------------------------------------------------------------------------------------------- analysis
def logit(p):
    p = np.clip(np.asarray(p, float), .005, .995)
    return np.log(p / (1 - p))


def fit_map(X, y, w0, lam, lam_b, lower=None):
    """Logistic regression, MAP with prior w ~ N(w0, 1/lam), b ~ N(0, 1/lam_b); lam may be per-feature.
    lower: optional per-feature lower bounds on w (0 keeps a calibration monotone). Works for dense or sparse X."""
    k = X.shape[1]

    def f(th):
        w, b = th[:k], th[k]
        m = X @ w + b
        r = expit(m) - y
        loss = np.logaddexp(0, m).sum() - y @ m + (lam * (w - w0) ** 2).sum() / 2 + lam_b / 2 * b * b
        return loss, np.append(X.T @ r + lam * (w - w0), r.sum() + lam_b * b)

    bounds = None if lower is None else [(lo if np.isfinite(lo) else None, None) for lo in lower] + [(None, None)]
    th = minimize(f, np.append(w0, 0.0), jac=True, method="L-BFGS-B", bounds=bounds).x
    return th[:k], th[k]


def learners(t, scratch_lams=(1.,)):
    """Feature scales are set so that a unit prior (lam = 1) gives margins of order 1: JEV logits as they are,
    TF-IDF rows are L2-normalised, embeddings are standardised and divided by sqrt(dim).  scratch_lams adds variants
    of the from-scratch learners with other prior strengths (sensitivity check)."""
    J = np.c_[logit(t.j_accept), logit(t.j_need), logit(t.j_disrupt)]
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(t.text).tocsr()
    inf = np.inf
    L = {"JEV + Platt": (J[:, :1], np.array([1.]), 1., 1., [0.]),
         "JEV 3Q + LR": (J, np.array([1., 0, 0]), 1., 1., [0., -inf, -inf])}
    E = None
    if os.path.exists(EMB):
        z = np.load(EMB, allow_pickle=False)
        pos = {i: k for k, i in enumerate(z["ids"])}
        E = z["emb"][[pos[i] for i in t.id]]
        E = (E - E.mean(0)) / (E.std(0) + 1e-8) / np.sqrt(E.shape[1])
    for lm in scratch_lams:
        tag = "" if lm == 1 else f" (lam={lm:g})"
        L["TF-IDF + LR" + tag] = (tf, np.zeros(tf.shape[1]), lm, 1e-3, None)
        if E is not None:
            L["Emb + LR" + tag] = (E, np.zeros(E.shape[1]), lm, 1e-3, None)
            L["Emb + JEV" + tag] = (np.c_[J, E], np.r_[1., 0, 0, np.zeros(E.shape[1])],
                                    np.r_[1., 1, 1, np.full(E.shape[1], lm)], 1., [0.] + [-inf] * (E.shape[1] + 2))
    if E is None:
        print("(no embeddings at %s: run src/embed_local.py to add the Emb baselines)" % EMB)
    return L


def metrics(y, s):
    n, P = len(y), int(y.sum())
    sh = s > 1 / 3
    out = [roc_auc_score(y, s) if 0 < P < n else np.nan,
           ((s > .5) == (y == 1)).mean(),
           ((sh & (y == 1)).sum() - .5 * (sh & (y == 0)).sum()) / n,
           ((s - y) ** 2).mean(),
           (s[y == 0] < np.quantile(s[y == 1], .1)).mean() if 0 < P < n else np.nan]  # E2's oracle-threshold metric
    return np.array(out)


def cluster_boot(groups, rng):
    u, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(u))]
    return np.concatenate([members[k] for k in rng.integers(0, len(u), len(u))])


# ---- parallel learning curves (fork: workers inherit _G)
_G = {}


def _init_worker():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def _job(job):
    proto, n, rep = job
    rng = np.random.default_rng([list(_G["splits"]).index(proto), _G["ns"].index(n), rep])
    y, L, ja = _G["y"], _G["L"], _G["j_accept"]
    res = []
    for pool, e in _G["splits"][proto]:
        pr = {"JEV zero-shot": ja[e], "always show": np.ones(len(e)), "never show": np.zeros(len(e))}
        if n != 0:
            tr = pool if n == "all" else rng.choice(pool, size=min(n, len(pool)), replace=False)
            for m, (X, w0, lam, lam_b, lo) in L.items():
                w, b = fit_map(X[tr], y[tr], w0, lam, lam_b, lo)
                pr[m] = expit(X[e] @ w + b)
        res.append(pr)
    return job, res


def learning_curves(t, y, L, splits, ns=NS, reps=REPS):
    _G.update(y=y, L=L, splits=splits, ns=list(ns), j_accept=t.j_accept.values.astype(float))
    jobs = [(p, n, r) for p in splits for n in ns for r in range(1 if n in (0, "all") else reps)]
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS, mp_context=mp.get_context("fork"), initializer=_init_worker) as ex:
        results = dict(ex.map(_job, jobs, chunksize=2))
    print(f"  {len(jobs)} fitting jobs in {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(1)
    rows = []
    for p, sp in splits.items():
        ys = [y[e] for _, e in sp]
        for n in ns:
            for rep in range(reps):
                res = results[(p, n, 0 if n in (0, "all") else rep)]
                if len(sp) == 1:  # one evaluation set: resample its sessions
                    bi = cluster_boot(_G["session"][sp[0][1]], rng)
                    for m, s in res[0].items():
                        full, boot = metrics(ys[0], s), metrics(ys[0][bi], s[bi])
                        rows += [(p, str(n), rep, m, k, full[j], boot[j]) for j, k in enumerate(METRICS)]
                else:  # one split per held-out session: average over sessions, resample sessions
                    pick = rng.integers(0, len(sp), len(sp))
                    for m in res[0]:
                        per = np.array([metrics(ys[i], res[i][m]) for i in range(len(sp))])
                        full, boot = np.nanmean(per, 0), np.nanmean(per[pick], 0)
                        rows += [(p, str(n), rep, m, k, full[j], boot[j]) for j, k in enumerate(METRICS)]
    long = pd.DataFrame(rows, columns=["protocol", "N", "rep", "method", "metric", "value", "value_b"])
    oof = {}
    if "P3 unseen session" in splits:
        for (pool, e), pr in zip(splits["P3 unseen session"], results[("P3 unseen session", "all", 0)]):
            for m, s in pr.items():
                oof.setdefault(m, np.full(len(y), np.nan))[e] = s
    return long, oof


def summarize(long, keys=("protocol", "N", "method", "metric")):
    g = long.groupby(list(keys), sort=False)
    return g.agg(mean=("value", "mean"), lo=("value_b", lambda v: v.quantile(.025)),
                 hi=("value_b", lambda v: v.quantile(.975))).reset_index()


def paired_diffs(long, pairs, keys=("protocol", "N", "metric"), rep="rep", col="method"):
    w = long.pivot_table(index=list(keys) + [rep], columns=col, values="value_b").reset_index()
    rows = []
    for a, b in pairs:
        if a not in w or b not in w:
            continue
        for kv, v in w.assign(diff=w[a] - w[b]).groupby(list(keys), sort=False)["diff"]:
            if v.notna().any():
                rows.append(dict(zip(keys, kv), pair=f"{a} - {b}", mean=v.mean(), lo=v.quantile(.025),
                                 hi=v.quantile(.975), p_le_0=(v <= 0).mean()))
    return pd.DataFrame(rows)


# ---- P4: sessions as stand-in users, online per-session calibration
def per_session(t, y, L, reps=20, seed=2):
    rng = np.random.default_rng(seed)
    tr = np.flatnonzero(t.split == "train")
    sess = t.session.values
    rows = []
    for s in np.unique(sess[tr]):
        own, pool = tr[sess[tr] == s], tr[sess[tr] != s]
        base = {"JEV zero-shot": logit(t.j_accept.values[own])}
        for m in ("JEV + Platt", "Emb + JEV"):
            if m in L:
                X, w0, lam, lam_b, lo = L[m]
                w, b = fit_map(X[pool], y[pool], w0, lam, lam_b, lo)
                base[m + " (other sessions)"] = X[own] @ w + b  # logits
        yo = y[own]
        for rep in range(reps):
            order = np.lexsort((rng.random(len(own)), t.step.values[own]))  # time order, ties shuffled
            hist, ev = order[:len(own) // 2], order[len(own) // 2:]
            rows.append([s, rep, "never show", 0] + list(metrics(yo[ev], np.zeros(len(ev)))))
            rows.append([s, rep, "always show", 0] + list(metrics(yo[ev], np.ones(len(ev)))))
            for m, z in base.items():
                for k in KS:
                    # per-user Platt on top of the base logit: slope prior N(1, 1/10) (kept near 1, >= 0),
                    # intercept prior N(0, 1) (the user's base rate may differ)
                    a, b = (np.array([1.]), 0.) if k == 0 else fit_map(z[hist[:k]][:, None], yo[hist[:k]],
                                                                     np.array([1.]), 10., 1., [0.])
                    sc = expit(a[0] * z[ev] + b)
                    rows.append([s, rep, m, k] + list(metrics(yo[ev], sc)))
    per = pd.DataFrame(rows, columns=["session", "rep", "method", "k"] + METRICS)
    # average the reps within session, then bootstrap sessions
    ps = per.groupby(["session", "method", "k"])[METRICS].mean().reset_index()
    sessions = ps.session.unique()
    longs = []
    for b in range(2000):
        pick = rng.choice(sessions, len(sessions))
        bs = pd.concat([ps[ps.session == x] for x in pick]) if b else ps
        longs.append(bs.groupby(["method", "k"])[METRICS].mean().assign(boot=b))
    lb = pd.concat(longs).reset_index().melt(id_vars=["method", "k", "boot"], var_name="metric")
    full = lb[lb.boot == 0].rename(columns={"value": "mean"}).drop(columns="boot")
    ci = lb[lb.boot > 0].groupby(["method", "k", "metric"]).value.quantile([.025, .975]).unstack()
    summ = full.merge(ci.rename(columns={.025: "lo", .975: "hi"}).reset_index(), on=["method", "k", "metric"])
    lbd = lb[lb.boot > 0].pivot_table(index=["method", "metric", "boot"], columns="k", values="value").reset_index()
    diffs = []
    for k in KS[1:]:
        dk = (lbd[k] - lbd[0]).groupby([lbd.method, lbd.metric])
        for (m, met), v in dk:
            if v.notna().any():
                diffs.append({"method": m, "metric": met, "pair": f"k={k} - k=0", "mean": v.mean(),
                              "lo": v.quantile(.025), "hi": v.quantile(.975)})
    return summ, pd.DataFrame(diffs), per


# ---- zero-shot on the large set, annotator agreement, individual votes
def zero_shot(d, rng, B=2000):
    tr = d[d.split == "train"].reset_index(drop=True)
    t = tr[tr.has_task].reset_index(drop=True)
    frac = t.annotation.apply(np.mean).values
    unan = t.annotation.apply(lambda a: len(set(a)) == 1).values
    yv, yn = t.valid.values.astype(int), tr.help_needed.values.astype(int)

    def stats(ti, ai):
        pa, u = t.j_accept.values[ti], unan[ti]
        out = {"accept_auroc (majority label)": roc_auc_score(yv[ti], pa),
               "accept_auroc, unanimous items": roc_auc_score(yv[ti][u], pa[u]),
               "accept_auroc, split-vote items": roc_auc_score(yv[ti][~u], pa[~u]),
               "spearman(P(accept), share of users accepting)": spearmanr(pa, frac[ti])[0],
               "need_auroc (help_needed, all items)": roc_auc_score(yn[ai], tr.j_need.values[ai])}
        out["accept_auroc - need_auroc"] = out["accept_auroc (majority label)"] - out["need_auroc (help_needed, all items)"]
        out["acc of show iff P(accept)>0.5"] = ((pa > .5) == yv[ti]).mean()
        out["acc of never showing"] = 1 - yv[ti].mean()
        return out

    point = stats(np.arange(len(t)), np.arange(len(tr)))
    ts, as_ = t.session.values, tr.session.values
    idx_t = {s: np.flatnonzero(ts == s) for s in np.unique(ts)}
    idx_a = {s: np.flatnonzero(as_ == s) for s in np.unique(as_)}
    boots = []
    for _ in range(B):
        pick = rng.choice(list(idx_a), len(idx_a))
        boots.append(stats(np.concatenate([idx_t[s] for s in pick]), np.concatenate([idx_a[s] for s in pick])))
    bt = pd.DataFrame(boots)
    zs = pd.DataFrame({"est": point, "lo": bt.quantile(.025), "hi": bt.quantile(.975)})
    lvl = t.assign(votes=t.annotation.apply(lambda a: f"{sum(a)}/{len(a)}")).groupby("votes").j_accept.agg(["mean", "count"])
    A = np.array([a for a in t.annotation if len(a) == 3])
    p1, n = A.mean(), 3
    po = np.mean((A.sum(1) * (A.sum(1) - 1) + (n - A.sum(1)) * (n - A.sum(1) - 1)) / (n * (n - 1)))
    agree = {"items with 3 users": len(A), "Fleiss kappa": (po - (p1 ** 2 + (1 - p1) ** 2)) / (1 - p1 ** 2 - (1 - p1) ** 2),
             "pairwise agreement": po, "accept rate per vote": p1, "split-vote share (all proposal items)": 1 - unan.mean(),
             "accept rate by annotation position": A.mean(0).round(3).tolist()}
    return zs, lvl, agree


def individual_votes(t, preds):
    """Would *this* user accept? Each annotator's own vote vs (a) the other annotators' votes, (b) model scores."""
    rows = []
    for i, a in enumerate(t.annotation):
        for j, v in enumerate(a):
            rows.append({"i": i, "vote": v, "n": len(a), "others": np.mean([x for k, x in enumerate(a) if k != j])})
    p = pd.DataFrame(rows)
    h = p[p.n == 3]
    out = {"other two users' votes (3-user items)": {
        "auroc": roc_auc_score(h.vote, h.others), "acc": np.where(h.others == .5, .5, (h.others > .5) == h.vote).mean(),
        "pairs": len(h)},
        "majority label incl. this user's own vote": {
        "auroc": np.nan, "acc": (p.vote == t.valid.values[p.i].astype(int)).mean(), "pairs": len(p)}}
    for m, s in preds.items():
        sc = s[p.i.values]
        out[m] = {"auroc": roc_auc_score(p.vote, sc), "acc": ((sc > .5) == p.vote).mean(), "pairs": len(p)}
    out = pd.DataFrame(out).T
    out["acc of never showing"] = 1 - p.vote.mean()
    return out


def cell(r):
    return f"{r['mean']:.3f} [{r.lo:.3f},{r.hi:.3f}]"


def analyze():
    pd.set_option("display.width", 250)
    rng = np.random.default_rng(0)
    d = pd.read_parquet(os.path.join(OUT, "jev_all.parquet"))
    dt = d[d.split == "train"]
    print(f"train {len(dt)} items / {dt.trace.nunique()} traces / {dt.session.nunique()} sessions; test "
          f"{sum(d.split == 'test')} items / {d[d.split == 'test'].trace.nunique()} traces / "
          f"{d[d.split == 'test'].session.nunique()} sessions; test traces also in train: "
          f"{d[d.split == 'test'].trace.isin(dt.trace).mean():.0%}")

    zs, lvl, agree = zero_shot(d, rng)
    zs.round(3).to_csv(os.path.join(OUT, "zero_shot.csv"))
    json.dump(agree, open(os.path.join(OUT, "agreement.json"), "w"), indent=1, default=float)
    print("\n== zero-shot JEV on the train set, no labels (CIs resample the 16 sessions)\n" + zs.round(3).to_string())
    print("\nmean JEV P(accept) by vote:\n" + lvl.round(3).T.to_string())
    print("\nannotator agreement on proposal items:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in agree.items()})

    t = d[d.has_task].reset_index(drop=True)
    y = t.valid.values.astype(float)
    L = learners(t)
    _G["session"] = t.session.values
    tr_idx, te_idx = np.flatnonzero(t.split == "train"), np.flatnonzero(t.split == "test")
    p2_pool = tr_idx[~np.isin(t.trace.values[tr_idx], t.trace.values[te_idx])]
    s_tr = t.session.values[tr_idx]
    splits = {"P1 as published": [(tr_idx, te_idx)],
              "P2 unseen moments": [(p2_pool, te_idx)],
              "P3 unseen session": [(tr_idx[s_tr != s], tr_idx[s_tr == s]) for s in np.unique(s_tr)]}
    print(f"\nP1 pool {len(tr_idx)}, P2 pool {len(p2_pool)} (dropped {len(tr_idx) - len(p2_pool)} items on test traces), "
          f"P3 = {len(splits['P3 unseen session'])} held-out sessions; learners {list(L)}; reps {REPS}")
    long, oof = learning_curves(t, y, L, splits)
    base = {p: (np.mean([1 - y[e].mean() for _, e in sp])) for p, sp in splits.items()}
    print("accuracy of never showing:", {p: round(v, 3) for p, v in base.items()})
    long.to_parquet(os.path.join(OUT, "curves_long.parquet"))
    summ = summarize(long)
    summ.round(4).to_csv(os.path.join(OUT, "curves.csv"), index=False)
    ref = "JEV zero-shot"
    pairs = [(m, ref) for m in L] + [("Emb + JEV", "Emb + LR"), ("Emb + JEV", "TF-IDF + LR")] + \
        [(m, tp) for m in ("JEV + Platt", "Emb + JEV") for tp in TRIVIAL]
    diffs = paired_diffs(long[long.N != "0"], pairs)
    diffs.round(4).to_csv(os.path.join(OUT, "diffs.csv"), index=False)

    # sensitivity of the from-scratch learners to their prior strength (P3 only)
    Ls = {k: v for k, v in learners(t, scratch_lams=(.1, 1., 10., 100.)).items() if not k.startswith("JEV")}
    sens, _ = learning_curves(t, y, Ls, {"P3 unseen session": splits["P3 unseen session"]}, ns=[50, 200, "all"], reps=20)
    sens = summarize(sens)
    sens.round(4).to_csv(os.path.join(OUT, "sensitivity_lambda.csv"), index=False)

    p4, p4d, p4raw = per_session(t, y, L)
    p4.round(4).to_csv(os.path.join(OUT, "per_session.csv"), index=False)
    p4d.round(4).to_csv(os.path.join(OUT, "per_session_diffs.csv"), index=False)

    tt = t.iloc[tr_idx].reset_index(drop=True)
    preds = {"JEV zero-shot": tt.j_accept.values.astype(float)}
    preds.update({m + " (all other sessions' labels)": v[tr_idx] for m, v in oof.items()
                  if m != ref and m not in TRIVIAL})
    ind = individual_votes(tt, preds)
    ind.round(3).to_csv(os.path.join(OUT, "individual_votes.csv"))

    for k in METRICS:
        s = summ[summ.metric == k].copy()
        s["cell"] = s.apply(cell, axis=1)
        print(f"\n== {k}\n" + s.pivot_table(index=["protocol", "method"], columns="N", values="cell", aggfunc="first",
                                             sort=False)[[str(n) for n in NS]].to_string())
    print("\n== paired differences vs JEV zero-shot etc. (* = 95% interval excludes 0)")
    for _, r in diffs[diffs.metric.isin(["auroc", "acc", "util@1/3"])].iterrows():
        star = "*" if (r.lo > 0 or r.hi < 0) else " "
        print(f"  {r.protocol:18s} N={r.N:>4s} {r.metric:7s} {r.pair:30s} {r['mean']:+.3f} [{r.lo:+.3f},{r.hi:+.3f}]{star}")
    print("\n== P4 per-session calibration on the first k labels of the session (sessions resampled)")
    p4["cell"] = p4.apply(cell, axis=1)
    print(p4.pivot_table(index=["method", "metric"], columns="k", values="cell", aggfunc="first").to_string())
    for _, r in p4d[p4d.metric.isin(["acc", "util@1/3", "brier"])].iterrows():
        star = "*" if (r.lo > 0 or r.hi < 0) else " "
        print(f"  {r.method:30s} {r.metric:7s} {r.pair:10s} {r['mean']:+.3f} [{r.lo:+.3f},{r.hi:+.3f}]{star}")
    sens["cell"] = sens.apply(cell, axis=1)
    print("\n== P3 sensitivity of from-scratch learners to prior strength (20 reps)\n" +
          sens[sens.metric.isin(["auroc", "brier"])].pivot_table(index=["metric", "method"], columns="N", values="cell",
                                                                 aggfunc="first").to_string())
    print("\n== would THIS user accept? (train proposal items, each annotator's own vote)\n" + ind.round(3).to_string())


if __name__ == "__main__":
    {"prep": prep, "analyze": analyze}[sys.argv[1]]()
