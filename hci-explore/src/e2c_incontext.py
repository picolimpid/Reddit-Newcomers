"""E2c (Idea 2, RQ2b proxy): can JEV learn a user's preferences from feedback placed in its state, beyond calibration?

Sessions stand in for users, as in E2b P4.  In each of the 16 train sessions the time points are split in half.  The
K = 10 most recent proposal items before the split, with their (majority) accepted / dismissed decision, go into the
JEV state as `feedback_history` (oldest first); every proposal item from the split on is then judged with that history.
Splitting by time point keeps sibling candidates offered at the same moment on one side.
Baselines on the same items, no API cost: zero-shot JEV (E2b), per-session Platt on the same K labels (prior as in
E2b P4), global Platt fitted on the other 15 sessions.
Control (only worth running if the history helps): the same K offers with their decisions permuted within the session,
which keeps the session's base rate but breaks which offer was accepted.
Signal check without JEV (no API cost): is there anything to learn from the K labels at all?  For each evaluated offer,
the share accepted among its 3 most similar history offers (bge-small embedding of the offer text, local CPU), alone
and as a second feature next to zero-shot JEV (fitted on the other sessions).

  python3 src/e2c_incontext.py prep [--control] [--dry]   # ~$0.035 per condition
  python3 src/e2c_incontext.py offers                     # writes results/e2c/offer_emb_in.jsonl, then:
  $PY src/embed_local.py results/e2c/offer_emb_in.jsonl results/e2c/offer_emb_bge-small.npz
  python3 src/e2c_incontext.py analyze
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import jev  # noqa: E402
import e2_proactive_gate as e2  # noqa: E402
import e2b_learning_curve as e2b  # noqa: E402

OUT = os.path.join(jev.ROOT, "results", "e2c")
os.makedirs(OUT, exist_ok=True)
K = 10
OFFER_EMB = os.path.join(OUT, "offer_emb_bge-small.npz")
Q_HIST = ("`feedback_history` lists earlier offers this assistant made to the same user in this session, oldest first, "
          "and whether the user accepted or dismissed each one. Taking into account what this user tends to accept or "
          "dismiss, if the assistant interrupted the user right now to offer `proposed_task`, would the user accept the "
          "offer as helpful (rather than dismiss it as unnecessary or off-target)?")


def design(seed=0, k=K):
    """Train proposal items; per session the k most recent items before the median time point (history; k=None: all
    of them) and all items from the median time point on (evaluation)."""
    d = pd.read_parquet(os.path.join(e2b.OUT, "jev_all.parquet"))
    t = d[(d.split == "train") & d.has_task].reset_index(drop=True)
    t["tie"] = np.random.default_rng(seed).random(len(t))
    hist, ev = {}, []
    for s, g in t.groupby("session"):
        steps = np.sort(g.step.unique())
        cut = steps[len(steps) // 2]
        h = g[g.step < cut].sort_values(["step", "tie"])
        hist[s] = h if k is None else h.tail(k)
        ev.append(g[g.step >= cut])
    return t, hist, pd.concat(ev).reset_index(drop=True)


def requests(control, seed=1):
    t, hist, ev = design()
    rows = {r["id"]: r for r in e2.load(e2b.TRAIN, prefix="t")}
    rng = np.random.default_rng(seed)
    hl = {}
    for s, h in hist.items():
        lab = rng.permutation(h.valid.values) if control else h.valid.values
        hl[s] = [{"offer": rows[i]["pred_task"], "user_response": "accepted" if v else "dismissed"}
                 for i, v in zip(h.id, lab)]
    tag = "e2c-control" if control else "e2c-history"
    its = [{"state": {"recent_activity": rows[i]["activity"], "feedback_history": hl[s],
                      "proposed_task": rows[i]["pred_task"]},
            "questions": {"accept": {"type": "noul", "instructions": Q_HIST}}, "tag": tag}
           for i, s in zip(ev.id, ev.session)]
    return ev, its, hl


def tokens_per_char():
    """From the E2b train run: JEV input tokens per character of the request JSON."""
    d = pd.read_parquet(os.path.join(e2b.OUT, "jev_all.parquet")).set_index("id")
    rows = e2.load(e2b.TRAIN, prefix="t")
    ch = sum(len(json.dumps({k: v for k, v in e2.jev_item(r).items() if k != "tag"}, ensure_ascii=False)) for r in rows)
    return d.loc[[r["id"] for r in rows], "n_tok"].sum() / ch


def prep(control, dry):
    ev, its, hl = requests(control)
    ch = sum(len(json.dumps({"state": it["state"], "questions": it["questions"]}, ensure_ascii=False)) for it in its)
    est = ch * tokens_per_char() * jev.USD_PER_TOKEN
    acc = {s: sum(x["user_response"] == "accepted" for x in h) for s, h in hl.items()}
    print(f"{'control' if control else 'history'}: {len(its)} requests over {ev.session.nunique()} sessions; "
          f"accepted offers in each history: {acc}; est cost ${est:.4f}; JEV spent so far ${jev.spent_usd():.4f}")
    if dry:
        return
    outs = jev.ask_many(its)
    ev = ev.assign(p=[jev.p_yes(o, "accept") for o in outs], n_tok=[o["usage"]["input_tokens"] for o in outs],
                   cached=[o["_cached"] for o in outs])
    name = "control" if control else "history"
    ev[["id", "session", "step", "valid", "j_accept", "p", "n_tok", "cached"]].to_parquet(os.path.join(OUT, name + ".parquet"))
    led = pd.read_json(jev.LEDGER, lines=True).query(f"tag == 'e2c-{name}'")
    print(f"done: ledger e2c-{name} {len(led)} requests, {led.input_tokens.sum()} tokens, ${led.cost_usd.sum():.4f}; "
          f"JEV spent now ${jev.spent_usd():.4f}")


def offers():
    t, _, _ = design()
    rows = {r["id"]: r for r in e2.load(e2b.TRAIN, prefix="t")}
    with open(os.path.join(OUT, "offer_emb_in.jsonl"), "w") as f:
        for i in t.id:
            f.write(json.dumps({"id": i, "text": rows[i]["pred_task"]}) + "\n")
    print(len(t), "offers written")


def knn_sensitivity(df, y, B=2000):
    """Is the null result about the history being short?  kNN signal with the whole first half and other k."""
    rng = np.random.default_rng(0)
    sessions = df.session.unique()
    rows = []
    for hk in (K, None):
        _, hist, _ = design(k=hk)
        for k in (1, 3, 5):
            f, _ = knn_feature(df, hist, k)
            a = np.array([roc_auc_score(y[df.session.values == s], f[df.session.values == s]) for s in sessions])
            boot = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(B)]
            rows.append({"history": f"{hk} most recent" if hk else "whole first half (%d per session on average)" %
                         np.mean([len(h) for h in hist.values()]), "k": k, "auroc": a.mean(),
                         "lo": np.quantile(boot, .025), "hi": np.quantile(boot, .975)})
    return pd.DataFrame(rows)


def knn_feature(df, hist, k=3):
    """Share accepted among the k history offers most similar to each evaluated offer (cosine on offer embeddings)."""
    z = np.load(OFFER_EMB, allow_pickle=False)
    pos = {i: j for j, i in enumerate(z["ids"])}
    E = z["emb"]
    out, base = np.empty(len(df)), np.empty(len(df))
    for s in df.session.unique():
        m = np.flatnonzero(df.session.values == s)
        h = hist[s]
        yh = h.valid.values.astype(float)
        S = E[[pos[i] for i in df.id.values[m]]] @ E[[pos[i] for i in h.id]].T
        out[m] = yh[np.argsort(-S, axis=1)[:, :k]].mean(1)
        base[m] = yh.mean()
    return out, base


def analyze(B=2000):
    t, hist, ev = design()
    df = pd.read_parquet(os.path.join(OUT, "history.parquet")).rename(columns={"p": "p_hist"})
    ctrl = os.path.join(OUT, "control.parquet")
    if os.path.exists(ctrl):
        df = df.merge(pd.read_parquet(ctrl)[["id", "p"]].rename(columns={"p": "p_ctrl"}), on="id")
    y = df.valid.values.astype(float)
    z0 = e2b.logit(df.j_accept)
    sc = {"JEV zero-shot": expit(z0), "never show": np.zeros(len(df))}
    glob, sess = np.empty(len(df)), np.empty(len(df))
    hist_glob = {}
    for s in df.session.unique():
        m = (df.session == s).values
        other = (t.session != s).values  # global Platt: all proposal items of the other sessions
        a, b = e2b.fit_map(e2b.logit(t.j_accept[other])[:, None], t.valid[other].values.astype(float),
                           np.array([1.]), 1., 1., [0.])
        glob[m] = expit(a[0] * z0[m] + b)
        h = hist[s]  # per-session Platt on the same K labels the history condition sees
        a, b = e2b.fit_map(e2b.logit(h.j_accept)[:, None], h.valid.values.astype(float), np.array([1.]), 10., 1., [0.])
        sess[m] = expit(a[0] * z0[m] + b)
        for col in ("p_hist", "p_ctrl"):  # history-JEV, then calibrated on the other sessions' history-JEV outputs
            if col in df:
                o = ~m
                a, b = e2b.fit_map(e2b.logit(df[col][o])[:, None], y[o], np.array([1.]), 1., 1., [0.])
                hist_glob.setdefault(col, np.empty(len(df)))[m] = expit(a[0] * e2b.logit(df[col][m]) + b)
    sc["JEV + global Platt (other sessions)"] = glob
    sc[f"JEV + session Platt ({K} labels)"] = sess
    sc[f"JEV with {K}-item history in state"] = df.p_hist.values
    sc["  ... + global Platt"] = hist_glob["p_hist"]
    if os.path.exists(OFFER_EMB):
        knn, base = knn_feature(df, hist)
        sc["kNN: share accepted among 3 most similar history offers"] = knn
        X = np.c_[z0, knn - base]
        comb = np.empty(len(df))
        for s in df.session.unique():
            m = (df.session == s).values
            w, b = e2b.fit_map(X[~m], y[~m], np.array([1., 0.]), 1., 1., [0., -np.inf])
            comb[m] = expit(X[m] @ w + b)
        sc["JEV zero-shot + kNN feature (fit on other sessions)"] = comb
    if "p_ctrl" in df:
        sc[f"JEV with permuted-label history (control)"] = df.p_ctrl.values
        sc["  ... control + global Platt"] = hist_glob["p_ctrl"]

    sessions = df.session.unique()
    idx = {s: np.flatnonzero(df.session.values == s) for s in sessions}
    per = {m: np.array([e2b.metrics(y[idx[s]], v[idx[s]]) for s in sessions]) for m, v in sc.items()}  # sess x metric
    rng = np.random.default_rng(0)
    picks = [rng.integers(0, len(sessions), len(sessions)) for _ in range(B)]
    rows = []
    for m, P in per.items():
        boot = np.array([np.nanmean(P[p], 0) for p in picks])
        for j, k in enumerate(e2b.METRICS):
            rows.append({"method": m, "metric": k, "mean": np.nanmean(P[:, j]), "lo": np.nanquantile(boot[:, j], .025),
                         "hi": np.nanquantile(boot[:, j], .975)})
    summ = pd.DataFrame(rows)
    pairs = [(f"JEV with {K}-item history in state", "JEV zero-shot"),
             ("JEV zero-shot + kNN feature (fit on other sessions)", "JEV zero-shot"),
             ("JEV zero-shot + kNN feature (fit on other sessions)", "JEV + global Platt (other sessions)"),
             (f"JEV with {K}-item history in state", f"JEV + session Platt ({K} labels)"),
             ("  ... + global Platt", "JEV + global Platt (other sessions)")]
    if "p_ctrl" in df:
        pairs += [(f"JEV with {K}-item history in state", "JEV with permuted-label history (control)"),
                  ("JEV with permuted-label history (control)", "JEV zero-shot")]
    drows = []
    for a, b in pairs:
        D = per[a] - per[b]
        boot = np.array([np.nanmean(D[p], 0) for p in picks])
        for j, k in enumerate(e2b.METRICS):
            drows.append({"pair": f"{a.strip()} - {b.strip()}", "metric": k, "mean": np.nanmean(D[:, j]),
                          "lo": np.nanquantile(boot[:, j], .025), "hi": np.nanquantile(boot[:, j], .975)})
    diffs = pd.DataFrame(drows)
    pooled = {m: roc_auc_score(y, v) for m, v in sc.items() if m != "never show"}
    summ.round(4).to_csv(os.path.join(OUT, "summary.csv"), index=False)
    sens = knn_sensitivity(df, y) if os.path.exists(OFFER_EMB) else None
    if sens is not None:
        sens.round(4).to_csv(os.path.join(OUT, "knn_sensitivity.csv"), index=False)
    diffs.round(4).to_csv(os.path.join(OUT, "diffs.csv"), index=False)

    pd.set_option("display.width", 250)
    print(f"eval items {len(df)} over {len(sessions)} sessions (accept rate {y.mean():.3f}); history size {K}; "
          f"sessions with both classes in eval: {sum(0 < y[idx[s]].sum() < len(idx[s]) for s in sessions)}")
    summ["cell"] = summ.apply(lambda r: f"{r['mean']:.3f} [{r.lo:.3f},{r.hi:.3f}]", axis=1)
    print("\n== per-session metrics averaged over sessions (CIs resample sessions)")
    print(summ.pivot_table(index="method", columns="metric", values="cell", aggfunc="first", sort=False)
          [["auroc", "acc", "util@1/3", "brier"]].to_string())
    print("\npooled AUROC over all eval items:", {k: round(v, 3) for k, v in pooled.items()})
    print("\n== paired differences (* = 95% CI excludes 0)")
    for _, r in diffs[diffs.metric.isin(["auroc", "acc", "util@1/3", "brier"])].iterrows():
        star = "*" if (r.lo > 0 or r.hi < 0) else " "
        print(f"  {r.pair:75s} {r.metric:8s} {r['mean']:+.3f} [{r.lo:+.3f},{r.hi:+.3f}]{star}")
    if sens is not None:
        print("\n== kNN signal check, per-session AUROC (CIs resample sessions)\n" + sens.round(3).to_string(index=False))
    print("\nmean P(accept) by true label:", df.groupby("valid")[["j_accept", "p_hist"] + (["p_ctrl"] if "p_ctrl" in df else [])]
          .mean().round(3).to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "offers", "analyze"])
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    {"prep": lambda: prep(a.control, a.dry), "offers": offers, "analyze": analyze}[a.cmd]()
