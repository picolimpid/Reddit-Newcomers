"""JEV vs a local open-weight judge (Qwen3-32B in bf16, vLLM, TP=2 on our own GPUs; src/local_judge.py) on the v6 data.
Zero-shot, same question wording; Qwen's P(Yes) comes from the first generated token's logprobs.  No API calls.

  E2b  ProactiveBench reward train set, 1,387 proposal items: accept AUROC (session bootstrap)
  E2e  RealHumanEval, 4,765 unrequested suggestions: accept / useful AUROC, pooled and within programmer (programmer
       bootstrap); plus 5-fold-over-programmers logistic regressions on JEV's, Qwen's and both judges' answers
  E1e  DICES-350: AUROC against the raters' majority and Spearman with the share of raters who said unsafe
  Latency at batch size 1 (median), from local_judge's latency files and JEV's ledger.

  python3 src/qwen_local_compare.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import jev  # noqa: E402

R = os.path.join(jev.ROOT, "results")
TAG = "qwen3-32b-local"
B = 2000


def qwen(exp):
    rows = [json.loads(l) for l in open(os.path.join(R, exp, f"llm_{TAG}.jsonl"))]
    q = pd.DataFrame({"id": [r["id"].rsplit("|", 1)[0] for r in rows], "q": [r["id"].rsplit("|", 1)[1] for r in rows],
                      "p": [r["probs"]["Yes"] for r in rows], "mass": [r["mass"] for r in rows]})
    return q.pivot(index="id", columns="q", values="p"), q.mass


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def cluster_boot(y, scores, cluster, rng, fn=roc_auc_score):
    """Point and 95% CI for each score and for the paired difference score[1] - score[0]."""
    cl = np.unique(cluster)
    members = {c: np.flatnonzero(cluster == c) for c in cl}
    pt = [fn(y, s) for s in scores]
    bt = []
    for _ in range(B):
        idx = np.concatenate([members[c] for c in rng.choice(cl, len(cl))])
        if y[idx].min() == y[idx].max():
            continue
        bt.append([fn(y[idx], s[idx]) for s in scores])
    bt = np.array(bt)
    fmt = lambda v, b: f"{v:.3f} [{np.quantile(b, .025):.3f}, {np.quantile(b, .975):.3f}]"
    return [fmt(pt[j], bt[:, j]) for j in range(len(scores))] + [
        f"{pt[1] - pt[0]:+.3f} [{np.quantile(bt[:, 1] - bt[:, 0], .025):+.3f}, {np.quantile(bt[:, 1] - bt[:, 0], .975):+.3f}]"]


def within(y, s, g):
    conc = tot = 0.
    for u in np.unique(g):
        k = g == u
        pos, neg = s[k][y[k] == 1], s[k][y[k] == 0]
        if len(pos) and len(neg):
            d = pos[:, None] - neg[None, :]
            conc, tot = conc + (d > 0).sum() + .5 * (d == 0).sum(), tot + d.size
    return conc / tot


def main():
    pd.set_option("display.width", 250)
    rng = np.random.default_rng(0)
    rows = []
    # ---- E2b
    d = pd.read_parquet(os.path.join(R, "e2b", "jev_all.parquet")).query("split == 'train' and has_task")
    q, mass = qwen("e2b")
    d = d.join(q, on="id")
    y = d.valid.values.astype(int)
    rows.append(["E2b ProactiveBench accept (1,387 proposals, 16 sessions)", "session"] +
                cluster_boot(y, [d.j_accept.values, d.accept.values], d.session.values, rng))
    # ---- E2e
    sys.argv = sys.argv[:1]
    import e2e_realhumaneval as e2e
    a = e2e.load()
    j = pd.read_parquet(os.path.join(R, "e2e", "jev.parquet")).set_index("row").loc[a.row]
    q2, mass2 = qwen("e2e")
    q2.index = q2.index.astype(int)
    q2 = q2.loc[a.row]
    y2, g2 = a.accepted.values.astype(int), a.programmer_id.values
    for k in ("accept", "useful"):
        rows.append([f"E2e RealHumanEval {k} (4,765 suggestions, 83 programmers)", "programmer"] +
                    cluster_boot(y2, [j["j_" + k].values, q2[k].values], g2, rng))
        rows.append([f"E2e RealHumanEval {k}, within programmer", "point only",
                     f"{within(y2, j['j_' + k].values, g2):.3f}", f"{within(y2, q2[k].values, g2):.3f}", ""])
    # logistic regressions, 5 folds over programmers
    ug = np.unique(g2)
    fold = dict(zip(np.random.default_rng(0).permutation(ug), np.arange(len(ug)) % 5))
    f = np.array([fold[u] for u in g2])
    Xs = {"JEV (2 answers)": logit(j[["j_accept", "j_useful"]].values),
          "Qwen3-32B (2 answers)": logit(q2[["accept", "useful"]].values)}
    Xs["JEV + Qwen3-32B"] = np.c_[Xs["JEV (2 answers)"], Xs["Qwen3-32B (2 answers)"]]
    oof = {}
    for n, X in Xs.items():
        oof[n] = np.zeros(len(y2))
        for k in range(5):
            m = LogisticRegression(C=100., max_iter=5000).fit(X[f != k], y2[f != k])
            oof[n][f == k] = m.decision_function(X[f == k])
    rows.append(["E2e LR on both judges vs JEV alone (5-fold over programmers)", "programmer"] +
                cluster_boot(y2, [oof["JEV (2 answers)"], oof["JEV + Qwen3-32B"]], g2, rng))
    rows.append(["E2e LR Qwen alone vs JEV alone", "programmer"] +
                cluster_boot(y2, [oof["JEV (2 answers)"], oof["Qwen3-32B (2 answers)"]], g2, rng))
    # ---- E1e
    import e1e_dices as e1e
    df, items = e1e.load()
    df = df[~df.rater_id.isin(e1e.REMOVED) & (df.Q_overall != "Unsure")]
    share = (df.Q_overall == "Yes").groupby(df.item_id).mean().loc[items.index]
    jd = pd.read_parquet(os.path.join(R, "e1e", "jev.parquet")).set_index("item_id").loc[items.index]
    q3, mass3 = qwen("e1e")
    q3.index = q3.index.astype(int)
    q3 = q3.loc[items.index]
    maj = (share.values > .5).astype(int)
    for k in e1e.QUESTIONS:
        rows.append([f"E1e DICES {k}: AUROC vs rater majority (350 conversations)", "conversation"] +
                     cluster_boot(maj, [jd["j_" + k].values, q3[k].values], items.index.values, rng))
    r = pd.DataFrame(rows, columns=["comparison", "bootstrap unit", "JEV", "Qwen3-32B local", "Qwen - JEV"])
    sp = pd.DataFrame([{"question": k, "JEV": spearmanr(jd["j_" + k], share).correlation,
                        "Qwen3-32B local": spearmanr(q3[k], share).correlation} for k in e1e.QUESTIONS])
    lat = {}
    for exp in ("e2b", "e2e", "e1e"):
        lat[exp] = json.load(open(os.path.join(R, exp, f"llm_{TAG}_latency.json")))
    led = pd.read_json(jev.LEDGER, lines=True)
    jl = {t: led[led.tag == t].latency_s.median() for t in ("e2b-train", "e2e-rhe", "e1e-dices")}
    out = os.path.join(R, "qwen_local_compare.csv")
    r.to_csv(out, index=False)
    print(r.to_string(index=False))
    print("\nDICES Spearman with the share of raters who said unsafe\n" + sp.round(3).to_string(index=False))
    print("\nlatency at batch size 1, median seconds: " + ", ".join(
        f"{e}: Qwen local {lat[e]['latency_median_s']:.3f} (batch run {lat[e]['n'] / lat[e]['batch_wall_s']:.0f} prompts/s), "
        f"JEV via OpenRouter {jl[t]:.3f}" for e, t in zip(("e2b", "e2e", "e1e"), ("e2b-train", "e2e-rhe", "e1e-dices"))))
    print("Qwen option mass below 0.5:", {e: round(lat[e]["mass_below_0.5"], 4) for e in lat})


if __name__ == "__main__":
    main()
