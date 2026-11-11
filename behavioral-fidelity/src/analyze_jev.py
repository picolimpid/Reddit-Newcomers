"""Jev System One vs Qwen3-8B on the same agents (jev_probes.py, probes.py main run).

    python3 analyze_jev.py   -> results/probes/jev/summary.md"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analyze_probes import boot, fmt, logit
from common import BENCH, RESULTS

warnings.filterwarnings("ignore")
EXTRA = ["probes-gpt-4o-mini"]


def main():
    B = pd.read_parquet(BENCH)
    J = pd.read_parquet(os.path.join(RESULTS, "probes", "jev", "scores.parquet"))
    Q = pd.read_parquet(os.path.join(RESULTS, "probes", "qwen3-8b", "scores.parquet"))
    A = pd.read_parquet(os.path.join(RESULTS, "qwen3-8b", "agent_scores.parquet"))
    A = A[(A.protocol == "intent") & (A.condition == "observed")].set_index("idx").p
    W = {"jev": J.pivot_table(index="idx", columns=["exp", "condition"], values="p"),
         "qwen3-8b": Q.pivot_table(index="idx", columns=["exp", "condition"], values="p")}
    for run in EXTRA:  # other models scored on the same agents (api_agents.py --task probes)
        f = os.path.join(RESULTS, "probes", run, "scores.parquet")
        if os.path.exists(f):
            W[run.replace("probes-", "")] = pd.read_parquet(f).pivot_table(index="idx", columns=["exp", "condition"], values="p")
    ids = W["jev"].index
    W = {m: w.loc[ids] for m, w in W.items()}
    H = B.loc[ids]
    post = H.post_id
    rows = []
    for m, w in W.items():
        none = logit(w[("ladder", "none")])
        obs = w[("observed", "observed")] if ("observed", "observed") in w else A.loc[ids]
        row = {"model": m, "P none": f"{w[('ladder', 'none')].mean():.3f}",
               "P observed": f"{obs.mean():.3f}", "AUROC (observed)": f"{roc_auc_score(H.return_90d, obs):.3f}"}
        for c in ("ack", "disagree", "praise", "praise_welcome", "praise_invite", "praise_welcome_invite", "template"):
            row[c] = fmt(*boot(logit(w[("ladder", c)]) - none, post))
        row["NC r/Y template"] = fmt(*boot(logit(w[("nc", "template")]) - logit(w[("nc", "none")]), post))
        row["P(r/Y) none"] = f"{w[('nc', 'none')].mean():.3f}"
        rr = H.reply_received == 1
        row[f"real reply (n={rr.sum()})"] = fmt(*boot((logit(obs) - none)[rr], post[rr]))
        rows.append(row)
    T = pd.DataFrame(rows).set_index("model").T
    sd = {m: logit(W[m][("ladder", "none")]).std() for m in W}
    L = [f"# Jev System One vs Qwen3-8B (and API chat models), same {len(ids)} agents", "",
         f"Human 90-day return on these agents: {H.return_90d.mean():.3f}; r/Y human base rate ≈ 0.059 (full sample).", "",
         "Δlogit vs no reply (within agent, post-cluster bootstrap 95% CI):", "", T.to_markdown(), "",
         "Between-agent SD of logit P(none): " + ", ".join(f"{m} {v:.2f}" for m, v in sd.items()), ""]
    ok = H.c2v_big.notna()
    for m in W:
        z = logit(W[m][("ladder", "none")])[ok]
        L.append(f"- {m}: Spearman(logit P none, c2v fit) = {pd.Series(z).corr(H.c2v_big[ok], method='spearman'):.3f}")
    L.append(f"- human: AUROC of c2v fit on these agents = {roc_auc_score(H.return_90d[ok], H.c2v_big[ok]):.3f}")
    txt = "\n".join(L) + "\n"
    open(os.path.join(RESULTS, "probes", "jev", "summary.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
