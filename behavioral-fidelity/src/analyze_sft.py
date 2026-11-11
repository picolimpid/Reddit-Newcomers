"""Base vs observationally fine-tuned Qwen3-8B on held-out communities (sft.py eval).

    python3 analyze_sft.py [sft-qwen3-8b]   -> results/probes/<run>/summary.md"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analyze_probes import boot, fmt, logit
from common import BENCH, RESULTS
from fingerprints import estimate

warnings.filterwarnings("ignore")


def main(run):
    out = os.path.join(RESULTS, "probes", run)
    B = pd.read_parquet(BENCH)
    S = pd.read_parquet(os.path.join(out, "scores.parquet"))
    H = B.loc[S.idx.unique()]
    post = H.post_id
    W = {m: S.pivot_table(index="idx", columns=["exp", "condition"], values=f"p_{m}").loc[H.index] for m in ("base", "sft")}
    L = [f"# Observational SFT: {run}", "", f"Held-out agents (val + test communities, never seen in training): {len(H)}. "
         f"Human 90-day return on them: {H.return_90d.mean():.1%}.", ""]

    # calibration + discrimination + fingerprints, observed condition
    _, r3, _ = estimate(H, "return_90d", ["c2v_big"])
    _, r1, _ = estimate(H, "return_90d", ["jw_d0_30", "jw_d30_90", "jw_d90_365"])
    _, r2, _ = estimate(H, "return_90d", ["reply_received"])
    rows = []
    for who in ("human", "base", "sft"):
        if who == "human":
            y, frac = "return_90d", False
        else:
            H["_p"] = W[who][("observed", "observed")]
            y, frac = "_p", True
        f3 = estimate(H, y, ["c2v_big"], rows=r3, frac=frac)[0]
        f1 = estimate(H, y, ["jw_d0_30", "jw_d30_90", "jw_d90_365"], rows=r1, frac=frac)[0]
        f2 = estimate(H, y, ["reply_received"], rows=r2, frac=frac)[0]
        rows.append({"who": who, "mean P / rate": H[y].mean(),
                     "AUROC 90d": np.nan if who == "human" else roc_auc_score(H.return_90d, H[y]),
                     "FP3 fit OR": f3.OR.iloc[0], "FP1 Jev windows": " / ".join(f"{x:.2f}" for x in f1.OR),
                     "FP2 reply OR": f2.OR.iloc[0]})
    ok = H.c2v_big.notna()
    L += ["## Calibration, discrimination and fingerprints (observed replies)", "",
          pd.DataFrame(rows).round(3).to_markdown(index=False), "",
          f"Reference: c2v fit alone AUROC {roc_auc_score(H.return_90d[ok], H.c2v_big[ok]):.3f}. "
          f"Human fingerprints on the held-out half are noisy (n ≈ {len(r3)}).", ""]

    # interventions
    rows = []
    rr = H.reply_received == 1
    for m in ("base", "sft"):
        w = W[m]
        none = logit(w[("ladder", "none")])
        row = {"model": m, "P none": w[("ladder", "none")].mean()}
        for c in ("ack", "disagree", "praise", "praise_welcome", "praise_invite", "praise_welcome_invite", "template"):
            row[c] = fmt(*boot(logit(w[("ladder", c)]) - none, post))
        row["real reply (observed − none, n=%d)" % rr.sum()] = fmt(*boot((logit(w[("observed", "observed")]) - none)[rr], post[rr]))
        row["NC r/Y: template"] = fmt(*boot(logit(w[("nc", "template")]) - logit(w[("nc", "none")]), post))
        rows.append(row)
    T = pd.DataFrame(rows).set_index("model").T
    L += ["## Reply interventions (Δlogit vs no reply, within agent)", "", T.to_markdown(), "",
          "Human reference (observational, full sample): reply received OR 1.07 → Δlogit +0.07; TOST rules out "
          "|Δlogit| > 0.41 at 90 days.", ""]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sft-qwen3-8b")
