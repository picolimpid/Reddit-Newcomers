"""Is observational SFT more than a temperature rescaling of the base model? (2026-11-02)

Platt scaling: logistic regression of the real 90-day return on logit P_base(observed), fit on the
train-split agents (results/qwen3-8b, intent protocol), applied to the held-out base scores from
sft.py eval. Platt keeps the base ranking and shrinks every within-agent Δlogit by one factor b, so
any difference between SFT and Platt is what fine-tuning learned beyond recalibration.

    python3 sft_vs_platt.py   -> appends to results/probes/sft-qwen3-8b/summary.md"""
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

from analyze_probes import boot, fmt, logit
from common import BENCH, RESULTS
from fingerprints import estimate

warnings.filterwarnings("ignore")
RUN = os.path.join(RESULTS, "probes", "sft-qwen3-8b")


def main():
    B = pd.read_parquet(BENCH)
    A = pd.read_parquet(os.path.join(RESULTS, "qwen3-8b", "agent_scores.parquet"))
    A = A[(A.protocol == "intent") & (A.condition == "observed")].set_index("idx")
    tr = B.index[B.split == "train"]
    X = sm.add_constant(logit(A.loc[tr, "p"]).rename("z"))
    platt = sm.GLM(B.loc[tr, "return_90d"], X, family=sm.families.Binomial()).fit()
    a, b = platt.params["const"], platt.params["z"]

    S = pd.read_parquet(os.path.join(RUN, "scores.parquet"))
    S["p_platt"] = 1 / (1 + np.exp(-(a + b * logit(S.p_base))))
    H = B.loc[S.idx.unique()]
    post = H.post_id
    W = {m: S.pivot_table(index="idx", columns=["exp", "condition"], values=f"p_{m}").loc[H.index]
         for m in ("base", "platt", "sft")}

    # how much of SFT is an affine map of the base logit?
    zb, zs = logit(S.p_base), logit(S.p_sft)
    aff = sm.OLS(zs, sm.add_constant(zb)).fit()

    L = ["", "## SFT vs Platt scaling of the base model", "",
         f"Platt fit on {len(tr)} train-split agents: logit P' = {a:+.2f} + {b:.3f} · logit P_base. "
         f"Affine fit of logit P_sft on logit P_base over all {len(S)} held-out prompts: "
         f"slope {aff.params.iloc[1]:.3f}, R² {aff.rsquared:.3f}.", ""]

    _, r3, _ = estimate(H, "return_90d", ["c2v_big"])
    _, r2, _ = estimate(H, "return_90d", ["reply_received"])
    rows = []
    for m in ("base", "platt", "sft"):
        H["_p"] = W[m][("observed", "observed")]
        rows.append({"model": m, "mean P": H._p.mean(), "AUROC 90d": roc_auc_score(H.return_90d, H._p),
                     "Brier": np.mean((H._p - H.return_90d) ** 2),
                     "FP3 fit OR (human 1.29)": estimate(H, "_p", ["c2v_big"], rows=r3, frac=True)[0].OR.iloc[0],
                     "FP2 reply OR (human 0.99)": estimate(H, "_p", ["reply_received"], rows=r2, frac=True)[0].OR.iloc[0]})
    L += [pd.DataFrame(rows).round(3).to_markdown(index=False), "",
          f"Human base-rate Brier (predict {H.return_90d.mean():.3f} for everyone): "
          f"{np.mean((H.return_90d.mean() - H.return_90d) ** 2):.3f}.", ""]

    rows = []
    rr = H.reply_received == 1
    for c in ("ack", "disagree", "praise", "praise_welcome", "praise_invite", "praise_welcome_invite", "template",
              "real reply", "NC r/Y: template"):
        row = {"condition": c}
        for m in ("base", "platt", "sft"):
            w = W[m]
            if c == "real reply":
                v, g = (logit(w[("observed", "observed")]) - logit(w[("ladder", "none")]))[rr], post[rr]
            elif c.startswith("NC"):
                v, g = logit(w[("nc", "template")]) - logit(w[("nc", "none")]), post
            else:
                v, g = logit(w[("ladder", c)]) - logit(w[("ladder", "none")]), post
            mu, lo, hi = boot(v, g)
            row[m] = fmt(mu, lo, hi)
            row[f"_{m}"] = mu
        row["sft / base"] = row["_sft"] / row["_base"]
        rows.append(row)
    T = pd.DataFrame(rows)
    L += ["Within-agent Δlogit vs no reply (Platt column = b × base):", "",
          T[["condition", "base", "platt", "sft", "sft / base"]].round(3).to_markdown(index=False), ""]
    # relative structure: ratio of each effect to the template effect
    ref = T.set_index("condition")
    rel = pd.DataFrame({m: ref[f"_{m}"] / ref.loc["template", f"_{m}"] for m in ("base", "sft")}).round(2)
    L += ["Each effect as a share of the template effect:", "", rel.T.to_markdown(), ""]

    txt = "\n".join(L) + "\n"
    with open(os.path.join(RUN, "summary.md"), "a") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
