"""Agent vs human fingerprints for one agent run, plus the RQ2 reply interventions.

    python evaluate.py qwen3-8b   -> results/qwen3-8b/{fingerprints.csv, interventions.csv, summary.md}

Agents are compared with the human 90-day return model on exactly the human estimation rows
(the intent question asks about the next 3 months). Verdicts follow FINGERPRINTS.md."""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import BENCH, RESULTS
from fingerprints import all_fingerprints

PROTOCOLS = ["intent", "choice"]
EPS = 1e-4


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def cluster_mean(d, col, B, n_boot=2000):
    """mean of col with a post-cluster bootstrap 95% CI."""
    d = d.dropna(subset=[col])
    g = d.groupby(B.loc[d.index, "post_id"])[col].agg(["sum", "count"])
    rng = np.random.default_rng(0)
    k = rng.integers(0, len(g), (n_boot, len(g)))
    boots = g["sum"].values[k].sum(1) / g["count"].values[k].sum(1)
    return d[col].mean(), *np.percentile(boots, [2.5, 97.5]), len(d)


def verdicts(F, who):
    """Scale-free criteria (agent outputs are near-deterministic, so agent p-values are tiny):
    FP1  recent window positive and the 3-12-month window < half its weight
    FP2  reply effect < 1/4 of a 1-SD fit effect (or CI covers 1)
    FP3  in the joint model, fit positive and larger than the reply effect"""
    f = F[F.who == who].set_index(["spec", "predictor"])
    out = {}
    for spec, (rec, old) in {"FP1_traj_c2v": ("fit_0_30", "fit_90_365"),
                             "FP1_traj_jev": ("jw_d0_30", "jw_d90_365")}.items():
        r, o = f.loc[(spec, rec)], f.loc[(spec, old)]
        out[spec] = bool(r.beta > 0 and r.p < .05 and o.beta < 0.5 * r.beta)
    fit_b = f.loc[("FP3_fit", "c2v_big")].beta
    rep = f.loc[("FP2_reply", "reply_received")]
    out["FP2_reply"] = bool(rep.ci_lo <= 1 <= rep.ci_hi or abs(rep.beta) < 0.25 * abs(fit_b))
    jf, jr = f.loc[("FP3_joint", "c2v_big")], f.loc[("FP3_joint", "reply_received")]
    out["FP3_joint"] = bool(jf.p < .05 and jf.beta > 0 and jf.beta > jr.beta)
    return out


def main(run):
    out = os.path.join(RESULTS, run)
    B = pd.read_parquet(BENCH)
    S = pd.read_parquet(os.path.join(out, "agent_scores.parquet"))
    for (proto, cond), g in S.groupby(["protocol", "condition"]):
        B[f"p_{proto}_{cond}"] = g.set_index("idx").p
    # the choice options are shuffled per agent (same order in every condition); the letter of
    # the r/X option is a randomized nuisance with a strong position bias, so it is a control
    pos = S[S.protocol == "choice"].drop_duplicates("idx").set_index("idx").target
    B["pos_choice"] = pos.map(lambda t: next(k for k, v in json.loads(t).items() if v == 1))
    present = [p for p in PROTOCOLS if f"p_{p}_observed" in B]
    sub = B.dropna(subset=[f"p_{p}_observed" for p in present])
    print(f"agents scored: {len(sub)} of {len(B)}")

    H, rows = all_fingerprints(sub, "return_90d", tag="human")
    H180, _ = all_fingerprints(sub, "return_180d", tag="human_180d")
    F = [H, H180]
    for proto in present:
        A, _ = all_fingerprints(sub, f"p_{proto}_observed", rows_by_spec=rows, frac=True, tag=f"agent_{proto}",
                                extra=["pos_choice"] if proto == "choice" else [])
        F.append(A)
    F = pd.concat(F, ignore_index=True)
    F.to_csv(os.path.join(out, "fingerprints.csv"), index=False)

    # RQ2: within-agent effect of the reply shown, in log-odds and in units of the agent's own
    # 1-SD fit effect (FP3_fit); the human analogue is observational only
    I = []
    for proto in present:
        fit_b = F[(F.who == f"agent_{proto}") & (F.spec == "FP3_fit")].beta.iloc[0]
        for cond, base, where in (("welcoming", "none", None), ("hostile", "none", None),
                                  ("observed", "none", "reply_received == 1")):
            if f"p_{proto}_{cond}" not in sub or f"p_{proto}_{base}" not in sub:
                continue
            d = sub.query(where) if where else sub
            d = pd.DataFrame({"dlogit": logit(d[f"p_{proto}_{cond}"]) - logit(d[f"p_{proto}_{base}"]),
                              "dp": d[f"p_{proto}_{cond}"] - d[f"p_{proto}_{base}"]}, index=d.index)
            m, lo, hi, n = cluster_mean(d, "dlogit", B)
            mp, plo, phi, _ = cluster_mean(d, "dp", B)
            I.append({"protocol": proto, "contrast": f"{cond} - {base}", "n": n, "dlogit": m, "dlogit_lo": lo,
                      "dlogit_hi": hi, "dp": mp, "dp_lo": plo, "dp_hi": phi, "agent_fit_beta": fit_b,
                      "in_fit_sd_units": m / fit_b if fit_b else np.nan})
    hb = F[(F.who == "human") & (F.spec == "FP2_reply")].beta.iloc[0]
    hf = F[(F.who == "human") & (F.spec == "FP3_fit")].beta.iloc[0]
    I.append({"protocol": "human (observational)", "contrast": "reply received vs not", "dlogit": hb,
              "agent_fit_beta": hf, "in_fit_sd_units": hb / hf})
    I = pd.DataFrame(I)
    I.to_csv(os.path.join(out, "interventions.csv"), index=False)

    # base rates and individual-level discrimination (secondary)
    base = [{"who": "human", "mean_return_90d": sub.return_90d.mean(), "auroc_90d": np.nan, "auroc_180d": np.nan}]
    base.append({"who": "c2v fit alone", "mean_return_90d": np.nan,
                 "auroc_90d": roc_auc_score(sub.dropna(subset=["c2v_big"]).return_90d, sub.c2v_big.dropna()),
                 "auroc_180d": roc_auc_score(*sub.dropna(subset=["c2v_big", "return_180d"])[["return_180d", "c2v_big"]].T.values)})
    for proto in present:
        p = sub[f"p_{proto}_observed"]
        s = logit(p) - logit(p).groupby(sub.pos_choice).transform("mean") if proto == "choice" else p
        r180 = sub.return_180d.notna()   # choice: AUROC on the position-adjusted logit
        base.append({"who": f"agent_{proto}", "mean_return_90d": p.mean(),
                     "auroc_90d": roc_auc_score(sub.return_90d, s),
                     "auroc_180d": roc_auc_score(sub.return_180d[r180], s[r180]),
                     **{f"mean_p_{c}": sub[f"p_{proto}_{c}"].mean() for c in ("none", "welcoming", "hostile")
                        if f"p_{proto}_{c}" in sub}})
    base = pd.DataFrame(base)
    base.to_csv(os.path.join(out, "base_rates.csv"), index=False)

    # summary
    whos = ["human"] + [f"agent_{p}" for p in present]
    V = {who: verdicts(F, who) for who in whos}
    for who in whos:   # the scale-free ratios behind the verdicts
        f = F[F.who == who].set_index(["spec", "predictor"]).beta
        V[who] |= {"ratio_old_recent_c2v": f[("FP1_traj_c2v", "fit_90_365")] / f[("FP1_traj_c2v", "fit_0_30")],
                   "ratio_old_recent_jev": f[("FP1_traj_jev", "jw_d90_365")] / f[("FP1_traj_jev", "jw_d0_30")],
                   "ratio_reply_fit": f[("FP2_reply", "reply_received")] / f[("FP3_fit", "c2v_big")]}
    show = F[F.who.isin(["human"] + [f"agent_{p}" for p in present])]
    piv = show.assign(cell=show.apply(lambda r: f"{r.OR:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}] p={r.p:.3f}", axis=1)) \
        .pivot_table(index=["spec", "predictor"], columns="who", values="cell", aggfunc="first")
    piv = piv.reindex(pd.MultiIndex.from_frame(F[F.who == "human"][["spec", "predictor"]]))
    lines = [f"# Agent run: {run}", "", f"Agents scored: {len(sub)}. Human = observed 90-day return; agents = "
             "P(return) from the observed condition, fractional logit on the same rows. Cells: OR per SD [95% CI].",
             "", piv.to_markdown(), "", "## Verdicts (match human pattern?)", "",
             pd.DataFrame(V).to_markdown(), "", "## Reply interventions (RQ2)", "",
             I.round(3).to_markdown(index=False), "", "## Base rates and AUROC", "", base.round(3).to_markdown(index=False)]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen3-8b")
