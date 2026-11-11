"""Summaries of the probe experiments (probes.py).

    python3 analyze_probes.py qwen3-8b qwen3-8b          # probe run, main run (fit SD units, repro check)
    python3 analyze_probes.py qwen3-32b-fp8 qwen3-32b
-> results/probes/<run>/summary.md"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import BENCH, RESULTS, SOURCE

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def boot(v, post, n=2000):
    """mean with a post-cluster bootstrap 95% CI"""
    d = pd.DataFrame({"v": v, "g": post}).dropna()
    g = d.groupby("g").v.agg(["sum", "count"])
    k = np.random.default_rng(0).integers(0, len(g), (n, len(g)))
    b = g["sum"].values[k].sum(1) / g["count"].values[k].sum(1)
    return d.v.mean(), *np.percentile(b, [2.5, 97.5])


def fmt(m, lo, hi, nd=2):
    return f"{m:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def main(run, main_run):
    out = os.path.join(RESULTS, "probes", run)
    B = pd.read_parquet(BENCH)
    S = pd.read_parquet(os.path.join(out, "scores.parquet"))
    P = S.pivot_table(index="idx", columns=["exp", "condition"], values="p")
    post = B.loc[P.index, "post_id"]
    fp = pd.read_csv(os.path.join(RESULTS, main_run, "fingerprints.csv"))
    fit_b = fp[(fp.who == "agent_intent") & (fp.spec == "FP3_fit")].beta.iloc[0]
    L = ["# Probe experiments: " + run, "", f"Agents: {len(P)}. Intent protocol, first-token P(Yes). Effects are "
         f"within-agent mean differences vs the no-reply condition, post-cluster bootstrap 95% CI. "
         f"Fit-SD units use the agent's own 1-SD c2v fit effect from `{main_run}` (beta = {fit_b:.2f}).", ""]

    # ladder
    none = P[("ladder", "none")]
    rows = [{"condition": "none", "mean P": none.mean(), "median P": none.median(), "Δlogit vs none": "–",
             "ΔP vs none": "–", "fit-SD units": "–"}]
    for c in [c for e, c in P.columns if e == "ladder" and c != "none"]:
        dl = boot(logit(P[("ladder", c)]) - logit(none), post)
        dp = boot(P[("ladder", c)] - none, post)
        rows.append({"condition": c, "mean P": P[("ladder", c)].mean(), "median P": P[("ladder", c)].median(),
                     "Δlogit vs none": fmt(*dl), "ΔP vs none": fmt(*dp, nd=3), "fit-SD units": f"{dl[0] / fit_b:.1f}"})
    L += ["## 1. Reply ladder: which part of the welcome moves the agent?", "",
          pd.DataFrame(rows).round(4).to_markdown(index=False), ""]
    pr = logit(P[("ladder", "praise")])
    dec = {"welcome cue (praise_welcome − praise)": logit(P[("ladder", "praise_welcome")]) - pr,
           "invitation (praise_invite − praise)": logit(P[("ladder", "praise_invite")]) - pr,
           "both (praise_welcome_invite − praise)": logit(P[("ladder", "praise_welcome_invite")]) - pr,
           "interaction (both − welcome − invite + praise)": logit(P[("ladder", "praise_welcome_invite")])
           - logit(P[("ladder", "praise_welcome")]) - logit(P[("ladder", "praise_invite")]) + pr,
           "valence (praise − disagree)": pr - logit(P[("ladder", "disagree")]),
           "reply presence (ack − none)": logit(P[("ladder", "ack")]) - logit(none)}
    L += ["Decomposition (Δlogit):", "", pd.DataFrame([{"component": k, "Δlogit": fmt(*boot(v, post))}
                                                       for k, v in dec.items()]).to_markdown(index=False), ""]

    # reproducibility vs the main run (identical prompts for none and template)
    M = pd.read_parquet(os.path.join(RESULTS, main_run, "agent_scores.parquet"))
    M = M[M.protocol == "intent"].pivot_table(index="idx", columns="condition", values="p").loc[P.index]
    rep = []
    for c_probe, c_main in (("none", "none"), ("template", "welcoming")):
        a, b = logit(P[("ladder", c_probe)]), logit(M[c_main])
        rep.append({"condition": c_probe, "corr(logit)": np.corrcoef(a, b)[0, 1], "mean P probe": P[("ladder", c_probe)].mean(),
                    f"mean P {main_run}": M[c_main].mean(), "max |ΔP|": (P[("ladder", c_probe)] - M[c_main]).abs().max()})
    L += [f"Reproducibility vs `{main_run}` (same prompts; differs only if weights/numerics differ):", "",
          pd.DataFrame(rep).round(4).to_markdown(index=False), ""]

    # negative control
    A = pd.read_parquet(os.path.join(SOURCE, "after_activity.parquet")).loc[P.index]
    ysub = S[S.exp == "nc"].drop_duplicates("idx").set_index("idx").outcome_sub.loc[P.index]
    after = [{k.lower() for k in {**json.loads(c or "{}"), **json.loads(p or "{}")}}
             for c, p in zip(A.comment_after, A.post_after)]
    human_y = pd.Series([y.lower() in s for y, s in zip(ysub, after)], index=P.index).astype(float)
    rows = []
    for c in [c for e, c in P.columns if e == "nc"]:
        dly = boot(logit(P[("nc", c)]) - logit(P[("nc", "none")]), post) if c != "none" else None
        dlx = boot(logit(P[("ladder", c)]) - logit(none), post) if c != "none" else None
        rows.append({"condition": c, "mean P(r/Y)": P[("nc", c)].mean(),
                     "Δlogit r/Y vs none": fmt(*dly) if dly else "–", "Δlogit r/X vs none (ladder)": fmt(*dlx) if dlx else "–",
                     "r/Y effect as share of r/X effect": f"{dly[0] / dlx[0]:.2f}" if dly else "–"})
    L += ["## 2. Negative-control outcome (r/Y: a random target community the user never visited)", "",
          f"Human base rate of any activity in r/Y within 90 days: {human_y.mean():.3%} "
          f"(r/X: {B.loc[P.index, 'return_90d'].mean():.1%}). "
          f"Human r/Y rate with vs without a reply in r/X: {human_y[B.loc[P.index, 'reply_received'] == 1].mean():.3%} vs "
          f"{human_y[B.loc[P.index, 'reply_received'] == 0].mean():.3%}.", "",
          pd.DataFrame(rows).round(4).to_markdown(index=False), ""]

    # outsider line
    rows = []
    for c in ("none", "template"):
        a, b = P[("ladder", c)], P[("outsider", c)]
        rows.append({"condition": c, "mean P with line": a.mean(), "mean P without line": b.mean(),
                     "Δlogit (without − with)": fmt(*boot(logit(b) - logit(a), post))})
    w_with = boot(logit(P[("ladder", "template")]) - logit(none), post)
    w_wo = boot(logit(P[("outsider", "template")]) - logit(P[("outsider", "none")]), post)
    L += ["## 3. Outsider line removed (\"You had not posted or commented in r/X ... past year.\")", "",
          pd.DataFrame(rows).round(4).to_markdown(index=False), "",
          f"Template welcome effect with the line {fmt(*w_with)}, without {fmt(*w_wo)}.", ""]

    # readout
    rp = os.path.join(out, "readout.parquet")
    if os.path.exists(rp):
        R = pd.read_parquet(rp)
        W = R.groupby(["idx", "readout", "condition"]).value.mean().unstack(["readout", "condition"])
        rows = []
        for ro in ("first_token", "verbalized", "thinking"):
            if (ro, "none") not in W:
                continue
            miss = R[R.readout == ro].value.isna().mean()
            d_t = boot(W[(ro, "template")] - W[(ro, "none")], B.loc[W.index, "post_id"])
            d_a = boot(W[(ro, "ack")] - W[(ro, "none")], B.loc[W.index, "post_id"])
            y = B.loc[W.index, "return_90d"]
            ok = W[(ro, "none")].notna()
            rows.append({"readout": ro, "mean (none)": W[(ro, "none")].mean(), "mean (ack)": W[(ro, "ack")].mean(),
                         "mean (template)": W[(ro, "template")].mean(), "ΔP ack − none": fmt(*d_a, nd=3),
                         "ΔP template − none": fmt(*d_t, nd=3), "unparsed": miss,
                         "AUROC vs human 90d (none)": roc_auc_score(y[ok], W[(ro, "none")][ok]),
                         "Spearman with first_token (none)": W[(ro, "none")].corr(W[("first_token", "none")], method="spearman")})
        T = R[R.readout == "thinking"]
        L += ["## 4. Readout: first-token P vs verbalized probability vs thinking-mode answer", "",
              f"{W.shape[0]} agents (random subsample). Human 90-day return on this subsample: "
              f"{B.loc[W.index, 'return_90d'].mean():.1%}. Thinking: 4 samples per prompt, T = 0.6, "
              f"median {T.n_gen_tokens.median():.0f} generated tokens.", "",
              pd.DataFrame(rows).round(3).to_markdown(index=False), ""]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
