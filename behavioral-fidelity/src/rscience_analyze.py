"""Retrodiction scorecard: agents vs the real r/science experiment (rscience_retro_collect.py,
rscience_agents.py). (2026-11-03)

Human side (randomized at the post level, block fixed effects, SEs clustered by post)
  attrition    identifiable commenters by arm (removed comments lose their author in Arctic Shift)
  newcomer     share of sampled commenters who are newcomers, by arm
  return       90/180-day return to r/science among newcomer commenters, by arm
Agent side (per run in results/probes/<run>/scores.parquet)
  arm effect      agent P under the condition each person actually saw, regressed on the arm
                  exactly like the human outcome -> directly comparable to the human estimate
  within-agent    Δlogit of full / welcome / rules sticky and of the template reply vs no sticky

    python3 rscience_analyze.py rscience-qwen3-8b [rscience-jev ...] -> results/probes/rscience_summary.md"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score

from analyze_probes import boot, fmt, logit
from common import RESULTS

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "rscience2016")


def arm_fit(d, y, fe=False):
    """logit of y on the arm, post-clustered SE -> (beta, lo, hi, p, n). Assignment is 50/50 within
    every block, so the plain difference is unbiased; block FE (fe=True) is unstable here because
    newcomers sit in few posts (median 1 per post) and few blocks have newcomers in both arms."""
    d = d.dropna(subset=[y])
    m = smf.glm(f"{y} ~ treat" + (" + C(block)" if fe else ""), d, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.post_id)[0]})
    b, se = m.params["treat"], m.bse["treat"]
    return b, b - 1.96 * se, b + 1.96 * se, m.pvalues["treat"], int(m.nobs)


def paired_gap(H, n=4000):
    """agent minus human arm effect (logit of pooled rates), post-cluster bootstrap on the same people"""
    lo = lambda m: np.log(m / (1 - m))  # noqa: E731
    g = H.groupby("post_id").agg(t=("treat", "first"), y=("return_90d", "sum"), a=("p_arm", "sum"), n=("p_arm", "size"))

    def est(g):
        t, c = g[g.t == 1], g[g.t == 0]
        return (lo(t.a.sum() / t.n.sum()) - lo(c.a.sum() / c.n.sum())) - (lo(t.y.sum() / t.n.sum()) - lo(c.y.sum() / c.n.sum()))
    rng = np.random.default_rng(0)
    b = [est(g.iloc[rng.integers(0, len(g), len(g))]) for _ in range(n)]
    return est(g), *np.percentile(b, [2.5, 97.5])


def main(runs):
    R = pd.read_parquet(os.path.join(D, "retro.parquet"))
    F = pd.read_parquet(os.path.join(D, "commenters.parquet"))
    C = pd.read_parquet(os.path.join(D, "thread_comments.parquet"))
    P = pd.read_csv(os.path.join(D, "civilservant", "r_science_experiment_1_posts.09.26.2016.csv"))
    L = ["# Retrodicting the r/science sticky-comment experiment (Matias 2019)", ""]

    # attrition
    from rscience_retro_collect import experiment_posts
    arm = experiment_posts().set_index("id").treat
    C["treat"] = C.post_id.map(arm)
    C = C[C.treat.notna()]
    att = C.assign(deleted=C.author.eq("[deleted]")).groupby("treat").agg(
        comments=("id", "size"), deleted_author=("deleted", "mean"))
    L += ["## Attrition and composition", "",
          "Comments fetched from Arctic Shift on non-AMA experiment posts, by arm (author `[deleted]` = removed or deleted, "
          "not identifiable):", "", att.round(3).to_markdown(), "",
          "CivilServant newcomer comments per post (all newcomers, including removed; paper's data): "
          + ", ".join(f"arm {t}: {v:.2f}" for t, v in P[~P['post.ama']].groupby('TREAT')['newcomer.comments'].mean().items()), "",
          f"Sampled identifiable commenters: {len(F)}; newcomers (no r/science activity in the prior year): "
          f"{int(F.newcomer.sum())} ({F.newcomer.mean():.1%}).", ""]
    b = arm_fit(F.assign(nc=F.newcomer.astype(int)), "nc")
    L.append(f"Newcomer share by arm: control {F[F.treat == 0].newcomer.mean():.3f}, treated {F[F.treat == 1].newcomer.mean():.3f}; "
             f"logit Δ {b[0]:+.3f} [{b[1]:+.3f}, {b[2]:+.3f}], p = {b[3]:.3f}.")
    L.append("")

    # human return effect
    R = R[R.n_hist > 0]
    rows = []
    for y in ("return_90d", "return_180d"):
        b, f = arm_fit(R, y), arm_fit(R, y, fe=True)
        rows.append({"outcome": y, "control": R[R.treat == 0][y].mean(), "treated": R[R.treat == 1][y].mean(),
                     "Δlogit [95% CI]": f"{b[0]:+.3f} [{b[1]:+.3f}, {b[2]:+.3f}]", "OR": np.exp(b[0]), "p": b[3],
                     "block FE Δlogit": f"{f[0]:+.3f} [{f[1]:+.3f}, {f[2]:+.3f}]", "n": b[4]})
    L += ["## Human effect of the sticky on newcomer return (conditional on commenting)", "",
          pd.DataFrame(rows).round(3).to_markdown(index=False), ""]

    # agents
    for run in runs:
        S = pd.read_parquet(os.path.join(RESULTS, "probes", run, "scores.parquet"))
        W = S.pivot_table(index="idx", columns="condition", values="p")
        H = R.loc[R.index.intersection(W.index)].copy()
        W = W.loc[H.index]
        H["p_arm"] = np.where(H.treat == 1, W["full"], W["off"])
        b = arm_fit(H, "p_arm")
        hb = arm_fit(H, "return_90d")
        L += [f"## Agents: {run} ({len(H)} newcomers)", "",
              f"Mean P: " + ", ".join(f"{c} {W[c].mean():.3f}" for c in W.columns) +
              f" | human 90-day return {H.return_90d.mean():.3f}", "",
              f"AUROC of P(arm) for the real 90-day return: {roc_auc_score(H.return_90d, H.p_arm):.3f}", "",
              f"**Arm effect, same regression as the human outcome:** agent Δlogit {b[0]:+.3f} [{b[1]:+.3f}, {b[2]:+.3f}] "
              f"vs human {hb[0]:+.3f} [{hb[1]:+.3f}, {hb[2]:+.3f}] (same {hb[4]} newcomers).", "",
              "Agent − human gap (paired post-cluster bootstrap): " + fmt(*paired_gap(H), nd=3), "",
              "Within-agent Δlogit vs no sticky (all agents):", ""]
        off = logit(W["off"])
        rows = [{"condition": c, "Δlogit": fmt(*boot(logit(W[c]) - off, H.post_id)), "n agents": int(W[c].notna().sum())}
                for c in W.columns if c != "off"]
        L += [pd.DataFrame(rows).to_markdown(index=False), ""]
    txt = "\n".join(L) + "\n"
    open(os.path.join(RESULTS, "probes", "rscience_summary.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main(sys.argv[1:])
