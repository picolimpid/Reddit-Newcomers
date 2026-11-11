"""P7 analysis (2026-11-06): does a model reproduce the direction of the one non-zero human effect (compliance)?

Human: CivilServant r/science 2016 posts (non-AMA, >= 1 newcomer comment). Primary (design-consistent, randomization was
blocked): binomial GLM of removed vs visible newcomer comments on TREAT + block FE, HC1 SEs over posts. Secondary: pooled
removed share by arm with a post bootstrap (no FE).
Agents: for each generator x judge, mean P(remove) per agent and condition (averaged over samples); within-agent
difference full - off (pp) and the logit of pooled rates; post-cluster bootstrap. Also judge false-positive check on the
agents' real (surviving) first comments, and comment length.

    python3 p7_analyze.py   -> results/p7/summary.md
"""
import glob
import os

import numpy as np
import pandas as pd

from common import RESULTS
from panel_summary import ROOT, inputs

lg = lambda p: np.log(p / (1 - p))  # noqa: E731


def human(B=4000):
    P = pd.read_csv(os.path.join(ROOT, "data", "rscience2016", "civilservant", "r_science_experiment_1_posts.09.26.2016.csv"))
    P = P[~P["post.ama"] & (P["newcomer.comments"] > 0)]
    t, n, k = P.TREAT.values, P["newcomer.comments"].values, P["newcomer.comments.removed"].values

    def est(t, n, k):
        a, b = k[t == 1].sum() / n[t == 1].sum(), k[t == 0].sum() / n[t == 0].sum()
        return a - b, lg(a) - lg(b), b, a
    e = est(t, n, k)
    rng = np.random.default_rng(0)
    bs = np.array([est(t[i], n[i], k[i])[:2] for i in (rng.integers(0, len(P), len(P)) for _ in range(B))])
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    D = P.assign(rem=k, vis=n - k, block=P["block.id"])
    g = smf.glm("rem + vis ~ TREAT + C(block)", D, family=sm.families.Binomial()).fit(cov_type="HC1")
    fe = (g.params["TREAT"], *g.conf_int().loc["TREAT"].values, g.pvalues["TREAT"])
    return {"pp": (e[0], *np.percentile(bs[:, 0], [2.5, 97.5])), "logit": (e[1], *np.percentile(bs[:, 1], [2.5, 97.5])),
            "control": e[2], "treated": e[3], "posts": len(P), "fe": fe}


def agents(B=2000):
    R = inputs("rscience")
    rows = []
    for d in sorted(glob.glob(os.path.join(RESULTS, "p7", "rscience-*"))):
        name = os.path.basename(d)[len("rscience-"):]
        G = pd.read_parquet(os.path.join(d, "gen.parquet"))
        for jf in sorted(glob.glob(os.path.join(d, "judge-*.parquet"))):
            jn = os.path.basename(jf)[len("judge-"):-len(".parquet")]
            J = pd.read_parquet(jf).merge(G[["idx", "condition", "sample", "n_tokens"]], on=["idx", "condition", "sample"])
            W = J.groupby(["idx", "condition"]).p_remove.mean().unstack()
            L = J.groupby(["idx", "condition"]).n_tokens.mean().unstack()
            pid = pd.factorize(R.loc[W.index, "post_id"])[0]
            for c in [x for x in W.columns if x != "off"]:
                a, b = W[c].values, W["off"].values

                def est(ix):
                    return a[ix].mean() - b[ix].mean(), lg(a[ix].mean()) - lg(b[ix].mean())
                e = est(np.arange(len(a)))
                groups = pd.Series(np.arange(len(a))).groupby(pid).indices
                keys = list(groups)
                rng = np.random.default_rng(0)
                bs = np.array([est(np.concatenate([groups[keys[j]] for j in rng.integers(0, len(keys), len(keys))]))
                               for _ in range(B)])
                rows.append({"generator": name, "judge": jn, "condition": c, "P(remove) off": b.mean(), "P(remove) cond": a.mean(),
                             "Δ pp": e[0] * 100, "Δ pp CI": np.percentile(bs[:, 0], [2.5, 97.5]) * 100,
                             "Δ logit": e[1], "Δ logit CI": np.percentile(bs[:, 1], [2.5, 97.5]),
                             "tokens off → cond": f"{L['off'].mean():.0f} → {L[c].mean():.0f}"})
    return pd.DataFrame(rows)


def main():
    h = human()
    L = ["# P7: compliance by generation (r/science 2016)", "",
         f"Human (CivilServant, {h['posts']} non-AMA posts with newcomer comments): removed share of newcomer comments "
         f"{h['control']:.3f} (control) → {h['treated']:.3f} (sticky); Δ {h['pp'][0] * 100:+.1f} pp "
         f"[{h['pp'][1] * 100:+.1f}, {h['pp'][2] * 100:+.1f}], Δ logit {h['logit'][0]:+.2f} [{h['logit'][1]:+.2f}, {h['logit'][2]:+.2f}] (pooled, no FE).",
         f"Primary, block FE binomial GLM: Δ logit(removed) {h['fe'][0]:+.2f} [{h['fe'][1]:+.2f}, {h['fe'][2]:+.2f}], p = {h['fe'][3]:.3f}. "
         "An agent matches the human direction if its within-agent Δ logit(P remove) for the full sticky is < 0 with a CI excluding 0.", ""]
    A = agents()
    if len(A):
        A["match"] = [("same direction" if (lo < 0 and hi < 0) else "opposite" if lo > 0 else "CI includes 0")
                      for lo, hi in A["Δ logit CI"]]
        A["Δ pp"] = [f"{e:+.1f} [{c[0]:+.1f}, {c[1]:+.1f}]" for e, c in zip(A["Δ pp"], A["Δ pp CI"])]
        A["Δ logit"] = [f"{e:+.2f} [{c[0]:+.2f}, {c[1]:+.2f}]" for e, c in zip(A["Δ logit"], A["Δ logit CI"])]
        A = A.drop(columns=["Δ pp CI", "Δ logit CI"]).round(3)
        L += ["Agents: within-agent change in judged P(remove) vs no sticky (mean over 2 samples per agent; post-cluster bootstrap).",
              "", A.to_markdown(index=False), ""]
    for jf in sorted(glob.glob(os.path.join(RESULTS, "p7", "real", "judge-*.parquet"))):
        J = pd.read_parquet(jf)
        L.append(f"Judge {os.path.basename(jf)[6:-8]} on the agents' real first comments (all survived moderation): "
                 f"mean P(remove) {J.p_remove.mean():.3f}, share > 0.5 {np.mean(J.p_remove > 0.5):.3f}.")
    txt = "\n".join(L) + "\n"
    open(os.path.join(RESULTS, "p7", "summary.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
