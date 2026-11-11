"""P5 analysis (2026-11-06): effect of positive feedback (a highly upvoted first comment) on newcomer return,
Lambert, Saha & Chandrasekharan (CHI 2025)-style, estimated the same way for real newcomers and for their agents.

Treatment: first-comment final score in the top quartile; control: at or below the median (middle quartile
dropped, as in Lambert et al.). Pre-entry and entry covariates -> logistic propensity -> ATT inverse-probability
weights. Effect = logit of weighted mean outcome, treated minus control. Agents: the same people, the same
weights, outcome = P(return) under `real` (their actual score shown). Post-cluster bootstrap (1,000).
Within agent: high (50 points) vs low (1 point), the agent's own causal response.

    python3 p5_analyze.py qwen3-8b qwen3-32b ...   -> results/p5/summary.md"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from analyze_probes import boot, fmt, logit
from common import RESULTS

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "rscience2016")
lo = lambda m: np.log(m / (1 - m))  # noqa: E731


def design():
    R = pd.read_parquet(os.path.join(D, "retro.parquet"))
    R = R[R.n_hist > 0].copy()
    sc = pd.read_parquet(os.path.join(D, "first_scores.parquet")).set_index("id").score
    R["score"] = R.comment_id.map(sc)
    R = R[R.score.notna()]
    C = pd.read_parquet(os.path.join(D, "thread_comments.parquet"))
    R["thread_size"] = R.post_id.map(C.post_id.value_counts()).fillna(0)
    q50, q75 = R.score.quantile([.5, .75])
    R["T"] = np.where(R.score >= q75, 1, np.where(R.score <= q50, 0, np.nan))
    R = R[R["T"].notna()].copy()
    X = pd.DataFrame({"delay": np.log1p(R.entry_delay_min.clip(lower=0)), "len": np.log1p(R.first_body.str.len()),
                      "top": R.top_level, "hist": np.log1p(R.n_hist), "sticky": R.treat,
                      "thread": np.log1p(R.thread_size), "hour": pd.to_datetime(R.post_utc, unit="s").dt.hour,
                      "nsubs": R.top_subs.str.count(",") + 1}, index=R.index)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    ps = make_pipeline(StandardScaler(), PolynomialFeatures(2), LogisticRegression(C=0.5, max_iter=5000))
    e = np.clip(ps.fit(X, R["T"]).predict_proba(X)[:, 1], 0.02, 0.98)  # quadratic propensity, trimmed weights
    R["w"] = np.where(R["T"] == 1, 1.0, e / (1 - e))
    smd = ((X[R["T"] == 1].mean() - X[R["T"] == 0].apply(lambda c: np.average(c, weights=R.w[R["T"] == 0])))
           / X.std()).round(3)
    return R, X, smd, (q50, q75)


def att(R, y):
    t, c = R[R["T"] == 1], R[R["T"] == 0]
    return lo(t[y].mean()) - lo(np.average(c[y], weights=c.w))


def boot_att(R, ys, n=1000):
    """post-cluster bootstrap of att for each y in ys and of the differences ys[i] - ys[0]"""
    posts = R.post_id.unique()
    g = {p: ix for p, ix in R.groupby("post_id").groups.items()}
    rng = np.random.default_rng(0)
    B = []
    for _ in range(n):
        ix = np.concatenate([g[p] for p in rng.choice(posts, len(posts))])
        b = R.loc[ix]
        B.append([att(b, y) for y in ys])
    B = np.array(B)
    est = np.array([att(R, y) for y in ys])
    ci = np.percentile(B, [2.5, 97.5], axis=0)
    dci = np.percentile(B - B[:, :1], [2.5, 97.5], axis=0)
    return est, ci, dci


def main(models):
    R, X, smd, (q50, q75) = design()
    L = ["# P5: positive feedback (highly upvoted first comment) and newcomer return, r/science 2016", "",
         f"Treated: first-comment score ≥ {q75:.0f} (top quartile); control: ≤ {q50:.0f} (median); "
         f"n treated {int((R['T'] == 1).sum())}, control {int((R['T'] == 0).sum())}.", "",
         "Weighted standardized differences after ATT weighting: " + ", ".join(f"{k} {v:+.3f}" for k, v in smd.items()), "",
         f"Unweighted 90-day return: treated {R[R['T'] == 1].return_90d.mean():.3f}, control {R[R['T'] == 0].return_90d.mean():.3f}", ""]
    ys, rows = ["return_90d"], []
    for m in models:
        f = os.path.join(RESULTS, "p5", f"rscience2016-{m}", "scores.parquet")
        if not os.path.exists(f):
            continue
        W = pd.read_parquet(f).pivot_table(index="idx", columns="condition", values="p")
        R[f"p_{m}"] = W["real"].reindex(R.index)
        ys.append(f"p_{m}")
        v = boot(logit(W["high"]) - logit(W["low"]), pd.read_parquet(os.path.join(D, "retro.parquet")).loc[W.index, "post_id"])
        rows.append({"model": m, "within: 50 pts vs 1 pt": fmt(*v), "mean P real": W["real"].mean()})
    R = R.dropna(subset=ys)
    est, ci, dci = boot_att(R, ys)
    T = [{"who": "human" if y == "return_90d" else y[2:], "ATT Δlogit [95% CI]": fmt(est[i], *ci[:, i]),
          "agent − human [95% CI]": "" if i == 0 else fmt(est[i] - est[0], *dci[:, i])} for i, y in enumerate(ys)]
    L += ["## Same design, same people: human return vs agent P(return) with the real score shown", "",
          pd.DataFrame(T).to_markdown(index=False), "",
          "## The agent's own causal response to feedback (within agent)", "",
          pd.DataFrame(rows).round(3).to_markdown(index=False) if rows else "(no agent runs yet)", "",
          "Human anchor from the literature: Lambert et al. CHI 2025, highly upvoted posts -> +2.24% daily post "
          "rate (largest for newer accounts); a different outcome, so only an order-of-magnitude comparison.", ""]
    txt = "\n".join(L) + "\n"
    os.makedirs(os.path.join(RESULTS, "p5"), exist_ok=True)
    open(os.path.join(RESULTS, "p5", "summary.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main(sys.argv[1:])
