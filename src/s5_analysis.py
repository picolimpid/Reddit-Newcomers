"""Stage 5: descriptive statistics, measurement validity, and fixed-effects
regressions of newcomer first-participation outcomes on prior-interest alignment."""
import json
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from api import DATA, RESULTS

warnings.filterwarnings("ignore")
OUT = RESULTS
df = pd.read_parquet(os.path.join(DATA, "analysis_table.parquet"))
res = {}

# --------------------------------------------------------------- covariates
df["log_hist_n"] = np.log1p(df.n_hist)
df["log_n_subs"] = np.log1p(df.hist_n_subs)
df["log_hist_len"] = np.log(df.hist_mean_len)
df["log_hist_rate"] = np.log(df.hist_rate)
df["hist_eng"] = np.arcsinh(df.hist_median_score)
df["log_delay"] = np.log1p(df.entry_delay_min.clip(lower=0))
df["log_prior"] = np.log1p(df.prior_thread_comments)
df["log_first_len"] = np.log1p(df.first_len)
CONTROLS = ("log_hist_n + log_n_subs + log_hist_len + log_hist_rate + hist_eng"
            " + log_delay + top_level + log_prior")

MEASURES = ["mini_mean", "mini_top5", "mini_recency", "mini_top10", "mpnet_mean", "mpnet_top5"]
OUTCOMES = {"reply_received": "logit", "n_replies": "poisson", "reengaged": "logit"}


def newcomers(lookback="12m", min_hist=20):
    d = df[(df[f"newcomer_{lookback}"] == 1) & (df.n_hist >= min_hist)].copy()
    return d


def fit(d, outcome, x, family, extra="", cluster="post_id"):
    d = d.dropna(subset=[x, outcome]).copy()
    # drop posts with no within-post outcome variation (uninformative under post FE)
    var = d.groupby("post_id")[outcome].transform(lambda s: s.nunique() > 1)
    d = d[var]
    d["z"] = (d[x] - d[x].mean()) / d[x].std()
    f = f"{outcome} ~ z + {CONTROLS}{extra} + C(post_id)"
    groups = pd.factorize(d[cluster])[0]
    if family == "logit":
        m = smf.logit(f, d).fit(disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": groups})
    elif family == "poisson":
        m = smf.poisson(f, d).fit(disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": groups})
    elif family == "negbin":
        m = smf.negativebinomial(f, d).fit(disp=0, maxiter=400, cov_type="cluster", cov_kwds={"groups": groups})
    else:  # linear probability model
        m = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": groups})
    b, se, p = m.params["z"], m.bse["z"], m.pvalues["z"]
    ratio = family in ("logit", "poisson", "negbin")
    return {"outcome": outcome, "measure": x, "model": family, "n": int(m.nobs),
            "n_posts": int(d.post_id.nunique()), "coef": b, "se": se, "p": p,
            "effect": float(np.exp(b)) if ratio else b,
            "ci_lo": float(np.exp(b - 1.96 * se)) if ratio else b - 1.96 * se,
            "ci_hi": float(np.exp(b + 1.96 * se)) if ratio else b + 1.96 * se}


# --------------------------------------------------------------- descriptives
nc = newcomers()
res["sample"] = {
    "posts": int(df.post_id.nunique()), "pairs_sampled": len(df),
    "pairs_with_history": int(df.hist_idx.notna().sum()),
    "newcomers_12m": int(len(nc)), "newcomers_6m": int(len(newcomers("6m"))),
    "newcomers_3m": int(len(newcomers("3m"))),
    "existing": int(df.existing.sum()),
    "unique_newcomer_users": int(nc.author.nunique()),
    "newcomer_share_of_sampled": float(df.newcomer_12m.mean()),
    "by_sub": nc.groupby("subreddit").size().to_dict(),
}
desc_cols = ["reply_received", "n_replies", "reengaged", "depth", "top_level", "entry_delay_min",
             "n_hist", "hist_n_subs", "mini_mean", "mini_top5", "mpnet_mean", "mpnet_top5"]
res["desc_newcomers"] = nc[desc_cols].describe().T[["mean", "50%", "std"]].round(4).to_dict("index")

# --------------------------------------------------------------- validity
v = {}
for m in ["mini_mean", "mini_top5"]:
    a, b = nc[m], nc[f"{m}_othersub"]
    v[f"{m}_vs_othersub"] = {"mean_true": a.mean(), "mean_other": b.mean(),
                             "share_true_gt_other": float((a > b).mean()),
                             "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue)}
ok = nc.mini_mean_samesub.notna()  # undefined when the subreddit has only one target post
a, b = nc.mini_mean[ok], nc.mini_mean_samesub[ok]
v["mini_mean_vs_samesub"] = {"n": int(ok.sum()), "mean_true": a.mean(), "mean_other": b.mean(),
                             "share_true_gt_other": float((a > b).mean()),
                             "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue)}
v["spearman"] = nc[MEASURES].corr("spearman").round(3).to_dict()
# newcomers vs existing members: are existing members' *outside* histories more aligned?
both = df[((df.newcomer_12m == 1) | (df.existing == 1)) & df.mini_mean.notna()]
v["align_newcomer_vs_existing"] = both.groupby("existing")[["mini_mean", "mini_top5"]].mean().to_dict()
res["validity"] = v

# --------------------------------------------------------------- main models
main = []
for m in MEASURES:
    for y, fam in OUTCOMES.items():
        main.append(fit(nc, y, m, fam))
main = pd.DataFrame(main)
main.to_csv(os.path.join(OUT, "main_models.csv"), index=False)

# --------------------------------------------------------------- robustness (primary measures)
rob = []
for m in ["mini_mean", "mini_top5"]:
    for y, fam in OUTCOMES.items():
        spec = [("lookback 6m", newcomers("6m"), fam, "", "post_id"),
                ("lookback 3m", newcomers("3m"), fam, "", "post_id"),
                ("min history 50", newcomers(min_hist=50), fam, "", "post_id"),
                ("+ first-comment length", nc, fam, " + log_first_len", "post_id"),
                ("cluster by user", nc, fam, "", "author"),
                ("first appearance per user", nc.sort_values("post_id").drop_duplicates("author"), fam, "", "post_id"),
                ("top-level comments only", nc[nc.top_level == 1], fam, "", "post_id"),
                ("linear probability / OLS", nc, "ols", "", "post_id")]
        if y == "n_replies":
            spec.append(("negative binomial", nc, "negbin", "", "post_id"))
        for name, d, f, extra, cl in spec:
            try:
                r = fit(d, y, m, f, extra, cl)
            except Exception as e:  # noqa: BLE001
                r = {"outcome": y, "measure": m, "model": f, "error": str(e)[:120]}
            r["spec"] = name
            rob.append(r)
rob = pd.DataFrame(rob)
rob.to_csv(os.path.join(OUT, "robustness.csv"), index=False)

# --------------------------------------------------------------- nonlinearity: within-post quartiles
q = []
for m in ["mini_mean", "mini_top5"]:
    d = nc[nc.groupby("post_id")[m].transform("size") >= 4].copy()  # need >= 4 newcomers to form quartiles
    d["q"] = d.groupby("post_id")[m].transform(
        lambda s: ((s.rank(method="first") - 1) / len(s) * 4).astype(int))
    for y in OUTCOMES:
        g = d.groupby("q")[y].agg(["mean", "sem", "size"]).reset_index()
        g["measure"], g["outcome"] = m, y
        q.append(g)
pd.concat(q).to_csv(os.path.join(OUT, "within_post_quartiles.csv"), index=False)

# --------------------------------------------------------------- secondary: newcomers vs existing
sec = []
d = df[((df.newcomer_12m == 1) | (df.existing == 1)) & df.hist_idx.notna()].copy()
d["newcomer"] = d.newcomer_12m
for y, fam in OUTCOMES.items():
    dd = d[d.groupby("post_id")[y].transform(lambda s: s.nunique() > 1)]
    f = f"{y} ~ newcomer + {CONTROLS} + C(post_id)"
    g = pd.factorize(dd.post_id)[0]
    fitter = smf.logit if fam == "logit" else smf.poisson
    mm = fitter(f, dd).fit(disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": g})
    b, se = mm.params["newcomer"], mm.bse["newcomer"]
    sec.append({"outcome": y, "n": int(mm.nobs), "rate_newcomer": d.loc[d.newcomer == 1, y].mean(),
                "rate_existing": d.loc[d.newcomer == 0, y].mean(), "ratio": np.exp(b),
                "ci_lo": np.exp(b - 1.96 * se), "ci_hi": np.exp(b + 1.96 * se), "p": mm.pvalues["newcomer"]})
res["newcomer_vs_existing"] = sec

with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(res, f, indent=2, default=float)

pd.set_option("display.width", 200)
print(json.dumps(res["sample"], indent=1))
print(json.dumps(res["validity"], indent=1, default=float)[:3000])
print(main.round(4).to_string())
print(rob[["spec", "measure", "outcome", "model", "n", "effect", "ci_lo", "ci_hi", "p"]].round(4).to_string())
print(pd.DataFrame(sec).round(4).to_string())
