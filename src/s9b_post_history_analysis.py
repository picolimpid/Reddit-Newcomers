"""Stage 9b: compare the post-based profile (posts U engaged with) with the earlier
comment-text profile: validity, density of user-Px similarity for newcomers vs
existing members, and the main newcomer models re-run with the new measure."""
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from api import DATA, RESULTS

warnings.filterwarnings("ignore")
df = pd.read_parquet(os.path.join(DATA, "analysis_table.parquet"))
df = df.join(pd.read_parquet(os.path.join(DATA, "post_alignment.parquet")))
for extra in ("comment_features.parquet", "return.parquet"):
    p = os.path.join(DATA, extra)
    if os.path.exists(p):
        df = df.join(pd.read_parquet(p).drop(columns=["return_error"], errors="ignore"))
df["group"] = np.select([df.newcomer_12m == 1, df.existing == 1], ["newcomer", "existing"], "other")

# ---------------------------------------------------------------- validity: comment- vs post-based profile
nc = df[(df.newcomer_12m == 1) & df.pp_mini_mean.notna()]
val = []
for label, true, other in (("comment profile, mean", "mini_mean", "mini_mean_othersub"),
                           ("comment profile, top-5", "mini_top5", "mini_top5_othersub"),
                           ("post profile, mean", "pp_mini_mean", "pp_mini_mean_othersub"),
                           ("post profile, top-5", "pp_mini_top5", "pp_mini_top5_othersub")):
    a, b = nc[true], nc[other]
    d = a - b
    val.append({"profile": label, "n": len(nc), "mean_true": a.mean(), "mean_random": b.mean(),
                "share_true_gt_random": (d > 0).mean(), "effect_size_d": d.mean() / d.std(),
                "wilcoxon_p": stats.wilcoxon(a, b).pvalue})
val = pd.DataFrame(val)
val.to_csv(os.path.join(RESULTS, "post_profile_validity.csv"), index=False)

# ---------------------------------------------------------------- density: all users by group
BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
dens = []
for ax, (col, title) in zip(axes, (("mini_mean", "Profile = text U wrote (earlier approach)"),
                                   ("pp_mini_mean", "Profile = posts U engaged with (new approach)"))):
    d = df[df.group.isin(["newcomer", "existing"]) & df[col].notna() & df.pp_mini_mean.notna()]
    xs = np.linspace(d[col].quantile(0.001), d[col].quantile(0.999), 300)
    for g, c in (("newcomer", BLUE), ("existing", ORANGE)):
        v = d.loc[d.group == g, col]
        ax.plot(xs, stats.gaussian_kde(v)(xs), color=c, lw=2, label=f"{'newcomers' if g == 'newcomer' else 'existing members'} (n={len(v):,}, median {v.median():.3f})")
        ax.axvline(v.median(), color=c, lw=1, ls=(0, (3, 3)))
        dens.append({"profile": col, "group": g, "n": len(v), "mean": v.mean(), "median": v.median(),
                     "sd": v.std()})
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel("Cosine similarity between user history and entry post Px (MiniLM, mean)", color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)  # headroom so the legend clears the curves
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
axes[0].set_ylabel("Density", color=MUTED)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "fig4_similarity_density.png"), dpi=160)
pd.DataFrame(dens).to_csv(os.path.join(RESULTS, "similarity_density_stats.csv"), index=False)

# ---------------------------------------------------------------- newcomer vs existing gap (same posts)
both = df[df.group.isin(["newcomer", "existing"])].copy()
both["newcomer"] = (both.group == "newcomer").astype(int)
gap = []
for col in ("mini_mean", "pp_mini_mean", "pp_mini_top5"):
    d = both.dropna(subset=[col]).copy()
    d["y"] = (d[col] - d[col].mean()) / d[col].std()
    m = smf.ols("y ~ newcomer + C(post_id)", d).fit(cov_type="cluster",
                                                    cov_kwds={"groups": pd.factorize(d.post_id)[0]})
    gap.append({"measure": col, "n": int(m.nobs), "newcomer_minus_existing_SD": m.params["newcomer"],
                "ci_lo": m.conf_int().loc["newcomer", 0], "ci_hi": m.conf_int().loc["newcomer", 1],
                "p": m.pvalues["newcomer"]})
gap = pd.DataFrame(gap)

# ---------------------------------------------------------------- main newcomer models with the new measure
df["log_hist_n"] = np.log1p(df.n_hist)
df["log_n_subs"] = np.log1p(df.hist_n_subs)
df["log_hist_len"] = np.log(df.hist_mean_len)
df["log_hist_rate"] = np.log(df.hist_rate)
df["hist_eng"] = np.arcsinh(df.hist_median_score)
df["log_delay"] = np.log1p(df.entry_delay_min.clip(lower=0))
df["log_prior"] = np.log1p(df.prior_thread_comments)
df["log_pp_n"] = np.log1p(df.pp_n_posts)
if "n_words" in df:
    df["log_words"] = np.log1p(df.n_words)
CONTROLS = ["log_hist_n", "log_n_subs", "log_hist_len", "log_hist_rate", "hist_eng",
            "log_delay", "top_level", "log_prior", "log_pp_n"]
OUTCOMES = [("reply_received", "logit", []), ("n_replies", "poisson", []), ("reengaged", "logit", []),
            ("return_90d", "logit", []), ("question", "logit", ["log_words"]),
            ("stance_disagree", "logit", ["log_words"]), ("sim_thread", "ols", ["sim_own_history", "log_words"])]


def fit(d, y, x, fam, extra):
    d = d.dropna(subset=[y, x] + extra).copy()
    if fam != "ols":
        d = d[d.groupby("post_id")[y].transform(lambda s: s.nunique() > 1)]
    else:
        d[y] = (d[y] - d[y].mean()) / d[y].std()
    d["z"] = (d[x] - d[x].mean()) / d[x].std()
    ctrl = [c for c in CONTROLS + extra if d[c].nunique() > 1]
    f = f"{y} ~ z + {' + '.join(ctrl)} + C(post_id)"
    kw = dict(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.post_id)[0]})
    m = {"logit": lambda: smf.logit(f, d).fit(disp=0, maxiter=300, **kw),
         "poisson": lambda: smf.poisson(f, d).fit(disp=0, maxiter=300, **kw),
         "ols": lambda: smf.ols(f, d).fit(**kw)}[fam]()
    b, se = m.params["z"], m.bse["z"]
    t = np.exp if fam != "ols" else (lambda v: v)
    return {"n": int(m.nobs), "effect": t(b), "ci_lo": t(b - 1.96 * se), "ci_hi": t(b + 1.96 * se),
            "p": m.pvalues["z"]}


ncd = df[(df.newcomer_12m == 1) & df.pp_mini_mean.notna()]
rows = []
for y, fam, extra in OUTCOMES:
    if y not in df:
        continue
    for x in ("mini_mean", "pp_mini_mean", "pp_mini_top5", "pp_mpnet_mean"):
        try:
            r = fit(ncd, y, x, fam, extra)
        except Exception as e:  # noqa: BLE001
            r = {"error": str(e)[:80]}
        rows.append({"outcome": y, "measure": x, "model": fam, **r})
models = pd.DataFrame(rows)
models.to_csv(os.path.join(RESULTS, "post_profile_models.csv"), index=False)

pd.set_option("display.width", 200)
print(val.round(4).to_string(), "\n")
print(pd.DataFrame(dens).round(4).to_string(), "\n")
print(gap.round(4).to_string(), "\n")
print("Spearman(comment-profile mean, post-profile mean) among newcomers:",
      round(nc[["mini_mean", "pp_mini_mean"]].corr("spearman").iloc[0, 1], 3), "\n")
print(models.round(4).to_string())
