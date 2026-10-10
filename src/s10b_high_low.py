"""Stage 10b: within newcomers, split into HIGH vs LOW similarity to the entry post
and compare their posting activity after the interaction.

Similarity = post-based profile (posts the user engaged with before Px vs Px, MiniLM
mean; s9). Groups are formed WITHIN each target post: top 25% = high, bottom 25% = low
(posts with >= 4 newcomers). Robustness: within-post median split.

Outcomes after the first comment on Px:
  in the target subreddit : returned within 30 / 90 days, # contributions in 90 days,
                            days until first return (cumulative return curve)
  across all of Reddit    : # comments + posts in 90 days, change vs the 90 days before,
                            active days, still active in days 60-90
Comparisons: raw difference, regression with post FE + controls, and 1:1 propensity
matching on pre-interaction characteristics (proposal §10)."""
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from api import DATA, RESULTS, get

warnings.filterwarnings("ignore")
DAY = 86400
rng = np.random.default_rng(0)

df = pd.read_parquet(os.path.join(DATA, "analysis_table.parquet"))
for f in ("post_alignment.parquet", "activity.parquet"):
    df = df.join(pd.read_parquet(os.path.join(DATA, f)))
tc = pd.read_parquet(os.path.join(DATA, "thread_comments.parquet"),
                     columns=["id", "created_utc"]).drop_duplicates("id").set_index("id")
nc = df[(df.newcomer_12m == 1) & df.pp_mini_mean.notna() & df.act_error.isna()].copy()
nc["first_utc"] = tc.loc[nc.first_comment_id, "created_utc"].values

# ---------------------------------------------------------------- in-subreddit return details
# same queries as s7a (cache hits): activity in the subreddit within 90 days, outside the thread
def sub_return(r):
    t0, t1 = int(r.first_utc) + 1, int(r.first_utc) + 90 * DAY
    cs = get("comments/search", author=r.author, subreddit=r.subreddit, after=t0, before=t1,
             sort="asc", limit=100, fields="link_id,created_utc")
    ps = get("posts/search", author=r.author, subreddit=r.subreddit, after=t0, before=t1,
             sort="asc", limit=100, fields="id,created_utc")
    ts = sorted([c["created_utc"] for c in cs if c["link_id"].split("_")[-1] != r.post_id]
                + [p["created_utc"] for p in ps])
    return len(ts), ((ts[0] - r.first_utc) / DAY if ts else np.nan)


nc[["sub_n90", "days_to_return"]] = [sub_return(r) for r in nc.itertuples()]
nc["return_30d"] = (nc.days_to_return <= 30).astype(int)
nc["return_90d"] = nc.days_to_return.notna().astype(int)
nc["pre90"] = nc.pre90_c + nc.pre90_p
nc["post90"] = nc.post90_c + nc.post90_p
nc["log_post90"] = np.log1p(nc.post90)
nc["activity_change"] = np.log1p(nc.post90) - np.log1p(nc.pre90)  # >0 = more active after

# ---------------------------------------------------------------- groups (within post)
size = nc.groupby("post_id").pp_mini_mean.transform("size")
# symmetric percentile (rank - 0.5) / n, so top and bottom 25% get equal counts within each post
pct = (nc.groupby("post_id").pp_mini_mean.rank(method="first") - 0.5) / size
nc["group_q"] = np.where(size < 4, None, np.where(pct >= 0.75, "high", np.where(pct <= 0.25, "low", "middle")))
nc["group_med"] = np.where(size < 2, None, np.where(pct > 0.5, "high", "low"))

# ---------------------------------------------------------------- controls (pre-interaction only)
nc["log_hist_n"] = np.log1p(nc.n_hist)
nc["log_n_subs"] = np.log1p(nc.hist_n_subs)
nc["log_hist_len"] = np.log(nc.hist_mean_len)
nc["log_hist_rate"] = np.log(nc.hist_rate)
nc["hist_eng"] = np.arcsinh(nc.hist_median_score)
nc["log_delay"] = np.log1p(nc.entry_delay_min.clip(lower=0))
nc["log_prior"] = np.log1p(nc.prior_thread_comments)
nc["log_pre90"] = np.log1p(nc.pre90)
CONTROLS = ["log_hist_n", "log_n_subs", "log_hist_len", "log_hist_rate", "hist_eng",
            "log_delay", "top_level", "log_prior", "log_pre90"]

OUTCOMES = [  # name, label, family
    ("return_30d", "Returned to the subreddit within 30 days", "logit"),
    ("return_90d", "Returned to the subreddit within 90 days", "logit"),
    ("sub_n90", "Contributions in the subreddit, 90 days", "poisson"),
    ("post90", "Reddit-wide comments + posts, 90 days after", "poisson"),
    ("activity_change", "Change in Reddit activity, log(after+1) - log(before+1)", "ols"),
    ("post90_active_days", "Active days on Reddit, 90 days after", "poisson"),
    ("active_last30", "Still active on Reddit in days 60-90", "logit"),
]


def boot_ci(a, b, n=2000):
    """95% bootstrap CI for mean(a) - mean(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return np.percentile(d, [2.5, 97.5])


def fe_model(d, y, fam):
    d = d.copy()
    d["high"] = (d.grp == "high").astype(int)
    if fam != "ols":
        d = d[d.groupby("post_id")[y].transform(lambda s: s.nunique() > 1)]
    f = f"{y} ~ high + {' + '.join(CONTROLS)} + C(post_id)"
    kw = dict(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.post_id)[0]})
    m = {"logit": lambda: smf.logit(f, d).fit(disp=0, maxiter=300, **kw),
         "poisson": lambda: smf.poisson(f, d).fit(disp=0, maxiter=300, **kw),
         "ols": lambda: smf.ols(f, d).fit(**kw)}[fam]()
    b, se = m.params["high"], m.bse["high"]
    t = np.exp if fam != "ols" else (lambda v: v)
    return t(b), t(b - 1.96 * se), t(b + 1.96 * se), m.pvalues["high"], int(m.nobs)


def matched(d, y):
    """1:1 nearest-neighbour propensity matching (high = treated), caliper 0.2 SD of logit."""
    d = d.dropna(subset=CONTROLS + [y]).copy()
    d["high"] = (d.grp == "high").astype(int)
    ps = smf.logit(f"high ~ {' + '.join(CONTROLS)}", d).fit(disp=0)
    d["lp"] = ps.fittedvalues  # linear predictor (log-odds of being high)
    cal = 0.2 * d.lp.std()
    treated, pool = d[d.high == 1], d[d.high == 0].copy()
    pairs = []
    for i, r in treated.sample(frac=1, random_state=0).iterrows():
        if pool.empty:
            break
        j = (pool.lp - r.lp).abs().idxmin()
        if abs(pool.at[j, "lp"] - r.lp) <= cal:
            pairs.append((i, j))
            pool = pool.drop(j)
    diff = np.array([d.at[i, y] - d.at[j, y] for i, j in pairs], float)
    boots = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
    return diff.mean(), *np.percentile(boots, [2.5, 97.5]), stats.ttest_1samp(diff, 0).pvalue, len(pairs)


rows = []
for split in ("group_q", "group_med"):
    d = nc[nc[split].isin(["high", "low"])].copy()
    d["grp"] = d[split]
    hi, lo = d[d.grp == "high"], d[d.grp == "low"]
    for y, label, fam in OUTCOMES:
        lo_ci = boot_ci(hi[y], lo[y])
        r = {"split": "top vs bottom 25% (within post)" if split == "group_q" else "median split (within post)",
             "outcome": y, "label": label, "n_high": len(hi), "n_low": len(lo),
             "mean_high": hi[y].mean(), "mean_low": lo[y].mean(), "raw_diff": hi[y].mean() - lo[y].mean(),
             "raw_ci_lo": lo_ci[0], "raw_ci_hi": lo_ci[1],
             "raw_p": (stats.mannwhitneyu(hi[y], lo[y]).pvalue if fam != "logit"
                       else stats.chi2_contingency(pd.crosstab(d.grp, d[y]))[1])}
        try:
            e, clo, chi, p, n = fe_model(d, y, fam)
            r.update({"adj_effect": e, "adj_ci_lo": clo, "adj_ci_hi": chi, "adj_p": p, "adj_n": n,
                      "adj_scale": "OR" if fam == "logit" else ("RR" if fam == "poisson" else "diff")})
        except Exception as ex:  # noqa: BLE001
            r["adj_error"] = str(ex)[:80]
        if split == "group_q":
            md, mlo, mhi, mp, npairs = matched(d, y)
            r.update({"matched_diff": md, "matched_ci_lo": mlo, "matched_ci_hi": mhi, "matched_p": mp,
                      "matched_pairs": npairs})
        rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(os.path.join(RESULTS, "high_low_comparison.csv"), index=False)

# balance check on pre-interaction characteristics (top vs bottom 25%)
d = nc[nc.group_q.isin(["high", "low"])]
bal = pd.DataFrame({c: [d.loc[d.group_q == "high", c].mean(), d.loc[d.group_q == "low", c].mean()]
                    for c in CONTROLS + ["pp_mini_mean", "pre90"]}, index=["high", "low"]).T
bal.to_csv(os.path.join(RESULTS, "high_low_balance.csv"))

# ---------------------------------------------------------------- figures
BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
# (a) cumulative return to the subreddit
ax = axes[0]
days = np.arange(0, 91)
for g, c in (("high", BLUE), ("low", ORANGE)):
    s = d[d.group_q == g]
    curve = [(s.days_to_return <= t).mean() * 100 for t in days]
    se = [np.sqrt(p / 100 * (1 - p / 100) / len(s)) * 100 for p in curve]
    ax.plot(days, curve, color=c, lw=2, label=f"{g} similarity (n={len(s)})")
    ax.fill_between(days, np.array(curve) - 1.96 * np.array(se), np.array(curve) + 1.96 * np.array(se),
                    color=c, alpha=0.12, lw=0)
ax.set_xlabel("Days after first comment on the entry post", color=MUTED)
ax.set_ylabel("% who returned to the subreddit", color=MUTED)
ax.set_title("(a) Cumulative return to the subreddit", loc="left", fontsize=10, color=INK)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_xlim(0, 90)
# (b) Reddit-wide activity before vs after
ax = axes[1]
for k, (g, c) in enumerate((("high", BLUE), ("low", ORANGE))):
    s = d[d.group_q == g]
    vals = [s.pre90.median(), s.post90.median()]
    ci = [np.percentile([rng.choice(s[col], len(s)).mean() for _ in range(1000)], [2.5, 97.5])
          for col in ("pre90", "post90")]
    means = [s.pre90.mean(), s.post90.mean()]
    x = np.array([0, 1]) + (k - 0.5) * 0.12
    ax.errorbar(x, means, yerr=[[m - lo for m, (lo, hi) in zip(means, ci)], [hi - m for m, (lo, hi) in zip(means, ci)]],
                color=c, lw=2, marker="o", ms=7, mec=SURF, mew=2, capsize=0, label=f"{g} similarity")
ax.set_xticks([0, 1], ["90 days before", "90 days after"])
ax.set_ylabel("Mean Reddit-wide comments + posts", color=MUTED)
ax.set_title("(b) Reddit-wide activity before vs after", loc="left", fontsize=10, color=INK)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "fig5_high_low_activity.png"), dpi=160)

pd.set_option("display.width", 250)
print("newcomers:", len(nc), "| top/bottom-25% groups:", (d.group_q == "high").sum(), "/", (d.group_q == "low").sum())
print(bal.round(3).to_string(), "\n")
cols = ["split", "outcome", "n_high", "n_low", "mean_high", "mean_low", "raw_diff", "raw_p",
        "adj_effect", "adj_ci_lo", "adj_ci_hi", "adj_p", "matched_diff", "matched_ci_lo", "matched_ci_hi",
        "matched_p", "matched_pairs"]
print(res[[c for c in cols if c in res]].round(3).to_string())
