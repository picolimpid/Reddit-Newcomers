"""Stage 8: Step 6 - does prior-interest alignment predict what newcomers write and
how they integrate? Same post-FE design as stage 5, new outcomes from stage 7.

Primary (pre-specified) outcomes: question, stance_disagree, sim_thread, return_90d.
Everything else is exploratory. Benjamini-Hochberg FDR within each family."""
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from api import DATA, RESULTS

warnings.filterwarnings("ignore")
df = pd.read_parquet(os.path.join(DATA, "analysis_table.parquet"))
df = df.join(pd.read_parquet(os.path.join(DATA, "comment_features.parquet")), how="inner")
ret = os.path.join(DATA, "return.parquet")
if os.path.exists(ret):
    df = df.join(pd.read_parquet(ret).drop(columns="return_error"))

df["log_hist_n"] = np.log1p(df.n_hist)
df["log_n_subs"] = np.log1p(df.hist_n_subs)
df["log_hist_len"] = np.log(df.hist_mean_len)
df["log_hist_rate"] = np.log(df.hist_rate)
df["hist_eng"] = np.arcsinh(df.hist_median_score)
df["log_delay"] = np.log1p(df.entry_delay_min.clip(lower=0))
df["log_prior"] = np.log1p(df.prior_thread_comments)
df["log_words"] = np.log1p(df.n_words)
CONTROLS = ("log_hist_n + log_n_subs + log_hist_len + log_hist_rate + hist_eng"
            " + log_delay + top_level + log_prior")
LEN = " + log_words"
OWN = " + sim_own_history"

# name, family, extra controls, sample restriction, primary?
SPECS = [
    ("question", "logit", LEN, None, True),
    ("stance_disagree", "logit", LEN, None, True),          # replies only (NaN otherwise)
    ("sim_thread", "ols", OWN + LEN, None, True),
    ("return_90d", "logit", "", None, True),
    ("return_30d", "logit", "", None, False),
    ("stance_agree", "logit", LEN, None, False),
    ("sentiment", "ols", LEN, None, False),
    ("insult", "ols", LEN, None, False),
    ("toxicity", "ols", LEN, None, False),
    ("log_words", "ols", "", None, False),
    ("rate_i", "ols", LEN, None, False),
    ("rate_you", "ols", LEN, None, False),
    ("rate_we", "ols", LEN, None, False),
    ("rate_hedge", "ols", LEN, None, False),
    ("exclaim", "logit", LEN, None, False),
    ("laugh", "logit", LEN, None, False),
    ("sim_post", "ols", OWN + LEN, None, False),
    ("sim_parent", "ols", OWN + LEN, None, False),
    ("reply_tone_sentiment", "ols", LEN, None, False),     # among subjects who got replies
    ("reply_tone_insult", "ols", LEN, None, False),
    ("reply_tone_disagree", "logit", LEN, None, False),
    ("answered_replier", "logit", LEN, "reply_received == 1", False),
    ("exchange_turns", "poisson", LEN, "reply_received == 1", False),
]


def fit(d, y, x, fam, extra, rhs_var="z"):
    d = d.dropna(subset=[y] + ([x] if x else [])).copy()
    if fam in ("logit", "poisson"):
        d = d[d.groupby("post_id")[y].transform(lambda s: s.nunique() > 1)]
    if x:
        d["z"] = (d[x] - d[x].mean()) / d[x].std()
    if fam == "ols":
        d[y] = (d[y] - d[y].mean()) / d[y].std()  # coefficient in outcome-SD units
    if "sim_own_history" in extra:
        d = d.dropna(subset=["sim_own_history"])
    # drop controls that are constant in this sample (e.g. top_level among replies-only outcomes)
    ctrl = [c for c in (CONTROLS + extra).split(" + ") if c.strip() and d[c.strip()].nunique() > 1]
    f = f"{y} ~ {rhs_var} + {' + '.join(ctrl)} + C(post_id)"
    g = pd.factorize(d.post_id)[0]
    kw = dict(cov_type="cluster", cov_kwds={"groups": g})
    m = {"logit": lambda: smf.logit(f, d).fit(disp=0, maxiter=300, **kw),
         "poisson": lambda: smf.poisson(f, d).fit(disp=0, maxiter=300, **kw),
         "ols": lambda: smf.ols(f, d).fit(**kw)}[fam]()
    b, se = m.params[rhs_var], m.bse[rhs_var]
    ratio = fam != "ols"
    return {"n": int(m.nobs), "n_posts": int(d.post_id.nunique()), "coef": b, "p": m.pvalues[rhs_var],
            "effect": np.exp(b) if ratio else b,
            "ci_lo": np.exp(b - 1.96 * se) if ratio else b - 1.96 * se,
            "ci_hi": np.exp(b + 1.96 * se) if ratio else b + 1.96 * se,
            "scale": "OR/RR" if ratio else "SD of outcome"}


nc = df[df.newcomer_12m == 1]
rows = []
for y, fam, extra, where, primary in SPECS:
    if y not in df:
        continue
    d = nc.query(where) if where else nc
    measures = ["mini_mean", "mini_top5", "mpnet_mean"] if primary else ["mini_mean"]
    for x in measures:
        try:
            r = fit(d, y, x, fam, extra)
        except Exception as e:  # noqa: BLE001
            r = {"error": str(e)[:100]}
        rows.append({"outcome": y, "measure": x, "model": fam, "primary": primary,
                     "mean_newcomers": d[y].mean(), **r})
res = pd.DataFrame(rows)
# FDR within family, on the primary alignment measure
for fam_flag in (True, False):
    m = (res.primary == fam_flag) & (res.measure == "mini_mean") & res.p.notna()
    res.loc[m, "p_fdr"] = multipletests(res.loc[m, "p"], method="fdr_bh")[1]
res.to_csv(os.path.join(RESULTS, "content_models.csv"), index=False)

# ---------------------------------------------------------------- newcomers vs existing members
both = df[(df.newcomer_12m == 1) | (df.existing == 1)].copy()
both["newcomer"] = both.newcomer_12m
cmp_rows = []
for y, fam, extra, where, _ in SPECS:
    if y not in df or y.startswith("return"):
        continue
    d = both.query(where) if where else both
    try:
        r = fit(d, y, None, fam, extra, rhs_var="newcomer")
    except Exception as e:  # noqa: BLE001
        r = {"error": str(e)[:100]}
    cmp_rows.append({"outcome": y, "model": fam, "mean_newcomer": d.loc[d.newcomer == 1, y].mean(),
                     "mean_existing": d.loc[d.newcomer == 0, y].mean(), **r})
cmp = pd.DataFrame(cmp_rows)
m = cmp.p.notna()
cmp.loc[m, "p_fdr"] = multipletests(cmp.loc[m, "p"], method="fdr_bh")[1]
cmp.to_csv(os.path.join(RESULTS, "content_newcomer_vs_existing.csv"), index=False)

# ---------------------------------------------------------------- human-validation sample
tc = pd.read_parquet(os.path.join(DATA, "thread_comments.parquet")).drop_duplicates("id").set_index("id")
v = nc.dropna(subset=["stance_disagree"]).sample(min(200, nc.stance_disagree.notna().sum()), random_state=1)
v = pd.DataFrame({
    "item": range(1, len(v) + 1),
    "parent_text": [tc.at[tc.at[c, "parent_id"][3:], "body"] for c in v.first_comment_id],
    "comment_text": v.first_body.values,
    "human_question_0_1": "", "human_stance_agree_disagree_neutral": "", "human_insult_0_1": ""})
v.to_csv(os.path.join(RESULTS, "validation_sample_to_label.csv"), index=False, encoding="utf-8-sig")
key = nc.loc[nc.stance_disagree.notna()].sample(len(v), random_state=1)
pd.DataFrame({"item": range(1, len(v) + 1), "auto_question": key.question.values,
              "auto_stance_disagree": key.stance_disagree.values, "auto_stance_agree": key.stance_agree.values,
              "auto_insult": key.insult.values}).to_csv(os.path.join(RESULTS, "validation_auto_labels.csv"),
                                                        index=False)

pd.set_option("display.width", 220)
cols = ["outcome", "measure", "model", "primary", "n", "mean_newcomers", "effect", "ci_lo", "ci_hi", "p", "p_fdr"]
print(res[[c for c in cols if c in res]].round(4).to_string())
print(cmp.round(4).to_string())
