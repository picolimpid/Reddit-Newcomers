"""Stage 13b: extended exploratory analyses on the s13a data.

  1. Longer horizon: do community-embedding fit / text alignment predict retention at
     180 days, presence in months 4-6, # active months, and becoming a regular?
  2. Who replies: replies from community insiders (>= 3 prior contributions in the
     subreddit in the 12 months before the post) vs outsiders (0) vs in-between.
     Does an insider reply predict return where a generic reply does not?
  3. Larger embedding training set: subreddit embeddings trained on sampled users +
     EXTRA_PER_POST extra commenters per post; does the fit measure get sharper?

Models: post FE logit / Poisson, s12 controls, SEs clustered by post; BH-FDR within
each block."""
import json
import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import c2v
from api import DATA, RESULTS

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

df = pd.read_parquet(os.path.join(DATA, "analysis_table.parquet"))
for f in ("post_alignment.parquet", "comment_features.parquet"):
    df = df.join(pd.read_parquet(os.path.join(DATA, f)))
df = df.join(pd.read_parquet(os.path.join(DATA, "return.parquet")).drop(columns="return_error"))
df = df.join(pd.read_parquet(os.path.join(DATA, "explore_features.parquet")))
r180 = os.path.join(DATA, "return180.parquet")
if os.path.exists(r180):
    df = df.join(pd.read_parquet(r180).drop(columns="r180_error"))

df["log_hist_n"] = np.log1p(df.n_hist)
df["log_n_subs"] = np.log1p(df.hist_n_subs)
df["log_hist_len"] = np.log(df.hist_mean_len)
df["log_hist_rate"] = np.log(df.hist_rate)
df["hist_eng"] = np.arcsinh(df.hist_median_score)
df["log_delay"] = np.log1p(df.entry_delay_min.clip(lower=0))
df["log_prior"] = np.log1p(df.prior_thread_comments)
CONTROLS = ["log_hist_n", "log_n_subs", "log_hist_len", "log_hist_rate", "hist_eng",
            "log_delay", "top_level", "log_prior"]
df["regular_90d"] = (df.return_n_90d >= 3).astype(float).where(df.return_90d.notna())
nc = df[(df.newcomer_12m == 1) & df.return_90d.notna()].copy()

# reuse s12's estimator
import statsmodels.formula.api as smf  # noqa: E402


def fit(d, y, x, fam, ctrl=CONTROLS, inter=None):
    cols = [y, x] + ctrl + ([inter] if inter else [])
    d = d.dropna(subset=cols).copy()
    if fam in ("logit", "poisson"):
        d = d[d.groupby("post_id")[y].transform(lambda s: s.nunique() > 1)]
    d["z"] = (d[x] - d[x].mean()) / d[x].std() if d[x].nunique() > 2 else d[x]
    ctrl = [c for c in ctrl if d[c].nunique() > 1 and c != x]
    rhs = "z" + (f" * {inter}" if inter else "")
    f = f"{y} ~ {rhs} + {' + '.join(ctrl)} + C(post_id)"
    kw = dict(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.post_id)[0]})
    m = (smf.logit if fam == "logit" else smf.poisson)(f, d).fit(disp=0, maxiter=300, **kw)
    term = f"z:{inter}" if inter else "z"
    b, se = m.params[term], m.bse[term]
    return {"n": int(m.nobs), "n_posts": d.post_id.nunique(), "effect": np.exp(b),
            "ci_lo": np.exp(b - 1.96 * se), "ci_hi": np.exp(b + 1.96 * se), "p": m.pvalues[term]}


def block(name, tests):
    rows = []
    for label, d, y, x, fam, ctrl, inter in tests:
        try:
            rows.append({"block": name, "label": label, "outcome": y, **fit(d, y, x, fam, ctrl, inter)})
        except Exception as e:  # noqa: BLE001
            print("FAILED", label, y, e)
    r = pd.DataFrame(rows)
    r = r[r.p.notna()].reset_index(drop=True) if len(r) else r  # drop failed fits (e.g. outcome missing)
    if len(r):
        r["fdr_p"] = multipletests(r.p, method="fdr_bh")[1]
    return r


fmt = {k: "{:.2f}".format for k in ("effect", "ci_lo", "ci_hi")} | {k: "{:.3f}".format for k in ("p", "fdr_p")}
out = []

# ---------------------------------------------------------------- 1. longer horizon
if "return_180d" in nc:
    nc["regular_180d"] = (nc.return_n_180d >= 3).astype(int)
    nc["months_3plus"] = (nc.n_active_months >= 3).astype(int)
    print("180-day base rates:", nc[["return_180d", "active_91_180", "regular_180d", "months_3plus"]].mean().round(3).to_dict())
    T = []
    for x, lab in (("c2v_fit", "community-embedding fit"), ("pp_mini_mean", "text alignment"),
                   ("struct_fit", "structural PPMI fit")):
        for y, fam in (("return_180d", "logit"), ("active_91_180", "logit"), ("regular_180d", "logit"),
                       ("months_3plus", "logit"), ("n_active_months", "poisson")):
            T.append((lab, nc, y, x, fam, CONTROLS, None))
    # among 90-day returners: does fit predict staying (months 4-6)?
    ret = nc[nc.return_90d == 1]
    T.append(("community-embedding fit | returned by 90d", ret, "active_91_180", "c2v_fit", "logit", CONTROLS, None))
    T.append(("text alignment | returned by 90d", ret, "active_91_180", "pp_mini_mean", "logit", CONTROLS, None))
    r = block("1 horizon", T)
    print("\n=== 1. Longer horizon ===")
    print(r[["label", "outcome", "n", "effect", "ci_lo", "ci_hi", "p", "fdr_p"]].to_string(formatters=fmt))
    out.append(r)

# ---------------------------------------------------------------- 2. who replies
rp = os.path.join(DATA, "repliers.parquet")
if os.path.exists(rp):
    rep = pd.read_parquet(rp)
    rep = rep[(rep.bot == 0) & rep.replier_in_sub.notna()]
    rep["insider"] = (rep.replier_in_sub >= 3).astype(int)
    rep["outsider"] = (rep.replier_in_sub == 0).astype(int)
    print(f"\nreplies to newcomers: {len(rep)}; insider {rep.insider.mean():.1%}, "
          f"outsider {rep.outsider.mean():.1%}, 1-2 prior {1 - rep.insider.mean() - rep.outsider.mean():.1%}, "
          f"OP {rep.is_op.mean():.1%}")
    g = rep.groupby("idx").agg(insider_reply=("insider", "max"), outsider_reply=("outsider", "max"),
                               n_insider=("insider", "sum"))
    nc = nc.join(g)
    for c in ("insider_reply", "outsider_reply", "n_insider"):
        nc[c] = nc[c].fillna(0)
    nc["only_outsider"] = ((nc.reply_received == 1) & (nc.insider_reply == 0)).astype(int)
    replied = nc[nc.reply_received == 1]
    print("newcomers with >= 1 insider reply:", int(nc.insider_reply.sum()), "of", int(nc.reply_received.sum()), "replied-to")
    raw = nc.assign(kind=np.select([nc.insider_reply == 1, nc.reply_received == 1], ["insider reply", "outsider reply only"],
                                   "no reply")).groupby("kind").agg(return_90d=("return_90d", "mean"), n=("return_90d", "size"),
                                                                    c2v=("c2v_fit", "mean"))
    print(raw.round(3))
    raw.to_csv(os.path.join(RESULTS, "explore_reply_source_raw.csv"))
    T = [("insider reply (vs all others)", nc, "return_90d", "insider_reply", "logit", CONTROLS, None),
         ("insider reply, among replied-to", replied, "return_90d", "insider_reply", "logit", CONTROLS, None),
         ("insider reply (vs all others)", nc, "return_n_90d", "insider_reply", "poisson", CONTROLS, None),
         ("insider reply, + c2v fit", nc, "return_90d", "insider_reply", "logit", CONTROLS + ["c2v_fit"], None),
         ("outsider-only reply (vs no reply)", nc[nc.insider_reply == 0], "return_90d", "only_outsider", "logit", CONTROLS, None),
         ("c2v fit x insider reply", nc, "return_90d", "c2v_fit", "logit", CONTROLS, "insider_reply")]
    if "return_180d" in nc:
        T.append(("insider reply (vs all others)", nc, "return_180d", "insider_reply", "logit", CONTROLS, None))
        T.append(("insider reply (vs all others)", nc, "regular_180d", "insider_reply", "logit", CONTROLS, None))
    # do insiders preferentially answer well-fitting newcomers? (gatekeeping, among replied-to)
    T.append(("c2v fit -> gets an insider reply (replied-to)", replied, "insider_reply", "c2v_fit", "logit", CONTROLS, None))
    T.append(("text alignment -> gets an insider reply (replied-to)", replied, "insider_reply", "pp_mini_mean", "logit", CONTROLS, None))
    r = block("2 reply source", T)
    print("\n=== 2. Who replies ===")
    print(r[["label", "outcome", "n", "effect", "ci_lo", "ci_hi", "p", "fdr_p"]].to_string(formatters=fmt))
    out.append(r)

# ---------------------------------------------------------------- 3. larger embedding training set
ex = os.path.join(DATA, "extra_users.parquet")
if os.path.exists(ex):
    users = pd.read_parquet(os.path.join(DATA, "users.parquet"))
    users = users[users.history.notna()].drop_duplicates(["post_id", "author"])
    base = [{x["sub"]: 1 for x in json.loads(h)} for h in users.history]
    extra = [json.loads(s) for s in pd.read_parquet(ex).subs.dropna()]
    extra = [c for c in extra if c]
    vid, E = c2v.train(base + extra)
    print(f"\nlarger embedding: {len(base)} + {len(extra)} users, vocabulary {len(vid)}")
    hist = dict(zip(zip(users.post_id, users.author), users.history))

    def counts(key, target):
        c = {}
        for x in json.loads(hist[key]):
            s = x["sub"].lower()
            if s != target:
                c[s] = c.get(s, 0) + 1
        return c

    nc["subs"] = [counts((p, a), s.lower()) if (p, a) in hist else None
                  for p, a, s in zip(nc.post_id, nc.author, nc.subreddit)]
    nc["c2v_big"] = [c2v.fit(c, s, vid, E) for c, s in zip(nc.subs, nc.subreddit)]
    targets = [s for s in nc.subreddit.str.lower().unique() if s in vid]
    nc["c2v_big_placebo"] = [c2v.fit(c, rng.choice([o for o in targets if o != s.lower()]), vid, E)
                             for c, s in zip(nc.subs, nc.subreddit)]
    nc[["c2v_big", "c2v_big_placebo"]].to_parquet(os.path.join(DATA, "c2v_big.parquet"))  # reused by s14
    ok = nc[["c2v_big", "c2v_big_placebo"]].notna().all(axis=1)
    print(f"validity own > random: {(nc.c2v_big[ok] > nc.c2v_big_placebo[ok]).mean():.3f}; "
          f"spearman with small c2v: {nc[['c2v_big', 'c2v_fit']].corr('spearman').iloc[0, 1]:.2f}; "
          f"own vs placebo fit: {nc[['c2v_big', 'c2v_big_placebo']].corr('spearman').iloc[0, 1]:.2f}")
    T = [("larger c2v fit", nc, "return_90d", "c2v_big", "logit", CONTROLS, None),
         ("larger c2v fit", nc, "regular_90d" if "regular_90d" in nc else "return_n_90d", "c2v_big",
          "logit" if "regular_90d" in nc else "poisson", CONTROLS, None),
         ("larger c2v fit, + text alignment", nc, "return_90d", "c2v_big", "logit", CONTROLS + ["pp_mini_mean"], None),
         ("placebo: larger c2v fit to random subreddit", nc, "return_90d", "c2v_big_placebo", "logit", CONTROLS, None),
         ("larger c2v fit, + placebo fit", nc, "return_90d", "c2v_big", "logit", CONTROLS + ["c2v_big_placebo"], None),
         ("placebo fit, + larger c2v fit", nc, "return_90d", "c2v_big_placebo", "logit", CONTROLS + ["c2v_big"], None)]
    if "return_180d" in nc:
        T += [("larger c2v fit", nc, "return_180d", "c2v_big", "logit", CONTROLS, None),
              ("larger c2v fit", nc, "regular_180d", "c2v_big", "logit", CONTROLS, None),
              ("larger c2v fit, + placebo fit", nc, "return_180d", "c2v_big", "logit", CONTROLS + ["c2v_big_placebo"], None),
              ("placebo fit, + larger c2v fit", nc, "return_180d", "c2v_big_placebo", "logit", CONTROLS + ["c2v_big"], None)]
    r = block("3 larger embedding", T)
    print("\n=== 3. Larger embedding training set ===")
    print(r[["label", "outcome", "n", "effect", "ci_lo", "ci_hi", "p", "fdr_p"]].to_string(formatters=fmt))
    out.append(r)
    nc["c2v_q"] = nc.groupby("post_id").c2v_big.transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=False) if s.notna().sum() >= 4 else np.nan)
    cols = [c for c in ("return_90d", "return_180d", "regular_180d") if c in nc]
    q = nc.groupby("c2v_q")[cols].mean()
    print("\nraw rates by within-post quartile of the larger c2v fit:\n", q.round(3))
    q.to_csv(os.path.join(RESULTS, "explore_c2v_big_quartiles.csv"))

if out:
    pd.concat(out).to_csv(os.path.join(RESULTS, "explore_extended.csv"), index=False)
