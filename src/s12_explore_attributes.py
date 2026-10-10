"""Stage 12 (exploratory): other newcomer attributes suggested by related work, tested on
existing data only (no new API calls). Outcome: return to the subreddit within 90 days
(plus # contributions and "became regular" = >= 3 contributions in 90 days).

  A. Structural (co-membership) fit  - Waller & Anderson 2019: are the newcomer's OTHER
     subreddits the ones the target community's members also use? (vs text similarity)
  B. Generalist vs specialist        - Waller & Anderson 2019: entropy / concentration of
     the newcomer's prior subreddit activity
  C. Community feedback on the first comment - Joyce & Kraut 2006, Cheng et al. 2014:
     reply received, OP replied, within-post score rank, downvoted, reply tone
  D. Norm fit of the first comment   - Danescu-Niculescu-Mizil et al. 2013,
     Rajadesingan et al. 2020: style distance to the thread's norm, adjusting for the
     user's usual distance (= pre-entry adaptation); semantic fit to prior discussion
  E. Heterogeneity                   - Zhang et al. 2017, Chan et al. 2024: alignment x
     reply, alignment x text post, alignment x newcomer share of the thread

Models: post fixed-effects logit / Poisson, s8 controls, SEs clustered by post.
BH-FDR across all main-effect tests (family A-D) and separately across interactions (E)."""
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

import c2v
from api import DATA, RESULTS

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

df = pd.read_parquet(os.path.join(DATA, "analysis_table.parquet"))
for f in ("post_alignment.parquet", "comment_features.parquet"):
    df = df.join(pd.read_parquet(os.path.join(DATA, f)))
df = df.join(pd.read_parquet(os.path.join(DATA, "return.parquet")).drop(columns="return_error"))
posts = pd.read_parquet(os.path.join(DATA, "target_posts.parquet")).drop_duplicates("post_id").set_index("post_id")
users = pd.read_parquet(os.path.join(DATA, "users.parquet"))
users = users[users.history.notna()].drop_duplicates(["post_id", "author"])

df["log_hist_n"] = np.log1p(df.n_hist)
df["log_n_subs"] = np.log1p(df.hist_n_subs)
df["log_hist_len"] = np.log(df.hist_mean_len)
df["log_hist_rate"] = np.log(df.hist_rate)
df["hist_eng"] = np.arcsinh(df.hist_median_score)
df["log_delay"] = np.log1p(df.entry_delay_min.clip(lower=0))
df["log_prior"] = np.log1p(df.prior_thread_comments)
df["log_words"] = np.log1p(df.n_words)
CONTROLS = ["log_hist_n", "log_n_subs", "log_hist_len", "log_hist_rate", "hist_eng",
            "log_delay", "top_level", "log_prior"]
df["regular_90d"] = (df.return_n_90d >= 3).astype(float).where(df.return_90d.notna())

# ---------------------------------------------------------------- per-user subreddit activity
def sub_counts(h, target):
    c = {}
    for x in json.loads(h):
        s = x["sub"].lower()
        if s != target:
            c[s] = c.get(s, 0) + 1
    return c


users["key"] = list(zip(users.post_id, users.author))
users["subs"] = [sub_counts(h, s.lower()) for h, s in zip(users.history, users.subreddit)]
subs_of = dict(zip(users.key, users.subs))
df["key"] = list(zip(df.post_id, df.author))
df["subs"] = df.key.map(subs_of)
df["sub_l"] = df.subreddit.str.lower()

# platform-wide baseline share of each subreddit (each user weighted equally)
base = pd.Series(dtype=float)
for c in users.subs:
    tot = sum(c.values())
    if tot:
        base = base.add(pd.Series(c) / tot, fill_value=0)
base /= len(users)

# ---------------------------------------------------------------- A. structural fit
# audience profile of S = mean activity share of S's prior participants (in_sub_count >= 1,
# i.e. non-newcomers who commented on S's target posts); PPMI vs the platform baseline;
# only subreddits used by >= 2 of those participants.
def profile(members):
    shares = [pd.Series(c) / sum(c.values()) for c in members if c]
    if len(shares) < 5:
        return None
    m = pd.concat(shares, axis=1).fillna(0)
    support = (m > 0).sum(axis=1)
    a = m.mean(axis=1)[support >= 2]
    return np.log(a / base[a.index]).clip(lower=0)


aud = {}
for s, g in df[(df.in_sub_count > 0) & df.subs.notna()].groupby("sub_l"):
    aud[s] = profile(g.subs.tolist())
print("subreddits with an audience profile:", sum(v is not None for v in aud.values()), "/", df.sub_l.nunique())


def fit_to(c, prof):
    if prof is None or not c:
        return np.nan
    sh = pd.Series(c) / sum(c.values())
    return float((sh * prof.reindex(sh.index).fillna(0)).sum())


def neighbor_share(c, prof):
    if prof is None or not c:
        return np.nan
    sh = pd.Series(c) / sum(c.values())
    return float(sh[sh.index.isin(prof[prof > np.log(2)].index)].sum())


df["struct_fit"] = [fit_to(c, aud.get(s)) for c, s in zip(df.subs, df.sub_l)]
df["struct_neighbor"] = [neighbor_share(c, aud.get(s)) for c, s in zip(df.subs, df.sub_l)]

# ---------------------------------------------------------------- B. generalist vs specialist
def entropy(c):
    if not c:
        return np.nan
    p = np.array(list(c.values()), float)
    p /= p.sum()
    return float(-(p * np.log(p)).sum())


df["sub_entropy"] = [entropy(c) for c in df.subs]
df["sub_top_share"] = [max(c.values()) / sum(c.values()) if c else np.nan for c in df.subs]

# ---------------------------------------------------------------- A2. community2vec-style fit
# subreddit embeddings from co-membership of all sampled users (their full 365-day histories,
# target subreddits included, so every target has a vector); the newcomer's vector never
# includes the target subreddit
vid, E = c2v.train([sub_counts(h, "") for h in users.history])
print("c2v vocabulary:", len(vid), "| targets covered:", sum(s in vid for s in df.sub_l.unique()))
df["c2v_fit"] = [c2v.fit(c, s, vid, E) for c, s in zip(df.subs, df.sub_l)]

# ---------------------------------------------------------------- C. feedback on first comment
tc = pd.read_parquet(os.path.join(DATA, "thread_comments.parquet")).drop_duplicates("id")
tc["parent"] = tc.parent_id.str.split("_").str[-1]
op = posts.author.to_dict()
kids = tc[tc.parent.isin(set(df.first_comment_id.dropna()))]
op_rep = kids[kids.author == kids.post_id.map(op)].parent.unique()
df["op_replied"] = df.first_comment_id.isin(op_rep).astype(int)
df["score_rank"] = df.groupby("post_id").first_score.rank(pct=True)  # among sampled commenters
df["downvoted"] = (df.first_score <= 0).astype(int)

# ---------------------------------------------------------------- D. norm fit (style)
WORD = re.compile(r"[A-Za-z']+")
LAUGH = re.compile(r"\b(lol|lmao|lmfao|rofl|haha\w*|hehe\w*)\b|[\U0001F600-\U0001F64F\U0001F923]", re.I)
I_W = {"i", "i'm", "i've", "i'd", "i'll", "me", "my", "mine"}
YOU_W = {"you", "you're", "your", "yours", "you've", "you'd", "u", "ur"}


def sty(t):
    t = t or ""
    w = [x.lower() for x in WORD.findall(t)]
    n = max(len(w), 1)
    return (np.log1p(len(w)), int("?" in t), sum(x in I_W for x in w) / n,
            sum(x in YOU_W for x in w) / n, int("!" in t), int(bool(LAUGH.search(t))))


SDIM = ["s_len", "s_q", "s_i", "s_you", "s_excl", "s_laugh"]
# thread norm: up to 2,000 random non-sample comments per post
samp = tc[~tc.body.isin(["[deleted]", "[removed]"])]
samp = samp.sample(frac=1, random_state=0)
samp = samp[samp.groupby("post_id").cumcount() < 2000]
samp[SDIM] = [sty(t) for t in samp.body]
norm_mu = samp.groupby("post_id")[SDIM].mean()
norm_sd = samp.groupby("post_id")[SDIM].std().replace(0, np.nan)

first = pd.DataFrame([sty(t) for t in df.first_body], columns=SDIM, index=df.index)
hist_sty = {}
for u in users.itertuples():
    texts = [x["text"] for x in json.loads(u.history) if x["kind"] == "c"
             and x["sub"].lower() != u.subreddit.lower() and x["text"] not in ("[deleted]", "[removed]")]
    if len(texts) >= 10:
        hist_sty[u.key] = np.mean([sty(t) for t in texts], axis=0)
hist = pd.DataFrame([hist_sty.get(k, [np.nan] * 6) for k in df.key], columns=SDIM, index=df.index)
mu, sd = norm_mu.loc[df.post_id].values, norm_sd.loc[df.post_id].values
df["style_dist_first"] = np.nanmean(np.abs((first.values - mu) / sd), axis=1)
df["style_dist_hist"] = np.nanmean(np.abs((hist.values - mu) / sd), axis=1)
df["style_fit_first"] = -df.style_dist_first   # higher = closer to the thread norm

# ---------------------------------------------------------------- E. moderators
df["text_post"] = df.post_id.map(posts.selftext.fillna("").str.strip().str.len() > 0).astype(int)
nshare = df.groupby("post_id").newcomer_12m.mean()
df["post_nc_share"] = df.post_id.map(nshare)
df["high_nc_share"] = (df.post_nc_share > nshare.median()).astype(int)


# ---------------------------------------------------------------- models
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


nc = df[(df.newcomer_12m == 1) & df.return_90d.notna()].copy()
print("newcomers with return outcome:", len(nc))
NO_NSUBS = [c for c in CONTROLS if c != "log_n_subs"]

TESTS = [  # family, attribute, label, sample, extra controls / control set
    ("A", "struct_fit", "Structural fit (co-membership PPMI)", None, CONTROLS),
    ("A", "struct_neighbor", "Share of activity in S's neighbour subreddits", None, CONTROLS),
    ("A", "struct_fit", "Structural fit, + text alignment (pp_mini_mean)", None, CONTROLS + ["pp_mini_mean"]),
    ("A", "c2v_fit", "Community-embedding fit (community2vec-style)", None, CONTROLS),
    ("A", "c2v_fit", "Community-embedding fit, + text alignment", None, CONTROLS + ["pp_mini_mean"]),
    ("B", "sub_entropy", "Subreddit entropy (generalist)", None, NO_NSUBS),
    ("B", "sub_top_share", "Share in top prior subreddit (specialist)", None, NO_NSUBS),
    ("C", "reply_received", "First comment got a reply", None, CONTROLS),
    ("C", "op_replied", "OP replied to first comment", None, CONTROLS),
    ("C", "score_rank", "Within-post score rank of first comment", None, CONTROLS),
    ("C", "downvoted", "First comment score <= 0", None, CONTROLS),
    ("C", "reply_tone_sentiment", "Reply tone: sentiment (replied only)", "reply_received == 1", CONTROLS),
    ("C", "reply_tone_insult", "Reply tone: max insult (replied only)", "reply_received == 1", CONTROLS),
    ("D", "style_fit_first", "Style fit to thread norm, | user's usual distance", None,
     CONTROLS + ["style_dist_hist", "log_words"]),
    ("D", "sim_thread", "Semantic fit to prior discussion", None, CONTROLS + ["sim_own_history", "log_words"]),
    ("D", "sim_post", "Semantic fit to the post", None, CONTROLS + ["sim_own_history", "log_words"]),
    ("D", "log_delay", "Entry delay (later = higher)", None, CONTROLS),
]
rows = []
for fam_, x, label, where, ctrl in TESTS:
    d = nc.query(where) if where else nc
    for y, fam in (("return_90d", "logit"), ("return_n_90d", "poisson"), ("regular_90d", "logit")):
        try:
            r = fit(d, y, x, fam, ctrl)
        except Exception as e:
            print("FAILED", x, y, e)
            continue
        rows.append({"family": fam_, "attribute": x, "label": label, "outcome": y, **r})
res = pd.DataFrame(rows)
for y in res.outcome.unique():
    m = res.outcome == y
    res.loc[m, "fdr_p"] = multipletests(res.loc[m, "p"], method="fdr_bh")[1]

INTER = [("pp_mini_mean", "reply_received", "Alignment x reply received (RQ2)"),
         ("pp_mini_mean", "op_replied", "Alignment x OP replied"),
         ("pp_mini_mean", "text_post", "Alignment x text post (RQ3)"),
         ("pp_mini_mean", "high_nc_share", "Alignment x high-newcomer thread"),
         ("struct_fit", "text_post", "Structural fit x text post"),
         ("reply_received", "high_nc_share", "Reply x high-newcomer thread")]
irows = []
for x, mod, label in INTER:
    try:
        irows.append({"label": label, **fit(nc, "return_90d", x, "logit", CONTROLS, inter=mod)})
    except Exception as e:
        print("FAILED", label, e)
ires = pd.DataFrame(irows)
ires["fdr_p"] = multipletests(ires.p, method="fdr_bh")[1]

# post-level: does a higher newcomer share in the thread lower newcomers' return? (subreddit FE)
pl = nc.dropna(subset=["return_90d"] + CONTROLS).copy()
pl = pl[pl.groupby("sub_l").post_id.transform("nunique") > 1]
pl["z_share"] = (pl.post_nc_share - pl.post_nc_share.mean()) / pl.post_nc_share.std()
pl["log_post_comments"] = np.log(pl.post_id.map(posts.num_comments))
mp = smf.logit("return_90d ~ z_share + log_post_comments + " + " + ".join(CONTROLS) + " + C(sub_l)", pl).fit(
    disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": pd.factorize(pl.post_id)[0]})

# ---------------------------------------------------------------- validity / descriptives
valid = {}
for name, fn in (("struct_fit", fit_to), ("struct_neighbor", neighbor_share)):
    own = nc[name].values
    other_subs = [s for s in aud if aud[s] is not None]
    rnd = np.array([fn(c, aud[rng.choice([o for o in other_subs if o != s])]) for c, s in zip(nc.subs, nc.sub_l)])
    ok = ~np.isnan(own) & ~np.isnan(rnd)
    valid[name] = {"own_mean": own[ok].mean(), "random_mean": rnd[ok].mean(),
                   "share_own_gt_random": (own[ok] > rnd[ok]).mean(), "n": int(ok.sum())}
ex = df[(df.existing == 1)]
valid["corr_struct_vs_text_newcomers"] = nc[["struct_fit", "pp_mini_mean"]].corr(method="spearman").iloc[0, 1]
valid["struct_fit_existing_mean (in-sample, optimistic)"] = ex.struct_fit.mean()
valid["struct_fit_newcomer_mean"] = nc.struct_fit.mean()
valid["op_replied_rate"] = nc.op_replied.mean()
valid["downvoted_rate"] = nc.downvoted.mean()
valid["regular_90d_rate"] = nc.regular_90d.mean()

pd.set_option("display.width", 220, "display.max_columns", 20)
fmt = {"effect": "{:.2f}".format, "ci_lo": "{:.2f}".format, "ci_hi": "{:.2f}".format,
       "p": "{:.3f}".format, "fdr_p": "{:.3f}".format}
print("\n=== Main effects (per +1 SD, or 0/1 for binary) ===")
print(res[["family", "label", "outcome", "n", "effect", "ci_lo", "ci_hi", "p", "fdr_p"]].to_string(formatters=fmt))
print("\n=== Interactions (return_90d; ratio of ORs) ===")
print(ires.to_string(formatters=fmt))
print("\n=== Post-level newcomer share (subreddit FE) ===")
b, se = mp.params["z_share"], mp.bse["z_share"]
print(f"OR {np.exp(b):.2f} [{np.exp(b-1.96*se):.2f}, {np.exp(b+1.96*se):.2f}] p={mp.pvalues['z_share']:.3f} n={int(mp.nobs)}")
print("\n=== Validity / descriptives ===")
print(json.dumps(valid, indent=1, default=float))

res.to_csv(os.path.join(RESULTS, "explore_attributes.csv"), index=False)
ires.to_csv(os.path.join(RESULTS, "explore_interactions.csv"), index=False)
with open(os.path.join(RESULTS, "explore_validity.json"), "w") as f:
    json.dump(valid, f, indent=1, default=float)

# ---------------------------------------------------------------- robustness for the two strongest signals
# placebo: fit to a RANDOM other subreddit should not predict return to S
others = [s for s in aud if aud[s] is not None]
targets = [s for s in df.sub_l.unique() if s in vid]
nc["struct_placebo"] = [fit_to(c, aud[rng.choice([o for o in others if o != s])]) for c, s in zip(nc.subs, nc.sub_l)]
nc["c2v_placebo"] = [c2v.fit(c, rng.choice([o for o in targets if o != s]), vid, E) for c, s in zip(nc.subs, nc.sub_l)]
ok = nc[["c2v_fit", "c2v_placebo"]].notna().all(axis=1)
valid["c2v_share_own_gt_random"] = float((nc.c2v_fit[ok] > nc.c2v_placebo[ok]).mean())
valid["corr_c2v_vs_struct_text"] = nc[["c2v_fit", "struct_fit", "pp_mini_mean"]].corr("spearman").iloc[0, 1:].to_dict()
nc["wave"] = (nc.entry_delay_min <= 24 * 60).astype(int)   # entered during the first day (r/popular wave)
NO_DELAY = [c for c in CONTROLS if c != "log_delay"]
rob = []
for label, x, ctrl, sel in (
        ("placebo: struct fit to random subreddit", "struct_placebo", CONTROLS, None),
        ("struct_fit + entropy + text alignment", "struct_fit", CONTROLS + ["sub_entropy", "pp_mini_mean"], None),
        ("struct_fit, top-level comments only", "struct_fit", NO_DELAY[:6] + ["log_delay", "log_prior"], "top_level == 1"),
        ("placebo: c2v fit to random subreddit", "c2v_placebo", CONTROLS, None),
        ("c2v_fit + placebo + entropy + struct + text", "c2v_fit",
         CONTROLS + ["c2v_placebo", "sub_entropy", "struct_fit", "pp_mini_mean"], None),
        ("c2v_fit, entered in first 24h", "c2v_fit", NO_DELAY, "wave == 1"),
        ("c2v_fit, entered after 24h", "c2v_fit", NO_DELAY, "wave == 0"),
        ("text alignment, entered in first 24h", "pp_mini_mean", NO_DELAY, "wave == 1"),
        ("text alignment, entered after 24h", "pp_mini_mean", NO_DELAY, "wave == 0")):
    d = nc.query(sel) if sel else nc
    rob.append({"label": label, **fit(d, "return_90d", x, "logit", ctrl)})
rob = pd.DataFrame(rob)
print("\n=== Robustness (return_90d) ===")
print(rob.to_string(formatters=fmt))
rob.to_csv(os.path.join(RESULTS, "explore_robustness.csv"), index=False)


def wq(col):
    return nc.groupby("post_id")[col].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=False) if s.notna().sum() >= 4 else np.nan)


nc["struct_q"], nc["c2v_q"], nc["text_q"] = wq("struct_fit"), wq("c2v_fit"), wq("pp_mini_mean")
nc["delay_h"] = pd.cut(nc.entry_delay_min / 60, [-1, 1, 3, 12, 48, 1e9], labels=["<1h", "1-3h", "3-12h", "12-48h", ">48h"])
quart = pd.DataFrame({q: nc.groupby(q).return_90d.mean() for q in ("text_q", "struct_q", "c2v_q")})
print("\n=== Raw 90-day return by within-post quartile ===")
print(quart.round(3))
quart.to_csv(os.path.join(RESULTS, "explore_quartiles.csv"))
delay = nc.groupby("delay_h").agg(ret=("return_90d", "mean"), n=("return_90d", "size"),
                                  reply=("reply_received", "mean"), c2v=("c2v_fit", "mean"),
                                  text=("pp_mini_mean", "mean"))
print("\n=== Raw 90-day return by entry delay ===")
print(delay.round(3))
delay.to_csv(os.path.join(RESULTS, "explore_entry_delay.csv"))
with open(os.path.join(RESULTS, "explore_validity.json"), "w") as f:
    json.dump(valid, f, indent=1, default=float)

# per-newcomer feature table for s13b
keep = ["struct_fit", "struct_neighbor", "c2v_fit", "sub_entropy", "sub_top_share", "op_replied", "score_rank",
        "style_fit_first", "style_dist_hist", "text_post", "post_nc_share", "wave"]
nc[keep].to_parquet(os.path.join(DATA, "explore_features.parquet"))
