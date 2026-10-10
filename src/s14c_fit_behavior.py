"""Stage 14c: do newcomers with different similarity BEHAVE differently?

Similarity measures compared side by side:
  c2v   community-embedding (network) fit, larger embedding (s13b)   <- primary
  text  post-based text alignment, MiniLM mean (s9)                  <- reference
Each outcome: post-FE model per +1 SD (continuous, all newcomers) and raw means for the
within-post top vs bottom quartile. BH-FDR within each family x measure.

  F1  entry and first comment: timing, position, content, reception (existing data)
  F2  Reddit-wide behavior 90 days after vs before (ANCOVA; s14b metrics) + movement of
      overall activity towards the community's network neighbourhood
  F3  what returners do INSIDE the community (s14a insub): volume, posts, conversation,
      reception (score), ordinary vs viral threads, timing of return
  F4  pre-entry trajectory: was the newcomer moving towards the community before entry?
  F5  social ties (s14a ties): did they talk to the community's members / to people close
      to it before entry, and does that explain the fit -> return association?"""
import json
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

import c2v
from api import DATA, RESULTS

warnings.filterwarnings("ignore")
DAY = 86400
ONLY = os.environ.get("ONLY", "")  # e.g. ONLY=F1,F4

df = pd.read_parquet(os.path.join(DATA, "analysis_table.parquet"))
for f in ("post_alignment.parquet", "comment_features.parquet", "c2v_big.parquet"):
    df = df.join(pd.read_parquet(os.path.join(DATA, f)))
df = df.join(pd.read_parquet(os.path.join(DATA, "return.parquet")).drop(columns="return_error"))
df = df.join(pd.read_parquet(os.path.join(DATA, "return180.parquet")).drop(columns="r180_error"))
posts = pd.read_parquet(os.path.join(DATA, "target_posts.parquet")).drop_duplicates("post_id").set_index("post_id")
tc = pd.read_parquet(os.path.join(DATA, "thread_comments.parquet"), columns=["id", "created_utc"]).drop_duplicates("id")

df["log_hist_n"] = np.log1p(df.n_hist)
df["log_n_subs"] = np.log1p(df.hist_n_subs)
df["log_hist_len"] = np.log(df.hist_mean_len)
df["log_hist_rate"] = np.log(df.hist_rate)
df["hist_eng"] = np.arcsinh(df.hist_median_score)
df["log_delay"] = np.log1p(df.entry_delay_min.clip(lower=0))
df["log_prior"] = np.log1p(df.prior_thread_comments)
df["log_words"] = np.log1p(df.n_words)
df["score_asinh"] = np.arcsinh(df.first_score)
HIST = ["log_hist_n", "log_n_subs", "log_hist_len", "log_hist_rate", "hist_eng"]
ENTRY = ["log_delay", "top_level", "log_prior"]
CONTROLS = HIST + ENTRY

nc = df[(df.newcomer_12m == 1) & df.c2v_big.notna() & df.pp_mini_mean.notna()].copy()
nc["first_utc"] = tc.set_index("id").loc[nc.first_comment_id, "created_utc"].values
MEASURES = {"c2v": "c2v_big", "text": "pp_mini_mean"}
# ---------------------------------------------------------------- shared: subreddit embedding
need_emb = True
if need_emb:
    users = pd.read_parquet(os.path.join(DATA, "users.parquet"))
    users = users[users.history.notna()].drop_duplicates(["post_id", "author"])
    extra = [json.loads(s) for s in pd.read_parquet(os.path.join(DATA, "extra_users.parquet")).subs.dropna()]
    vid, E = c2v.train([{x["sub"]: 1 for x in json.loads(h)} for h in users.history] + [c for c in extra if c])
    hist = dict(zip(zip(users.post_id, users.author), users.history))
    print("embedding vocabulary:", len(vid), flush=True)



def window_fit(key, sub, T, lo, hi):
    """fit of the user's activity between lo and hi days before the post (>= 5 items)"""
    c = {}
    for x in json.loads(hist[key]):
        if lo <= (T - x["t"]) / DAY < hi and x["sub"].lower() != sub.lower():
            c[x["sub"].lower()] = c.get(x["sub"].lower(), 0) + 1
    return c2v.fit(c, sub, vid, E) if sum(c.values()) >= 5 else np.nan


T0 = nc.post_id.map(posts.created_utc)
nc["fit_recent30"] = [window_fit((p, a), s, t, 0, 30) for p, a, s, t in zip(nc.post_id, nc.author, nc.subreddit, T0)]
MEASURES["recent"] = "fit_recent30"   # network fit of the last 30 days before entry (F4)

for name, col in MEASURES.items():
    size = nc.groupby("post_id")[col].transform("count")
    pct = (nc.groupby("post_id")[col].rank(method="first") - 0.5) / size
    nc[f"grp_{name}"] = np.where(size < 4, None, np.where(pct >= 0.75, "high", np.where(pct <= 0.25, "low", None)))
print("newcomers:", len(nc), "| high/low per measure:",
      {m: int((nc[f"grp_{m}"] == "high").sum()) for m in MEASURES}, flush=True)


def fit(d, y, x, fam, ctrl, extra_rhs=""):
    d = d.dropna(subset=[y, x] + ctrl).copy()
    if fam in ("logit", "poisson"):
        d = d[d.groupby("post_id")[y].transform(lambda s: s.nunique() > 1)]
    if len(d) < 30:
        raise ValueError("too few observations")
    d["z"] = (d[x] - d[x].mean()) / d[x].std()
    if fam == "ols":
        d[y] = (d[y] - d[y].mean()) / d[y].std()
    ctrl = [c for c in ctrl if d[c].nunique() > 1 and c != y]
    f = f"Q('{y}') ~ z + {' + '.join(ctrl) if ctrl else '1'}{extra_rhs} + C(post_id)"
    kw = dict(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.post_id)[0]})
    m = {"logit": smf.logit, "poisson": smf.poisson, "ols": smf.ols}[fam](f, d)
    m = m.fit(**kw) if fam == "ols" else m.fit(disp=0, maxiter=300, **kw)
    b, se = m.params["z"], m.bse["z"]
    ratio = fam != "ols"
    return {"n": int(m.nobs), "effect": np.exp(b) if ratio else b,
            "ci_lo": np.exp(b - 1.96 * se) if ratio else b - 1.96 * se,
            "ci_hi": np.exp(b + 1.96 * se) if ratio else b + 1.96 * se, "p": m.pvalues["z"],
            "scale": "OR/RR" if ratio else "SD"}


def family(name, d, specs):
    """specs: (outcome, label, fam, controls, sample query or None, extra rhs)."""
    rows = []
    for meas, col in MEASURES.items():
        for y, label, fam, ctrl, where, extra in specs:
            dd = d.query(where) if where else d
            if y not in dd:
                continue
            try:
                r = fit(dd, y, col, fam, ctrl, extra)
            except Exception as e:  # noqa: BLE001
                print("  skip", meas, y, str(e)[:80])
                continue
            g = dd.groupby(f"grp_{meas}")[y].mean()
            rows.append({"family": name, "measure": meas, "outcome": y, "label": label, **r,
                         "high_mean": g.get("high", np.nan), "low_mean": g.get("low", np.nan)})
    r = pd.DataFrame(rows)
    for meas in MEASURES:
        m = r.measure == meas
        if m.any():
            r.loc[m, "fdr_p"] = multipletests(r.loc[m, "p"], method="fdr_bh")[1]
    return r


def show(r):
    if r.empty:
        return
    w = r.pivot_table(index=["outcome", "label"], columns="measure",
                      values=["effect", "p", "fdr_p", "high_mean", "low_mean"], aggfunc="first")
    out = pd.DataFrame(index=w.index)
    for m in MEASURES:
        if ("effect", m) not in w:
            continue
        out[f"{m}: hi/lo"] = [f"{h:.3g} / {lo:.3g}" for h, lo in zip(w[("high_mean", m)], w[("low_mean", m)])]
        out[f"{m}: eff"] = w[("effect", m)].map("{:.2f}".format)
        out[f"{m}: p"] = w[("p", m)].map("{:.3f}".format)
        out[f"{m}: fdr"] = w[("fdr_p", m)].map("{:.3f}".format)
    print(out.to_string())


pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 46)
results = []

# ---------------------------------------------------------------- F1 entry and first comment
if not ONLY or "F1" in ONLY:
    LEN = ["log_words"]
    specs = [
        ("log_delay", "Entry: log minutes after post creation", "ols", HIST, None, ""),
        ("top_level", "Entry: first comment is top-level", "logit", HIST + ["log_delay"], None, ""),
        ("log_prior", "Entry: log # comments already in thread", "ols", HIST + ["log_delay"], None, ""),
        ("log_words", "Comment: length (log words)", "ols", CONTROLS, None, ""),
        ("question", "Comment: asks a question", "logit", CONTROLS + LEN, None, ""),
        ("stance_disagree", "Comment: disagrees with parent (replies)", "logit", CONTROLS + LEN, None, ""),
        ("sentiment", "Comment: sentiment", "ols", CONTROLS + LEN, None, ""),
        ("insult", "Comment: insult", "ols", CONTROLS + LEN, None, ""),
        ("rate_i", "Comment: 'I' rate", "ols", CONTROLS + LEN, None, ""),
        ("rate_you", "Comment: 'you' rate", "ols", CONTROLS + LEN, None, ""),
        ("rate_we", "Comment: 'we' rate", "ols", CONTROLS + LEN, None, ""),
        ("laugh", "Comment: laughter/emoji", "logit", CONTROLS + LEN, None, ""),
        ("sim_post", "Comment: similarity to post", "ols", CONTROLS + LEN + ["sim_own_history"], None, ""),
        ("sim_thread", "Comment: similarity to prior discussion", "ols", CONTROLS + LEN + ["sim_own_history"], None, ""),
        ("score_asinh", "Reception: comment score (asinh)", "ols", CONTROLS + LEN, None, ""),
        ("reply_received", "Reception: got a reply", "logit", CONTROLS, None, ""),
        ("n_replies", "Reception: # replies", "poisson", CONTROLS, None, ""),
        ("reply_tone_insult", "Reception: insult in replies", "ols", CONTROLS + LEN, None, ""),
        ("reengaged", "Thread: commented again in thread", "logit", CONTROLS, None, ""),
        ("answered_replier", "Thread: answered a replier", "logit", CONTROLS, "reply_received == 1", ""),
    ]
    r = family("F1 entry & first comment", nc, specs)
    print("\n=== F1. Entry and first comment ===")
    show(r)
    results.append(r)

# ---------------------------------------------------------------- F2 Reddit-wide behavior after
bm = os.path.join(DATA, "behavior_metrics.parquet")
if (not ONLY or "F2" in ONLY) and os.path.exists(bm):
    M = pd.read_parquet(bm)
    d = nc.join(M, how="inner")
    # movement of overall activity towards the community's neighbourhood
    for w in ("before", "after"):
        d[f"nbr_fit_{w}"] = [c2v.fit(json.loads(s), sub, vid, E) if isinstance(s, str) else np.nan
                             for s, sub in zip(d[f"subs_{w}"], d.subreddit)]
    TF = {"n_subreddits": np.log1p, "comments_per_day": np.log1p, "posts_per_day": np.log1p}
    METRICS = ["log_words", "question", "rate_i", "rate_you", "rate_we", "rate_hedge", "exclaim", "laugh",
               "sentiment", "insult", "topic_sim_px", "post_share", "log_comment_post_ratio", "n_subreddits",
               "sub_entropy", "subs_per_10_items", "comments_per_day", "posts_per_day", "active_days", "nbr_fit"]
    specs = []
    for m in METRICS:
        b, a = f"{m}_before", f"{m}_after"
        if a not in d:
            continue
        if m in TF:
            d[a], d[b] = TF[m](d[a]), TF[m](d[b])
        specs.append((a, f"after: {m} (| before)", "ols", CONTROLS + [b], None, ""))
    d["log_new_subs"] = np.log1p(d.new_subreddits_after)
    specs += [("log_new_subs", "after: new subreddits (log)", "ols", CONTROLS + ["n_subreddits_before"], None, ""),
              ("target_sub_share_after", "after: share of activity in target subreddit", "ols", CONTROLS, None, "")]
    r = family("F2 Reddit-wide after (ANCOVA)", d, specs)
    print("\n=== F2. Reddit-wide behavior in the 90 days after (adjusted for before) ===")
    show(r)
    results.append(r)

# ---------------------------------------------------------------- F3 behavior inside the community
ins = os.path.join(DATA, "insub_items.parquet")
if (not ONLY or "F3" in ONLY) and os.path.exists(ins):
    it = pd.read_parquet(ins)
    it = it[~it.text.isin(["[deleted]", "[removed]"])]
    it = it.merge(nc[["first_utc"]], left_on="idx", right_index=True)
    it["days"] = (it.t - it.first_utc) / DAY
    it["thread_age_h"] = (it.t - it.thread_utc) / 3600
    it["viral"] = (it.thread_comments >= 1000).astype(float)
    it["ordinary"] = (it.thread_comments < 100).astype(float)
    it["is_post"] = (it.kind == "p").astype(float)
    it["is_reply"] = ((it.kind == "c") & (it.top_level == 0)).astype(float)
    it["early"] = (it.thread_age_h <= 6).astype(float)
    it["score_asinh"] = np.arcsinh(it.score.astype(float))
    it["log_len"] = np.log1p(it.text.str.split().str.len())
    it["question"] = it.text.str.contains(r"\?").astype(float)
    g = it.groupby("idx")
    A = pd.DataFrame({
        "insub_n": g.size(), "insub_threads": g.thread.nunique(), "insub_post_share": g.is_post.mean(),
        "insub_reply_share": g.is_reply.mean(), "insub_score": g.score_asinh.mean(),
        "insub_viral_share": g.viral.mean(), "insub_ordinary_share": g.ordinary.mean(),
        "insub_early_share": g.early.mean(), "insub_len": g.log_len.mean(), "insub_question": g.question.mean(),
        "insub_days_first": g.days.min(), "insub_span_days": g.days.max() - g.days.min(),
        "insub_months": g.days.apply(lambda s: len(set((s // 30).astype(int))))})
    A["insub_log_n"] = np.log1p(A.insub_n)
    d = nc.join(A, how="inner")
    print(f"\nreturners with in-community items: {len(d)}; items {len(it)}")
    specs = [("insub_log_n", "# contributions (log)", "ols", CONTROLS, None, ""),
             ("insub_threads", "# distinct threads", "poisson", CONTROLS, None, ""),
             ("insub_months", "# active 30-day blocks", "poisson", CONTROLS, None, ""),
             ("insub_days_first", "days until first return", "ols", CONTROLS, None, ""),
             ("insub_span_days", "days between first and last contribution", "ols", CONTROLS, None, ""),
             ("insub_post_share", "share that are posts (submissions)", "ols", CONTROLS, None, ""),
             ("insub_reply_share", "share that are replies to comments (conversation)", "ols", CONTROLS, None, ""),
             ("insub_score", "mean score received (asinh)", "ols", CONTROLS, None, ""),
             ("insub_viral_share", "share in threads with >= 1000 comments", "ols", CONTROLS, None, ""),
             ("insub_ordinary_share", "share in threads with < 100 comments", "ols", CONTROLS, None, ""),
             ("insub_early_share", "share within 6 h of thread creation (browsing new)", "ols", CONTROLS, None, ""),
             ("insub_len", "mean length (log words)", "ols", CONTROLS, None, ""),
             ("insub_question", "share asking a question", "ols", CONTROLS, None, "")]
    r = family("F3 inside the community (returners)", d, specs)
    print("\n=== F3. What returners do inside the community (180 days) ===")
    show(r)
    results.append(r)

# ---------------------------------------------------------------- F4 pre-entry trajectory
if (not ONLY or "F4" in ONLY) and need_emb:
    nc["fit_older"] = [window_fit((p, a), s, t, 30, 365) for p, a, s, t in zip(nc.post_id, nc.author, nc.subreddit, T0)]
    nc["approach"] = nc.fit_recent30 - nc.fit_older
    d = nc.dropna(subset=["approach"])
    print(f"\nF4: newcomers with >= 5 items in both windows: {len(d)}")
    print("mean approach (recent - older fit) by c2v group:", d.groupby("grp_c2v").approach.mean().round(4).to_dict())
    rows = []
    for y, fam in (("return_90d", "logit"), ("return_180d", "logit")):
        for x, ctrl, lab in (("approach", CONTROLS, "approach (recent - older fit)"),
                             ("approach", CONTROLS + ["fit_older"], "approach | older fit"),
                             ("fit_recent30", CONTROLS + ["fit_older"], "recent fit | older fit"),
                             ("c2v_big", CONTROLS + ["approach"], "overall fit | approach")):
            try:
                rows.append({"family": "F4 trajectory", "measure": "c2v", "outcome": y, "label": lab,
                             **fit(d, y, x, fam, ctrl)})
            except Exception as e:  # noqa: BLE001
                print("  skip", lab, y, e)
    r = pd.DataFrame(rows)
    r["fdr_p"] = multipletests(r.p, method="fdr_bh")[1]
    print(r[["outcome", "label", "n", "effect", "ci_lo", "ci_hi", "p", "fdr_p"]].round(3).to_string())
    results.append(r)

    # recency vs measurement noise: three windows entered together; if the effect were only
    # 'better measurement of a stable fit', the coefficients would track window reliability
    # (largest for the long 90-365 d window), not recency
    nc["fit_30_90"] = [window_fit((p, a), s, t, 30, 90) for p, a, s, t in zip(nc.post_id, nc.author, nc.subreddit, T0)]
    nc["fit_90_365"] = [window_fit((p, a), s, t, 90, 365) for p, a, s, t in zip(nc.post_id, nc.author, nc.subreddit, T0)]
    W3 = ["fit_recent30", "fit_30_90", "fit_90_365"]
    d3 = nc.dropna(subset=W3 + ["return_180d"] + CONTROLS).copy()
    for c in W3:
        d3[c] = (d3[c] - d3[c].mean()) / d3[c].std()
    print(f"\nF4b: three windows together (n = {len(d3)}); spearman between windows:")
    print(d3[W3].corr("spearman").round(2).to_string())
    for y in ("return_90d", "return_180d"):
        dd = d3[d3.groupby("post_id")[y].transform(lambda s: s.nunique() > 1)]
        m = smf.logit(f"{y} ~ {' + '.join(W3 + CONTROLS)} + C(post_id)", dd).fit(
            disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.post_id)[0]})
        print(y, " | ".join(f"{c}: OR {np.exp(m.params[c]):.2f} [{np.exp(m.conf_int().loc[c, 0]):.2f}, "
                            f"{np.exp(m.conf_int().loc[c, 1]):.2f}] p={m.pvalues[c]:.3f}" for c in W3))
        b = m.params[["fit_recent30", "fit_90_365"]]
        V = m.cov_params().loc[["fit_recent30", "fit_90_365"], ["fit_recent30", "fit_90_365"]]
        se = np.sqrt(V.iloc[0, 0] + V.iloc[1, 1] - 2 * V.iloc[0, 1])
        from scipy.stats import norm
        print(f"   recent - oldest: diff {b.iloc[0] - b.iloc[1]:.3f}, p = {2 * norm.sf(abs((b.iloc[0] - b.iloc[1]) / se)):.3f}")
    # items per window (reliability proxy)
    print("median # items per window: recent30 / 30-90 / 90-365 =",
          [int(np.median([sum(1 for x in json.loads(hist[(p, a)]) if lo <= (t - x['t']) / DAY < hi)
                          for p, a, t in zip(d3.post_id, d3.author, d3.post_id.map(posts.created_utc))]))
           for lo, hi in ((0, 30), (30, 90), (90, 365))])

# ---------------------------------------------------------------- F5 social ties
ti = os.path.join(DATA, "ties.parquet")
if (not ONLY or "F5" in ONLY) and os.path.exists(ti) and need_emb:
    T = pd.read_parquet(ti)
    d = nc.join(T, how="inner")
    rows = []
    for i, r in zip(d.index, d.itertuples()):
        inter = [x for x in json.loads(r.interlocutors) if x["subs"] is not None]
        tgt = r.subreddit.lower()
        ins = [sum(n for s, n in x["subs"].items() if s.lower() == tgt) for x in inter]
        fits = [c2v.fit(x["subs"], r.subreddit, vid, E) for x in inter]
        fits = [f for f in fits if not np.isnan(f)]
        rows.append({"i": i, "n_profiled": len(inter),
                     "tie_any_member": float(any(n >= 1 for n in ins)) if inter else np.nan,
                     "tie_any_regular": float(any(n >= 3 for n in ins)) if inter else np.nan,
                     "tie_share_member": float(np.mean([n >= 1 for n in ins])) if inter else np.nan,
                     "social_fit": float(np.mean(fits)) if fits else np.nan})
    d = d.join(pd.DataFrame(rows).set_index("i"))
    d["log_interlocutors"] = np.log1p(d.n_interlocutors)
    d["log_profiled"] = np.log1p(d.n_profiled)  # only part of the interlocutors could be profiled
    print(f"\nF5: newcomers with >= 1 profiled interlocutor: {(d.n_profiled > 0).sum()} of {len(d)}")
    print("by c2v group:", d.groupby("grp_c2v")[["tie_any_member", "tie_any_regular", "social_fit"]].mean().round(3).to_dict())
    print("spearman own fit vs social fit:", round(d[["c2v_big", "social_fit"]].corr("spearman").iloc[0, 1], 3))
    C5 = CONTROLS + ["log_interlocutors", "log_profiled"]
    rows = []
    for y, x, fam, ctrl, lab, where in (
            ("tie_any_member", "c2v_big", "logit", C5, "own fit -> talked to a member before", None),
            ("social_fit", "c2v_big", "ols", C5, "own fit -> interlocutors' fit", None),
            ("return_90d", "tie_any_member", "logit", C5, "talked to a member -> return 90d", None),
            ("return_90d", "tie_any_regular", "logit", C5, "talked to a regular -> return 90d", None),
            ("return_90d", "social_fit", "logit", C5, "interlocutors' fit -> return 90d", None),
            ("return_180d", "social_fit", "logit", C5, "interlocutors' fit -> return 180d", None),
            ("return_90d", "c2v_big", "logit", C5 + ["social_fit", "tie_any_member"], "own fit | social fit + ties -> return 90d", None),
            ("return_180d", "c2v_big", "logit", C5 + ["social_fit", "tie_any_member"], "own fit | social fit + ties -> return 180d", None),
            ("return_90d", "social_fit", "logit", C5 + ["c2v_big"], "social fit | own fit -> return 90d", None)):
        dd = d.query(where) if where else d
        try:
            rows.append({"family": "F5 social ties", "measure": "c2v", "outcome": y, "label": lab,
                         **fit(dd, y, x, fam, ctrl)})
        except Exception as e:  # noqa: BLE001
            print("  skip", lab, e)
    r = pd.DataFrame(rows)
    r["fdr_p"] = multipletests(r.p, method="fdr_bh")[1]
    print(r[["label", "n", "effect", "ci_lo", "ci_hi", "p", "fdr_p"]].round(3).to_string())
    results.append(r)

if results:
    tag = ("_" + ONLY.replace(",", "")) if ONLY else ""
    pd.concat(results).to_csv(os.path.join(RESULTS, f"fit_behavior{tag}.csv"), index=False)
