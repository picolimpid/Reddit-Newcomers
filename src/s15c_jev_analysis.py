"""Stage 15c (Jev): analyses of the s15b Jev measurements (Study 2).

A. Jev alignment as a measure (proposal Method 2 / RQ3)
   A1 discriminant validity: entry post vs placebo post (newcomers), vs embedding and c2v fit
   A2 known groups: existing members vs newcomers on the same post
   A3 convergent validity: Spearman with embedding alignment and community-embedding fit
   A4 predictive validity: first-comment outcomes and return / regular at 90 and 180 days,
      alone and net of community-embedding fit; subreddit-names-only variant for comparison
B. Social signals in the first comment: newcomers vs existing members; do they track fit;
   do they predict reception and return?
C. Replies: gatekeeping / welcoming toward newcomers vs existing members; insiders vs outsiders;
   does the reply a newcomer gets predict return?
D. Post genre and insider knowledge needed: heterogeneity of return and of the fit -> return link.

Models: post FE logit (binary), post FE OLS (continuous / group gaps), s13b controls, SEs
clustered by post, BH-FDR within each block. Run with STUDY=global."""
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from api import DATA, RESULTS

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

df = pd.read_parquet(os.path.join(DATA, "analysis_table.parquet"))
for f in ("post_alignment.parquet", "comment_features.parquet"):
    df = df.join(pd.read_parquet(os.path.join(DATA, f)))
df = df.join(pd.read_parquet(os.path.join(DATA, "return.parquet")).drop(columns="return_error"))
df = df.join(pd.read_parquet(os.path.join(DATA, "explore_features.parquet")))
df = df.join(pd.read_parquet(os.path.join(DATA, "return180.parquet")).drop(columns="r180_error"))
df = df.join(pd.read_parquet(os.path.join(DATA, "c2v_big.parquet")))

df["log_hist_n"] = np.log1p(df.n_hist)
df["log_n_subs"] = np.log1p(df.hist_n_subs)
df["log_hist_len"] = np.log(df.hist_mean_len)
df["log_hist_rate"] = np.log(df.hist_rate)
df["hist_eng"] = np.arcsinh(df.hist_median_score)
df["log_delay"] = np.log1p(df.entry_delay_min.clip(lower=0))
df["log_prior"] = np.log1p(df.prior_thread_comments)
df["log_len"] = np.log1p(df.first_len)
CONTROLS = ["log_hist_n", "log_n_subs", "log_hist_len", "log_hist_rate", "hist_eng",
            "log_delay", "top_level", "log_prior"]
df["regular_90d"] = (df.return_n_90d >= 3).astype(float).where(df.return_90d.notna())
df["regular_180d"] = (df.return_n_180d >= 3).astype(float).where(df.return_180d.notna())
df["newcomer"] = df.newcomer_12m


def load(name):
    p = os.path.join(DATA, name + ".parquet")
    return pd.read_parquet(p) if os.path.exists(p) else None


def fit(d, y, x, fam="logit", ctrl=CONTROLS, inter=None, standardize=True):
    """Effect of +1 SD x (or of a 0/1 x) on y; post FE, SEs clustered by post."""
    cols = [y, x] + ctrl + ([inter] if inter else [])
    d = d.dropna(subset=cols).copy()
    if fam == "logit":
        d = d[d.groupby("post_id")[y].transform(lambda s: s.nunique() > 1)]
    d["z"] = (d[x] - d[x].mean()) / d[x].std() if standardize and d[x].nunique() > 2 else d[x]
    ctrl = [c for c in ctrl if d[c].nunique() > 1 and c != x]
    # a post-level moderator's main effect is absorbed by the post FE
    post_level = inter and (d.groupby("post_id")[inter].nunique() <= 1).all()
    rhs = " + ".join(["z" + ((f" + z:{inter}" if post_level else f" * {inter}") if inter else "")]
                     + ctrl + ["C(post_id)"])
    kw = dict(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.post_id)[0]})
    m = (smf.logit if fam == "logit" else smf.ols)(f"{y} ~ {rhs}", d).fit(
        **({"disp": 0, "maxiter": 300} if fam == "logit" else {}), **kw)
    term = f"z:{inter}" if inter else "z"
    b, se = m.params[term], m.bse[term]
    tr = np.exp if fam == "logit" else (lambda v: v)
    return {"n": int(m.nobs), "n_posts": d.post_id.nunique(), "effect": tr(b),
            "ci_lo": tr(b - 1.96 * se), "ci_hi": tr(b + 1.96 * se), "p": m.pvalues[term]}


def block(name, tests):
    rows = []
    for label, d, y, x, fam, ctrl, inter in tests:
        try:
            rows.append({"block": name, "label": label, "outcome": y, "x": x, "model": fam,
                         **fit(d, y, x, fam, ctrl, inter)})
        except Exception as e:  # noqa: BLE001
            print("FAILED", label, y, e)
    r = pd.DataFrame(rows)
    if len(r):
        r["fdr_p"] = multipletests(r.p, method="fdr_bh")[1]
    print(f"\n=== {name} ===")
    print(r[["label", "outcome", "n", "effect", "ci_lo", "ci_hi", "p", "fdr_p"]].to_string(
        formatters={k: "{:.2f}".format for k in ("effect", "ci_lo", "ci_hi")} |
                   {k: "{:.3f}".format for k in ("p", "fdr_p")}))
    out.append(r)
    return r


def paired(a, b):
    ok = a.notna() & b.notna()
    a, b = a[ok], b[ok]
    win = ((a > b) + 0.5 * (a == b)).mean()
    d = (a - b).mean() / (a - b).std()
    return {"n": int(ok.sum()), "entry_gt_placebo": win, "d": d, "p": stats.wilcoxon(a, b).pvalue}


out, summary = [], {}
nc = df[df.newcomer == 1].copy()

# ================================================================ A. Jev alignment
JEV = ["align_post", "align_community", "familiar"]
for variant, suffix in (("full", ""), ("subs", "_subs")):
    al = load("jev_align" + suffix)
    if al is None:
        continue
    piv = al.pivot_table(index=["idx", "role"], values=JEV).unstack("role")
    piv.columns = [f"{m}{suffix}_{r}" for m, r in piv.columns]
    df = df.join(piv)
    nc = nc.join(piv)
    # A1 discriminant validity
    print(f"\n=== A1 validity, Jev {variant}: entry post vs random post from another subreddit ===")
    rows = []
    for m in JEV:
        rows.append({"measure": f"jev_{variant}:{m}", **paired(nc[f"{m}{suffix}_entry"], nc[f"{m}{suffix}_placebo"])})
    if variant == "full":
        rows += [{"measure": "embedding: text mean (MiniLM)", **paired(nc.mini_mean, nc.mini_mean_othersub)},
                 {"measure": "embedding: post-based mean", **paired(nc.pp_mini_mean, nc.pp_mini_mean_othersub)},
                 {"measure": "community-embedding fit (c2v_big)", **paired(nc.c2v_big, nc.c2v_big_placebo)}]
    v = pd.DataFrame(rows)
    print(v.round(3).to_string())
    summary[f"validity_{variant}"] = v
    # A2 known groups (existing vs newcomer, same post), full variant only
    if variant == "full":
        kg = df[((df.newcomer == 1) | (df.existing == 1))].copy()
        T = []
        for m in JEV:
            kg[m + "_z"] = (kg[m + "_entry"] - kg[m + "_entry"].mean()) / kg[m + "_entry"].std()
            T.append((f"newcomer - existing, Jev {m} (SD)", kg, m + "_z", "newcomer", "ols", [], None))
        for m in ("mini_mean", "pp_mini_mean"):
            kg[m + "_z"] = (kg[m] - kg[m].mean()) / kg[m].std()
            T.append((f"newcomer - existing, {m} (SD)", kg, m + "_z", "newcomer", "ols", [], None))
        block("A2 known groups", T)
        # A3 convergent validity
        cols = ["align_post_entry", "align_community_entry", "familiar_entry", "mini_mean", "mini_top5",
                "pp_mini_mean", "c2v_big", "struct_fit"]
        rho = nc[cols].corr("spearman").loc[cols[:3]].round(2)
        print("\n=== A3 Spearman correlations (newcomers) ===")
        print(rho.to_string())
        rho.to_csv(os.path.join(RESULTS, "jev_convergent.csv"))
    # A4 predictive validity
    T = []
    for m in JEV:
        x = f"{m}{suffix}_entry"
        for y in ("return_90d", "return_180d", "regular_180d"):
            T.append((f"Jev {variant} {m}", nc, y, x, "logit", CONTROLS, None))
            T.append((f"Jev {variant} {m}, + c2v fit", nc, y, x, "logit", CONTROLS + ["c2v_big"], None))
        if variant == "full":
            for y in ("reply_received", "reengaged"):
                T.append((f"Jev {variant} {m}", nc, y, x, "logit", CONTROLS, None))
        T.append((f"placebo: Jev {variant} {m} to random post", nc, "return_180d", f"{m}{suffix}_placebo",
                  "logit", CONTROLS, None))
    if variant == "full":
        for y in ("return_90d", "return_180d", "regular_180d"):
            T.append(("c2v fit (reference)", nc, y, "c2v_big", "logit", CONTROLS, None))
            T.append(("c2v fit, + Jev familiar + align_community", nc, y, "c2v_big", "logit",
                      CONTROLS + ["familiar_entry", "align_community_entry"], None))
    block(f"A4 predictive validity, Jev {variant}", T)

if "familiar_entry" in nc:
    for m in ("familiar_entry", "align_community_entry", "c2v_big"):
        nc["q"] = nc.groupby("post_id")[m].transform(
            lambda s: pd.qcut(s.rank(method="first"), 4, labels=False) + 1 if len(s) >= 4 else np.nan)
        q = nc.groupby("q")[["return_90d", "return_180d", "regular_180d"]].mean().round(3)
        print(f"\nwithin-post quartiles of {m}:\n{q.T.to_string()}")
        q.assign(measure=m).to_csv(os.path.join(RESULTS, f"jev_quartiles_{m}.csv"))

if "familiar_entry" in nc:
    # A5: Jev scores to a *random* post also predict return, i.e. they carry a user-level component
    # (e.g. how "viral-content-aligned" a history is). Target-specific fit = entry score net of placebo.
    T = []
    for suffix, lab in (("", "full"), ("_subs", "subs")):
        for m in JEV:
            for y in ("return_90d", "return_180d", "regular_180d"):
                T.append((f"Jev {lab} {m}, + placebo score", nc, y, f"{m}{suffix}_entry", "logit",
                          CONTROLS + [f"{m}{suffix}_placebo"], None))
                T.append((f"Jev {lab} {m}, + placebo score + c2v fit", nc, y, f"{m}{suffix}_entry", "logit",
                          CONTROLS + [f"{m}{suffix}_placebo", "c2v_big"], None))
        T.append((f"Jev {lab} align_post, + placebo score", nc, "reply_received", f"align_post{suffix}_entry",
                  "logit", CONTROLS + [f"align_post{suffix}_placebo"], None))
    for y in ("return_90d", "return_180d", "regular_180d"):
        T.append(("c2v fit, + Jev subs align_community entry & placebo", nc, y, "c2v_big", "logit",
                  CONTROLS + ["align_community_subs_entry", "align_community_subs_placebo"], None))
    block("A5 target-specific Jev alignment (net of the user's score for a random post)", T)
    # A6: ablations on the same 600 newcomers: what carries Jev's signal?
    abl = {}
    for v, suffix in (("full", ""), ("subs", "_subs"), ("text", "_text"), ("noimg", "_noimg")):
        a = load("jev_align" + suffix)
        if a is None:
            continue
        a = a[a.group == "newcomer"].pivot_table(index="idx", columns="role", values=JEV)
        abl[v] = a
    if "text" in abl:
        ids = abl["text"].index.intersection(abl["noimg"].index)
        rows = []
        for v, a in abl.items():
            for m in JEV:
                r = paired(a.loc[ids, (m, "entry")], a.loc[ids, (m, "placebo")])
                rows.append({"variant": v, "measure": m, **r,
                             "rho_with_full": stats.spearmanr(a.loc[ids, (m, "entry")],
                                                              abl["full"].loc[ids, (m, "entry")]).statistic})
        a6 = pd.DataFrame(rows)
        print("\n=== A6 ablations (same 600 newcomers): entry vs placebo discrimination ===")
        print(a6.round(3).to_string())
        a6.to_csv(os.path.join(RESULTS, "jev_ablation.csv"), index=False)

wd = load("jev_window")
if wd is not None:
    # A7: pre-entry trajectory (cf. s14c F4, which used c2v fit by window)
    w = wd.pivot_table(index="idx", columns="role", values="align_community")
    w.columns = ["jw_" + c for c in w.columns]
    nc = nc.join(w)
    full3 = nc.dropna(subset=list(w.columns))
    print(f"\n=== A7 pre-entry trajectory (Jev, subreddit names per window) ===\n"
          f"newcomers with all three windows: {len(full3)}; mean align_community 90-365 / 30-90 / 0-30 d: "
          f"{full3.jw_d90_365.mean():.2f} / {full3.jw_d30_90.mean():.2f} / {full3.jw_d0_30.mean():.2f}")
    print("mean by 180-day return:\n", full3.groupby("return_180d")[list(w.columns)].mean().round(3).to_string())
    T = []
    for y in ("return_90d", "return_180d", "regular_180d"):
        for c in ("jw_d0_30", "jw_d30_90", "jw_d90_365"):
            T.append((f"{c} alone", full3, y, c, "logit", CONTROLS, None))
            T.append((f"{c} | other windows", full3, y, c, "logit",
                      CONTROLS + [o for o in w.columns if o != c], None))
    block("A7 pre-entry trajectory: which window's community alignment predicts return?", T)

# ================================================================ B. first-comment signals
cm = load("jev_comment")
if cm is not None:
    sig = ["insider_knowledge", "outsider_signal", "member_signal", "hostile", "invites_response"]
    cm = cm.set_index("idx")
    for c in sig:
        cm[c + "_hi"] = (cm[c] > 0.5).astype(int)
    fun = pd.get_dummies(cm.function, prefix="fn").astype(int)
    cm = cm.join(fun)
    keep = [c for c in cm.columns if c.endswith("_hi") or c in sig or c.startswith("fn_")
            or c in ("stance", "p_disagree")]
    df = df.join(cm[keep].rename(columns={"stance": "jev_stance", "p_disagree": "jev_p_disagree"}))
    nc = nc.join(cm[keep].rename(columns={"stance": "jev_stance", "p_disagree": "jev_p_disagree"}))
    both = df[(df.newcomer == 1) | (df.existing == 1)]
    print("\n=== B0 first-comment signals: raw rates (share with p > .5; mean p) ===")
    rates = both.groupby("newcomer")[[c + "_hi" for c in sig] + sig].mean().T.round(3)
    rates.columns = ["existing", "newcomer"]
    print(rates.to_string())
    print("\nfunction distribution:\n",
          both.groupby("newcomer")[[c for c in both if c.startswith("fn_")]].mean().T.round(3).to_string())
    rates.to_csv(os.path.join(RESULTS, "jev_comment_rates.csv"))
    T = [(f"newcomer - existing: {c}", both, c, "newcomer", "ols", ["log_delay", "top_level", "log_prior", "log_len"], None)
         for c in sig + [c for c in both if c.startswith("fn_")]]
    block("B1 newcomer vs existing first comments (pp, post FE)", T)
    T = []
    for x in ("c2v_big", "familiar_entry", "pp_mini_mean"):
        if x in nc:
            for c in sig:
                T.append((f"{x} -> {c}", nc, c, x, "ols", CONTROLS + ["log_len"], None))
    block("B2 does fit show in first-comment signals? (newcomers; change in p per SD)", T)
    T = []
    for c in sig:
        for y in ("return_90d", "return_180d", "regular_180d"):
            T.append((f"{c} -> {y}", nc, y, c, "logit", CONTROLS + ["log_len"], None))
            T.append((f"{c} -> {y}, + c2v fit", nc, y, c, "logit", CONTROLS + ["log_len", "c2v_big"], None))
        T.append((f"{c} -> reply_received", nc, "reply_received", c, "logit", CONTROLS + ["log_len"], None))
    block("B3 do first-comment signals predict reception and return? (newcomers)", T)
    # agreement with the earlier Qwen-3B stance classifier
    s = both.dropna(subset=["jev_stance", "stance_disagree"])
    if len(s):
        from sklearn.metrics import cohen_kappa_score
        print(f"\nJev vs Qwen stance: n = {len(s)}; Jev disagree {(s.jev_stance == 'disagree').mean():.1%}, "
              f"Qwen disagree {s.stance_disagree.mean():.1%}; agreement on disagree "
              f"{((s.jev_stance == 'disagree') == (s.stance_disagree == 1)).mean():.1%}; kappa "
              f"{cohen_kappa_score(s.jev_stance == 'disagree', s.stance_disagree == 1):.3f}")

# ================================================================ C. replies
rp = load("jev_reply")
if rp is not None:
    rs = ["gatekeeping", "welcoming", "explains_context", "hostile"]
    for c in rs:
        rp[c + "_hi"] = (rp[c] > 0.5).astype(int)
    rp["disagree"] = (rp.stance == "disagree").astype(int)
    rp = rp.merge(df[["post_id", "log_delay", "top_level", "log_prior", "log_len"]], left_on="idx", right_index=True,
                  suffixes=("", "_u"))
    rp["newcomer"] = (rp.group == "newcomer").astype(int)
    print("\n=== C0 replies to first comments: share with p > .5 ===")
    rr = rp.groupby("group")[[c + "_hi" for c in rs] + ["disagree"]].mean().T.round(3)
    print(rr.to_string(), "\nn replies:", rp.group.value_counts().to_dict())
    rr.to_csv(os.path.join(RESULTS, "jev_reply_rates.csv"))
    T = [(f"reply to newcomer - to existing: {c}", rp, c, "newcomer", "ols",
          ["log_delay", "top_level", "log_prior", "log_len"], None) for c in rs + ["disagree"]]
    block("C1 how replies treat newcomers vs existing members (pp, post FE)", T)
    # replier status (newcomers' replies only)
    reps = load("repliers")
    if reps is not None:
        q = rp[rp.newcomer == 1].merge(reps[["id", "replier_in_sub", "is_op"]], left_on="reply_id", right_on="id", how="left")
        q["insider"] = (q.replier_in_sub >= 3).astype(float).where(q.replier_in_sub.notna())
        print("\nreplies to newcomers by replier status (share with p > .5):")
        print(q.groupby("insider")[[c + "_hi" for c in rs] + ["disagree"]].agg(["mean"]).round(3).T.to_string())
        T = [(f"insider replier -> {c}", q, c, "insider", "ols", [], None) for c in rs]
        block("C2 insider vs outsider repliers (replies to newcomers)", T)
    # newcomer-level: does what the reply says predict return? (replied-to newcomers)
    agg = rp[rp.newcomer == 1].groupby("idx")[rs].max().add_prefix("got_")
    nc = nc.join(agg)
    rt = nc[nc.reply_received == 1]
    T = []
    for c in rs:
        for y in ("return_90d", "return_180d"):
            T.append((f"got {c} reply -> {y} (replied-to newcomers)", rt, y, "got_" + c, "logit", CONTROLS, None))
    if "outsider_signal" in nc:
        T.append(("outsider_signal -> got gatekeeping (replied-to)", rt, "got_gatekeeping", "outsider_signal", "ols",
                  CONTROLS, None))
        T.append(("c2v fit -> got gatekeeping (replied-to)", rt, "got_gatekeeping", "c2v_big", "ols", CONTROLS, None))
        T.append(("insider_knowledge -> got welcoming (replied-to)", rt, "got_welcoming", "insider_knowledge", "ols",
                  CONTROLS, None))
    block("C3 reply content and newcomer return", T)

# ================================================================ D. post genre / insider knowledge
po = load("jev_post")
if po is not None:
    po = po.set_index("post_id")
    nc = nc.join(po[["genre", "insider_needed", "invites_discussion"]], on="post_id")
    print("\n=== D0 post genres ===")
    g = nc.groupby("genre").agg(posts=("post_id", "nunique"), newcomers=("post_id", "size"),
                                return_90d=("return_90d", "mean"), return_180d=("return_180d", "mean"),
                                insider_needed=("insider_needed", "mean")).round(3).sort_values("posts", ascending=False)
    print(g.to_string())
    g.to_csv(os.path.join(RESULTS, "jev_post_genres.csv"))
    sh = df[(df.newcomer == 1) | (df.existing == 1)].groupby("post_id").newcomer.mean().rename("nc_share")
    pp = po.join(sh)
    print("\nposts: Spearman(insider_needed, newcomer share) =",
          round(pp[["insider_needed", "nc_share"]].corr("spearman").iloc[0, 1], 3),
          "| genre-mean newcomer share:", pp.groupby("genre").nc_share.mean().round(2).to_dict())
    nc["insider_post"] = (nc.insider_needed >= nc.insider_needed.median()).astype(int)
    nc["discussion_post"] = (nc.invites_discussion > 0.5).astype(int)
    T = [("c2v fit x insider-knowledge post", nc, "return_180d", "c2v_big", "logit", CONTROLS, "insider_post"),
         ("Jev familiar x insider-knowledge post", nc, "return_180d", "familiar_entry", "logit", CONTROLS, "insider_post"),
         ("c2v fit x discussion post", nc, "return_180d", "c2v_big", "logit", CONTROLS, "discussion_post")]
    block("D1 does the fit -> return link differ by post type? (ratio of ORs)", T)
    # return base rate by post type, between posts (subreddit-clustered logit, no post FE)
    m = smf.logit("return_180d ~ insider_needed + invites_discussion + " + " + ".join(CONTROLS), nc.dropna(
        subset=["return_180d", "insider_needed"] + CONTROLS)).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(nc.dropna(
            subset=["return_180d", "insider_needed"] + CONTROLS).subreddit)[0]})
    print("\nbetween posts: return_180d ~ insider_needed + invites_discussion (OR per unit):",
          np.exp(m.params[["insider_needed", "invites_discussion"]]).round(2).to_dict(),
          "p:", m.pvalues[["insider_needed", "invites_discussion"]].round(3).to_dict())

res = pd.concat(out, ignore_index=True)
res.to_csv(os.path.join(RESULTS, "jev_models.csv"), index=False)
if summary:
    pd.concat(summary, names=["variant"]).to_csv(os.path.join(RESULTS, "jev_validity.csv"))
nc.to_parquet(os.path.join(DATA, "jev_newcomers.parquet"))
print("\nsaved jev_models.csv, jev_validity.csv")
