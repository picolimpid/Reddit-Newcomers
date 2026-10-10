"""Stage 11b: high- vs low-similarity newcomers, behavior 90 days BEFORE vs AFTER the
interaction (professor's list): language, interactivity, interaction diversity, activity.

Groups: within-post top vs bottom 25% on post-based similarity (as s10b).
Each metric is measured in both windows; the main estimate is an ANCOVA with post FE:
    after_m ~ high + before_m + pre-interaction controls + C(post_id)
plus the difference-in-differences of the raw group means. BH-FDR across metrics."""
import json
import os
import re
import warnings
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sentence_transformers import SentenceTransformer
from statsmodels.stats.multitest import multipletests
from transformers import pipeline

from api import DATA, RESULTS, get

warnings.filterwarnings("ignore")
DAY, WIN, MIN_ITEMS = 86400, 90, 5

df = pd.read_parquet(os.path.join(DATA, "analysis_table.parquet"))
for f in ("post_alignment.parquet", "activity.parquet", "after_activity.parquet"):
    df = df.join(pd.read_parquet(os.path.join(DATA, f)))
users = pd.read_parquet(os.path.join(DATA, "users.parquet"), columns=["post_id", "author", "history"])
posts = pd.read_parquet(os.path.join(DATA, "target_posts.parquet")).set_index("post_id")
tc = pd.read_parquet(os.path.join(DATA, "thread_comments.parquet"),
                     columns=["id", "created_utc"]).drop_duplicates("id").set_index("id")

nc = df[(df.newcomer_12m == 1) & df.pp_mini_mean.notna() & df.after_error.isna() & df.act_error.isna()].copy()
nc["first_utc"] = tc.loc[nc.first_comment_id, "created_utc"].values
nc = nc.reset_index().merge(users, on=["post_id", "author"], how="left").set_index("index")

# ---------------------------------------------------------------- groups (identical to s10b)
size = nc.groupby("post_id").pp_mini_mean.transform("size")
pct = (nc.groupby("post_id").pp_mini_mean.rank(method="first") - 0.5) / size
nc["grp"] = np.where(size < 4, None, np.where(pct >= 0.75, "high", np.where(pct <= 0.25, "low", None)))
nc = nc[nc.grp.notna()].copy()
print("high / low:", (nc.grp == "high").sum(), "/", (nc.grp == "low").sum(), flush=True)

# ---------------------------------------------------------------- text samples per window
def windows(r):
    before = [h for h in json.loads(r.history) if r.first_utc - WIN * DAY <= h["t"] < r.first_utc]
    after = [h for h in json.loads(r.after_items) if h["thread"] != r.post_id]  # drop the entry thread itself
    return before, after


W = {i: windows(r) for i, r in zip(nc.index, nc.itertuples())}
BAD = {"", "[deleted]", "[removed]"}
texts, owner = [], []  # comments only, for language measures
for i, (b, a) in W.items():
    for w, items in (("before", b), ("after", a)):
        for h in items:
            if h["kind"] == "c" and h["text"].strip() not in BAD:
                texts.append(h["text"][:1000])
                owner.append((i, w))
print(len(texts), "comments to score", flush=True)

WORD = re.compile(r"[A-Za-z']+")
QSTART = re.compile(r"^\s*(who|what|when|where|why|how|is|are|was|were|do|does|did|can|could|would|"
                    r"should|will|has|have|anyone|any)\b", re.I)
HEDGE = {"maybe", "perhaps", "probably", "possibly", "might", "seems", "guess", "think", "suppose",
         "apparently", "kinda", "sorta"}
LAUGH = re.compile(r"\b(lol|lmao|lmfao|rofl|haha\w*|hehe\w*)\b|[\U0001F600-\U0001F64F\U0001F923]", re.I)


def style(t):
    w = [x.lower() for x in WORD.findall(t)]
    n = max(len(w), 1)
    return [np.log1p(len(w)),
            float("?" in t or any(QSTART.match(s) for s in re.split(r"[.!\n]+", t) if s.strip())),
            sum(x in ("i", "i'm", "i've", "i'd", "i'll", "me", "my", "mine") for x in w) / n,
            sum(x in ("you", "you're", "your", "yours", "you've", "you'd", "u", "ur") for x in w) / n,
            sum(x in ("we", "we're", "our", "ours", "us", "we've") for x in w) / n,
            sum(x in HEDGE for x in w) / n, float("!" in t), float(bool(LAUGH.search(t)))]


def cached_scores(name, fn):
    """Score texts once and keep the result on disk so an interrupted run resumes."""
    path = os.path.join(DATA, f"s11_{name}_{len(texts)}.npy")
    if os.path.exists(path):
        return np.load(path)
    arr = fn()
    np.save(path, arr)
    print(f"  {name} scored", flush=True)
    return arr


def run_sentiment():
    sent = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0,
                    top_k=None, truncation=True, max_length=128)
    return np.array([{d["label"]: d["score"] for d in r}["positive"] - {d["label"]: d["score"] for d in r}["negative"]
                     for r in sent(texts, batch_size=256)])


def run_insult():
    tox = pipeline("text-classification", model="unitary/unbiased-toxic-roberta", device=0, top_k=None,
                   function_to_apply="sigmoid", truncation=True, max_length=128)
    return np.array([{d["label"]: d["score"] for d in r}["insult"] for r in tox(texts, batch_size=256)])


S = cached_scores("style", lambda: np.array([style(t) for t in texts]))
sentiment = cached_scores("sentiment", run_sentiment)
insult = cached_scores("insult", run_insult)
emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
E = cached_scores("embed", lambda: emb.encode(texts, batch_size=256, normalize_embeddings=True,
                                               show_progress_bar=False))
Px = emb.encode([(posts.at[p, "title"] + "\n" + (posts.at[p, "selftext"] or "")[:1000]).strip()
                 for p in posts.index], normalize_embeddings=True)
pidx = {p: k for k, p in enumerate(posts.index)}
print("scored", flush=True)

LANG = ["log_words", "question", "rate_i", "rate_you", "rate_we", "rate_hedge", "exclaim", "laugh"]
lang = pd.DataFrame(S, columns=LANG)
lang["sentiment"], lang["insult"] = sentiment, insult
lang["i"], lang["w"] = [o[0] for o in owner], [o[1] for o in owner]
lang_n = lang.groupby(["i", "w"]).size()
lang_m = lang.groupby(["i", "w"]).mean()
# topical similarity of the user's comments to the entry post Px
topic = {}
for (i, w), g in pd.DataFrame({"i": lang.i, "w": lang.w, "k": range(len(lang))}).groupby(["i", "w"]):
    m = E[g.k.values].mean(0)
    topic[(i, w)] = float(m @ Px[pidx[nc.at[i, "post_id"]]] / np.linalg.norm(m))

# ---------------------------------------------------------------- per-user metrics in both windows
def entropy(counts):
    v = np.array([c for c in counts.values() if c > 0], float)
    if v.sum() == 0:
        return np.nan
    p = v / v.sum()
    return float(-(p * np.log(p)).sum())


def daily_counts(endpoint, author, t0):  # same query as s10a -> cache hit
    d = lambda t: datetime.fromtimestamp(int(t), timezone.utc).strftime("%Y-%m-%d")  # noqa: E731
    rows = get(endpoint, aggregate="created_utc", frequency="day", author=author,
               after=d(t0 - (WIN + 1) * DAY), before=d(t0 + (WIN + 2) * DAY))
    return {int(datetime.fromisoformat(r["created_utc"].replace("Z", "+00:00")).timestamp()): int(r["count"])
            for r in rows}


rows, weekly = [], []
for i, r in zip(nc.index, nc.itertuples()):
    rec = {"i": i}
    c = daily_counts("comments/search/aggregate", r.author, r.first_utc)
    p = daily_counts("posts/search/aggregate", r.author, r.first_utc)
    for w, sel in (("before", lambda t: r.first_utc - WIN * DAY <= t and t + DAY <= r.first_utc),
                   ("after", lambda t: r.first_utc < t <= r.first_utc + WIN * DAY)):
        nc_, np_ = sum(n for t, n in c.items() if sel(t)), sum(n for t, n in p.items() if sel(t))
        act = {t for s in (c, p) for t, n in s.items() if n > 0 and sel(t)}
        # Activity: frequency of posting
        rec[f"comments_per_day_{w}"] = nc_ / WIN
        rec[f"posts_per_day_{w}"] = np_ / WIN
        rec[f"active_days_{w}"] = len(act)
        # Interactivity: comments vs posts
        rec[f"post_share_{w}"] = np_ / (nc_ + np_) if nc_ + np_ else np.nan
        rec[f"log_comment_post_ratio_{w}"] = np.log((nc_ + 1) / (np_ + 1))
        # Interaction diversity: subreddits engaged
        subs = {}
        for k in ("comment", "post"):
            for s, n in json.loads(getattr(r, f"{k}_{w}")).items():
                subs[s] = subs.get(s, 0) + n
        rec[f"n_subreddits_{w}"] = len(subs)
        rec[f"sub_entropy_{w}"] = entropy(subs)
        rec[f"subs_per_10_items_{w}"] = len(subs) / max(sum(subs.values()), 1) * 10
        if w == "before":
            before_subs = set(subs)
        else:
            rec["new_subreddits_after"] = len(set(subs) - before_subs)
            tot = sum(subs.values())
            rec["target_sub_share_after"] = subs.get(r.subreddit, 0) / tot if tot else np.nan
        # Language
        if lang_n.get((i, w), 0) >= MIN_ITEMS:
            for m in LANG + ["sentiment", "insult"]:
                rec[f"{m}_{w}"] = lang_m.at[(i, w), m]
            rec[f"topic_sim_px_{w}"] = topic[(i, w)]
    for t, n in list(c.items()) + list(p.items()):
        wk = int(np.floor((t - r.first_utc) / (7 * DAY)))
        if -13 <= wk <= 12:
            weekly.append((i, r.grp, wk, n))
    rows.append(rec)
M = pd.DataFrame(rows).set_index("i")
nc = nc.join(M)

# ---------------------------------------------------------------- comparisons
nc["log_hist_n"] = np.log1p(nc.n_hist)
nc["log_n_subs"] = np.log1p(nc.hist_n_subs)
nc["log_hist_len"] = np.log(nc.hist_mean_len)
nc["log_hist_rate"] = np.log(nc.hist_rate)
nc["hist_eng"] = np.arcsinh(nc.hist_median_score)
nc["log_delay"] = np.log1p(nc.entry_delay_min.clip(lower=0))
nc["log_prior"] = np.log1p(nc.prior_thread_comments)
CONTROLS = ["log_hist_n", "log_n_subs", "log_hist_len", "log_hist_rate", "hist_eng", "log_delay",
            "top_level", "log_prior"]
nc["high"] = (nc.grp == "high").astype(int)

METRICS = [  # (family, metric, label, transform)
    ("Language", "log_words", "Comment length, log words", None),
    ("Language", "question", "Share of comments asking a question", None),
    ("Language", "rate_i", "First-person 'I' rate", None),
    ("Language", "rate_you", "Second-person 'you' rate", None),
    ("Language", "rate_we", "'We' rate", None),
    ("Language", "rate_hedge", "Hedging rate", None),
    ("Language", "exclaim", "Share with exclamation", None),
    ("Language", "laugh", "Share with laughter/emoji", None),
    ("Language", "sentiment", "Sentiment (pos - neg)", None),
    ("Language", "insult", "Insult probability", None),
    ("Language", "topic_sim_px", "Topical similarity of comments to Px", None),
    ("Interactivity", "post_share", "Posts / (comments + posts)", None),
    ("Interactivity", "log_comment_post_ratio", "log((comments+1)/(posts+1))", None),
    ("Diversity", "n_subreddits", "Distinct subreddits engaged (log)", np.log1p),
    ("Diversity", "sub_entropy", "Subreddit entropy", None),
    ("Diversity", "subs_per_10_items", "Subreddits per 10 contributions", None),
    ("Activity", "comments_per_day", "Comments per day (log)", np.log1p),
    ("Activity", "posts_per_day", "Posts per day (log)", np.log1p),
    ("Activity", "active_days", "Active days in 90", None),
]
out = []
for fam, m, label, tf in METRICS:
    b, a = f"{m}_before", f"{m}_after"
    d = nc.dropna(subset=[b, a]).copy()
    if tf is not None:
        d[b], d[a] = tf(d[b]), tf(d[a])
    g = d.groupby("grp")[[b, a]].mean()
    sd = d[a].std()
    f = f"Q('{a}') ~ high + Q('{b}') + {' + '.join(CONTROLS)} + C(post_id)"
    mod = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.post_id)[0]})
    out.append({"family": fam, "metric": m, "label": label, "n": len(d),
                "high_before": g.at["high", b], "high_after": g.at["high", a],
                "low_before": g.at["low", b], "low_after": g.at["low", a],
                "did_raw": (g.at["high", a] - g.at["high", b]) - (g.at["low", a] - g.at["low", b]),
                "ancova_diff": mod.params["high"], "ci_lo": mod.conf_int().loc["high", 0],
                "ci_hi": mod.conf_int().loc["high", 1], "diff_in_sd": mod.params["high"] / sd,
                "p": mod.pvalues["high"]})
# after-only outcomes (no 'before' counterpart for newcomers)
for m, label in (("new_subreddits_after", "New subreddits after (not engaged before), log"),
                 ("target_sub_share_after", "Share of after-activity in the target subreddit")):
    d = nc.dropna(subset=[m]).copy()
    if m.startswith("new"):
        d[m] = np.log1p(d[m])
    mod = smf.ols(f"{m} ~ high + {' + '.join(CONTROLS)} + C(post_id)", d).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.post_id)[0]})
    g = d.groupby("grp")[m].mean()
    out.append({"family": "Diversity", "metric": m, "label": label, "n": len(d),
                "high_after": g["high"], "low_after": g["low"], "ancova_diff": mod.params["high"],
                "ci_lo": mod.conf_int().loc["high", 0], "ci_hi": mod.conf_int().loc["high", 1],
                "diff_in_sd": mod.params["high"] / d[m].std(), "p": mod.pvalues["high"]})
res = pd.DataFrame(out)
res["p_fdr"] = multipletests(res.p, method="fdr_bh")[1]
res.to_csv(os.path.join(RESULTS, "after_behavior_high_low.csv"), index=False)

# ---------------------------------------------------------------- figures
BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})
# (1) weekly activity trajectory
wk = pd.DataFrame(weekly, columns=["i", "grp", "week", "n"]).groupby(["i", "grp", "week"]).n.sum().unstack(fill_value=0)
wk = wk.reindex(columns=range(-13, 13), fill_value=0).stack().rename("n").reset_index()
fig, ax = plt.subplots(figsize=(7.5, 3.4))
for g, col in (("high", BLUE), ("low", ORANGE)):
    s = wk[wk.grp == g].groupby("week").n
    mu, se = s.mean(), s.sem()
    ax.plot(mu.index, mu.values, color=col, lw=2, marker="o", ms=4, label=f"{g} similarity")
    ax.fill_between(mu.index, mu - 1.96 * se, mu + 1.96 * se, color=col, alpha=0.12, lw=0)
ax.axvline(-0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
ax.text(-0.3, ax.get_ylim()[1] * 0.97, "interaction with Px", color=MUTED, fontsize=8.5, va="top")
ax.set_xlabel("Weeks relative to first comment on Px", color=MUTED)
ax.set_ylabel("Mean Reddit comments + posts per week", color=MUTED)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "fig6_weekly_activity.png"), dpi=160)

# (2) before -> after for key metrics
KEY = [("n_subreddits", "Distinct subreddits", None), ("sub_entropy", "Subreddit entropy", None),
       ("post_share", "Posts / all contributions", None), ("comments_per_day", "Comments per day", None),
       ("sentiment", "Comment sentiment", None), ("topic_sim_px", "Topical similarity to Px", None)]
fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
for ax, (m, label, _) in zip(axes.flat, KEY):
    for k, (g, col) in enumerate((("high", BLUE), ("low", ORANGE))):
        d = nc[nc.grp == g].dropna(subset=[f"{m}_before", f"{m}_after"])
        mu = [d[f"{m}_before"].mean(), d[f"{m}_after"].mean()]
        se = [d[f"{m}_before"].sem(), d[f"{m}_after"].sem()]
        x = np.array([0, 1]) + (k - 0.5) * 0.1
        ax.errorbar(x, mu, yerr=1.96 * np.array(se), color=col, lw=2, marker="o", ms=6, mec=SURF, mew=1.5,
                    capsize=0, label=f"{g} similarity")
    r = res.set_index("metric").loc[m]
    ax.set_title(f"{label}\nadj. diff after = {r.diff_in_sd:+.2f} SD, p = {r.p:.3f}", loc="left", fontsize=9,
                 color=INK)
    ax.set_xticks([0, 1], ["90 days before", "90 days after"])
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
axes[0, 0].legend(frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "fig7_before_after_metrics.png"), dpi=160)

pd.set_option("display.width", 250)
cols = ["family", "metric", "n", "high_before", "high_after", "low_before", "low_after", "did_raw",
        "ancova_diff", "ci_lo", "ci_hi", "diff_in_sd", "p", "p_fdr"]
print(res[cols].round(4).to_string())
