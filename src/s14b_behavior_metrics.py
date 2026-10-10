"""Stage 14b: Reddit-wide behavior metrics in the 90 days BEFORE and AFTER the interaction
for ALL newcomers (s11b computed them only for the top/bottom quartile on text similarity).
Metric definitions are identical to s11b (language, interactivity, diversity, activity), so
any similarity measure can be compared on them in s14c, continuously or by group.

Text scores (sentiment, insult, MiniLM embedding) are cached per text hash; the cache is
seeded from s11b's saved arrays so previously scored comments are not scored again."""
import hashlib
import json
import os
import re
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from api import DATA, get

warnings.filterwarnings("ignore")
DAY, WIN, MIN_ITEMS = 86400, 90, 5
BAD = {"", "[deleted]", "[removed]"}

df = pd.read_parquet(os.path.join(DATA, "analysis_table.parquet"))
for f in ("post_alignment.parquet", "activity.parquet", "after_activity.parquet"):
    df = df.join(pd.read_parquet(os.path.join(DATA, f)))
users = pd.read_parquet(os.path.join(DATA, "users.parquet"), columns=["post_id", "author", "history"])
users = users.drop_duplicates(["post_id", "author"])
posts = pd.read_parquet(os.path.join(DATA, "target_posts.parquet")).drop_duplicates("post_id").set_index("post_id")
tc = pd.read_parquet(os.path.join(DATA, "thread_comments.parquet"),
                     columns=["id", "created_utc"]).drop_duplicates("id").set_index("id")

nc = df[(df.newcomer_12m == 1) & df.after_error.isna() & df.act_error.isna()].copy()
nc["first_utc"] = tc.loc[nc.first_comment_id, "created_utc"].values
nc = nc.reset_index().merge(users, on=["post_id", "author"], how="left").set_index("index")
print("newcomers:", len(nc), flush=True)


def windows(r):
    before = [h for h in json.loads(r.history) if r.first_utc - WIN * DAY <= h["t"] < r.first_utc]
    after = [h for h in json.loads(r.after_items) if h["thread"] != r.post_id]
    return before, after


def comment_texts(rows):
    texts, owner = [], []
    for i, r in zip(rows.index, rows.itertuples()):
        for w, items in zip(("before", "after"), windows(r)):
            for h in items:
                if h["kind"] == "c" and h["text"].strip() not in BAD:
                    texts.append(h["text"][:1000])
                    owner.append((i, w))
    return texts, owner


texts, owner = comment_texts(nc)
H = [hashlib.sha1(t.encode("utf-8", "surrogatepass")).hexdigest() for t in texts]
print(len(texts), "comments;", len(set(H)), "unique", flush=True)

# ---------------------------------------------------------------- score cache (seeded from s11b)
CACHE = os.path.join(DATA, "s14_textscores.parquet")
EMB = os.path.join(DATA, "s14_textemb.npy")
if os.path.exists(CACHE):
    cache = pd.read_parquet(CACHE)
    cemb = np.load(EMB)
else:
    cache, cemb = pd.DataFrame(columns=["h", "sentiment", "insult"]), np.zeros((0, 384), np.float32)
    # rebuild s11b's text list (top/bottom quartile within post on pp_mini_mean) to reuse its scores
    old = nc[nc.pp_mini_mean.notna()].copy()
    size = old.groupby("post_id").pp_mini_mean.transform("size")
    pct = (old.groupby("post_id").pp_mini_mean.rank(method="first") - 0.5) / size
    old = old[(size >= 4) & ((pct >= 0.75) | (pct <= 0.25))]
    old_texts, _ = comment_texts(old)
    paths = [os.path.join(DATA, f"s11_{k}_{len(old_texts)}.npy") for k in ("sentiment", "insult", "embed")]
    if all(os.path.exists(p) for p in paths):
        s, ins, e = (np.load(p) for p in paths)
        oh = [hashlib.sha1(t.encode("utf-8", "surrogatepass")).hexdigest() for t in old_texts]
        seed = pd.DataFrame({"h": oh, "sentiment": s, "insult": ins, "k": range(len(oh))}).drop_duplicates("h")
        cache, cemb = seed[["h", "sentiment", "insult"]].reset_index(drop=True), e[seed.k.values].astype(np.float32)
        print("seeded score cache from s11b:", len(cache), flush=True)
    else:
        print("s11b arrays not found for", len(old_texts), "texts; scoring from scratch", flush=True)

have = set(cache.h)
todo = sorted({h: t for h, t in zip(H, texts) if h not in have}.items())
print("to score:", len(todo), flush=True)
if todo:
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline
    tt = [t for _, t in todo]
    sent = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0,
                    top_k=None, truncation=True, max_length=128)
    sv = np.array([{d["label"]: d["score"] for d in r}["positive"] - {d["label"]: d["score"] for d in r}["negative"]
                   for r in sent(tt, batch_size=256)])
    print("  sentiment scored", flush=True)
    tox = pipeline("text-classification", model="unitary/unbiased-toxic-roberta", device=0, top_k=None,
                   function_to_apply="sigmoid", truncation=True, max_length=128)
    iv = np.array([{d["label"]: d["score"] for d in r}["insult"] for r in tox(tt, batch_size=256)])
    print("  insult scored", flush=True)
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    ev = emb.encode(tt, batch_size=256, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    cache = pd.concat([cache, pd.DataFrame({"h": [h for h, _ in todo], "sentiment": sv, "insult": iv})],
                      ignore_index=True)
    cemb = np.vstack([cemb, ev])
    cache.to_parquet(CACHE)
    np.save(EMB, cemb)
pos = dict(zip(cache.h, range(len(cache))))
k = np.array([pos[h] for h in H])

# ---------------------------------------------------------------- language per user-window
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


LANG = ["log_words", "question", "rate_i", "rate_you", "rate_we", "rate_hedge", "exclaim", "laugh"]
lang = pd.DataFrame([style(t) for t in texts], columns=LANG)
lang["sentiment"], lang["insult"] = cache.sentiment.values[k].astype(float), cache.insult.values[k].astype(float)
lang["i"], lang["w"] = [o[0] for o in owner], [o[1] for o in owner]
lang_n = lang.groupby(["i", "w"]).size()
lang_m = lang.groupby(["i", "w"]).mean()

from sentence_transformers import SentenceTransformer  # noqa: E402
emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
Px = emb.encode([(posts.at[p, "title"] + "\n" + (posts.at[p, "selftext"] or "")[:1000]).strip()
                 for p in posts.index], normalize_embeddings=True)
pidx = {p: j for j, p in enumerate(posts.index)}
topic = {}
for (i, w), g in pd.DataFrame({"i": lang.i, "w": lang.w, "k": k}).groupby(["i", "w"]):
    m = cemb[g.k.values].mean(0)
    topic[(i, w)] = float(m @ Px[pidx[nc.at[i, "post_id"]]] / np.linalg.norm(m))


# ---------------------------------------------------------------- per-user metrics (as s11b)
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


rows = []
for i, r in zip(nc.index, nc.itertuples()):
    rec = {"i": i}
    try:
        c = daily_counts("comments/search/aggregate", r.author, r.first_utc)
        p = daily_counts("posts/search/aggregate", r.author, r.first_utc)
    except RuntimeError:
        continue
    for w, sel in (("before", lambda t: r.first_utc - WIN * DAY <= t and t + DAY <= r.first_utc),
                   ("after", lambda t: r.first_utc < t <= r.first_utc + WIN * DAY)):
        nc_, np_ = sum(n for t, n in c.items() if sel(t)), sum(n for t, n in p.items() if sel(t))
        act = {t for s in (c, p) for t, n in s.items() if n > 0 and sel(t)}
        rec[f"comments_per_day_{w}"] = nc_ / WIN
        rec[f"posts_per_day_{w}"] = np_ / WIN
        rec[f"active_days_{w}"] = len(act)
        rec[f"post_share_{w}"] = np_ / (nc_ + np_) if nc_ + np_ else np.nan
        rec[f"log_comment_post_ratio_{w}"] = np.log((nc_ + 1) / (np_ + 1))
        subs = {}
        for kk in ("comment", "post"):
            for s, n in json.loads(getattr(r, f"{kk}_{w}")).items():
                subs[s] = subs.get(s, 0) + n
        rec[f"n_subreddits_{w}"] = len(subs)
        rec[f"sub_entropy_{w}"] = entropy(subs)
        rec[f"subs_per_10_items_{w}"] = len(subs) / max(sum(subs.values()), 1) * 10
        rec[f"subs_{w}"] = json.dumps(subs)
        if w == "before":
            before_subs = set(subs)
        else:
            rec["new_subreddits_after"] = len(set(subs) - before_subs)
            tot = sum(subs.values())
            rec["target_sub_share_after"] = subs.get(r.subreddit, 0) / tot if tot else np.nan
        if lang_n.get((i, w), 0) >= MIN_ITEMS:
            for m in LANG + ["sentiment", "insult"]:
                rec[f"{m}_{w}"] = lang_m.at[(i, w), m]
            rec[f"topic_sim_px_{w}"] = topic[(i, w)]
    rows.append(rec)
M = pd.DataFrame(rows).set_index("i")
M.to_parquet(os.path.join(DATA, "behavior_metrics.parquet"))
print("metrics:", M.shape, flush=True)
