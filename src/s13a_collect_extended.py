"""Stage 13a: data for the extended exploratory analyses (s13b).

PART=return180  newcomers' activity in the target subreddit (outside the entry thread)
                within 180 days of the first comment -> longer-horizon retention
PART=repliers   prior activity in the target subreddit of everyone who directly replied
                to a newcomer's first comment (12 months before the post): insider vs
                outsider replies
PART=extra      subreddit activity counts (365 days before the post, uncapped, comments)
                of EXTRA_PER_POST additional commenters per target post, not in the
                stage-3 sample -> larger training set for subreddit (community2vec-style)
                embeddings
PART=all        all three (default)"""
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

from api import DATA, get

DAY = 86400
YEAR = 365 * DAY
PART = os.environ.get("PART", "all")
THREADS = int(os.environ.get("THREADS", 6))
EXTRA_PER_POST = int(os.environ.get("EXTRA_PER_POST", 40))
BOT_RE = re.compile(r"(bot|b0t)$|^bot[-_]|automod", re.I)
SKIP = {"[deleted]", "AutoModerator", None}
day = lambda t: datetime.fromtimestamp(int(t), timezone.utc).strftime("%Y-%m-%d")  # noqa: E731

df = pd.read_parquet(os.path.join(DATA, "analysis_table.parquet"))
posts = pd.read_parquet(os.path.join(DATA, "target_posts.parquet")).drop_duplicates("post_id").set_index("post_id")
tc = pd.read_parquet(os.path.join(DATA, "thread_comments.parquet")).drop_duplicates("id")
nc = df[df.newcomer_12m == 1].copy()
nc["first_utc"] = tc.set_index("id").loc[nc.first_comment_id, "created_utc"].values


def run(fn, items):
    with ThreadPoolExecutor(THREADS) as ex:
        return list(ex.map(fn, items))


# ---------------------------------------------------------------- 180-day return
def return180(r):
    t0, t1 = int(r.first_utc) + 1, int(r.first_utc) + 180 * DAY
    try:
        cs = get("comments/search", author=r.author, subreddit=r.subreddit, after=t0, before=t1,
                 sort="asc", limit=100, fields="link_id,created_utc")
        ps = get("posts/search", author=r.author, subreddit=r.subreddit, after=t0, before=t1,
                 sort="asc", limit=100, fields="id,created_utc")
    except RuntimeError as e:
        return {"idx": r.Index, "r180_error": str(e)[-120:]}
    ts = sorted([c["created_utc"] for c in cs if c["link_id"].split("_")[-1] != r.post_id]
                + [p["created_utc"] for p in ps])
    d = [(t - r.first_utc) / DAY for t in ts]
    return {"idx": r.Index, "r180_error": None,
            "return_180d": int(len(d) > 0),
            "return_n_180d": len(d),
            "active_91_180": int(any(x > 90 for x in d)),          # still there in months 4-6
            "n_active_months": len({int(x // 30) for x in d}),     # distinct 30-day blocks with activity
            "n_posts_180d": len(ps)}


if PART in ("return180", "all"):
    out = pd.DataFrame(run(return180, list(nc.itertuples()))).set_index("idx")
    out.to_parquet(os.path.join(DATA, "return180.parquet"))
    print("return180:", len(out), "errors:", out.r180_error.notna().sum(), flush=True)

# ---------------------------------------------------------------- repliers
if PART in ("repliers", "all"):
    tc["parent"] = tc.parent_id.str.split("_").str[-1]
    first_ids = dict(zip(nc.first_comment_id, nc.index))
    rep = tc[tc.parent.isin(first_ids.keys())].copy()
    rep["idx"] = rep.parent.map(first_ids)
    rep["newcomer"] = rep.idx.map(nc.author)
    rep = rep[~rep.author.isin(SKIP) & (rep.author != rep.newcomer)]
    rep["is_op"] = (rep.author == rep.post_id.map(posts.author)).astype(int)
    rep["bot"] = rep.author.map(lambda a: bool(BOT_RE.search(a))).astype(int)
    keys = rep[["author", "subreddit", "post_id"]].drop_duplicates()

    def in_sub(k):
        T = int(posts.at[k.post_id, "created_utc"])
        try:
            c = get("comments/search", author=k.author, subreddit=k.subreddit, after=T - YEAR, before=T,
                    sort="desc", limit=100, fields="created_utc")
            p = get("posts/search", author=k.author, subreddit=k.subreddit, after=T - YEAR, before=T,
                    sort="desc", limit=100, fields="created_utc")
            return (k.author, k.post_id, len(c) + len(p), None)
        except RuntimeError as e:
            return (k.author, k.post_id, None, str(e)[-120:])

    res = pd.DataFrame(run(in_sub, list(keys.itertuples())),
                       columns=["author", "post_id", "replier_in_sub", "replier_error"])
    rep = rep.merge(res, on=["author", "post_id"], how="left")
    rep[["idx", "id", "author", "post_id", "subreddit", "created_utc", "body", "score", "is_op", "bot",
         "replier_in_sub", "replier_error"]].to_parquet(os.path.join(DATA, "repliers.parquet"))
    print("repliers:", len(rep), "replies,", len(keys), "replier-post pairs; errors:",
          res.replier_error.notna().sum(), flush=True)

# ---------------------------------------------------------------- extra users for subreddit embeddings
if PART in ("extra", "all"):
    rng = random.Random(11)
    sampled = set(zip(df.post_id, df.author))
    tasks = []
    for pid, g in tc.groupby("post_id"):
        if pid not in posts.index:
            continue
        elig = sorted(a for a in g.author.unique() if a not in SKIP and a != posts.at[pid, "author"]
                      and not BOT_RE.search(a) and (pid, a) not in sampled)
        for a in rng.sample(elig, min(EXTRA_PER_POST, len(elig))):
            tasks.append((pid, int(posts.at[pid, "created_utc"]), a))
    print("extra users:", len(tasks), flush=True)

    def extra(t):
        pid, T, a = t
        try:
            rows = get("comments/search/aggregate", aggregate="subreddit", author=a,
                       after=day(T - YEAR), before=day(T), limit="")
            return {"post_id": pid, "author": a, "subs": json.dumps({r["key"]: int(r["count"]) for r in rows}),
                    "error": None}
        except RuntimeError as e:
            return {"post_id": pid, "author": a, "subs": None, "error": str(e)[-120:]}

    out = pd.DataFrame(run(extra, tasks))
    out.to_parquet(os.path.join(DATA, "extra_users.parquet"))
    print("extra:", len(out), "errors:", out.error.notna().sum(), flush=True)
