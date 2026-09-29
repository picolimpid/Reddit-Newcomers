"""Stage 1: enumerate all posts in each target subreddit-month and select
high-visibility posts (>= 95th percentile of num_comments)."""
import os
import random
from datetime import datetime, timezone

import pandas as pd

from api import DATA, paginate

SUBREDDITS = ["technology", "science", "movies", "gaming"]
MONTH_START = int(datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp())
MONTH_END = int(datetime(2026, 4, 1, tzinfo=timezone.utc).timestamp())
PCTL = 0.95
POSTS_PER_SUB = 20
SEED = 42

rows = []
for sub in SUBREDDITS:
    posts = paginate("posts/search", MONTH_START, MONTH_END, subreddit=sub)
    for p in posts:
        rows.append({
            "post_id": p["id"], "subreddit": sub, "author": p.get("author"),
            "created_utc": p["created_utc"], "num_comments": p.get("num_comments", 0),
            "score": p.get("score"), "title": p.get("title", ""),
            "selftext": p.get("selftext", "") or "", "url": p.get("url"),
            "over_18": p.get("over_18"), "removed_by_category": p.get("removed_by_category"),
        })
    print(sub, len(posts), "posts")

allp = pd.DataFrame(rows)
allp.to_parquet(os.path.join(DATA, "all_posts.parquet"))

sel = []
rng = random.Random(SEED)
for sub, g in allp.groupby("subreddit"):
    thr = g["num_comments"].quantile(PCTL)
    top = g[(g["num_comments"] >= thr) & g["removed_by_category"].isna() & (g["over_18"] != True)]  # noqa: E712
    ids = sorted(top["post_id"])
    pick = rng.sample(ids, min(POSTS_PER_SUB, len(ids)))
    s = g[g["post_id"].isin(pick)].copy()
    s["p95_threshold"] = thr
    sel.append(s)
    print(f"{sub}: n={len(g)} p95={thr:.0f} eligible={len(top)} picked={len(s)} "
          f"median_comments_picked={s['num_comments'].median():.0f}")

pd.concat(sel).to_parquet(os.path.join(DATA, "target_posts.parquet"))
