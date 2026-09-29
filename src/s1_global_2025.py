"""Stage 1 (global study): target posts = Reddit's site-wide "Top Posts of 2025"
list (reddit.com/posts/2025/global/), saved manually from a browser because the
page is not available to scripts. Post metadata is then looked up in Arctic Shift.

Run with STUDY=global."""
import os
import re

import pandas as pd

from api import DATA, ROOT, get

html = open(os.path.join(ROOT, "data", "Top Posts of 2025.html"), encoding="utf-8").read()
ids = list(dict.fromkeys(re.findall(r"/r/[A-Za-z0-9_]+/comments/([a-z0-9]+)", html)))
print(len(ids), "post ids on the saved page")

found = get("posts/ids", ids=",".join(ids))
posts = pd.DataFrame([{
    "post_id": p["id"], "subreddit": p["subreddit"], "author": p.get("author"),
    "created_utc": p["created_utc"], "num_comments": p.get("num_comments", 0),
    "score": p.get("score"), "title": p.get("title", ""),
    "selftext": p.get("selftext", "") or "", "url": p.get("url"),
    "over_18": p.get("over_18"), "is_video": p.get("is_video"),
} for p in found])
print(len(posts), "found in archive;", posts.subreddit.nunique(), "subreddits;",
      "median comments", posts.num_comments.median())
posts.to_parquet(os.path.join(DATA, "target_posts.parquet"))
