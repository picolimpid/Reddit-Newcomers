"""Stage 3: for sampled commenters on each target post, retrieve
 (a) prior activity inside the target subreddit (newcomer status), and
 (b) prior activity across Reddit in the 365 days before the post (history)."""
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from api import DATA, get

USERS_PER_POST = int(os.environ.get("USERS_PER_POST", 40))
SEED = 7
YEAR = 365 * 86400
BOT_RE = re.compile(r"(bot|b0t)$|^bot[-_]|automod", re.I)
SKIP = {"[deleted]", "AutoModerator", None}

posts = pd.read_parquet(os.path.join(DATA, "target_posts.parquet"))
comments = pd.read_parquet(os.path.join(DATA, "thread_comments.parquet"))

rng = random.Random(SEED)
tasks = []
for p in posts.itertuples():
    authors = comments.loc[comments.post_id == p.post_id, "author"].unique()
    elig = sorted(a for a in authors if a not in SKIP and a != p.author and not BOT_RE.search(a))
    for a in rng.sample(elig, min(USERS_PER_POST, len(elig))):
        tasks.append((p.post_id, p.subreddit, int(p.created_utc), a))
print(len(tasks), "user-post pairs")


def collect(t):
    post_id, sub, T, user = t
    rec = {"post_id": post_id, "subreddit": sub, "author": user, "error": None}
    try:
        in_c = get("comments/search", author=user, subreddit=sub, after=T - YEAR, before=T,
                   sort="desc", limit=3, fields="id,created_utc")
        in_p = get("posts/search", author=user, subreddit=sub, after=T - YEAR, before=T,
                   sort="desc", limit=3, fields="id,created_utc")
        hc = get("comments/search", author=user, after=T - YEAR, before=T, sort="desc",
                 limit=100, fields="id,body,subreddit,created_utc,score")
        hp = get("posts/search", author=user, after=T - YEAR, before=T, sort="desc",
                 limit=100, fields="id,title,selftext,subreddit,created_utc,score")
    except Exception as e:  # noqa: BLE001
        rec["error"] = str(e)[:300]
        return rec
    ins = [x["created_utc"] for x in in_c + in_p]
    rec["in_sub_count_capped"] = len(ins)  # up to 3 comments + 3 posts
    rec["last_in_sub_utc"] = max(ins) if ins else None
    rec["history"] = json.dumps(
        [{"kind": "c", "text": x.get("body") or "", "sub": x["subreddit"],
          "t": x["created_utc"], "score": x.get("score")} for x in hc]
        + [{"kind": "p", "text": ((x.get("title") or "") + "\n" + (x.get("selftext") or "")).strip(),
            "sub": x["subreddit"], "t": x["created_utc"], "score": x.get("score")} for x in hp])
    return rec


out = []
with ThreadPoolExecutor(int(os.environ.get("THREADS", 6))) as ex:
    futs = [ex.submit(collect, t) for t in tasks]
    for i, f in enumerate(as_completed(futs), 1):
        out.append(f.result())
        if i % 250 == 0:
            print(i, "done", flush=True)

df = pd.DataFrame(out)
df.to_parquet(os.path.join(DATA, "users.parquet"))
print("errors:", df["error"].notna().sum(), "of", len(df))
