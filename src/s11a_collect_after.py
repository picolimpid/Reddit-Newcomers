"""Stage 11a: newcomers' behavior in the 90 days BEFORE vs AFTER the interaction.

Per newcomer:
  - text sample after: first <=100 comments and <=100 posts in the 90 days after the
    first comment (the 'before' sample is the cached stage-3 history)
  - per-subreddit activity counts (uncapped) for both windows, comments and posts
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

from api import DATA, get

DAY = 86400
WIN = 90
df = pd.read_parquet(os.path.join(DATA, "analysis_table.parquet"))
tc = pd.read_parquet(os.path.join(DATA, "thread_comments.parquet"),
                     columns=["id", "created_utc"]).drop_duplicates("id").set_index("id")
nc = df[df.newcomer_12m == 1].copy()
nc["first_utc"] = tc.loc[nc.first_comment_id, "created_utc"].values
day = lambda t: datetime.fromtimestamp(int(t), timezone.utc).strftime("%Y-%m-%d")  # noqa: E731


def by_sub(endpoint, author, start, end):
    rows = get(endpoint, aggregate="subreddit", author=author, after=day(start), before=day(end), limit="")
    return {r["key"]: int(r["count"]) for r in rows}


def collect(r):
    t0 = int(r.first_utc)
    try:
        ca = get("comments/search", author=r.author, after=t0, before=t0 + WIN * DAY, sort="asc",
                 limit=100, fields="id,body,subreddit,created_utc,link_id")
        pa = get("posts/search", author=r.author, after=t0, before=t0 + WIN * DAY, sort="asc",
                 limit=100, fields="id,title,selftext,subreddit,created_utc")
        # whole calendar days strictly before / after the entry day
        subs = {f"{k}_{w}": by_sub(f"{k}s/search/aggregate", r.author, *rng)
                for k in ("comment", "post")
                for w, rng in (("before", (t0 - WIN * DAY, t0)), ("after", (t0 + DAY, t0 + (WIN + 1) * DAY)))}
    except RuntimeError as e:
        return {"idx": r.Index, "after_error": str(e)[-120:]}
    after_items = ([{"kind": "c", "text": c.get("body") or "", "sub": c["subreddit"], "t": c["created_utc"],
                     "thread": c["link_id"].split("_")[-1]} for c in ca]
                   + [{"kind": "p", "text": ((p.get("title") or "") + "\n" + (p.get("selftext") or "")).strip(),
                       "sub": p["subreddit"], "t": p["created_utc"], "thread": p["id"]} for p in pa])
    return {"idx": r.Index, "after_error": None, "after_items": after_items, **subs}


with ThreadPoolExecutor(int(os.environ.get("THREADS", 4))) as ex:
    out = pd.DataFrame(list(ex.map(collect, nc.itertuples()))).set_index("idx")
for col in out.columns:
    if col != "after_error":
        out[col] = out[col].map(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else None)
out.to_parquet(os.path.join(DATA, "after_activity.parquet"))
print(len(out), "newcomers; errors:", out.after_error.notna().sum())
