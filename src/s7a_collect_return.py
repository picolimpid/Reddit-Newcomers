"""Stage 7a: community return. For each observed newcomer, did they post or comment
in the target subreddit again (outside the target thread) within 30 / 90 days of
their first comment?"""
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from api import DATA, get

DAY = 86400
df = pd.read_parquet(os.path.join(DATA, "analysis_table.parquet"))
comments = pd.read_parquet(os.path.join(DATA, "thread_comments.parquet"),
                           columns=["id", "created_utc"]).set_index("id")
nc = df[df.newcomer_12m == 1].copy()
nc["first_utc"] = comments.loc[nc.first_comment_id, "created_utc"].values


def collect(r):
    t0, t1 = int(r.first_utc) + 1, int(r.first_utc) + 90 * DAY
    try:
        cs = get("comments/search", author=r.author, subreddit=r.subreddit, after=t0, before=t1,
                 sort="asc", limit=100, fields="link_id,created_utc")
        ps = get("posts/search", author=r.author, subreddit=r.subreddit, after=t0, before=t1,
                 sort="asc", limit=100, fields="id,created_utc")
    except RuntimeError as e:
        return {"idx": r.Index, "return_error": str(e)[-120:]}
    # activity outside the thread the newcomer entered through
    ts = [c["created_utc"] for c in cs if c["link_id"].split("_")[-1] != r.post_id]
    ts += [p["created_utc"] for p in ps]
    first = min(ts) if ts else None
    return {"idx": r.Index, "return_error": None,
            "return_30d": int(first is not None and first <= r.first_utc + 30 * DAY),
            "return_90d": int(first is not None),
            "return_n_90d": len(ts)}


with ThreadPoolExecutor(int(os.environ.get("THREADS", 4))) as ex:
    out = pd.DataFrame(list(ex.map(collect, nc.itertuples()))).set_index("idx")
out.to_parquet(os.path.join(DATA, "return.parquet"))
print(len(out), "newcomers; errors:", out.return_error.notna().sum(),
      "| return_30d:", round(out.return_30d.mean(), 3), "| return_90d:", round(out.return_90d.mean(), 3))
