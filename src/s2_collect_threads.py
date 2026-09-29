"""Stage 2: collect every archived comment under each target post."""
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from api import DATA, paginate

posts = pd.read_parquet(os.path.join(DATA, "target_posts.parquet"))
FIELDS = "id,author,parent_id,link_id,created_utc,body,score"


def fetch(row):
    start = int(row.created_utc) - 1
    end = start + 90 * 86400  # comments within 90 days of posting
    try:
        cs = paginate("comments/search", start, end, link_id=row.post_id, fields=FIELDS)
    except RuntimeError as e:  # skip a post the archive keeps timing out on; re-run resumes from cache
        print("FAILED", row.post_id, str(e)[-80:], flush=True)
        return []
    for c in cs:
        c["post_id"] = row.post_id
        c["subreddit"] = row.subreddit
    return cs


with ThreadPoolExecutor(int(os.environ.get("THREADS", 4))) as ex:
    results = list(ex.map(fetch, posts.itertuples()))

df = pd.DataFrame([c for r in results for c in r])
df.to_parquet(os.path.join(DATA, "thread_comments.parquet"))
cov = df.groupby("post_id").size().rename("archived").to_frame().join(
    posts.set_index("post_id")["num_comments"])
print(len(df), "comments;", "archived/num_comments coverage median =",
      round((cov.archived / cov.num_comments).median(), 3))
