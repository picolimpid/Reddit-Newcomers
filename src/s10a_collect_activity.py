"""Stage 10a: each newcomer's Reddit-wide activity (comments + posts per day) in the
90 days before and after their first comment on the target post."""
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


def daily(endpoint, author, t0):
    # the aggregate endpoint needs calendar dates; epoch timestamps silently return all-zero buckets
    day = lambda t: datetime.fromtimestamp(int(t), timezone.utc).strftime("%Y-%m-%d")  # noqa: E731
    rows = get(endpoint, aggregate="created_utc", frequency="day", author=author,
               after=day(t0 - (WIN + 1) * DAY), before=day(t0 + (WIN + 2) * DAY))
    return {int(datetime.fromisoformat(r["created_utc"].replace("Z", "+00:00")).timestamp()): int(r["count"])
            for r in rows}


def collect(r):
    try:
        c = daily("comments/search/aggregate", r.author, r.first_utc)
        p = daily("posts/search/aggregate", r.author, r.first_utc)
    except RuntimeError as e:
        return {"idx": r.Index, "act_error": str(e)[-120:]}
    rec = {"idx": r.Index, "act_error": None}
    # day buckets strictly before / after the bucket containing the first comment
    for name, series in (("c", c), ("p", p)):
        pre = sum(n for t, n in series.items() if r.first_utc - WIN * DAY <= t and t + DAY <= r.first_utc)
        post = sum(n for t, n in series.items() if r.first_utc < t <= r.first_utc + WIN * DAY)
        rec[f"pre90_{name}"], rec[f"post90_{name}"] = pre, post
    days_after = sorted({(t - r.first_utc) // DAY for s in (c, p) for t, n in s.items()
                         if n > 0 and r.first_utc < t <= r.first_utc + WIN * DAY})
    rec["post90_active_days"] = len(days_after)
    rec["active_last30"] = int(any(d >= WIN - 30 for d in days_after))  # still active on Reddit in days 60-90
    return rec


with ThreadPoolExecutor(int(os.environ.get("THREADS", 4))) as ex:
    out = pd.DataFrame(list(ex.map(collect, nc.itertuples()))).set_index("idx")
out.to_parquet(os.path.join(DATA, "activity.parquet"))
print(len(out), "newcomers; errors:", out.act_error.notna().sum())
print(out.drop(columns="act_error").describe().round(2).T[["mean", "50%"]])
