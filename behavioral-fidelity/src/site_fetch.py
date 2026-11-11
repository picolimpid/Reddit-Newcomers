"""Fetch raw material for further CivilServant sticky-comment sites (2026-11-06): every post in the bot's
active window plus CivilServantBot's comments. Assignment, inputs and sealed outcomes: site_collect.py.

    python3 site_fetch.py ffxiv2019 | science2020"""
import datetime as dt
import json
import os
import sys

import pandas as pd

sys.path.insert(0, "/data/hanqim2/Reddit Newcomers/src")
from api import paginate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITES = {  # subreddit, fetch window (UTC dates, padded around the bot's activity)
    "ffxiv2019": ("ffxiv", (2019, 7, 1), (2019, 9, 20)),
    "science2020": ("science", (2020, 2, 14), (2020, 8, 20)),
}
FIELDS = ("id,created_utc,author,title,selftext,removed_by_category,link_flair_text,num_comments,is_self,domain,"
          "over_18,stickied,distinguished")


def main(site):
    sub, s, e = SITES[site]
    d = os.path.join(ROOT, "data", "sites", site)
    os.makedirs(d, exist_ok=True)
    s, e = (int(dt.datetime(*x, tzinfo=dt.timezone.utc).timestamp()) for x in (s, e))
    B = paginate("comments/search", s, e, author="CivilServantBot", subreddit=sub)
    json.dump(B, open(os.path.join(d, "bot_comments.json"), "w"))
    print(f"{site}: bot comments {len(B)}", flush=True)
    # full objects: removed_by_category is not a valid `fields` value in Arctic Shift
    P = pd.DataFrame([{k: x.get(k) for k in FIELDS.split(",")} for x in paginate("posts/search", s, e, subreddit=sub)])
    P.to_parquet(os.path.join(d, "raw_posts.parquet"))
    print(f"{site}: posts {len(P)}; removed_by_category:\n{P.removed_by_category.value_counts(dropna=False)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
