"""Human causal anchor (step 1): the r/science sticky-comment experiment (Matias 2019, PNAS),
recovered from the Arctic Shift archive. CivilServantBot posted the rules sticky
("Welcome to r/science! Comments will be removed if ...") on treated posts; posts in the same
window without it are the candidates for control. Writes data/rscience2016/{posts,bot}.parquet.

    python3 rscience_collect.py"""
import os
import time

import pandas as pd
import requests

API = "https://arctic-shift.photon-reddit.com/api"
H = {"User-Agent": "newcomer-alignment-study/0.1 (academic research)"}
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "rscience2016")
os.makedirs(OUT, exist_ok=True)


def page(ep, after, before, **p):
    rows = []
    while True:
        for k in range(5):
            try:
                r = requests.get(f"{API}/{ep}", params={**p, "after": after, "before": before, "limit": "100", "sort": "asc"},
                                 headers=H, timeout=90)
                d = r.json().get("data")
                break
            except Exception:  # noqa: BLE001
                time.sleep(3 * (k + 1))
        if not d:
            return rows
        rows += d
        after = d[-1]["created_utc"]
        if len(d) < 100:
            return rows
        time.sleep(0.3)


bot = pd.DataFrame(page("comments/search", "2016-08-01", "2016-10-15", author="CivilServantBot", subreddit="science",
                        fields="id,link_id,body,created_utc"))
bot["post_id"] = bot.link_id.str[3:]
bot["arm"] = bot.body.str.startswith("Welcome to r/science!").map({True: "rules", False: "ama_note"})
bot.to_parquet(os.path.join(OUT, "bot.parquet"))
t0, t1 = bot.created_utc.min() - 3600, bot.created_utc.max() + 3600
posts = pd.DataFrame(page("posts/search", int(t0), int(t1), subreddit="science",
                          fields="id,created_utc,num_comments,title,author,link_flair_text,score"))
posts["treated"] = posts.id.isin(bot.loc[bot.arm == "rules", "post_id"])
posts["ama"] = posts.id.isin(bot.loc[bot.arm == "ama_note", "post_id"]) | posts.title.str.contains(r"\bAMA\b", na=False)
posts.to_parquet(os.path.join(OUT, "posts.parquet"))
print("bot comments", len(bot), bot.arm.value_counts().to_dict())
print("posts in window", len(posts), "| treated", int(posts.treated.sum()), "| AMA", int(posts.ama.sum()))
d = posts[~posts.ama]
print("non-AMA posts with >= 1 comment: treated", int((d.treated & (d.num_comments > 0)).sum()),
      "control", int((~d.treated & (d.num_comments > 0)).sum()))
print("comments: treated mean", round(d[d.treated].num_comments.mean(), 1), "control mean", round(d[~d.treated].num_comments.mean(), 1))
