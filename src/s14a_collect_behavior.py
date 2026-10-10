"""Stage 14a: data for comparing the BEHAVIOR of high- vs low-fit newcomers (s14c).

PART=insub  What returning newcomers do inside the community: every comment and post they
            made in the target subreddit within 180 days of the first comment (outside the
            entry thread), with text, score and the thread it was made in; plus the threads'
            metadata (size, age), so 'joining ordinary discussion' can be separated from
            'commenting on the next viral post'.
PART=ties   Who newcomers talked to before entry (social-tie mechanism): the authors of the
            comments each newcomer replied to in their 365-day history (comments/ids lookups),
            and, for up to TIES_PER_USER of those interlocutors, their own subreddit activity
            in the 365 days before the target post (-> were they members of the target
            community? how close are they to it in the subreddit network?).
PART=all    both (default)"""
import json
import os
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

from api import DATA, get

DAY = 86400
YEAR = 365 * DAY
PART = os.environ.get("PART", "all")
THREADS = int(os.environ.get("THREADS", 6))
TIES_PER_USER = int(os.environ.get("TIES_PER_USER", 8))
BOT_RE = re.compile(r"(bot|b0t)$|^bot[-_]|automod", re.I)
SKIP = {"[deleted]", "AutoModerator", None, ""}
day = lambda t: datetime.fromtimestamp(int(t), timezone.utc).strftime("%Y-%m-%d")  # noqa: E731

df = pd.read_parquet(os.path.join(DATA, "analysis_table.parquet"))
posts = pd.read_parquet(os.path.join(DATA, "target_posts.parquet")).drop_duplicates("post_id").set_index("post_id")
tc = pd.read_parquet(os.path.join(DATA, "thread_comments.parquet"), columns=["id", "created_utc"]).drop_duplicates("id")
nc = df[(df.newcomer_12m == 1) & df.hist_idx.notna()].copy()
nc["first_utc"] = tc.set_index("id").loc[nc.first_comment_id, "created_utc"].values


def run(fn, items):
    with ThreadPoolExecutor(THREADS) as ex:
        return list(ex.map(fn, items))


def batch(endpoint, ids, fields, size=500):
    chunks = [ids[k:k + size] for k in range(0, len(ids), size)]
    return [x for r in run(lambda ch: get(endpoint, ids=",".join(ch), fields=fields), chunks) for x in r]


# ---------------------------------------------------------------- in-community behavior of returners
if PART in ("insub", "all"):
    r180 = pd.read_parquet(os.path.join(DATA, "return180.parquet"))
    ret = nc[nc.index.isin(r180.index[r180.return_180d == 1])]

    def insub(r):
        t0, t1 = int(r.first_utc) + 1, int(r.first_utc) + 180 * DAY
        try:
            cs = get("comments/search", author=r.author, subreddit=r.subreddit, after=t0, before=t1, sort="asc",
                     limit=100, fields="id,body,score,created_utc,link_id,parent_id")
            ps = get("posts/search", author=r.author, subreddit=r.subreddit, after=t0, before=t1, sort="asc",
                     limit=100, fields="id,title,selftext,score,created_utc,num_comments")
        except RuntimeError:
            return []
        rows = [{"idx": r.Index, "kind": "c", "id": c["id"], "text": c.get("body") or "", "score": c.get("score"),
                 "t": c["created_utc"], "thread": c["link_id"].split("_")[-1],
                 "top_level": int(c["parent_id"].startswith("t3_")), "parent": c["parent_id"].split("_")[-1]}
                for c in cs if c["link_id"].split("_")[-1] != r.post_id]
        rows += [{"idx": r.Index, "kind": "p", "id": p["id"],
                  "text": ((p.get("title") or "") + "\n" + (p.get("selftext") or "")).strip(),
                  "score": p.get("score"), "t": p["created_utc"], "thread": p["id"], "top_level": None,
                  "parent": None, "num_comments": p.get("num_comments")} for p in ps]
        return rows

    items = pd.DataFrame([x for rows in run(insub, list(ret.itertuples())) for x in rows])
    # metadata of the threads they commented in (size and creation time)
    th = pd.DataFrame(batch("posts/ids", sorted(set(items.thread)), "id,num_comments,score,created_utc,author"))
    th = th.rename(columns={"id": "thread", "num_comments": "thread_comments", "score": "thread_score",
                            "created_utc": "thread_utc", "author": "thread_author"})
    items = items.drop(columns="num_comments", errors="ignore").merge(th, on="thread", how="left")
    items.to_parquet(os.path.join(DATA, "insub_items.parquet"))
    print("insub:", len(ret), "returners;", len(items), "items;", items.thread.nunique(), "threads", flush=True)

# ---------------------------------------------------------------- interlocutors before entry
if PART in ("ties", "all"):
    def history_ids(r):  # identical parameters to stage 3 / s9 -> cache hit
        T = int(posts.at[r.post_id, "created_utc"])
        hc = get("comments/search", author=r.author, after=T - YEAR, before=T, sort="desc",
                 limit=100, fields="id,body,subreddit,created_utc,score")
        return [c["id"] for c in hc]

    hid = dict(zip(nc.index, run(history_ids, list(nc.itertuples()))))
    parent = {c["id"]: c.get("parent_id") or "" for c in
              batch("comments/ids", sorted({i for v in hid.values() for i in v}), "id,parent_id")}
    pids = sorted({p.split("_")[-1] for p in parent.values() if p.startswith("t1_")})
    pauth = {c["id"]: c.get("author") for c in batch("comments/ids", pids, "id,author")}
    print("ties: history comments", len(parent), "| replies to comments", len(pids), flush=True)

    rng = random.Random(3)
    tasks, talk = [], {}
    for i, r in zip(nc.index, nc.itertuples()):
        cnt = Counter(pauth.get(parent.get(c, "").split("_")[-1]) for c in hid[i]
                      if parent.get(c, "").startswith("t1_"))
        cnt = Counter({a: n for a, n in cnt.items() if a not in SKIP and a != r.author and not BOT_RE.search(a)})
        talk[i] = {"n_replies_to_comments": int(sum(cnt.values())), "n_interlocutors": len(cnt)}
        # most frequent interlocutors first, ties broken at random
        chosen = sorted(cnt, key=lambda a: (-cnt[a], rng.random()))[:TIES_PER_USER]
        talk[i]["chosen"] = {a: cnt[a] for a in chosen}
        T = int(posts.at[r.post_id, "created_utc"])
        tasks += [(a, day(T - YEAR), day(T)) for a in chosen]
    tasks = sorted(set(tasks))
    print("ties: interlocutor profiles to fetch:", len(tasks), flush=True)

    CACHED_ONLY = os.environ.get("CACHED_ONLY") == "1"  # use only profiles already in the cache

    def is_cached(endpoint, **params):
        import hashlib
        from api import CACHE
        key = endpoint + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
        return os.path.exists(os.path.join(CACHE, hashlib.sha1(key.encode()).hexdigest() + ".json"))

    def profile(t):
        a, d0, d1 = t
        if CACHED_ONLY and not is_cached("comments/search/aggregate", aggregate="subreddit", author=a,
                                         after=d0, before=d1, limit=""):
            return t, None
        try:
            rows = get("comments/search/aggregate", aggregate="subreddit", author=a, after=d0, before=d1, limit="")
            return t, {x["key"]: int(x["count"]) for x in rows}
        except RuntimeError:
            return t, None

    prof = dict(run(profile, tasks))
    out = []
    for i, r in zip(nc.index, nc.itertuples()):
        T = int(posts.at[r.post_id, "created_utc"])
        key = (day(T - YEAR), day(T))
        inter = [{"author": a, "n": n, "subs": prof.get((a,) + key)} for a, n in talk[i]["chosen"].items()]
        out.append({"idx": i, "n_replies_to_comments": talk[i]["n_replies_to_comments"],
                    "n_interlocutors": talk[i]["n_interlocutors"], "interlocutors": json.dumps(inter)})
    pd.DataFrame(out).set_index("idx").to_parquet(os.path.join(DATA, "ties.parquet"))
    print("ties: done;", sum(v is None for v in prof.values()), "profile errors", flush=True)
