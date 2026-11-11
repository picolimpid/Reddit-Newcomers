"""Human causal anchor (step 2): newcomers in the Matias (2019) r/science sticky-comment experiment.

The CivilServant replication files (data/rscience2016/civilservant/, from
github.com/citizensandtech/CivilServant-Analysis, papers/r_science_2016) give every experiment post
with its block and random assignment but no post ids and no authors. Post ids are recovered by
matching `created` (US Eastern, UTC-4 in Aug-Sep 2016) to Arctic Shift; authors, histories and
later activity come from Arctic Shift through the cached client of the Reddit Newcomers repo, with
the same history parameters as its stage 3, so the agents are built exactly like the benchmark's.

Per sampled commenter (first comment in an experiment thread, non-AMA posts, valid blocks):
  newcomer      no r/science comment or post in the 365 days before the post
  return_90d    any r/science comment or post outside this thread within 90 days of the comment
  return_180d   same within 180 days
Writes data/rscience2016/retro.parquet (newcomers, agent inputs + outcomes + arm) and
data/rscience2016/commenters.parquet (every sampled commenter with the newcomer flag).

    python3 rscience_retro_collect.py [--authors 8000]"""
import argparse
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, "/data/hanqim2/Reddit Newcomers/src")
from api import get, paginate  # noqa: E402  (disk-cached Arctic Shift client)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "rscience2016")
CS = os.path.join(D, "civilservant")
DAY, YEAR = 86400, 365 * 86400
WINDOWS = [(0, 30), (30, 90), (90, 365)]
BAD_BLOCKS = {"nonama.block058", "nonama.block222", "nonama.block037"}   # dropped in the paper's R code
SKIP = {"[deleted]", "AutoModerator", "CivilServantBot"}


def run(fn, items, threads=6):
    with ThreadPoolExecutor(threads) as ex:
        return list(ex.map(fn, items))


def experiment_posts():
    P = pd.read_csv(os.path.join(CS, "r_science_experiment_1_posts.09.26.2016.csv"))
    P = P[~P["block.id"].isin(BAD_BLOCKS) & ~P["post.ama"]].copy()
    P["created_utc"] = ((pd.to_datetime(P.created) + pd.Timedelta(hours=4)) - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    S = pd.read_parquet(os.path.join(D, "posts.parquet"))[["id", "created_utc"]]
    m = P.merge(S, on="created_utc", how="left")
    dup = m.duplicated("created_utc", keep=False)
    print(f"experiment posts {len(P)} | matched {m.id.notna().sum()} | same-second collisions {dup.sum()}", flush=True)
    m = m[m.id.notna() & ~dup]
    meta = []
    ids = m.id.tolist()
    for k in range(0, len(ids), 500):
        meta += get("posts/ids", ids=",".join(ids[k:k + 500]),
                    fields="id,title,selftext,author,link_flair_text,num_comments")
    meta = pd.DataFrame(meta).set_index("id")
    m = m.join(meta, on="id")
    return m[["id", "block.id", "TREAT", "created_utc", "title", "selftext", "author",
              "link_flair_text", "num_comments", "newcomer.comments"]].rename(
        columns={"block.id": "block", "TREAT": "treat", "author": "post_author"})


def thread(pid_t):
    pid, t = pid_t
    return paginate("comments/search", int(t) - 10, int(t) + 60 * DAY, link_id=pid,
                    fields="id,author,body,parent_id,created_utc,link_id")


def check(r):
    T = int(r.post_utc)
    c = get("comments/search", author=r.author, subreddit="science", after=T - YEAR, before=T, sort="desc",
            limit=3, fields="id,created_utc")
    p = get("posts/search", author=r.author, subreddit="science", after=T - YEAR, before=T, sort="desc",
            limit=3, fields="id,created_utc")
    return not c and not p


def details(r):
    T, t1 = int(r.post_utc), int(r.first_utc)
    # identical parameters to Reddit Newcomers stage 3 (history)
    hc = get("comments/search", author=r.author, after=T - YEAR, before=T, sort="desc",
             limit=100, fields="id,body,subreddit,created_utc,score")
    hp = get("posts/search", author=r.author, after=T - YEAR, before=T, sort="desc",
             limit=100, fields="id,title,selftext,subreddit,created_utc,score")
    oc = get("comments/search", author=r.author, subreddit="science", after=t1, before=t1 + 180 * DAY, sort="asc",
             limit=100, fields="id,link_id,created_utc")
    op = get("posts/search", author=r.author, subreddit="science", after=t1, before=t1 + 180 * DAY, sort="asc",
             limit=100, fields="id,created_utc")
    hist = ([{"kind": "c", "text": x.get("body") or "", "sub": x["subreddit"], "t": x["created_utc"], "score": x.get("score")}
             for x in hc] +
            [{"kind": "p", "text": ((x.get("title") or "") + "\n" + (x.get("selftext") or "")).strip(), "sub": x["subreddit"],
              "t": x["created_utc"], "score": x.get("score")} for x in hp])
    later = [x["created_utc"] for x in oc if x["link_id"][3:] != r.post_id] + [x["created_utc"] for x in op]
    gap = min(later) - t1 if later else None
    return hist, gap, len(later)


def main(a):
    P = experiment_posts()
    print("non-AMA posts with ids:", len(P), "| treated share", round(P.treat.mean(), 3), flush=True)
    th = run(thread, list(zip(P.id, P.created_utc)))
    rows = [dict(c, post_id=pid) for pid, cs in zip(P.id, th) for c in cs]
    C = pd.DataFrame(rows)
    C.to_parquet(os.path.join(D, "thread_comments.parquet"))
    print("thread comments", len(C), flush=True)

    Pi = P.set_index("id")
    C = C[~C.author.isin(SKIP)]
    first = C.sort_values("created_utc").drop_duplicates("author")          # each author's first experiment comment
    authors = sorted(first.author)
    random.Random(0).shuffle(authors)
    F = first.set_index("author").loc[authors[: a.authors]].reset_index()
    F["post_utc"] = F.post_id.map(Pi.created_utc)
    print(f"unique commenters {len(first)} | sampled {len(F)}", flush=True)
    F["newcomer"] = run(check, list(F.itertuples()))
    F.merge(P[["id", "block", "treat"]].rename(columns={"id": "post_id"}), on="post_id").to_parquet(
        os.path.join(D, "commenters.parquet"))
    N = F[F.newcomer].copy()
    print(f"newcomers {len(N)} of {len(F)}", flush=True)

    det = run(details, list(N.rename(columns={"created_utc": "first_utc"}).itertuples()))
    byid = C.set_index("id")
    kids = C.groupby("parent_id")
    out = []
    for (_, r), (hist, gap, n_later) in zip(N.iterrows(), det):
        p = Pi.loc[r.post_id]
        T = int(p.created_utc)
        top = r.parent_id.startswith("t3_")
        parent = None if top else byid.body.get(r.parent_id[3:])
        reps = kids.get_group("t1_" + r.id) if ("t1_" + r.id) in kids.groups else pd.DataFrame(columns=C.columns)
        reps = reps.sort_values("created_utc")
        rec = {"author": r.author, "post_id": r.post_id, "comment_id": r.id, "block": p.block, "treat": int(p.treat),
               "subreddit": "science", "post_utc": T, "first_utc": int(r.created_utc),
               "entry_delay_min": (r.created_utc - T) / 60, "post_title": p.title, "post_selftext": p.selftext or "",
               "post_is_video": False, "flair": p.link_flair_text, "top_level": int(top),
               "parent_body": parent or "", "first_body": r.body,
               "replies": json.dumps([{"author": x.author, "body": x.body, "is_op": x.author == p.post_author}
                                      for x in reps.itertuples() if x.author not in SKIP][:3]),
               "n_replies": int(sum(a_ not in SKIP for a_ in reps.author)),
               "history": json.dumps(hist), "n_hist": len(hist), "n_later": n_later,
               "days_to_return": None if gap is None else gap / DAY}
        for lo, hi in WINDOWS:
            rec[f"subs_{lo}_{hi}"] = json.dumps(dict(Counter(x["sub"] for x in hist if lo <= (T - x["t"]) / DAY < hi)))
        rec["top_subs"] = json.dumps([s for s, _ in Counter(x["sub"] for x in hist).most_common(10)])
        for d in (90, 180):
            rec[f"return_{d}d"] = int(gap is not None and gap <= d * DAY)
        out.append(rec)
    R = pd.DataFrame(out)
    R.to_parquet(os.path.join(D, "retro.parquet"))
    print(R.groupby("treat")[["return_90d", "return_180d", "n_hist", "n_replies"]].agg(["mean", "size"]).round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors", type=int, default=8000)
    main(ap.parse_args())
