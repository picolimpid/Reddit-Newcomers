"""Pre-registered retrodiction, step 1: newcomers in the r/Futurology 2020 sticky-comment experiment
(CivilServant replication of Matias 2019; results post reddit.com/r/Futurology/comments/x0p5lx). (2026-11-03)

No post-level assignment file is public. Assignment is reconstructed from two public pieces:
  - the pre-generated randomization sequence (CivilServant-Analysis, research-designs/reddit/randomizations/
    outputs/sticky_comment_futurology_0.randomizations.csv: blocks of 6, 50/50 within block), which the bot
    consumed in the order it processed new posts;
  - CivilServantBot's sticky comments in Arctic Shift (treated posts; control posts got nothing).
Exact alignment of the two fails (see assign()), so `assign` uses an as-treated design with an
eligibility restriction and checks it by the treated share and covariate balance.

Stages (outcomes are a separate, sealed stage, run only after the pre-registration is filed):
  assign    -> data/futurology2020/posts.parquet       experiment posts with treat / block / position
  collect   -> data/futurology2020/commenters.parquet  sampled commenters with the newcomer flag
               data/futurology2020/inputs.parquet      newcomer agent inputs (no outcome columns)
  outcomes  -> data/futurology2020/outcomes.parquet    90/180-day return (SEALED: see PREREG_futurology.md)

    python3 futurology_collect.py assign | collect [--authors 8000] | outcomes"""
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
SRC = os.path.join(ROOT, "data", "civilservant_more")
D = os.path.join(ROOT, "data", "futurology2020")
SUB = "Futurology"
DAY, YEAR = 86400, 365 * 86400
WINDOWS = [(0, 30), (30, 90), (90, 365)]
SKIP = {"[deleted]", "AutoModerator", "CivilServantBot"}
os.makedirs(D, exist_ok=True)


def run(fn, items, threads=12):
    with ThreadPoolExecutor(threads) as ex:
        return list(ex.map(fn, items))


# ---------------------------------------------------------------- assignment
def assign(a):
    """As-treated design with an eligibility restriction (PREREG_futurology.md §2.2).

    Exact reconstruction failed: aligning the bot's stickies to the public sequence breaks every few
    blocks (offsets jump; most 10-40-stretch windows match nowhere), consistent with the reported
    software glitch. Instead:
      - posts in the bot's active window that were not removed or deleted (same rule in both arms);
      - drop runs of more than 6 consecutive unstickied posts: impossible under blocks of 6 with 3
        treated, so the bot was not processing posts then (downtime);
      - treat = the post carries the CivilServantBot sticky (what commenters saw).
    Validity is checked by the treated share (design: 0.5) and pre-treatment covariate balance."""
    P = pd.read_parquet(os.path.join(SRC, "futurology_posts_2020.parquet")).sort_values(["created_utc", "id"])
    B = json.load(open(os.path.join(SRC, "futurology_bot_comments.json")))
    P["bot"] = P.id.isin({c["link_id"][3:] for c in B}).astype(int)
    t0, t1 = min(c["created_utc"] for c in B) - 600, max(c["created_utc"] for c in B)
    P = P[(P.created_utc >= t0) & (P.created_utc <= t1)]
    print(f"posts in the bot window {len(P)} | with sticky {P.bot.sum()} | bot comments {len(B)}")
    Q = P[P.removed_by_category.isna()].copy().reset_index(drop=True)
    run = (Q.bot != Q.bot.shift()).cumsum()
    Q["downtime"] = (Q.bot == 0) & (Q.groupby(run).bot.transform("size") > 6)
    n_runs = Q[Q.downtime].groupby(run).ngroups
    E = Q[~Q.downtime].copy()
    E["treat"] = E.bot
    print(f"not removed {len(Q)} | dropped {Q.downtime.sum()} posts in {n_runs} downtime runs | kept {len(E)} | "
          f"treated share {E.treat.mean():.3f}")
    ts = pd.to_datetime(E.created_utc, unit="s")
    cov = pd.DataFrame({"hour": ts.dt.hour, "weekday": ts.dt.dayofweek, "title_len": E.title.str.len(),
                        "self_post": E.is_self.astype(float),
                        "top10_domain": E.domain.isin(E.domain.value_counts().index[:10]).astype(float),
                        "no_flair": E.link_flair_text.isna().astype(float),
                        "author_gt10_posts": E.author.map(E.author.value_counts()).gt(10).astype(float)})
    bal = cov.groupby(E.treat.values).mean().T
    bal["std diff"] = (bal[1] - bal[0]) / cov.std()
    print(bal.round(3).to_string())
    E["block"] = ts.dt.strftime("%Y-%m")  # month strata, for the block-FE robustness column only
    E.to_parquet(os.path.join(D, "posts.parquet"))
    bal.round(4).to_csv(os.path.join(D, "post_balance.csv"))


# ---------------------------------------------------------------- commenters and agent inputs
def thread(pid_t):
    pid, t = pid_t
    return paginate("comments/search", int(t) - 10, int(t) + 60 * DAY, link_id=pid,
                    fields="id,author,body,parent_id,created_utc,link_id")


def check(r):
    T = int(r.post_utc)
    c = get("comments/search", author=r.author, subreddit=SUB, after=T - YEAR, before=T, sort="desc",
            limit=3, fields="id,created_utc")
    p = get("posts/search", author=r.author, subreddit=SUB, after=T - YEAR, before=T, sort="desc",
            limit=3, fields="id,created_utc")
    return not c and not p


def history(r):
    T = int(r.post_utc)  # identical parameters to Reddit Newcomers stage 3 and rscience_retro_collect
    hc = get("comments/search", author=r.author, after=T - YEAR, before=T, sort="desc",
             limit=100, fields="id,body,subreddit,created_utc,score")
    hp = get("posts/search", author=r.author, after=T - YEAR, before=T, sort="desc",
             limit=100, fields="id,title,selftext,subreddit,created_utc,score")
    return ([{"kind": "c", "text": x.get("body") or "", "sub": x["subreddit"], "t": x["created_utc"], "score": x.get("score")}
             for x in hc] +
            [{"kind": "p", "text": ((x.get("title") or "") + "\n" + (x.get("selftext") or "")).strip(), "sub": x["subreddit"],
              "t": x["created_utc"], "score": x.get("score")} for x in hp])


def collect(a):
    P = pd.read_parquet(os.path.join(D, "posts.parquet"))
    P = P[P.treat.notna()].copy()
    th = run(thread, list(zip(P.id, P.created_utc)))
    C = pd.DataFrame([dict(c, post_id=pid) for pid, cs in zip(P.id, th) for c in cs])
    C.to_parquet(os.path.join(D, "thread_comments.parquet"))
    print("thread comments", len(C), flush=True)
    Pi = P.set_index("id")
    C = C[~C.author.isin(SKIP)]
    first = C.sort_values("created_utc").drop_duplicates("author")
    authors = sorted(first.author)
    random.Random(0).shuffle(authors)
    F = first.set_index("author").loc[authors[: a.authors]].reset_index()
    F["post_utc"] = F.post_id.map(Pi.created_utc)
    print(f"unique commenters {len(first)} | sampled {len(F)}", flush=True)
    F["newcomer"] = run(check, list(F.itertuples()))
    F = F.merge(P[["id", "block", "treat"]].rename(columns={"id": "post_id"}), on="post_id")
    F.to_parquet(os.path.join(D, "commenters.parquet"))
    N = F[F.newcomer].copy()
    print(f"newcomers {len(N)} of {len(F)}", flush=True)
    hist = run(history, list(N.itertuples()))
    byid = C.set_index("id")
    kids = C.groupby("parent_id")
    out = []
    for (_, r), h in zip(N.iterrows(), hist):
        p = Pi.loc[r.post_id]
        T = int(p.created_utc)
        top = r.parent_id.startswith("t3_")
        parent = None if top else byid.body.get(r.parent_id[3:])
        reps = kids.get_group("t1_" + r.id) if ("t1_" + r.id) in kids.groups else pd.DataFrame(columns=C.columns)
        reps = reps.sort_values("created_utc")
        rec = {"author": r.author, "post_id": r.post_id, "comment_id": r.id, "block": p.block, "treat": int(p.treat),
               "subreddit": SUB, "post_utc": T, "first_utc": int(r.created_utc),
               "entry_delay_min": (r.created_utc - T) / 60, "post_title": p.title, "post_selftext": p.selftext or "",
               "post_is_video": False, "flair": p.link_flair_text, "top_level": int(top),
               "parent_body": parent or "", "first_body": r.body,
               "replies": json.dumps([{"author": x.author, "body": x.body, "is_op": x.author == p.author}
                                      for x in reps.itertuples() if x.author not in SKIP][:3]),
               "n_replies": int(sum(a_ not in SKIP for a_ in reps.author)),
               "history": json.dumps(h), "n_hist": len(h)}
        for lo, hi in WINDOWS:
            rec[f"subs_{lo}_{hi}"] = json.dumps(dict(Counter(x["sub"] for x in h if lo <= (T - x["t"]) / DAY < hi)))
        rec["top_subs"] = json.dumps([s for s, _ in Counter(x["sub"] for x in h).most_common(10)])
        out.append(rec)
    R = pd.DataFrame(out)
    R.to_parquet(os.path.join(D, "inputs.parquet"))
    print(R.groupby("treat")[["n_hist", "n_replies"]].agg(["mean", "size"]).round(3).to_string())


# ---------------------------------------------------------------- sealed outcomes
def later(r):
    t1 = int(r.first_utc)
    oc = get("comments/search", author=r.author, subreddit=SUB, after=t1, before=t1 + 180 * DAY, sort="asc",
             limit=100, fields="id,link_id,created_utc")
    op = get("posts/search", author=r.author, subreddit=SUB, after=t1, before=t1 + 180 * DAY, sort="asc",
             limit=100, fields="id,created_utc")
    ts = [x["created_utc"] for x in oc if x["link_id"][3:] != r.post_id] + [x["created_utc"] for x in op]
    return (min(ts) - t1) if ts else None, len(ts)


def outcomes(a):
    R = pd.read_parquet(os.path.join(D, "inputs.parquet"))
    got = run(later, list(R.itertuples()))
    O = R[["author", "post_id"]].copy()
    O["days_to_return"] = [None if g is None else g / DAY for g, _ in got]
    O["n_later"] = [n for _, n in got]
    for d in (90, 180):
        O[f"return_{d}d"] = [int(g is not None and g <= d * DAY) for g, _ in got]
    O.to_parquet(os.path.join(D, "outcomes.parquet"))
    print(f"outcomes written for {len(O)} newcomers (not summarized here; analysis: futurology_analyze.py)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["assign", "collect", "outcomes"])
    ap.add_argument("--authors", type=int, default=8000)
    a = ap.parse_args()
    {"assign": assign, "collect": collect, "outcomes": outcomes}[a.stage](a)
