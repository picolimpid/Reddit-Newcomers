"""Further sites of the CivilServant sticky-comment experiments, same pipeline as futurology_collect.py
(2026-11-06). Raw posts and bot comments come from site_fetch.py.

Design rule copied unchanged from PREREG_futurology.md §2.2 (as-treated with an eligibility restriction):
posts in the bot's active window, not removed or deleted; runs of more than 6 consecutive unstickied posts
dropped (impossible under blocks of 6 with 3 treated, i.e. bot downtime); treat = carries the sticky.

Stages (outcomes are SEALED until the scores of every model are frozen in results/sites/FREEZE.txt):
    python3 site_collect.py assign  ffxiv2019
    python3 site_collect.py collect ffxiv2019 [--authors 8000]
    python3 site_collect.py outcomes ffxiv2019      # only after the freeze"""
import argparse
import json
import os
import random
import sys
from collections import Counter

import pandas as pd

import futurology_collect as FC  # thread / check / history / later: identical Arctic Shift queries

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITES = {
    "ffxiv2019": {
        "sub": "ffxiv", "welcome": "Welcome to /r/ffxiv!",
        "rules": ("Gyorin the Gunbreaker asks you to keep in mind Reddiquette, follow our community rules, and be "
                  "civil with your fellow Warriors of Light. Threads on bad experiences with other players (even "
                  "anonymous) as well as hate-based comments such as personal attacks, bigotry, hate speech, and "
                  "name shaming are subject to removal by the moderator team under rule 1. Please report any rule "
                  "violations; the moderator team will review them as soon as possible.")},
    "science2020": {
        "sub": "science", "welcome": "Welcome to r/science!",
        "rules": ("Our team of 1,500+ moderators will remove comments if they are jokes, anecdotes, memes, off-topic "
                  "or medical advice (rules). We encourage respectful discussion about the science of the post.")},
}


def D(site):
    return os.path.join(ROOT, "data", "sites", site)


def assign(site, a):
    P = pd.read_parquet(os.path.join(D(site), "raw_posts.parquet")).sort_values(["created_utc", "id"])
    B = json.load(open(os.path.join(D(site), "bot_comments.json")))
    P["bot"] = P.id.isin({c["link_id"][3:] for c in B}).astype(int)
    t0, t1 = min(c["created_utc"] for c in B) - 600, max(c["created_utc"] for c in B)
    P = P[(P.created_utc >= t0) & (P.created_utc <= t1)]
    print(f"posts in the bot window {len(P)} | with sticky {P.bot.sum()} | bot comments {len(B)}")
    # removed or deleted, same rule as Futurology; 2019 Arctic Shift posts lack removed_by_category, so the
    # proxies [deleted] author, [removed]/[deleted] body and the moderators' "Removed" flair are added
    gone = (P.removed_by_category.notna() | P.author.eq("[deleted]") | P.selftext.fillna("").isin(["[removed]", "[deleted]"])
            | P.link_flair_text.fillna("").str.contains("Removed"))
    print("removed/deleted share by sticky:", P.groupby("bot").apply(lambda g: gone[g.index].mean()).round(3).to_dict())
    Q = P[~gone].copy().reset_index(drop=True)
    run = (Q.bot != Q.bot.shift()).cumsum()
    Q["downtime"] = (Q.bot == 0) & (Q.groupby(run).bot.transform("size") > 6)
    n_runs = Q[Q.downtime].groupby(run).ngroups
    E = Q[~Q.downtime].copy()
    E["treat"] = E.bot
    print(f"not removed {len(Q)} | dropped {Q.downtime.sum()} posts in {n_runs} downtime runs | kept {len(E)} | "
          f"treated share {E.treat.mean():.3f}")
    # removed posts by arm: a sticky could change removal, so report it (design check, no outcome involved)
    ts = pd.to_datetime(E.created_utc, unit="s")
    cov = pd.DataFrame({"hour": ts.dt.hour, "weekday": ts.dt.dayofweek, "title_len": E.title.str.len(),
                        "self_post": E.is_self.astype(float),
                        "top10_domain": E.domain.isin(E.domain.value_counts().index[:10]).astype(float),
                        "no_flair": E.link_flair_text.isna().astype(float),
                        "author_gt10_posts": E.author.map(E.author.value_counts()).gt(10).astype(float)})
    bal = cov.groupby(E.treat.values).mean().T
    bal["std diff"] = (bal[1] - bal[0]) / cov.std()
    print(bal.round(3).to_string())
    E["block"] = ts.dt.strftime("%G-W%V")  # ISO-week strata (block-FE robustness column only)
    E.to_parquet(os.path.join(D(site), "posts.parquet"))
    bal.round(4).to_csv(os.path.join(D(site), "post_balance.csv"))


def collect(site, a):
    """= futurology_collect.collect with the site's subreddit and paths."""
    FC.SUB = SITES[site]["sub"]
    d = D(site)
    P = pd.read_parquet(os.path.join(d, "posts.parquet"))
    P = P[P.treat.notna()].copy()
    th = FC.run(FC.thread, list(zip(P.id, P.created_utc)))
    C = pd.DataFrame([dict(c, post_id=pid) for pid, cs in zip(P.id, th) for c in cs])
    C.to_parquet(os.path.join(d, "thread_comments.parquet"))
    print("thread comments", len(C), flush=True)
    Pi = P.set_index("id")
    C = C[~C.author.isin(FC.SKIP)]
    first = C.sort_values("created_utc").drop_duplicates("author")
    authors = sorted(first.author)
    random.Random(0).shuffle(authors)
    F = first.set_index("author").loc[authors[: a.authors]].reset_index()
    F["post_utc"] = F.post_id.map(Pi.created_utc)
    print(f"unique commenters {len(first)} | sampled {len(F)}", flush=True)
    F["newcomer"] = FC.run(FC.check, list(F.itertuples()))
    F = F.merge(P[["id", "block", "treat"]].rename(columns={"id": "post_id"}), on="post_id")
    F.to_parquet(os.path.join(d, "commenters.parquet"))
    N = F[F.newcomer].copy()
    print(f"newcomers {len(N)} of {len(F)}", flush=True)
    hist = FC.run(FC.history, list(N.itertuples()))
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
               "subreddit": FC.SUB, "post_utc": T, "first_utc": int(r.created_utc),
               "entry_delay_min": (r.created_utc - T) / 60, "post_title": p.title, "post_selftext": p.selftext or "",
               "post_is_video": False, "flair": p.link_flair_text, "top_level": int(top),
               "parent_body": parent or "", "first_body": r.body,
               "replies": json.dumps([{"author": x.author, "body": x.body, "is_op": x.author == p.author}
                                      for x in reps.itertuples() if x.author not in FC.SKIP][:3]),
               "n_replies": int(sum(a_ not in FC.SKIP for a_ in reps.author)),
               "history": json.dumps(h), "n_hist": len(h)}
        for lo, hi in FC.WINDOWS:
            rec[f"subs_{lo}_{hi}"] = json.dumps(dict(Counter(x["sub"] for x in h if lo <= (T - x["t"]) / FC.DAY < hi)))
        rec["top_subs"] = json.dumps([s for s, _ in Counter(x["sub"] for x in h).most_common(10)])
        out.append(rec)
    R = pd.DataFrame(out)
    R.to_parquet(os.path.join(d, "inputs.parquet"))
    print(R.groupby("treat")[["n_hist", "n_replies"]].agg(["mean", "size"]).round(3).to_string())


def outcomes(site, a):
    freeze = os.path.join(ROOT, "results", "sites", "FREEZE.txt")  # see PREREG_sites.md §5
    assert os.path.exists(freeze) and f"scores {site}" in open(freeze).read(), "freeze the agent scores first"
    FC.SUB = SITES[site]["sub"]
    R = pd.read_parquet(os.path.join(D(site), "inputs.parquet"))
    got = FC.run(FC.later, list(R.itertuples()))
    O = R[["author", "post_id"]].copy()
    O["days_to_return"] = [None if g is None else g / FC.DAY for g, _ in got]
    O["n_later"] = [n for _, n in got]
    for d in (90, 180):
        O[f"return_{d}d"] = [int(g is not None and g <= d * FC.DAY) for g, _ in got]
    O.to_parquet(os.path.join(D(site), "outcomes.parquet"))
    print(f"outcomes written for {len(O)} newcomers")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["assign", "collect", "outcomes"])
    ap.add_argument("site", choices=list(SITES))
    ap.add_argument("--authors", type=int, default=8000)
    a = ap.parse_args()
    {"assign": assign, "collect": collect, "outcomes": outcomes}[a.stage](a.site, a)
