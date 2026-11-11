"""Build the behavioral benchmark from Reddit Newcomers Study 2.

One row per observed newcomer (2,413). Two kinds of columns:
  agent inputs     pre-entry history (dated items), entry post, first comment and its parent,
                   the replies actually received
  human reference  return outcomes, community fit (overall and per pre-entry window),
                   Jev window scores, reply codings, model controls
plus a community-level split (by subreddit) for later post-training / generalization tests.
Writes data/benchmark.parquet."""
import json
import os

import numpy as np
import pandas as pd

import c2v
from common import BENCH, DAY, SOURCE, WINDOWS


def src(name, **kw):
    return pd.read_parquet(os.path.join(SOURCE, name), **kw)


nc = src("jev_newcomers.parquet")          # merged newcomer table (index = analysis_table row)
posts = src("target_posts.parquet").drop_duplicates("post_id").set_index("post_id")
users = src("users.parquet")
tc = src("thread_comments.parquet", columns=["id", "parent_id", "body", "author", "created_utc"]).drop_duplicates("id").set_index("id")
print("newcomers:", len(nc), flush=True)

# ---------------------------------------------------------------- c2v window fits (as s14c)
hist_users = users[users.history.notna()].drop_duplicates(["post_id", "author"])
extra = [json.loads(s) for s in src("extra_users.parquet").subs.dropna()]
vid, E = c2v.train([{x["sub"]: 1 for x in json.loads(h)} for h in hist_users.history] + [c for c in extra if c])
print("c2v vocabulary:", len(vid), flush=True)
hist = dict(zip(zip(hist_users.post_id, hist_users.author), hist_users.history))


def window_counts(items, sub, T, lo, hi):
    c = {}
    for x in items:
        if lo <= (T - x["t"]) / DAY < hi and x["sub"].lower() != sub.lower():
            c[x["sub"]] = c.get(x["sub"], 0) + 1
    return c


rows = []
for i, r in nc.iterrows():
    T = int(posts.loc[r.post_id, "created_utc"])
    items = [x for x in json.loads(hist[(r.post_id, r.author)])
             if x["t"] < T and x["sub"].lower() != r.subreddit.lower()]
    row = {"idx": i, "history": json.dumps(items)}
    for lo, hi in WINDOWS:
        c = window_counts(items, r.subreddit, T, lo, hi)
        row[f"subs_{lo}_{hi}"] = json.dumps(c)
        row[f"n_{lo}_{hi}"] = sum(c.values())
        row[f"fit_{lo}_{hi}"] = c2v.fit(c, r.subreddit, vid, E) if sum(c.values()) >= 5 else np.nan
    year = window_counts(items, r.subreddit, T, 0, 365)
    row["top_subs"] = json.dumps([k for k, _ in sorted(year.items(), key=lambda kv: -kv[1])[:10]])
    fc = tc.loc[r.first_comment_id] if r.first_comment_id in tc.index else None
    row["first_utc"] = int(fc.created_utc) if fc is not None else np.nan
    rows.append(row)
B = pd.DataFrame(rows).set_index("idx")

# parent comment of first comments that are replies, and the direct replies received (full
# thread dump; same reply definition as n_replies: AutoModerator and self-replies excluded)
par = nc.first_comment_id.map(tc.parent_id)
B["parent_body"] = [tc.body.get(p[3:]) if isinstance(p, str) and p.startswith("t1_") else None for p in par]
d = tc[tc.parent_id.str.startswith("t1_")].reset_index().sort_values("created_utc")
d = d[d.parent_id.str[3:].isin(set(nc.first_comment_id)) & (d.author != "AutoModerator")]
by_parent = {k: g for k, g in d.groupby(d.parent_id.str[3:])}
ops = nc.post_id.map(posts.author)
B["replies"] = [json.dumps([{"author": a, "body": b, "is_op": int(a == op)}
                            for a, b in zip(g.author, g.body) if a != me] if g is not None else [])
                for g, me, op in ((by_parent.get(c), a, o) for c, a, o in zip(nc.first_comment_id, nc.author, ops))]

keep = ["post_id", "subreddit", "author", "first_body", "first_comment_id", "depth", "top_level",
        "entry_delay_min", "reply_received", "n_replies", "op_replied",
        "return_30d", "return_90d", "return_180d", "regular_90d", "regular_180d", "n_active_months",
        "c2v_big", "c2v_big_placebo", "pp_mini_mean", "jw_d0_30", "jw_d30_90", "jw_d90_365",
        "align_community_entry", "align_community_placebo",
        "got_welcoming", "got_gatekeeping", "got_hostile", "got_explains_context", "genre",
        "log_hist_n", "log_n_subs", "log_hist_len", "log_hist_rate", "hist_eng", "log_delay", "log_prior"]
B = nc[keep].join(B)
B["post_title"] = B.post_id.map(posts.title)
B["post_selftext"] = B.post_id.map(posts.selftext).fillna("").str.slice(0, 600)
B["post_utc"] = B.post_id.map(posts.created_utc).astype(int)
B["post_is_video"] = B.post_id.map(posts.is_video)

# community-level split for RQ3 (train on some subreddits, test on unseen ones)
subs = sorted(B.subreddit.unique())
rng = np.random.default_rng(0)
rng.shuffle(subs)
cut1, cut2 = int(0.6 * len(subs)), int(0.8 * len(subs))
split = {s: "train" if k < cut1 else "val" if k < cut2 else "test" for k, s in enumerate(subs)}
B["split"] = B.subreddit.map(split)

B.to_parquet(BENCH)
print(f"wrote {BENCH}: {B.shape}")
print("split (subreddits / newcomers):",
      {k: (sum(v == k for v in split.values()), int((B.split == k).sum())) for k in ("train", "val", "test")})
print("all three c2v windows (>= 5 items):", int(B[[f"fit_{lo}_{hi}" for lo, hi in WINDOWS]].notna().all(1).sum()))
print("median items per window:", {f"{lo}-{hi}": float(B[f"n_{lo}_{hi}"].median()) for lo, hi in WINDOWS})
print("with replies text:", int((B.replies != "[]").sum()), "| reply_received:", int(B.reply_received.sum()))
print("first comments that are replies, parent found:", int(B.parent_body.notna().sum()), "/", int((B.depth > 0).sum()))
