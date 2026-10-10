"""Stage 4: build the analysis table.
 - newcomer / existing-member status (3/6/12-month lookbacks)
 - first-comment outcomes from the thread tree
 - user-history controls
 - embedding-based prior-interest alignment (mean, recency-weighted, top-k)
"""
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from api import DATA

DAY = 86400
MIN_HIST = 20
BAD = {"[deleted]", "[removed]", ""}
rng = np.random.default_rng(0)

posts = pd.read_parquet(os.path.join(DATA, "target_posts.parquet")).set_index("post_id")
comments = pd.read_parquet(os.path.join(DATA, "thread_comments.parquet"))
users = pd.read_parquet(os.path.join(DATA, "users.parquet"))
users = users[users["error"].isna()].copy()

# ---------------------------------------------------------------- outcomes
thread_rows = {}
for pid, g in comments.groupby("post_id"):
    g = g.sort_values("created_utc")
    children = defaultdict(list)
    for c in g.itertuples():
        children[c.parent_id].append(c)
    thread_rows[pid] = (g, children)


def subtree_depth(cid, children):
    stack, best = [(cid, 0)], 0
    while stack:
        node, d = stack.pop()
        best = max(best, d)
        for ch in children.get("t1_" + node, []):
            stack.append((ch.id, d + 1))
    return best


def outcomes(pid, author):
    g, children = thread_rows[pid]
    mine = g[g.author == author]
    first = mine.iloc[0]
    replies = [c for c in children.get("t1_" + first.id, [])
               if c.author not in (author, "AutoModerator")]
    T = posts.at[pid, "created_utc"]
    return {
        "first_comment_id": first.id,
        "first_body": first.body,
        "reply_received": int(len(replies) > 0),
        "n_replies": len(replies),
        "reengaged": int((mine.created_utc > first.created_utc).any()),
        "n_own_comments": len(mine),
        "depth": subtree_depth(first.id, children),
        "first_score": first.score,
        "first_len": len(first.body or ""),
        "top_level": int(first.parent_id.startswith("t3_")),
        "entry_delay_min": (first.created_utc - T) / 60,
        "prior_thread_comments": int((g.created_utc < first.created_utc).sum()),
    }


# ---------------------------------------------------------------- histories
rows, hist_texts = [], []
for u in users.itertuples():
    T = int(posts.at[u.post_id, "created_utc"])
    hist = [h for h in json.loads(u.history)
            if h["sub"].lower() != u.subreddit.lower() and h["text"].strip() not in BAD
            and len(h["text"].strip()) >= 20]
    last = u.last_in_sub_utc if pd.notna(u.last_in_sub_utc) else None
    r = {"post_id": u.post_id, "subreddit": u.subreddit, "author": u.author,
         "in_sub_count": u.in_sub_count_capped,
         "n_hist": len(hist)}
    for m, days in (("12m", 365), ("6m", 182), ("3m", 91)):
        r[f"newcomer_{m}"] = int((last is None or last < T - days * DAY) and len(hist) >= MIN_HIST)
    r["existing"] = int(u.in_sub_count_capped >= 3 and len(hist) >= MIN_HIST)
    if len(hist) >= MIN_HIST:
        ts = np.array([h["t"] for h in hist])
        scores = [h["score"] for h in hist if h["kind"] == "c" and h["score"] is not None]
        r.update({
            "hist_n_subs": len({h["sub"].lower() for h in hist}),
            "hist_mean_len": float(np.mean([len(h["text"]) for h in hist])),
            "hist_span_days": max((T - ts.min()) / DAY, 1.0),
            "hist_median_score": float(np.median(scores)) if scores else 0.0,
        })
        r["hist_rate"] = len(hist) / r["hist_span_days"]
        r["hist_idx"] = len(hist_texts)
        hist_texts.append([(h["text"][:1000], (T - h["t"]) / DAY) for h in hist])
    r.update(outcomes(u.post_id, u.author))
    rows.append(r)

df = pd.DataFrame(rows)
print("pairs:", len(df), "| newcomers(12m):", df.newcomer_12m.sum(),
      "| existing:", df.existing.sum(), "| with history:", df.hist_idx.notna().sum())

# ---------------------------------------------------------------- embeddings
post_ids = list(posts.index)
post_txt = [(posts.at[p, "title"] + "\n" + (posts.at[p, "selftext"] or "")[:1000]).strip()
            for p in post_ids]
pidx = {p: i for i, p in enumerate(post_ids)}


def alignment(model_name, mask, prefix, extra=True):
    model = SentenceTransformer(model_name)
    model.max_seq_length = 256
    P = model.encode(post_txt, normalize_embeddings=True, batch_size=64)
    sub = df[mask]
    flat, owner = [], []
    for i, hi in zip(sub.index, sub.hist_idx.astype(int)):
        for txt, age in hist_texts[hi]:
            flat.append(txt)
            owner.append(i)
    print(prefix, "encoding", len(flat), "history items", flush=True)
    # encode in chunks cached on disk so an interrupted run resumes instead of restarting
    cache_dir = os.path.join(DATA, "emb_cache")
    os.makedirs(cache_dir, exist_ok=True)
    chunks, size = [], 20000
    for k, s in enumerate(range(0, len(flat), size)):
        path = os.path.join(cache_dir, f"{prefix}_{len(flat)}_{k}.npy")
        if not os.path.exists(path):
            np.save(path, model.encode(flat[s:s + size], normalize_embeddings=True, batch_size=128,
                                       show_progress_bar=False).astype(np.float16))
            print(f"  {prefix} chunk {k + 1}/{-(-len(flat) // size)}", flush=True)
        chunks.append(np.load(path).astype(np.float32))
    E = np.vstack(chunks)
    owner = np.array(owner)
    subs = posts.subreddit.values
    centroids = {}
    for i in sub.index:
        H = E[owner == i]
        centroids[i] = H.mean(axis=0) / np.linalg.norm(H.mean(axis=0))
        ages = np.array([a for _, a in hist_texts[int(df.at[i, "hist_idx"])]])
        p = P[pidx[df.at[i, "post_id"]]]

        def mean_sim(pv, w=None):
            m = np.average(H, axis=0, weights=w)
            return float(m @ pv / np.linalg.norm(m))

        sims = H @ p
        df.at[i, f"{prefix}_mean"] = mean_sim(p)
        df.at[i, f"{prefix}_top5"] = float(np.sort(sims)[-5:].mean())
        if extra:
            df.at[i, f"{prefix}_recency"] = mean_sim(p, 0.5 ** (ages / 30))
            df.at[i, f"{prefix}_top10"] = float(np.sort(sims)[-10:].mean())
            df.at[i, f"{prefix}_max"] = float(sims.max())
            # validity check: same user vs. a random post from a *different* subreddit
            other = rng.choice(np.where(subs != df.at[i, "subreddit"])[0])
            df.at[i, f"{prefix}_mean_othersub"] = mean_sim(P[other])
            df.at[i, f"{prefix}_top5_othersub"] = float(np.sort(H @ P[other])[-5:].mean())
            # and a random other high-visibility post from the *same* subreddit
            same = np.where((subs == df.at[i, "subreddit"])
                            & (np.array(post_ids) != df.at[i, "post_id"]))[0]
            df.at[i, f"{prefix}_mean_samesub"] = mean_sim(P[rng.choice(same)]) if len(same) else np.nan
    # user history centroids (row index -> vector), reused by the comment-content stage
    np.save(os.path.join(DATA, f"user_centroids_{prefix}.npy"),
            np.vstack([centroids[i] for i in sub.index]).astype(np.float32))
    np.save(os.path.join(DATA, f"user_centroids_{prefix}_idx.npy"), np.array(sub.index))


has_hist = df.hist_idx.notna()
alignment("sentence-transformers/all-MiniLM-L6-v2", has_hist, "mini")
alignment("sentence-transformers/all-mpnet-base-v2", has_hist & (df.newcomer_12m == 1), "mpnet", extra=False)

df.to_parquet(os.path.join(DATA, "analysis_table.parquet"))
print("saved", df.shape)
