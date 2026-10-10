"""Stage 9: post-based user profile (professor's definition).

User U's history = the posts U engaged with before Px (the posts U commented on,
plus U's own submissions), represented by those posts' title + body. Similarity is
then post-to-post: <Pa, Pb, Pc, Pd> vs Px. Computed for ALL users with history
(newcomers and existing members), alongside the earlier comment-text profile."""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from api import DATA, get

YEAR = 365 * 86400
DAY = 86400
MIN_POSTS = 5
rng = np.random.default_rng(0)

df = pd.read_parquet(os.path.join(DATA, "analysis_table.parquet"))
posts = pd.read_parquet(os.path.join(DATA, "target_posts.parquet")).set_index("post_id")
subj = df[df.hist_idx.notna()]

# ---------------------------------------------------------------- 1. re-read cached history queries
# identical parameters to stage 3, so these are cache hits (no new requests)
def history(r):
    T = int(posts.at[r.post_id, "created_utc"])
    hc = get("comments/search", author=r.author, after=T - YEAR, before=T, sort="desc",
             limit=100, fields="id,body,subreddit,created_utc,score")
    hp = get("posts/search", author=r.author, after=T - YEAR, before=T, sort="desc",
             limit=100, fields="id,title,selftext,subreddit,created_utc,score")
    return [(c["id"], c["created_utc"]) for c in hc], [(p["id"], p["created_utc"]) for p in hp]


hist = {i: history(r) for i, r in zip(subj.index, subj.itertuples())}
comment_ids = sorted({cid for c, _ in hist.values() for cid, _ in c})
print(len(subj), "users;", len(comment_ids), "history comments", flush=True)

# ---------------------------------------------------------------- 2. comment -> post it was made under
def batch(endpoint, ids, fields):
    chunks = [ids[k:k + 500] for k in range(0, len(ids), 500)]
    with ThreadPoolExecutor(int(os.environ.get("THREADS", 4))) as ex:
        res = ex.map(lambda ch: get(endpoint, ids=",".join(ch), fields=fields), chunks)
    return [x for r in res for x in r]


link = {c["id"]: c["link_id"].split("_")[-1] for c in batch("comments/ids", comment_ids, "id,link_id")}
own_post_ids = {pid for _, p in hist.values() for pid, _ in p}
post_ids = sorted(set(link.values()) | own_post_ids)
print(len(link), "comments resolved;", len(post_ids), "unique engaged posts", flush=True)

# ---------------------------------------------------------------- 3. engaged posts' content
meta = {p["id"]: p for p in batch("posts/ids", post_ids, "id,title,selftext,subreddit")}
print(len(meta), "posts found", flush=True)


def post_text(p):
    body = p.get("selftext") or ""
    body = "" if body in ("[deleted]", "[removed]") else body
    return ((p.get("title") or "") + "\n" + body[:1000]).strip()


# ---------------------------------------------------------------- 4. per-user engaged-post list
eng = {}  # subject index -> list of (post_id, age_days) ; one entry per distinct post
for i, r in zip(subj.index, subj.itertuples()):
    T = int(posts.at[r.post_id, "created_utc"])
    seen = {}
    for cid, t in hist[i][0]:
        pid = link.get(cid)
        if pid:
            seen[pid] = max(seen.get(pid, 0), t)  # most recent engagement with that post
    for pid, t in hist[i][1]:
        seen[pid] = max(seen.get(pid, 0), t)
    keep = [(pid, (T - t) / DAY) for pid, t in seen.items()
            if pid in meta and pid != r.post_id
            and (meta[pid].get("subreddit") or "").lower() != r.subreddit.lower()
            and len(post_text(meta[pid])) >= 5]
    eng[i] = keep
n_posts = pd.Series({i: len(v) for i, v in eng.items()})
print("engaged posts per user: median", n_posts.median(), "| users with >=", MIN_POSTS, ":",
      int((n_posts >= MIN_POSTS).sum()), flush=True)

# ---------------------------------------------------------------- 5. embed + score
uniq = sorted({pid for v in eng.values() for pid, _ in v})
target_ids = list(posts.index)
target_txt = [(posts.at[p, "title"] + "\n" + (posts.at[p, "selftext"] or "")[:1000]).strip() for p in target_ids]
tpos = {p: k for k, p in enumerate(target_ids)}
tsubs = posts.subreddit.values
out = pd.DataFrame(index=subj.index)
out["pp_n_posts"] = n_posts

for name, model_id in (("pp_mini", "sentence-transformers/all-MiniLM-L6-v2"),
                       ("pp_mpnet", "sentence-transformers/all-mpnet-base-v2")):
    model = SentenceTransformer(model_id, device="cuda")
    model.max_seq_length = 256
    E = model.encode([post_text(meta[p]) for p in uniq], batch_size=256, normalize_embeddings=True,
                     show_progress_bar=False)
    upos = {p: k for k, p in enumerate(uniq)}
    P = model.encode(target_txt, normalize_embeddings=True)
    print(name, "embedded", len(uniq), "posts", flush=True)
    for i, r in zip(subj.index, subj.itertuples()):
        if n_posts[i] < MIN_POSTS:
            continue
        H = E[[upos[p] for p, _ in eng[i]]]
        ages = np.array([a for _, a in eng[i]])
        p = P[tpos[r.post_id]]

        def mean_sim(pv, w=None):
            m = np.average(H, axis=0, weights=w)
            return float(m @ pv / np.linalg.norm(m))

        sims = H @ p
        out.at[i, f"{name}_mean"] = mean_sim(p)
        out.at[i, f"{name}_top5"] = float(np.sort(sims)[-5:].mean())
        if name == "pp_mini":
            out.at[i, f"{name}_recency"] = mean_sim(p, 0.5 ** (ages / 30))
            other = rng.choice(np.where(tsubs != r.subreddit)[0])  # validity: random other-subreddit target
            out.at[i, f"{name}_mean_othersub"] = mean_sim(P[other])
            out.at[i, f"{name}_top5_othersub"] = float(np.sort(H @ P[other])[-5:].mean())

out.to_parquet(os.path.join(DATA, "post_alignment.parquet"))
print("saved", out.shape, "| scored users:", int(out.pp_mini_mean.notna().sum()))
