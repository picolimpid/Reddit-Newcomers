"""Community2vec-style subreddit embeddings (after Waller & Anderson 2019): subreddits are
similar when the same users are active in both. Co-membership counts -> PPMI -> truncated
SVD; a user's position = activity-weighted mean of the vectors of the subreddits they used."""
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import randomized_svd


def train(user_subs, min_users=10, dim=64, seed=0):
    """user_subs: iterable of {subreddit: count}. Returns (vocab index dict, unit-norm vectors)."""
    sets = [{s.lower() for s in c} for c in user_subs if c]
    freq = {}
    for s in sets:
        for k in s:
            freq[k] = freq.get(k, 0) + 1
    vid = {k: i for i, k in enumerate(sorted(k for k, n in freq.items() if n >= min_users))}
    rows, cols = zip(*[(u, vid[k]) for u, s in enumerate(sets) for k in s if k in vid])
    M = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(sets), len(vid)))
    co = (M.T @ M).toarray().astype(float)
    np.fill_diagonal(co, 0)
    tot = co.sum()
    pr = co.sum(1, keepdims=True) / tot
    ppmi = np.log(np.maximum(co / tot / (pr @ pr.T), 1e-12)).clip(min=0)
    U, S, _ = randomized_svd(ppmi, dim, random_state=seed)
    E = U * np.sqrt(S)
    return vid, E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)


def fit(c, target, vid, E):
    """cosine(activity-weighted mean of the user's subreddit vectors, target subreddit vector);
    the target itself is never part of the user's vector."""
    target = target.lower()
    if not c or target not in vid:
        return np.nan
    lc = {}
    for k, n in c.items():
        lc[k.lower()] = lc.get(k.lower(), 0) + n
    ks = [k for k in lc if k in vid and k != target]
    if not ks:
        return np.nan
    w = np.array([lc[k] for k in ks], float)
    v = (E[[vid[k] for k in ks]] * w[:, None]).sum(0)
    return float(v @ E[vid[target]] / (np.linalg.norm(v) + 1e-12))
