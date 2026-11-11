"""Reply-ladder and negative-control probes on a non-Qwen model: TypeSafe Jev System One. (2026-11-03)

Jev is a structured judgment API: a JSON `state` plus typed questions; a "noul" question returns a
probability-like score in [0, 1], which is read as P(Yes). The agent text is exactly the one the
open models see (persona, entry, reply block from agent_sim / probes); the role-play system prompt
goes into the question instructions. Calls go through the cached, budget-capped client of the
Reddit Newcomers repo (src/jev.py; ledger in its data/jev_ledger.jsonl).

    python3 jev_probes.py --n 500          -> results/probes/jev/scores.parquet"""
import argparse
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, "/data/hanqim2/Reddit Newcomers/src")
import jev  # noqa: E402

from agent_sim import SYSTEM, entry, persona, replies  # noqa: E402
from common import BENCH, RESULTS  # noqa: E402
from probes import ask, ask_other, control_sub, reply_block  # noqa: E402

LADDER = ["none", "ack", "disagree", "praise", "praise_welcome", "praise_invite", "praise_welcome_invite", "template"]
SPECS = ([("ladder", c, "X") for c in LADDER] + [("observed", "observed", "X")]
         + [("nc", c, "Y") for c in ("none", "template")])


def score(job):
    state, q, tag = job
    out = jev.ask(state, {"ret": {"type": "noul", "instructions": f"{SYSTEM}\n\n{q}"}}, tag=tag)
    return out["answers"]["ret"]["noul"], out["usage"]["input_tokens"]


def main(a):
    B = pd.read_parquet(BENCH)
    subs = sorted(B.subreddit.unique())
    H = B.sample(a.n, random_state=1) if a.n else B
    jobs, meta = [], []
    for i, r in H.iterrows():
        k = int(i)
        p, e, y = persona(r, random.Random(k)), entry(r), control_sub(r, subs)
        for exp, c, tgt in SPECS:
            rep = replies(r, "observed", k) if c == "observed" else reply_block(c, r.subreddit, k)
            q = ask(r.subreddit) if tgt == "X" else ask_other(y)
            jobs.append(({"your_reddit_activity": p, "what_just_happened": e, "replies_to_your_comment": rep},
                         f"# Question\n{q}", f"probe_{exp}"))
            meta.append({"idx": i, "exp": exp, "condition": c, "outcome_sub": r.subreddit if tgt == "X" else y})
    print(f"{len(jobs)} calls for {len(H)} agents; spent so far ${jev.spent()[1]:.3f} of ${jev.BUDGET}", flush=True)
    # identical requests (e.g. "observed" for a user nobody replied to == "none") share one cache file;
    # score each distinct request once so threads never read a cache file another thread is writing
    key = [repr(j[:2]) for j in jobs]          # the cache key ignores the tag
    uniq = dict(zip(key, jobs))
    with ThreadPoolExecutor(a.threads) as ex:
        got = dict(zip(uniq, ex.map(score, uniq.values())))
    res = [got[k] for k in key]
    print(f"{len(uniq)} distinct requests", flush=True)
    S = pd.DataFrame(meta)
    S["p"], S["tokens"] = [x[0] for x in res], [x[1] for x in res]
    out = os.path.join(RESULTS, "probes", a.out)
    os.makedirs(out, exist_ok=True)
    S.to_parquet(os.path.join(out, "scores.parquet"))
    print(S.groupby(["exp", "condition"]).p.mean().round(4).to_string())
    print(f"tokens {S.tokens.sum():,}; spent ${jev.spent()[1]:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default="jev")
    ap.add_argument("--threads", type=int, default=8)
    main(ap.parse_args())
