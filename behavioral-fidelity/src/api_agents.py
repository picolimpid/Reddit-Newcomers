"""Agents on API models through OpenRouter (budget-capped client: Reddit Newcomers/src/openrouter.py,
$10 cap). (2026-11-03)

Backends
  chat model (e.g. openai/gpt-4o-mini): exactly the open-model prompt (system = agent_sim.SYSTEM, user =
      persona + entry + replies + question), first output token, P(Yes) from the top-20 logprobs.
  jev: Jev 1.14 (typesafe/jev-1.14) with the jev_probes.py request format (state = persona / entry /
      replies, the system prompt and question in the "noul" instructions).
Tasks
  rscience  the r/science retrodiction agents (rscience_agents.texts); --conds, --limit (random agents)
  probes    the reply-ladder / negative-control probes on the 400-agent Jev sample (jev_probes.SPECS)

    python3 api_agents.py --task rscience --backend openai/gpt-4o-mini --conds off,full --dry
    python3 api_agents.py --task probes --backend jev --out jev-or"""
import argparse
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/hanqim2/Reddit Newcomers/src")
import openrouter as orc  # noqa: E402

from agent_sim import SYSTEM, entry, persona, replies  # noqa: E402
from common import BENCH, RESULTS  # noqa: E402

PRICE = {"jev": 0.042, "openai/gpt-4o-mini": 0.15}  # $/M input, for the dry-run estimate


def rscience_jobs(a):
    from rscience_agents import QUESTION, RETRO, texts
    R = pd.read_parquet(RETRO)
    R = R[R.n_hist > 0]
    if a.limit:
        R = R.sample(a.limit, random_state=0)
    T = texts(R)
    T = T[T.condition.isin(a.conds.split(","))].reset_index(drop=True)
    T["question"] = f"# Question\n{QUESTION}"
    return T


def probe_jobs(a):
    from jev_probes import SPECS
    from probes import ask, ask_other, control_sub, reply_block
    B = pd.read_parquet(BENCH)
    subs = sorted(B.subreddit.unique())
    rows = []
    for i, r in B.sample(a.n, random_state=1).iterrows():
        k = int(i)
        p, e, y = persona(r, random.Random(k)), entry(r), control_sub(r, subs)
        for exp, c, tgt in SPECS:
            rep = replies(r, "observed", k) if c == "observed" else reply_block(c, r.subreddit, k)
            rows.append({"idx": i, "exp": exp, "condition": c, "outcome_sub": r.subreddit if tgt == "X" else y,
                         "persona": p, "entry": e, "replies": rep,
                         "question": f"# Question\n{ask(r.subreddit) if tgt == 'X' else ask_other(y)}"})
    return pd.DataFrame(rows)


def main(a):
    T = rscience_jobs(a) if a.task == "rscience" else probe_jobs(a)
    jev = a.backend == "jev"
    if jev:
        reqs = [({"your_reddit_activity": x.persona, "what_just_happened": x.entry, "replies_to_your_comment": x.replies},
                 {"ret": {"type": "noul", "instructions": f"{SYSTEM}\n\n{x.question}"}}) for x in T.itertuples()]
    else:
        reqs = [(SYSTEM, f"{x.persona}\n\n{x.entry}\n\n{x.replies}\n\n{x.question}") for x in T.itertuples()]
    key = [repr(r) for r in reqs]
    uniq = dict(zip(key, reqs))
    chars = sum(len(repr(r)) for r in uniq.values())
    est = chars / 3.5 * PRICE.get(a.backend, 3.0) / 1e6 * (1.6 if jev else 1.0)  # Jev adds its own overhead
    print(f"{a.task}/{a.backend}: {len(T)} prompts, {len(uniq)} distinct, ~{chars / 3.5 / 1e6:.1f}M tokens, "
          f"est ${est:.2f}; spent so far ${orc.spent():.3f} of ${orc.BUDGET}", flush=True)
    if a.dry:
        return
    tag = f"{a.task}-{a.out}"
    f = ((lambda r: (orc.jev(*r, tag=tag)["answers"]["ret"]["noul"], np.nan)) if jev
         else (lambda r: orc.chat_yes(a.backend, *r, tag=tag)))
    with ThreadPoolExecutor(a.threads) as ex:
        got = dict(zip(uniq, ex.map(f, uniq.values())))
    T["p"], T["mass"] = [got[k][0] for k in key], [got[k][1] for k in key]
    T = T.drop(columns=["persona", "entry", "replies", "question"])
    out = os.path.join(RESULTS, "probes", a.out)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "scores.parquet")
    if os.path.exists(path):  # runs with different --conds / --limit accumulate in one file
        ids = ["idx", "condition"] + (["exp"] if a.task == "probes" else [])
        T = pd.concat([pd.read_parquet(path), T]).drop_duplicates(ids, keep="last").reset_index(drop=True)
    T.to_parquet(path)
    by = ["condition"] if a.task == "rscience" else ["exp", "condition"]
    print(T.groupby(by)[["p", "mass"]].mean().round(4).to_string())
    print(f"spent ${orc.spent():.3f} of ${orc.BUDGET}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["rscience", "probes"], required=True)
    ap.add_argument("--backend", default="openai/gpt-4o-mini", help="an OpenRouter chat model id, or jev")
    ap.add_argument("--out", required=False)
    ap.add_argument("--conds", default="off,full,welcome,rules,off+tmpl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    a.out = a.out or f"{a.task}-{a.backend.split('/')[-1]}"
    main(a)
