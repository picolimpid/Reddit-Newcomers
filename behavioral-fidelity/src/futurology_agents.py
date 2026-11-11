"""Pre-registered retrodiction, step 2: agents for the newcomers of the r/Futurology 2020 sticky-comment
experiment (inputs from futurology_collect.py collect; no outcome is read here). (2026-11-03)

Identical to rscience_agents.py except the subreddit and the sticky text (PREREG_futurology.md §2.4):
  off        no pinned comment
  full       the real treatment text (markdown removed)
  welcome    only "Welcome to /r/Futurology!"
  rules      the rest of the treatment text
  off+tmpl   no pinned comment, plus the benchmark's template welcome reply
Backends: vLLM (--model, first-token P(Yes)); OpenRouter chat model (--api openai/gpt-4o-mini); Jev (--api jev).
API models score every newcomer in off/full and a random 1,000 (seed 0) in the other conditions.

    CUDA_VISIBLE_DEVICES=2 python futurology_agents.py --model Qwen/Qwen3-8B --out futurology-qwen3-8b
    python3 futurology_agents.py --api openai/gpt-4o-mini --out futurology-gpt-4o-mini"""
import argparse
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from agent_sim import REPLIERS, SYSTEM, TEMPLATES, clip, persona, readout, replies
from common import RESULTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS = os.path.join(ROOT, "data", "futurology2020", "inputs.parquet")
SUB = "Futurology"
WELCOME = "Welcome to /r/Futurology!"
RULES = ("To maintain a healthy, vibrant community, comments will be removed if they are disrespectful, off-topic, "
         "or spread misinformation (rules). While thousands of people comment daily and follow the rules, mods do "
         "remove a few hundred comments per day. Replies to this announcement are auto-removed.")
STICKY = {"off": None, "full": f"{WELCOME} {RULES}", "welcome": WELCOME, "rules": RULES}
CONDS = ["off", "full", "welcome", "rules", "off+tmpl"]
QUESTION = (f"Over the next 3 months, will you post or comment in r/{SUB} again "
            "(anywhere other than this thread)? Answer Yes or No.")


def entry(r, sticky):  # = rscience_agents.entry with the subreddit changed
    hours = max(r.entry_delay_min, 0) / 60
    when = f"{hours:.0f} hours" if hours >= 1.5 else f"{max(hours * 60, 1):.0f} minutes"
    lines = ["# What just happened", f"You opened a post in r/{SUB}: \"{clip(r.post_title, 300)}\""]
    if str(r.post_selftext).strip():
        lines.append(f"Post text: \"{clip(r.post_selftext, 400)}\"")
    if sticky:
        lines.append(f"The first comment in the thread, pinned by the r/{SUB} moderators, said: \"{sticky}\"")
    lines.append(f"You had not posted or commented in r/{SUB} at any point in the past year.")
    if r.top_level == 0 and r.parent_body:
        lines.append(f"{when} after it was posted, you replied to another user's comment "
                     f"\"{clip(r.parent_body, 300)}\" with:")
    else:
        lines.append(f"{when} after it was posted, you commented on the post:")
    lines.append(f"\"{clip(r.first_body, 500)}\"")
    return "\n".join(lines)


def texts(R, conds_for):
    rows = []
    for i, r in R.iterrows():
        k = int(i)
        p = persona(r, random.Random(k))
        for c in conds_for(i):
            rep = replies(r, "observed", k)
            if c == "off+tmpl":
                t = f"- {REPLIERS[k % 3]}: \"{TEMPLATES['welcoming'][k % 3].format(sub=SUB)}\""
                rep = (rep + "\n" + t) if rep.startswith("Replies") else "Replies to your comment:\n" + t
            rows.append({"idx": i, "condition": c, "persona": p, "entry": entry(r, STICKY.get(c.split("+")[0])),
                         "replies": rep})
    return pd.DataFrame(rows)


def run_vllm(T, a):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"{x.persona}\n\n{x.entry}\n\n{x.replies}\n\n# Question\n{QUESTION}"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False) for x in T.itertuples()]
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_util, max_model_len=8192, enable_prefix_caching=True,
              seed=0, quantization=a.quant)
    outs = llm.generate([{"prompt_token_ids": tok(s, add_special_tokens=False).input_ids} for s in prompts],
                        SamplingParams(max_tokens=1, temperature=0, logprobs=20))
    res = [readout(o.outputs[0].logprobs[0], {"Yes": 1, "No": 0}) for o in outs]
    T["p"], T["mass"] = [x[0] for x in res], [x[1] for x in res]
    return T


def run_api(T, a):
    sys.path.insert(0, "/data/hanqim2/Reddit Newcomers/src")
    import openrouter as orc
    tag = f"futurology-{a.out}"
    if a.api == "jev":
        reqs = [({"your_reddit_activity": x.persona, "what_just_happened": x.entry, "replies_to_your_comment": x.replies},
                 {"ret": {"type": "noul", "instructions": f"{SYSTEM}\n\n# Question\n{QUESTION}"}}) for x in T.itertuples()]
        f = lambda r: (orc.jev(*r, tag=tag)["answers"]["ret"]["noul"], np.nan)  # noqa: E731
    else:
        reqs = [(SYSTEM, f"{x.persona}\n\n{x.entry}\n\n{x.replies}\n\n# Question\n{QUESTION}") for x in T.itertuples()]
        f = lambda r: orc.chat_yes(a.api, *r, tag=tag)  # noqa: E731
    print(f"{len(reqs)} requests; spent so far ${orc.spent():.3f} of ${orc.BUDGET}", flush=True)
    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(f, reqs))
    T["p"], T["mass"] = [x[0] for x in res], [x[1] for x in res]
    print(f"spent ${orc.spent():.3f}")
    return T


def main(a):
    R = pd.read_parquet(INPUTS)
    assert not any(c.startswith("return_") for c in R.columns), "inputs must not contain outcomes"
    R = R[R.n_hist > 0]
    if a.api:
        sub = set(R.sample(min(1000, len(R)), random_state=0).index)
        conds_for = lambda i: CONDS if i in sub else ["off", "full"]  # noqa: E731
    else:
        conds_for = lambda i: CONDS  # noqa: E731
    T = texts(R, conds_for)
    print(f"{len(T)} prompts for {len(R)} newcomer agents", flush=True)
    S = run_api(T, a) if a.api else run_vllm(T, a)
    S = S.drop(columns=["persona", "entry", "replies"])
    out = os.path.join(RESULTS, "futurology", a.out)
    os.makedirs(out, exist_ok=True)
    S.to_parquet(os.path.join(out, "scores.parquet"))
    print(S.groupby("condition").p.mean().round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--api", default=None, help="OpenRouter chat model id, or jev")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--threads", type=int, default=16)
    main(ap.parse_args())
