"""Retrodiction of a field experiment: agents for the newcomers of the Matias (2019) r/science
sticky-comment experiment (data from rscience_retro_collect.py). (2026-11-03)

Each newcomer becomes an agent exactly as in the benchmark (agent_sim.persona, the same system
prompt and question), with an r/science entry block: the post, the moderators' pinned comment
when the condition shows it, the user's own comment and the replies the user actually received.
Sticky conditions (all with the observed replies):
  off        no pinned comment
  full       the real treatment text ("Welcome to r/science! Comments will be removed if ...")
  welcome    only "Welcome to r/science!"
  rules      only the removal warning and the moderators line, without "Welcome to r/science!"
  off+tmpl   no pinned comment, plus the benchmark's template welcome reply (scale reference)
`arm` (what the person actually saw) is `full` for treated posts and `off` for control posts.

    CUDA_VISIBLE_DEVICES=9 python rscience_agents.py --model Qwen/Qwen3-8B --out rscience-qwen3-8b
    python3 rscience_agents.py --jev --out rscience-jev      # Jev System One (budget-capped API)"""
import argparse
import json
import os
import random
import sys

import pandas as pd

from agent_sim import REPLIERS, SYSTEM, TEMPLATES, clip, persona, readout, replies
from common import RESULTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RETRO = os.path.join(ROOT, "data", "rscience2016", "retro.parquet")
RULES = ("Comments will be removed if they are jokes, memes, abusive, off-topic, or medical advice (rules). "
         "Our ~1200 moderators encourage respectful discussion.")
STICKY = {"off": None, "full": "Welcome to r/science! " + RULES, "welcome": "Welcome to r/science!", "rules": RULES}
CONDS = ["off", "full", "welcome", "rules", "off+tmpl"]
QUESTION = ("Over the next 3 months, will you post or comment in r/science again "
            "(anywhere other than this thread)? Answer Yes or No.")


def entry(r, sticky):
    hours = max(r.entry_delay_min, 0) / 60
    when = f"{hours:.0f} hours" if hours >= 1.5 else f"{max(hours * 60, 1):.0f} minutes"
    lines = ["# What just happened", f"You opened a post in r/science: \"{clip(r.post_title, 300)}\""]
    if str(r.post_selftext).strip():
        lines.append(f"Post text: \"{clip(r.post_selftext, 400)}\"")
    if sticky:
        lines.append(f"The first comment in the thread, pinned by the r/science moderators, said: \"{sticky}\"")
    lines.append("You had not posted or commented in r/science at any point in the past year.")
    if r.top_level == 0 and r.parent_body:
        lines.append(f"{when} after it was posted, you replied to another user's comment "
                     f"\"{clip(r.parent_body, 300)}\" with:")
    else:
        lines.append(f"{when} after it was posted, you commented on the post:")
    lines.append(f"\"{clip(r.first_body, 500)}\"")
    return "\n".join(lines)


def texts(R):
    rows = []
    for i, r in R.iterrows():
        k = int(i)
        p = persona(r, random.Random(k))
        for c in CONDS:
            rep = replies(r, "observed", k)
            if c == "off+tmpl":
                t = f"- {REPLIERS[k % 3]}: \"{TEMPLATES['welcoming'][k % 3].format(sub='science')}\""
                rep = (rep + "\n" + t) if rep.startswith("Replies") else "Replies to your comment:\n" + t
            rows.append({"idx": i, "condition": c, "persona": p, "entry": entry(r, STICKY.get(c.split("+")[0])),
                         "replies": rep})
    return pd.DataFrame(rows)


def run_vllm(T, a):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model)
    T["prompt"] = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"{x.persona}\n\n{x.entry}\n\n{x.replies}\n\n# Question\n{QUESTION}"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False) for x in T.itertuples()]
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_util, max_model_len=8192, enable_prefix_caching=True,
              seed=0, quantization=a.quant)
    outs = llm.generate([{"prompt_token_ids": tok(s, add_special_tokens=False).input_ids} for s in T.prompt],
                        SamplingParams(max_tokens=1, temperature=0, logprobs=20))
    res = [readout(o.outputs[0].logprobs[0], {"Yes": 1, "No": 0}) for o in outs]
    T["p"], T["mass"] = [x[0] for x in res], [x[1] for x in res]
    return T.drop(columns="prompt")


def run_jev(T, a):
    from concurrent.futures import ThreadPoolExecutor
    sys.path.insert(0, "/data/hanqim2/Reddit Newcomers/src")
    import jev
    T = T[T.condition.isin(a.jev_conds.split(","))].copy()
    jobs = [({"your_reddit_activity": x.persona, "what_just_happened": x.entry, "replies_to_your_comment": x.replies},
             f"# Question\n{QUESTION}") for x in T.itertuples()]
    q = {"ret": {"type": "noul", "instructions": f"{SYSTEM}\n\n" + "{}"}}
    print(f"{len(jobs)} Jev calls; spent so far ${jev.spent()[1]:.3f} of ${jev.BUDGET}", flush=True)

    def score(j):
        out = jev.ask(j[0], {"ret": {"type": "noul", "instructions": q["ret"]["instructions"].format(j[1])}},
                      tag="rscience")
        return out["answers"]["ret"]["noul"]
    key = [json.dumps(j, sort_keys=True) for j in jobs]
    uniq = dict(zip(key, jobs))
    with ThreadPoolExecutor(8) as ex:
        got = dict(zip(uniq, ex.map(score, uniq.values())))
    T["p"] = [got[k] for k in key]
    print(f"spent ${jev.spent()[1]:.3f}")
    return T


def main(a):
    R = pd.read_parquet(RETRO)
    R = R[R.n_hist > 0]
    if a.limit:
        R = R.sample(a.limit, random_state=0)
    T = texts(R)
    print(f"{len(T)} prompts for {len(R)} newcomer agents", flush=True)
    S = run_jev(T, a) if a.jev else run_vllm(T, a)
    S = S.drop(columns=["persona", "entry", "replies"])
    out = os.path.join(RESULTS, "probes", a.out)
    os.makedirs(out, exist_ok=True)
    S.to_parquet(os.path.join(out, "scores.parquet"))
    print(S.groupby("condition").p.mean().round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default="rscience-qwen3-8b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--jev", action="store_true")
    ap.add_argument("--jev-conds", default="off,full")
    main(ap.parse_args())
