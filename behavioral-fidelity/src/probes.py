"""Probe experiments on the welcome effect (2026-11-02). Same agents, persona and entry event as
agent_sim.py, intent protocol only; only the reply shown, the outcome community or the
outsider line changes, so every contrast is within-agent.

  ladder   which part of the template welcome moves the agent?
           none | ack | disagree | praise | praise_welcome | praise_invite |
           praise_welcome_invite | template (the original agent_sim welcoming template)
  nc       negative-control outcome (Lin et al. 2026, user drift): the same question about
           r/Y, a random target community the user has no activity in. A reply in r/X should
           not move the intent to post in r/Y. Conditions: none | ack | praise_welcome_invite | template
  outsider the entry text without "You had not posted or commented in r/X at any point in the
           past year." Conditions: none | template
  readout  (--mode readout) 400-agent subsample, conditions none | ack | template:
           Qwen3 thinking mode with a parsed final Yes/No, and a verbalized 0-100 probability

    CUDA_VISIBLE_DEVICES=0 python probes.py --model Qwen/Qwen3-8B --out qwen3-8b
    CUDA_VISIBLE_DEVICES=0 python probes.py --model Qwen/Qwen3-8B --out qwen3-8b --mode readout"""
import argparse
import json
import os
import random
import re

import numpy as np
import pandas as pd

from agent_sim import REPLIERS, SYSTEM, TEMPLATES, clip, entry, persona, readout
from common import BENCH, RESULTS

LADDER = {
    "ack": ["Fair enough.", "Huh, interesting.", "Yeah, I can see that."],
    "disagree": ["Nah, I don't think that's right.", "I disagree, but okay.", "That's not really how it works."],
    "praise": ["Great comment, this is spot on.", "Ha, love this.", "This is exactly right, well put."],
    "praise_welcome": ["Great comment, this is spot on. Welcome to r/{sub}!",
                       "Ha, love this. Don't think I've seen you around r/{sub} before, welcome!",
                       "This is exactly right, well put. Welcome to r/{sub}!"],
    "praise_invite": ["Great comment, this is spot on. You should stick around, we get threads like this all the time.",
                      "Ha, love this. Hope you stick around.",
                      "This is exactly right, well put. Hope we see more comments from you."],
    "praise_welcome_invite": ["Great comment, this is spot on. Welcome to r/{sub}! Stick around, we get threads like this all the time.",
                              "Ha, love this. Don't think I've seen you around r/{sub} before, welcome! Hope you stay.",
                              "This is exactly right, well put. Welcome to r/{sub}! Hope we see more of you here."],
    "template": TEMPLATES["welcoming"],
}
OUTSIDER = "You had not posted or commented in r/{sub} at any point in the past year.\n"


def reply_block(cond, sub, k):
    if cond == "none":
        return "Nobody replied to your comment."
    return f"Replies to your comment:\n- {REPLIERS[k % 3]}: \"{clip(LADDER[cond][k % 3].format(sub=sub), 300)}\""


def ask(sub):
    return (f"Over the next 3 months, will you post or comment in r/{sub} again "
            "(anywhere other than this thread)? Answer Yes or No.")


def ask_other(sub):  # negative control: no "again", the user has never been there
    return f"Over the next 3 months, will you post or comment in r/{sub}? Answer Yes or No."


def ask_verbal(sub):
    return (f"How likely is it that you will post or comment in r/{sub} again over the next 3 months "
            "(anywhere other than this thread)? Answer with a number from 0 to 100 only.")


def control_sub(r, subs):
    """a target community (one of the benchmark's 83) the user had no activity in during the past year"""
    seen = {x["sub"].lower() for x in json.loads(r.history)} | {r.subreddit.lower()}
    pool = [s for s in subs if s.lower() not in seen]
    return random.Random(20_000 + int(r.name)).choice(pool)


def build(B, tok, specs, question=ask, thinking=False):
    subs = sorted(pd.read_parquet(BENCH, columns=["subreddit"]).subreddit.unique())  # all 83 target communities
    rows = []
    for i, r in B.iterrows():
        k = int(i)
        p = persona(r, random.Random(k))
        e_full = entry(r)
        y = control_sub(r, subs)
        for exp, cond, outcome, drop_outsider in specs:
            e = e_full.replace(OUTSIDER.format(sub=r.subreddit), "") if drop_outsider else e_full
            target = y if outcome == "Y" else r.subreddit
            q = (ask_other(target) if outcome == "Y" else question(target))
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"{p}\n\n{e}\n\n{reply_block(cond, r.subreddit, k)}\n\n# Question\n{q}"}]
            rows.append({"idx": i, "exp": exp, "condition": cond, "outcome_sub": target,
                         "prompt": tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                           enable_thinking=thinking)})
    return pd.DataFrame(rows)


MAIN_SPECS = ([("ladder", c, "X", False) for c in ["none"] + list(LADDER)]
              + [("nc", c, "Y", False) for c in ["none", "ack", "praise_welcome_invite", "template"]]
              + [("outsider", c, "X", True) for c in ["none", "template"]])
READOUT_SPECS = [("readout", c, "X", False) for c in ["none", "ack", "template"]]


def final_answer(text):
    """Yes/No after the thinking block (None if missing)"""
    tail = text.split("</think>")[-1]
    m = re.findall(r"\b(yes|no)\b", tail, flags=re.I)
    return None if not m else int(m[0].lower() == "yes")


def number(text):
    m = re.search(r"\d+(\.\d+)?", text.split("</think>")[-1])
    return float(m.group()) if m else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default="qwen3-8b")
    ap.add_argument("--mode", choices=["main", "readout"], default="main")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-readout", type=int, default=400)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--quant", default=None, help="e.g. fp8 (weight-only on Ampere) to fit 32B on one GPU")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model)
    B = pd.read_parquet(BENCH)
    if a.mode == "readout":
        B = B.sample(a.n_readout, random_state=0).sort_index()
    elif a.limit:
        B = B.sample(a.limit, random_state=0).sort_index()
    out = os.path.join(RESULTS, "probes", a.out)
    os.makedirs(out, exist_ok=True)
    if a.tp > 1:
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_util, max_model_len=a.max_len, tensor_parallel_size=a.tp,
              disable_custom_all_reduce=a.tp > 1, enable_prefix_caching=True, seed=0, quantization=a.quant)
    ids = lambda s: {"prompt_token_ids": tok(s, add_special_tokens=False).input_ids}  # noqa: E731

    if a.mode == "main":
        P = build(B, tok, MAIN_SPECS)
        U = P.drop_duplicates("prompt")[["prompt"]].reset_index(drop=True)
        print(f"{len(P)} prompts, {len(U)} unique, {len(B)} agents", flush=True)
        outs = llm.generate([ids(s) for s in U.prompt], SamplingParams(max_tokens=1, temperature=0, logprobs=20))
        lp_of = dict(zip(U.prompt, (o.outputs[0].logprobs[0] for o in outs)))
        res = [readout(lp_of[s], {"Yes": 1, "No": 0}) for s in P.prompt]
        P["p"], P["mass"] = [x[0] for x in res], [x[1] for x in res]
        P.drop(columns="prompt").to_parquet(os.path.join(out, "scores.parquet"))
        print(P.groupby(["exp", "condition"]).agg(p=("p", "mean"), mass=("mass", "mean")).round(4))
        return

    # readout mode: (1) first-token P (non-thinking), (2) verbalized probability (non-thinking),
    # (3) thinking mode, 4 samples per prompt at Qwen's recommended thinking settings
    R = []
    P1 = build(B, tok, READOUT_SPECS)
    o1 = llm.generate([ids(s) for s in P1.prompt], SamplingParams(max_tokens=1, temperature=0, logprobs=20))
    P1["value"] = [readout(o.outputs[0].logprobs[0], {"Yes": 1, "No": 0})[0] for o in o1]
    R.append(P1.assign(readout="first_token"))
    P2 = build(B, tok, READOUT_SPECS, question=ask_verbal)
    o2 = llm.generate([ids(s) for s in P2.prompt], SamplingParams(max_tokens=6, temperature=0))
    P2["text"] = [o.outputs[0].text for o in o2]
    P2["value"] = P2.text.map(number) / 100
    R.append(P2.assign(readout="verbalized"))
    P3 = build(B, tok, READOUT_SPECS, thinking=True)
    o3 = llm.generate([ids(s) for s in P3.prompt],
                      SamplingParams(max_tokens=2048, temperature=0.6, top_p=0.95, top_k=20, n=4, seed=0))
    P3 = P3.loc[P3.index.repeat(4)].reset_index(drop=True)
    P3["text"] = [c.text for o in o3 for c in o.outputs]
    P3["value"] = P3.text.map(final_answer)
    P3["n_gen_tokens"] = [len(c.token_ids) for o in o3 for c in o.outputs]
    R.append(P3.assign(readout="thinking"))
    R = pd.concat(R, ignore_index=True).drop(columns="prompt")
    R.to_parquet(os.path.join(out, "readout.parquet"))
    print(R.groupby(["readout", "condition"]).value.agg(["mean", lambda s: s.isna().mean()]).round(3))


if __name__ == "__main__":
    main()
