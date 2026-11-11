"""P9 (2026-11-07): entry, the largest real effect of the r/science 2016 sticky, as a calibration target.

Human (CivilServant, non-AMA posts, n = 2214): newcomer comments per post 6.34 (control) -> 10.57 (sticky),
negative-binomial IRR 1.67 [1.53, 1.82] (Matias 2019 PNAS H3, unadjusted). The sticky is only visible after a post
is opened, so views per post should not depend on the arm and comments ≈ views × P(comment | opened). The agent
analogue is therefore the ratio mean P(Yes | full) / mean P(Yes | off) for "would you comment on this post?".

Agents: the same 1,000 r/science 2016 newcomers as P7 (persona = real past year). Each sees 2 posts drawn (seeded by
agent) from the experiment posts that had a newcomer, never the post they actually commented on, so the agent pool is
not selected on commenting on that post. Conditions as in the panel: off, full, welcome, rules.

    CUDA_VISIBLE_DEVICES=4 python p9_entry.py --model /data/hanqim2/models/Qwen3-8B --name qwen3-8b
    -> results/p9/rscience-<name>/scores.parquet (idx, post_id, condition, p, mass)
Analysis: p9_analyze.py. Plan frozen in PREREG_addendum_P9.md before any P9 score exists."""
import argparse
import os
import random

import pandas as pd

import rscience_agents as RA
from agent_sim import SYSTEM, persona, readout
from common import RESULTS

CONDS = ["off", "full", "welcome", "rules"]
QUESTION = "Would you write a comment on this post? Answer Yes or No."


def entry(post, sticky):
    lines = ["# What just happened", f"You opened a post in r/science: \"{RA.clip(post.post_title, 300)}\""]
    if str(post.post_selftext).strip():
        lines.append(f"Post text: \"{RA.clip(post.post_selftext, 400)}\"")
    if sticky:
        lines.append(f"The first comment in the thread, pinned by the r/science moderators, said: \"{sticky}\"")
    lines.append("You had not posted or commented in r/science at any point in the past year.")
    return "\n".join(lines)


def build(limit=1000, per=2):
    R = pd.read_parquet(RA.RETRO)
    posts = R.drop_duplicates("post_id").set_index("post_id")[["post_title", "post_selftext"]]
    A = R[R.n_hist > 0].sample(limit, random_state=0)
    rows = []
    for i, r in A.iterrows():
        k = int(i)
        rng = random.Random(k)
        pool = [p for p in posts.index if p != r.post_id]
        p_txt = persona(r, random.Random(k))
        for pid in rng.sample(pool, per):
            for c in CONDS:
                rows.append({"idx": i, "post_id": pid, "condition": c, "persona": p_txt,
                             "entry": entry(posts.loc[pid], RA.STICKY[c])})
    return pd.DataFrame(rows)


def main(a):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    out = os.path.join(RESULTS, "p9", f"rscience-{a.name}")
    if os.path.exists(os.path.join(out, "scores.parquet")):
        print(f"skip {out} (exists)")
        return
    T = build(a.limit)
    tok = AutoTokenizer.from_pretrained(a.model)

    def render(m):
        try:
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            m = [{"role": "user", "content": m[0]["content"] + "\n\n" + m[1]["content"]}]
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    prompts = [render([{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": f"{x.persona}\n\n{x.entry}\n\n# Question\n{QUESTION}"}])
               for x in T.itertuples()]
    print(f"{a.name}: {len(prompts)} prompts -> {out}\n{prompts[1][-600:]}", flush=True)
    kw = dict(tokenizer_mode="mistral", config_format="mistral", load_format="mistral") if a.mistral else {}
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_util, max_model_len=8192, enable_prefix_caching=True,
              seed=0, quantization=a.quant, tensor_parallel_size=a.tp, **kw)
    outs = llm.generate([{"prompt_token_ids": tok(s, add_special_tokens=False).input_ids} for s in prompts],
                        SamplingParams(max_tokens=1, temperature=0, logprobs=20))
    res = [readout(o.outputs[0].logprobs[0], {"Yes": 1, "No": 0}) for o in outs]
    T["p"], T["mass"] = [x[0] for x in res], [x[1] for x in res]
    os.makedirs(out, exist_ok=True)
    T.drop(columns=["persona", "entry"]).to_parquet(os.path.join(out, "scores.parquet"))
    print(T.groupby("condition").p.mean().round(4).to_dict(), "mass", round(T.mass.mean(), 3), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--mistral", action="store_true")
    main(ap.parse_args())
