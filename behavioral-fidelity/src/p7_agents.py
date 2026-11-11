"""P7 (2026-11-06): compliance, the one non-zero human effect of the r/science 2016 sticky, simulated by generation.

Human (CivilServant data, non-AMA posts): visible share of newcomer comments 0.429 (control) -> 0.504 (sticky), i.e.
violation logit about -0.30. Agents: the same r/science 2016 newcomers (persona = their real past year) open the same
post, see the sticky or not, and write the comment they would post. A judge (p7_judge.py) then scores each comment
for removal under the rules the sticky itself lists. Within agent, conditions share the sampling seed (common random
numbers), so only the pinned text differs.

    CUDA_VISIBLE_DEVICES=4 python p7_agents.py --model /data/hanqim2/models/Qwen3-8B --name qwen3-8b
    -> results/p7/rscience-<name>/gen.parquet  (idx, condition, sample, text, n_tokens)
Caveat: the agents' real first comments all survived moderation (removed comments lose their author), so the agent
population is selected on compliance; the comparison is of direction and rough size, not of levels."""
import argparse
import os
import random

import pandas as pd

import rscience_agents as RA
from agent_sim import persona
from common import RESULTS
RETRO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "rscience2016", "retro.parquet")

GEN_SYSTEM = ("You are role-playing one specific, real Reddit user. You are shown this user's own Reddit activity "
              "from the past year and then a post they just opened on Reddit. Write exactly as this particular user "
              "would actually write, not as a helpful assistant and not as an average person. Reply with only the "
              "comment text.")
CONDS = ["off", "full", "rules"]


def prompt(r, k, sticky):
    lines = [persona(r, random.Random(k)), "", "# What just happened",
             f"You opened a post in r/science: \"{RA.clip(r.post_title, 300)}\""]
    if str(r.post_selftext).strip():
        lines.append(f"Post text: \"{RA.clip(r.post_selftext, 400)}\"")
    if sticky:
        lines.append(f"The first comment in the thread, pinned by the r/science moderators, said: \"{sticky}\"")
    lines += ["You had not posted or commented in r/science at any point in the past year.",
              "You decide to write a comment on this post.", "", "# Task",
              "Write the comment you post on this thread. Reply with only the comment text."]
    return [{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def main(a):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    out = os.path.join(RESULTS, "p7", f"rscience-{a.name}")
    if os.path.exists(os.path.join(out, "gen.parquet")):
        print(f"skip {out} (exists)")
        return
    tok = AutoTokenizer.from_pretrained(a.model)

    def render(m):
        try:
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            m = [{"role": "user", "content": m[0]["content"] + "\n\n" + m[1]["content"]}]
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    R = pd.read_parquet(RETRO)
    R = R[R.n_hist > 0].sample(a.limit, random_state=0)
    rows, prompts, params = [], [], []
    for i, r in R.iterrows():
        k = int(i)
        for c in CONDS:
            p = render(prompt(r, k, RA.STICKY[c]))
            for s in range(a.samples):
                rows.append({"idx": i, "condition": c, "sample": s})
                prompts.append(p)
                params.append(SamplingParams(max_tokens=200, temperature=1.0, top_p=0.95, seed=k * 10 + s))
    print(f"{a.name}: {len(prompts)} generations for {len(R)} agents -> {out}", flush=True)
    print(prompts[1][-700:], flush=True)
    kw = dict(tokenizer_mode="mistral", config_format="mistral", load_format="mistral") if a.mistral else {}
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_util, max_model_len=a.max_len, enable_prefix_caching=True,
              seed=0, quantization=a.quant, tensor_parallel_size=a.tp, **kw)
    ids = [tok(s, add_special_tokens=False).input_ids for s in prompts]
    outs = llm.generate([{"prompt_token_ids": x} for x in ids], params)
    S = pd.DataFrame(rows)
    S["text"] = [o.outputs[0].text.strip() for o in outs]
    S["n_tokens"] = [len(o.outputs[0].token_ids) for o in outs]
    os.makedirs(out, exist_ok=True)
    S.to_parquet(os.path.join(out, "gen.parquet"))
    for c in CONDS:
        print(c, "|", S[S.condition == c].text.iloc[0][:300].replace("\n", " "), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--mistral", action="store_true")
    main(ap.parse_args())
