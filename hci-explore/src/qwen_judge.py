"""Local baseline judge: Qwen3-8B (non-thinking) via vLLM; first-token probabilities over given options.

Input jsonl rows: {"id": str, "system": str (optional), "user": str, "options": ["Yes", "No"] | ["1",...,"5"]}
Output jsonl rows: {"id": str, "probs": {option: p}} (renormalised over the options; mass on options reported too)
Also times 30 single-prompt requests (batch size 1) to estimate per-decision latency.

usage: CUDA_VISIBLE_DEVICES=0 $PY src/qwen_judge.py in.jsonl out.jsonl
"""
import json
import math
import sys
import time

from vllm import LLM, SamplingParams

MODEL = "/data/hanqim2/models/Qwen3-8B"
inp, outp = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(inp)]
llm = LLM(MODEL, max_model_len=8192, gpu_memory_utilization=0.85, enable_prefix_caching=False)
tok = llm.get_tokenizer()


def render(r):
    msgs = ([{"role": "system", "content": r["system"]}] if r.get("system") else []) + [
        {"role": "user", "content": r["user"]}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


prompts = [render(r) for r in rows]
sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
outs = llm.generate(prompts, sp)

with open(outp, "w") as f:
    for r, o in zip(rows, outs):
        lp = o.outputs[0].logprobs[0]
        cand = {}
        for tid, l in lp.items():
            s = (l.decoded_token or tok.decode([tid])).strip()
            for opt in r["options"]:
                if s.lower() == opt.lower():
                    cand[opt] = cand.get(opt, 0.0) + math.exp(l.logprob)
        mass = sum(cand.values())
        probs = {opt: (cand.get(opt, 0.0) / mass if mass > 0 else 1.0 / len(r["options"])) for opt in r["options"]}
        f.write(json.dumps({"id": r["id"], "probs": probs, "mass": mass, "n_prompt_tokens": len(o.prompt_token_ids)}) + "\n")

# latency at batch size 1 (warm), on a spread of prompts
lat = []
for p in prompts[:: max(1, len(prompts) // 30)][:30]:
    t0 = time.time()
    llm.generate([p], sp, use_tqdm=False)
    lat.append(time.time() - t0)
lat.sort()
json.dump({"median_s": lat[len(lat) // 2], "p90_s": lat[int(len(lat) * 0.9)], "n": len(lat)},
          open(outp.replace(".jsonl", "_latency.json"), "w"))
print("done", len(rows), "median latency", lat[len(lat) // 2])
