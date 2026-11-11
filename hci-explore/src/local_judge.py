"""Local judge for open-weight models through vLLM.  Open-weight models are run locally, never through an API
(user instruction, 2026-11-03).  Same input/output format as src/llm_judge.py, so the E1/E2/E3 analysis scripts pick
the output up as one more judge.

Input rows:  {"id", "user", "system"?, "options": ["Yes","No"] | ["1",...,"5"]}
Output:      <input dir>/llm_<tag>.jsonl, rows {"id", "probs", "mass", "prompt_tokens"}
             <input dir>/llm_<tag>_latency.json, batch-size-1 latency on ~30 prompts, measured before the batch run
Probabilities: the first generated token's top-20 logprobs, summed per option and renormalised (same rule as the API
judge).  Qwen3 runs in non-thinking mode (enable_thinking=False), as the API run did.

usage (Qwen3-32B in bf16 needs two 48 GB GPUs; pick two idle ones from nvidia-smi, never GPU 0 -- it is faulty and
makes CUDA initialisation fail for the whole process):
  CUDA_VISIBLE_DEVICES=1,8 /data/hanqim2/.conda/envs/vllm311/bin/python src/local_judge.py \
      --model /data/hanqim2/models/Qwen3-32B --tp 2 --tag qwen3-32b-local \
      results/e1/qwen_in.jsonl results/e2/qwen_in.jsonl results/e3/llm_in_digit.jsonl
"""
import argparse
import json
import math
import os
import time

from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("inputs", nargs="+")
ap.add_argument("--model", default="/data/hanqim2/models/Qwen3-32B")
ap.add_argument("--tag", default="qwen3-32b-local")
ap.add_argument("--tp", type=int, default=2)
ap.add_argument("--max-model-len", type=int, default=8192)
ap.add_argument("--n-latency", type=int, default=30)
ap.add_argument("--gpu-mem", type=float, default=0.88)
ap.add_argument("--custom-all-reduce", action="store_true",
                help="off by default: with the two GPUs on different CPU sockets it hung after graph capture (2026-11-05)")
a = ap.parse_args()

llm = LLM(a.model, tensor_parallel_size=a.tp, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
          enable_prefix_caching=True, seed=0, disable_custom_all_reduce=not a.custom_all_reduce)
tok = llm.get_tokenizer()
sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)


def render(r):
    msgs = ([{"role": "system", "content": r["system"]}] if r.get("system") else []) + [
        {"role": "user", "content": r["user"]}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def option_probs(o, options):
    cand = {}
    for tid, l in o.outputs[0].logprobs[0].items():
        s = (l.decoded_token if l.decoded_token is not None else tok.decode([tid])).strip().lower()
        for opt in options:
            if s == opt.lower():
                cand[opt] = cand.get(opt, 0.0) + math.exp(l.logprob)
    mass = sum(cand.values())
    return {opt: (cand.get(opt, 0.0) / mass if mass > 0 else 1.0 / len(options)) for opt in options}, mass


llm.generate([render({"user": "Say yes."})], sp, use_tqdm=False)  # warm-up (CUDA graphs, kernels)
for inp in a.inputs:
    rows = [json.loads(l) for l in open(inp)]
    prompts = [render(r) for r in rows]
    # latency at batch size 1 first, before the batch run puts these prompts in the prefix cache
    lat = []
    for p in prompts[:: max(1, len(prompts) // a.n_latency)][: a.n_latency]:
        t0 = time.time()
        llm.generate([p], sp, use_tqdm=False)
        lat.append(time.time() - t0)
    lat.sort()
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    wall = time.time() - t0
    outp = os.path.join(os.path.dirname(inp), f"llm_{a.tag}.jsonl")
    masses = []
    with open(outp, "w") as f:
        for r, o in zip(rows, outs):
            probs, mass = option_probs(o, r["options"])
            masses.append(mass)
            f.write(json.dumps({"id": r["id"], "probs": probs, "mass": mass,
                                "prompt_tokens": len(o.prompt_token_ids)}) + "\n")
    info = {"model": a.model, "tp": a.tp, "gpus": os.environ.get("CUDA_VISIBLE_DEVICES"), "n": len(rows),
            "latency_median_s": lat[len(lat) // 2], "latency_p90_s": lat[int(len(lat) * 0.9)], "n_latency": len(lat),
            "batch_wall_s": wall, "mass_median": sorted(masses)[len(masses) // 2],
            "mass_below_0.5": sum(m < 0.5 for m in masses) / len(masses)}
    json.dump(info, open(outp.replace(".jsonl", "_latency.json"), "w"), indent=1)
    print(outp, json.dumps(info))
