"""Diagnose a model whose Yes/No readout mass is 0: greedy-generate 12 tokens for 3 real panel prompts and print the
top-20 first-token candidates. Usage: python debug_firsttok.py --model allenai/Olmo-3.1-32B-Instruct [--quant fp8]"""
import argparse
import random

import pandas as pd

import rscience_agents as RA
from agent_sim import SYSTEM, persona, replies


def main(a):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model)
    R = pd.read_parquet(RA.RETRO)
    R = R[R.n_hist > 0].sample(3, random_state=0)
    ps = []
    for i, r in R.iterrows():
        u = f"{persona(r, random.Random(int(i)))}\n\n{RA.entry(r, None)}\n\n{replies(r, 'observed', int(i))}\n\n# Question\n{RA.QUESTION}"
        ps.append(tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": u}],
                                          tokenize=False, add_generation_prompt=True))
    print(repr(ps[0][-300:]), flush=True)
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_util, max_model_len=8192, seed=0, quantization=a.quant,
              tensor_parallel_size=a.tp)
    outs = llm.generate([{"prompt_token_ids": tok(s, add_special_tokens=False).input_ids} for s in ps],
                        SamplingParams(max_tokens=12, temperature=0, logprobs=20))
    for o in outs:
        print("GEN:", repr(o.outputs[0].text))
        print("TOP1:", sorted(((round(lp.logprob, 2), lp.decoded_token) for lp in o.outputs[0].logprobs[0].values()),
                               reverse=True)[:20], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--gpu-util", type=float, default=0.88)
    ap.add_argument("--tp", type=int, default=1)
    main(ap.parse_args())
