"""P7 judge (2026-11-06): P(removed) for every generated comment, under the rules the r/science 2016 sticky lists.

Scores every results/p7/rscience-*/gen.parquet that has no judge-<jname>.parquet yet, plus the real first comments of
the same agents (all of which survived moderation: the judge's false-positive check).

    CUDA_VISIBLE_DEVICES=4 python p7_judge.py --model /data/hanqim2/models/Qwen3-32B --jname qwen3-32b --quant fp8
    -> results/p7/rscience-<name>/judge-<jname>.parquet (idx, condition, sample, p_remove, mass)
       results/p7/real/judge-<jname>.parquet"""
import argparse
import glob
import os

import pandas as pd

import rscience_agents as RA
from agent_sim import readout
from common import RESULTS
RETRO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "rscience2016", "retro.parquet")

JSYS = ("You are a moderator of r/science in 2016. Comments in r/science are removed if they are jokes, memes, "
        "abusive, off-topic, or medical advice. Judge only by these rules. Answer only Yes or No.")


def msgs(title, text):
    return [{"role": "system", "content": JSYS},
            {"role": "user", "content": f"Post title: \"{RA.clip(title, 300)}\"\n\nComment: \"{RA.clip(text, 1200)}\"\n\n"
                                        "Would the r/science moderators remove this comment under these rules? "
                                        "Answer Yes or No."}]


def main(a):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model)

    def render(m):
        try:
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            m = [{"role": "user", "content": m[0]["content"] + "\n\n" + m[1]["content"]}]
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    R = pd.read_parquet(RETRO)
    jobs = []
    for f in sorted(glob.glob(os.path.join(RESULTS, "p7", "rscience-*", "gen.parquet"))):
        out = os.path.join(os.path.dirname(f), f"judge-{a.jname}.parquet")
        if not os.path.exists(out):
            jobs.append((pd.read_parquet(f), out))
    real_out = os.path.join(RESULTS, "p7", "real", f"judge-{a.jname}.parquet")
    if not os.path.exists(real_out):
        idx = R[R.n_hist > 0].sample(1000, random_state=0).index
        jobs.append((pd.DataFrame({"idx": idx, "condition": "real", "sample": 0, "text": R.loc[idx, "first_body"].values}),
                     real_out))
    if not jobs:
        print("nothing to judge")
        return
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_util, max_model_len=4096, enable_prefix_caching=True, seed=0,
              quantization=a.quant)
    for S, out in jobs:
        prompts = [render(msgs(R.loc[i, "post_title"], t)) for i, t in zip(S.idx, S.text)]
        ids = [tok(s, add_special_tokens=False).input_ids for s in prompts]
        res = llm.generate([{"prompt_token_ids": x} for x in ids], SamplingParams(max_tokens=1, temperature=0, logprobs=20))
        r = [readout(o.outputs[0].logprobs[0], {"Yes": 1, "No": 0}) for o in res]
        J = S[["idx", "condition", "sample"]].copy()
        J["p_remove"], J["mass"] = [x[0] for x in r], [x[1] for x in r]
        os.makedirs(os.path.dirname(out), exist_ok=True)
        J.to_parquet(out)
        print(out, J.groupby("condition").p_remove.mean().round(3).to_dict(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--jname", required=True)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    main(ap.parse_args())
