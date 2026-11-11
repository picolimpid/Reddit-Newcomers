"""P5 agents (2026-11-06): does the simulated newcomer respond to positive feedback (its first comment's score)
like real newcomers do? r/science 2016 newcomers (outcomes public), sticky as actually seen (arm).

Conditions (within agent; the entry block gets one extra line after the user's own comment):
  arm       sticky as seen, no score line (baseline)
  real      + "By the next day, your comment had N points (upvotes minus downvotes)."  N = the real final score
  low       + same line with N = 1 (no one upvoted it)
  high      + same line with N = 50
`real` lets the agent analysis use exactly the human design (same people, same observed scores, same matching).

    CUDA_VISIBLE_DEVICES=4 python p5_agents.py --model Qwen/Qwen3-8B --name qwen3-8b
    -> results/p5/rscience2016-<name>/scores.parquet"""
import argparse
import os

import pandas as pd

import panel_agents as PA
import rscience_agents as RA
from agent_sim import readout
from common import RESULTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def score_line(n):
    return f"By the next day, your comment had {n} point{'s' if abs(n) != 1 else ''} (upvotes minus downvotes)."


def texts(R, S):
    base = PA.texts(R, "science", RA.STICKY, ["off", "full"])
    base = base[[(c == "full") == bool(R.loc[i, "treat"]) for i, c in zip(base.idx, base.condition)]]  # arm only
    rows = []
    for x in base.itertuples():
        n = int(S.get(x.idx, 1))
        for c, line in [("arm", None), ("real", score_line(n)), ("low", score_line(1)), ("high", score_line(50))]:
            rows.append({"idx": x.idx, "condition": c, "persona": x.persona, "replies": x.replies,
                         "entry": x.entry + ("\n" + line if line else "")})
    return pd.DataFrame(rows)


def main(a):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    R = pd.read_parquet(RA.RETRO)
    R = R[R.n_hist > 0].drop(columns=[c for c in R.columns if c.startswith("return_")])
    sc = pd.read_parquet(os.path.join(ROOT, "data", "rscience2016", "first_scores.parquet"))
    S = R.comment_id.map(sc.set_index("id").score).fillna(1)  # missing (removed) -> the default 1 point
    T = texts(R, S)
    tok = AutoTokenizer.from_pretrained(a.model)

    def render(m):
        try:
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            m = [{"role": "user", "content": m[0]["content"] + "\n\n" + m[1]["content"]}]
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    prompts = [render(PA.messages(x, "science", "base")) for x in T.itertuples()]
    print(len(T), prompts[1][-600:], flush=True)
    kw = dict(tokenizer_mode="mistral", config_format="mistral", load_format="mistral") if a.mistral else {}
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_util, max_model_len=8192, enable_prefix_caching=True,
              seed=0, quantization=a.quant, tensor_parallel_size=a.tp, **kw)
    outs = llm.generate([{"prompt_token_ids": tok(s, add_special_tokens=False).input_ids} for s in prompts],
                        SamplingParams(max_tokens=1, temperature=0, logprobs=20))
    r = [readout(o.outputs[0].logprobs[0], {"Yes": 1, "No": 0}) for o in outs]
    T["p"], T["mass"] = [x[0] for x in r], [x[1] for x in r]
    out = os.path.join(RESULTS, "p5", f"rscience2016-{a.name}")
    os.makedirs(out, exist_ok=True)
    T.drop(columns=["persona", "entry", "replies"]).to_parquet(os.path.join(out, "scores.parquet"))
    print(T.groupby("condition").p.mean().round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--mistral", action="store_true")
    main(ap.parse_args())
