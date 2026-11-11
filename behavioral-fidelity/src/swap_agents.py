"""P8 (2026-11-06): text x population swap. Is a model's direction a property of the model, the people, or the words?

Same agents and prompts as panel_agents.py (base instrument), but each agent sees, in turn, the sticky text of every
CivilServant site (r/science 2016, r/Futurology 2020, r/ffxiv 2019, r/science 2020), with the subreddit name in the
text set to the agent's own community, plus a content-free pinned comment (placebo). Within agent, every condition is
compared with no sticky. If the direction tracks the model whatever the people and words, the simulation is a model
prior, not a community model (cf. community-specific norms, Chandrasekharan et al. CSCW 2018).

    CUDA_VISIBLE_DEVICES=4 python swap_agents.py --pop rscience,futurology --limit 1000 --model ... --name qwen3-8b
    -> results/swap/<pop>-<name>/scores.parquet   (conditions: off, placebo, text_<site> for each site)
Sanity: text_<own site> is the same prompt as the panel's 'full' condition (asserted)."""
import argparse
import os

import pandas as pd

import futurology_agents as FA
import panel_agents as PA
import rscience_agents as RA
import site_collect as SC
from common import RESULTS

SUBS = {"rscience": "science", "futurology": FA.SUB, "ffxiv2019": "ffxiv", "science2020": "science"}
FULL = {"rscience": RA.STICKY["full"], "futurology": FA.STICKY["full"],
        **{s: f"{c['welcome']} {c['rules']}" for s, c in SC.SITES.items()}}
PLACEBO = "This is the comment thread for this post."


def stickies(pop):
    sub = SUBS[pop]
    st = {"off": None, "placebo": PLACEBO}
    for t, txt in FULL.items():
        st[f"text_{t}"] = txt.replace(f"r/{SUBS[t]}", f"r/{sub}")
    return sub, st


def main(a):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from agent_sim import readout
    tok, llm = AutoTokenizer.from_pretrained(a.model), None

    def render(m):
        try:
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            m = [{"role": "user", "content": m[0]["content"] + "\n\n" + m[1]["content"]}] if m[0]["role"] == "system" else m
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    for pop in a.pop.split(","):
        out = os.path.join(RESULTS, "swap", f"{pop}-{a.name}")
        if os.path.exists(os.path.join(out, "scores.parquet")):
            print(f"skip {out} (exists)", flush=True)
            continue
        R, sub0, st0, _, _ = PA.site_spec(pop)
        if pop != "rscience":
            assert not any(c.startswith("return_") for c in R.columns), "inputs must not contain outcomes"
        R = R[R.n_hist > 0]
        R = R.sample(min(a.limit, len(R)), random_state=0) if a.limit else R
        sub, st = stickies(pop)
        assert sub == sub0 and st[f"text_{pop}"] == st0["full"], "own text must equal the panel's full sticky"
        T = PA.texts(R, sub, st, list(st))
        prompts = [render(PA.messages(x, sub, "base")) for x in T.itertuples()]
        print(f"{pop} {a.name}: {len(T)} prompts for {len(R)} agents -> {out}", flush=True)
        if llm is None:
            kw = dict(tokenizer_mode="mistral", config_format="mistral", load_format="mistral") if a.mistral else {}
            llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_util, max_model_len=a.max_len,
                      enable_prefix_caching=True, seed=0, quantization=a.quant, tensor_parallel_size=a.tp, **kw)
        ids = [tok(s, add_special_tokens=False).input_ids for s in prompts]
        outs = llm.generate([{"prompt_token_ids": x} for x in ids], SamplingParams(max_tokens=1, temperature=0, logprobs=20))
        res = [readout(o.outputs[0].logprobs[0], {"Yes": 1, "No": 0}) for o in outs]
        T["p"], T["mass"] = [x[0] for x in res], [x[1] for x in res]
        S = T.drop(columns=["persona", "entry", "replies"])
        os.makedirs(out, exist_ok=True)
        S.to_parquet(os.path.join(out, "scores.parquet"))
        print(S.groupby("condition")[["p", "mass"]].mean().round(4).to_string(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--mistral", action="store_true")
    main(ap.parse_args())
