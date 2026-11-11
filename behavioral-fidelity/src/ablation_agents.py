"""P4 (2026-11-06): which sentence of the sticky drives each model's direction, plus two model-level probes.

1. Sentence ablation, within agent (same agents and prompts as panel_agents.py, only the pinned text changes).
2. --probes (no persona; one prompt each, first-token readout over the listed options):
   knowledge  does the model know the CivilServant / Matias sticky-comment result?
   expert     the model's own aggregate prediction for this sticky ("Higher / Lower / Same" return), to
              compare with what its simulated newcomers do (coherence of expert vs simulated answers).

    CUDA_VISIBLE_DEVICES=4 python ablation_agents.py --site futurology --model Qwen/Qwen3-8B --name qwen3-8b --limit 1000
    -> results/ablation/<site>-<name>/scores.parquet and results/ablation/<site>-<name>/probes.json"""
import argparse
import json
import os

import numpy as np

import panel_agents as PA
from common import RESULTS

PARTS = {  # sticky sentences; the full text is the concatenation of all parts in order
    "rscience": {"W": "Welcome to r/science!",
                 "T": "Comments will be removed if they are jokes, memes, abusive, off-topic, or medical advice (rules).",
                 "M": "Our ~1200 moderators encourage respectful discussion."},
    "futurology": {"W": "Welcome to /r/Futurology!",
                   "P": "To maintain a healthy, vibrant community,",
                   "T": "comments will be removed if they are disrespectful, off-topic, or spread misinformation (rules).",
                   "N": "While thousands of people comment daily and follow the rules, mods do remove a few hundred "
                        "comments per day.",
                   "A": "Replies to this announcement are auto-removed."},
}


def combos(site):
    """leave-one-out and single parts; T without P is capitalized so it reads as a sentence."""
    parts = PARTS[site]
    keys = list(parts)
    out = {"off": None, "all": " ".join(parts.values())}
    for k in keys:
        rest = [x for x in keys if x != k]
        out[f"no{k}"] = " ".join(parts[x] for x in rest)
        out[f"only{k}"] = parts[k]
    if site == "futurology":  # the threat clause on its own must start a sentence
        out["onlyT"] = "Comments" + parts["T"][len("comments"):]
        out["noP"] = out["noP"].replace(" comments will", " Comments will")
        out["onlyP"] = None  # a dangling clause, not a sentence
        out["noT"] = " ".join(parts[x] for x in ("W", "N", "A"))  # dropping T also drops its clause P
        out = {k: v for k, v in out.items() if k == "off" or v}
    return out


def probes(site, sticky_full):
    sub = "science" if site == "rscience" else "Futurology"
    year = 2016 if site == "rscience" else 2020
    return {
        "knowledge": (f"In {year}, the CivilServant project (J. Nathan Matias) ran a randomized field experiment on "
                      f"r/{sub}: some new discussion threads got a pinned comment announcing the community rules. "
                      "What did the study report about the effect of that pinned comment on newcomer participation? "
                      "Answer with one word: Increased, Decreased, or Unchanged.", ["Increased", "Decreased", "Unchanged"]),
        "expert": (f"A large subreddit, r/{sub}, pins this comment as the first comment on half of its new posts: "
                   f"\"{sticky_full}\" Consider people who had never participated in r/{sub} before and who leave a "
                   "comment on one of these posts. Compared with similar newcomers on posts without the pinned comment, "
                   f"will the share who come back to post or comment in r/{sub} again within 3 months be Higher, Lower, "
                   "or Same? Answer with one word: Higher, Lower, or Same.", ["Higher", "Lower", "Same"]),
    }


def readout_opts(lps, opts):
    mass = {o: 0.0 for o in opts}
    for lp in lps.values():
        t = lp.decoded_token.strip().rstrip(".:,").lower()
        for o in opts:
            if t and o.lower().startswith(t) and len(t) >= 2:
                mass[o] += float(np.exp(lp.logprob))
    tot = sum(mass.values())
    return {o: (v / tot if tot else np.nan) for o, v in mass.items()} | {"mass": tot}


def main(a):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from agent_sim import readout
    tok, llm = AutoTokenizer.from_pretrained(a.model), None
    sp = SamplingParams(max_tokens=1, temperature=0, logprobs=20)

    def render(m):
        try:
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            m = [{"role": "user", "content": m[0]["content"] + "\n\n" + m[1]["content"]}] if m[0]["role"] == "system" else m
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)

    def gen(prompts):
        return llm.generate([{"prompt_token_ids": tok(s, add_special_tokens=False).input_ids} for s in prompts], sp)
    for site in a.site.split(","):
        out = os.path.join(RESULTS, "ablation", f"{site}-{a.name}")
        if os.path.exists(os.path.join(out, "scores.parquet")):
            print(f"skip {out}", flush=True)
            continue
        os.makedirs(out, exist_ok=True)
        R, sub, sticky, conds, _ = PA.site_spec(site)
        R = R[R.n_hist > 0]
        R = R.sample(min(a.limit, len(R)), random_state=0) if a.limit else R
        C = combos(site)
        T = PA.texts(R, sub, C, list(C))
        prompts = [render(PA.messages(x, sub, "base")) for x in T.itertuples()]
        P = probes(site, C["all"])
        if llm is None:
            kw = dict(tokenizer_mode="mistral", config_format="mistral", load_format="mistral") if a.mistral else {}
            llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_util, max_model_len=8192, enable_prefix_caching=True,
                      seed=0, quantization=a.quant, tensor_parallel_size=a.tp, **kw)
        got = gen([render([{"role": "user", "content": q}]) for q, _ in P.values()])
        res = {k: readout_opts(o.outputs[0].logprobs[0], opts) for (k, (_, opts)), o in zip(P.items(), got)}
        json.dump(res, open(os.path.join(out, "probes.json"), "w"), indent=1)
        print(site, json.dumps(res), flush=True)
        r = [readout(o.outputs[0].logprobs[0], {"Yes": 1, "No": 0}) for o in gen(prompts)]
        T["p"], T["mass"] = [x[0] for x in r], [x[1] for x in r]
        S = T.drop(columns=["persona", "entry", "replies"])
        S.to_parquet(os.path.join(out, "scores.parquet"))
        lo = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
        W = S.pivot_table(index="idx", columns="condition", values="p")
        print((lo(W).sub(lo(W["off"]), axis=0)).mean().round(2).to_string(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="rscience,futurology", help="comma list of " + ", ".join(PARTS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--mistral", action="store_true")
    main(ap.parse_args())
