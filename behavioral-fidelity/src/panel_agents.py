"""Many-model panel and prompt multiverse for the sticky-comment retrodictions (2026-11-06).

Builds exactly the prompts of rscience_agents.py / futurology_agents.py (asserted below) for any local
vLLM model, so more model families can be scored without touching the frozen Futurology scripts.
Optional --variant changes only the measurement instrument, never the sticky manipulation:
  base      identical to the frozen scripts
  q2        paraphrased question
  nosys     no system message (instructions moved into the user turn)
  order     "Answer No or Yes." (option order)
  observer  third-person prediction instead of role-play
Sites: rscience (2016, real assignment), futurology (2020), and sites built by site_collect.py.

    CUDA_VISIBLE_DEVICES=4 python panel_agents.py --site futurology --model /data/hanqim2/models/gemma-3-12b-it --name gemma3-12b
    ... --mistral   (Ministral-3: mistral tokenizer/config/load format)"""
import argparse
import os
import random
import re

import pandas as pd

import futurology_agents as FA
import rscience_agents as RA
from agent_sim import REPLIERS, SYSTEM, TEMPLATES, persona, readout, replies
from common import RESULTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def site_spec(site):
    """(inputs, subreddit, sticky texts by condition, conditions, output dir prefix)."""
    if site == "rscience":
        R = pd.read_parquet(RA.RETRO)
        return R, "science", RA.STICKY, RA.CONDS, os.path.join(RESULTS, "probes", "rscience-")
    if site == "futurology":
        R = pd.read_parquet(FA.INPUTS)
        return R, FA.SUB, FA.STICKY, FA.CONDS, os.path.join(RESULTS, "futurology", "futurology-")
    import site_collect as SC
    cfg = SC.SITES[site]
    R = pd.read_parquet(os.path.join(SC.D(site), "inputs.parquet"))
    st = {"off": None, "full": f"{cfg['welcome']} {cfg['rules']}", "welcome": cfg["welcome"], "rules": cfg["rules"]}
    return R, cfg["sub"], st, ["off", "full", "welcome", "rules", "off+tmpl"], os.path.join(RESULTS, "sites", site, f"{site}-")


def entry(r, sticky, sub):  # = rscience_agents.entry / futurology_agents.entry with the subreddit as an argument
    hours = max(r.entry_delay_min, 0) / 60
    when = f"{hours:.0f} hours" if hours >= 1.5 else f"{max(hours * 60, 1):.0f} minutes"
    lines = ["# What just happened", f"You opened a post in r/{sub}: \"{RA.clip(r.post_title, 300)}\""]
    if str(r.post_selftext).strip():
        lines.append(f"Post text: \"{RA.clip(r.post_selftext, 400)}\"")
    if sticky:
        lines.append(f"The first comment in the thread, pinned by the r/{sub} moderators, said: \"{sticky}\"")
    lines.append(f"You had not posted or commented in r/{sub} at any point in the past year.")
    if r.top_level == 0 and r.parent_body:
        lines.append(f"{when} after it was posted, you replied to another user's comment "
                     f"\"{RA.clip(r.parent_body, 300)}\" with:")
    else:
        lines.append(f"{when} after it was posted, you commented on the post:")
    lines.append(f"\"{RA.clip(r.first_body, 500)}\"")
    return "\n".join(lines)


def texts(R, sub, sticky, conds):
    rows = []
    for i, r in R.iterrows():
        k = int(i)
        p = persona(r, random.Random(k))
        for c in conds:
            rep = replies(r, "observed", k)
            if c == "off+tmpl":
                t = f"- {REPLIERS[k % 3]}: \"{TEMPLATES['welcoming'][k % 3].format(sub=sub)}\""
                rep = (rep + "\n" + t) if rep.startswith("Replies") else "Replies to your comment:\n" + t
            rows.append({"idx": i, "condition": c, "persona": p, "entry": entry(r, sticky.get(c.split("+")[0]), sub),
                         "replies": rep})
    return pd.DataFrame(rows)


def question(sub, variant):
    if variant == "q2":
        return (f"In the next three months, do you think you will come back and take part in r/{sub} "
                "(post, or comment somewhere other than this thread)? Reply with Yes or No.")
    if variant == "observer":
        return (f"Over the next 3 months, will this user post or comment in r/{sub} again "
                "(anywhere other than this thread)? Answer Yes or No.")
    q = (f"Over the next 3 months, will you post or comment in r/{sub} again "
         "(anywhere other than this thread)? Answer Yes or No.")
    return q.replace("Answer Yes or No.", "Answer No or Yes.") if variant == "order" else q


OBS_SYSTEM = ("You are an expert at predicting the behavior of Reddit users. You are shown one specific, real "
              "Reddit user's activity from the past year and then something that just happened to them on Reddit. "
              "Predict what this particular user will actually do, not what an average person would do. "
              "Reply with only the requested answer.")
THIRD = [("# Your Reddit activity in the past year", "# This user's Reddit activity in the past year"),
         ("Some of your comments and posts", "Some of this user's comments and posts"),
         ("You opened a post", "The user opened a post"), ("You had not posted", "The user had not posted"),
         ("you replied to another user's comment", "the user replied to another user's comment"),
         ("you commented on the post", "the user commented on the post"),
         ("Replies to your comment", "Replies to the user's comment"),
         ("Nobody replied to your comment", "Nobody replied to the user's comment")]


def messages(x, sub, variant):
    body = f"{x.persona}\n\n{x.entry}\n\n{x.replies}\n\n# Question\n{question(sub, variant)}"
    if variant == "observer":
        for a, b in THIRD:
            body = body.replace(a, b)
        body = re.sub(r"\bYour\b(?= (comment|post))", "Their", body)
        return [{"role": "system", "content": OBS_SYSTEM}, {"role": "user", "content": body}]
    if variant == "nosys":
        return [{"role": "user", "content": f"{SYSTEM}\n\n{body}"}]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]


def check_identical(site, R, sub, sticky, conds):
    """The base prompts must equal the frozen scripts' prompts (first 30 agents)."""
    if site not in ("rscience", "futurology"):
        return
    head = R.head(30)
    mine = texts(head, sub, sticky, conds)
    ref = RA.texts(head) if site == "rscience" else FA.texts(head, lambda i: FA.CONDS)
    cols = ["idx", "condition", "persona", "entry", "replies"]
    assert mine[cols].reset_index(drop=True).equals(ref[cols].reset_index(drop=True)), "prompt drift vs frozen script"


def main(a):
    """--site and --variant take comma lists; the model is loaded once and existing outputs are skipped."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok, llm = AutoTokenizer.from_pretrained(a.model), None

    def render(m):
        try:
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:  # templates without a system role: fold it into the user turn
            m = [{"role": "user", "content": m[0]["content"] + "\n\n" + m[1]["content"]}] if m[0]["role"] == "system" else m
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    for site in a.site.split(","):
        R, sub, sticky, conds, prefix = site_spec(site)
        if site != "rscience":  # r/science 2016 outcomes are public (Matias 2019); every other site is sealed
            assert not any(c.startswith("return_") for c in R.columns), "inputs must not contain outcomes"
        R = R[R.n_hist > 0]
        if a.limit:
            R = R.sample(min(a.limit, len(R)), random_state=0)
        check_identical(site, R, sub, sticky, conds)
        T0 = texts(R, sub, sticky, a.conds.split(",") if a.conds else conds)
        for variant in a.variant.split(","):
            out = prefix + a.name + ("" if variant == "base" else f"-{variant}")
            if os.path.exists(os.path.join(out, "scores.parquet")):
                print(f"skip {out} (exists)", flush=True)
                continue
            T = T0.copy()
            print(f"{site} {a.name} {variant}: {len(T)} prompts for {len(R)} agents -> {out}", flush=True)
            prompts = [render(messages(x, sub, variant)) for x in T.itertuples()]
            print(prompts[0][-500:], flush=True)
            if llm is None:
                kw = dict(tokenizer_mode="mistral", config_format="mistral", load_format="mistral") if a.mistral else {}
                llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_util, max_model_len=a.max_len,
                          enable_prefix_caching=True, seed=0, quantization=a.quant, tensor_parallel_size=a.tp, **kw)
            ids = [tok(s, add_special_tokens=False).input_ids for s in prompts]
            outs = llm.generate([{"prompt_token_ids": x} for x in ids],
                                SamplingParams(max_tokens=1, temperature=0, logprobs=20))
            res = [readout(o.outputs[0].logprobs[0], {"Yes": 1, "No": 0}) for o in outs]
            T["p"], T["mass"] = [x[0] for x in res], [x[1] for x in res]
            S = T.drop(columns=["persona", "entry", "replies"])
            os.makedirs(out, exist_ok=True)
            S.to_parquet(os.path.join(out, "scores.parquet"))
            print(S.groupby("condition")[["p", "mass"]].mean().round(4).to_string(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--variant", default="base", help="comma list of base, q2, nosys, order, observer")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conds", default="")
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--mistral", action="store_true")
    main(ap.parse_args())
