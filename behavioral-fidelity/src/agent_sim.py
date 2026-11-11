"""Base LLM agents on the benchmark: each real newcomer becomes an agent conditioned on that
user's own pre-entry history, experiences the real entry event, and makes a return decision.

Protocols (single forward pass each; the decision is read from first-token probabilities)
  intent   "Over the next 3 months, will you post or comment in r/X again?"  -> P(Yes)
  choice   "Two weeks later you open Reddit. What do you do first?" with r/X among the user's
           own top subreddits, r/popular and logging off (shuffled per agent) -> P(r/X)
Reply conditions (what the agent saw under its first comment)
  observed   the replies the real user actually received (none for 79% of newcomers)
  none       no reply
  welcoming  a template welcome from the community   } interventions the human data
  hostile    a template gatekeeping / hostile reply  } cannot provide (RQ2)

    CUDA_VISIBLE_DEVICES=5 python agent_sim.py --model Qwen/Qwen3-8B --out qwen3-8b [--limit 50]
    python agent_sim.py --dry-run      # render prompts and count tokens, no GPU
Mechanism tests: --history reverse (mirror item ages), --persona timeline (no window summary),
--conditions observed (score only some reply conditions)."""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd

from common import BENCH, DAY, RESULTS, WINDOWS

SYSTEM = ("You are role-playing one specific, real Reddit user. You are shown this user's own Reddit "
          "activity from the past year and then something that just happened to them on Reddit. "
          "Decide exactly as this particular user would actually behave, not as a helpful assistant and "
          "not as an average person. Reply with only the requested answer.")
WINDOW_LABEL = {(0, 30): "Last 30 days", (30, 90): "1-3 months ago", (90, 365): "3-12 months ago"}
REPLIERS = ["u/quiet_otter_88", "u/MapleCrunchTime", "u/not_a_robot_42"]
TEMPLATES = {
    "welcoming": ["Ha, love this. Welcome to r/{sub}! Stick around, we get threads like this all the time.",
                  "Great comment! Don't think I've seen you around r/{sub} before, glad you found us.",
                  "This is exactly the kind of comment this sub needs. Hope you stay, r/{sub} is a fun place."],
    "hostile": ["Nobody asked. Go back to r/popular, this sub isn't for tourists.",
                "Typical outsider take. You clearly have no idea what r/{sub} is actually about.",
                "Lol, another drive-by from the front page. Lurk more before commenting here."],
}
CONDITIONS = ["observed", "none", "welcoming", "hostile"]
N_RECENT, N_OLDER, MAX_CHARS = 25, 15, 160


def clip(s, n=MAX_CHARS):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def ago(days):
    d = int(days)
    return "today" if d == 0 else "1 day ago" if d == 1 else f"{d} days ago"


def persona(r, rng, history="real", fmt="windows"):
    """history='reverse' mirrors every item's age (a -> 365 - a): same content, opposite
    trajectory, so a user drifting towards the community now drifts away (mechanism test).
    fmt='timeline' drops the per-window subreddit summary and keeps only the dated items."""
    hist = json.loads(r.history)
    for x in hist:
        age = (r.post_utc - x["t"]) / DAY
        x["age"] = min(max(365 - age, 0), 364.9) if history == "reverse" else age
    lines = ["# Your Reddit activity in the past year"]
    if fmt == "windows":
        lines.append("Where you were active (comments + posts per subreddit, most active first):")
        for lo, hi in WINDOWS:
            if history == "real":
                c = json.loads(r[f"subs_{lo}_{hi}"])
            else:
                c = {}
                for x in hist:
                    if lo <= x["age"] < hi:
                        c[x["sub"]] = c.get(x["sub"], 0) + 1
            top = sorted(c.items(), key=lambda kv: -kv[1])[:12]
            lines.append(f"- {WINDOW_LABEL[(lo, hi)]}: " + (", ".join(f"r/{k} ({v})" for k, v in top) or "(no activity)"))
        lines.append("")
    items = [x for x in hist if len(str(x["text"]).strip()) >= 5
             and str(x["text"]).strip() not in ("[deleted]", "[removed]")]
    items.sort(key=lambda x: x["age"])
    older = items[N_RECENT:]
    shown = items[:N_RECENT] + (rng.sample(older, N_OLDER) if len(older) > N_OLDER else older)
    shown.sort(key=lambda x: x["age"])
    lines.append("Some of your comments and posts (newest first):")
    for x in shown:
        kind = "post" if x["kind"] == "p" else "comment"
        lines.append(f"[{ago(x['age'])}] r/{x['sub']}, {kind}: \"{clip(x['text'])}\"")
    return "\n".join(lines)


def entry(r):
    hours = max(r.entry_delay_min, 0) / 60
    when = f"{hours:.0f} hours" if hours >= 1.5 else f"{max(hours * 60, 1):.0f} minutes"
    lines = ["# What just happened",
             f"A post from r/{r.subreddit} reached the front page: \"{clip(r.post_title, 300)}\""
             + (" (video)" if r.post_is_video else "")]
    if r.post_selftext.strip():
        lines.append(f"Post text: \"{clip(r.post_selftext, 400)}\"")
    lines.append(f"You had not posted or commented in r/{r.subreddit} at any point in the past year.")
    if r.top_level == 0 and r.parent_body:
        lines.append(f"{when} after it was posted, you replied to another user's comment "
                     f"\"{clip(r.parent_body, 300)}\" with:")
    else:
        lines.append(f"{when} after it was posted, you commented on the post:")
    lines.append(f"\"{clip(r.first_body, 500)}\"")
    return "\n".join(lines)


def replies(r, cond, k):
    if cond == "observed":
        rs = [(f"u/{x['author']}" + (" (the original poster)" if x["is_op"] else ""), x["body"])
              for x in json.loads(r.replies)][:3]
    elif cond == "none":
        rs = []
    else:
        rs = [(REPLIERS[k % 3], TEMPLATES[cond][k % 3].format(sub=r.subreddit))]
    if not rs:
        return "Nobody replied to your comment."
    return "Replies to your comment:\n" + "\n".join(f"- {a}: \"{clip(b, 300)}\"" for a, b in rs)


def question(r, proto, rng):
    if proto == "intent":
        return (f"Over the next 3 months, will you post or comment in r/{r.subreddit} again "
                "(anywhere other than this thread)? Answer Yes or No."), {"Yes": 1, "No": 0}
    top = [s for s in json.loads(r.top_subs) if s.lower() != r.subreddit.lower()][:3]
    opts = [(f"Check what's new in r/{r.subreddit}", 1)] + [(f"Browse r/{s}", 0) for s in top] + \
           [("Scroll r/popular", 0), ("Close Reddit and do something else", 0)]
    rng.shuffle(opts)
    letters = "ABCDEF"[: len(opts)]
    q = ("Two weeks later, you open Reddit with a few minutes to spare. What do you do first?\n"
         + "\n".join(f"{l}) {o}" for l, (o, _) in zip(letters, opts)) + "\nAnswer with the letter only.")
    return q, {l: v for l, (_, v) in zip(letters, opts)}


def build(B, tok, history="real", fmt="windows", conditions=CONDITIONS):
    rows = []
    for i, r in B.iterrows():
        k = int(i)
        p = persona(r, random.Random(k), history, fmt)
        e = entry(r)
        for proto in ("intent", "choice"):
            q, target = question(r, proto, random.Random(10_000 + k))
            for cond in conditions:
                msgs = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": f"{p}\n\n{e}\n\n{replies(r, cond, k)}\n\n# Question\n{q}"}]
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
                rows.append({"idx": i, "protocol": proto, "condition": cond, "prompt": text,
                             "target": json.dumps(target)})
    return pd.DataFrame(rows)


def readout(lps, target):
    """P(answer coded 1), normalized over the answer options found in the top-k first tokens."""
    mass = {}
    for lp in lps.values():
        t = lp.decoded_token.strip().rstrip(").:").strip()
        t = t.capitalize() if t.lower() in ("yes", "no") else t
        if t in target:
            mass[t] = mass.get(t, 0.0) + float(np.exp(lp.logprob))
    tot = sum(mass.values())
    if tot == 0:
        return np.nan, 0.0
    return sum(v for t, v in mass.items() if target[t] == 1) / tot, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default="qwen3-8b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-util", type=float, default=0.45)
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel size (GPUs per model)")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--history", choices=["real", "reverse"], default="real")
    ap.add_argument("--persona", choices=["windows", "timeline"], default="windows")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    # Mistral tokenizers need the pre-tokenizer regex fix in recent transformers
    tok = AutoTokenizer.from_pretrained(a.model, **({"fix_mistral_regex": True} if "mistral" in a.model.lower() else {}))
    B = pd.read_parquet(BENCH)
    if a.limit:
        B = B.sample(a.limit, random_state=0).sort_index()
    P = build(B, tok, a.history, a.persona, a.conditions.split(","))
    # chat templates already contain any BOS token, so encode without adding special tokens and hand
    # vLLM the ids (otherwise Mistral/Llama-style templates get a second BOS)
    ids_of = {s: tok(s, add_special_tokens=False).input_ids for s in P.prompt.unique()}
    n_tok = P.prompt.map(lambda s: len(ids_of[s]))
    print(f"{len(P)} prompts for {len(B)} agents; tokens median {n_tok.median():.0f}, max {n_tok.max()}", flush=True)
    if a.dry_run:
        r = P[(P.condition == "observed") & B.loc[P.idx].reply_received.values.astype(bool)]
        for _, x in (r if len(r) else P).head(2).iterrows():
            print("=" * 100, "\n", x.prompt)
        return

    from vllm import LLM, SamplingParams
    U = P.drop_duplicates("prompt")[["prompt"]].reset_index(drop=True)
    print(f"unique prompts: {len(U)}", flush=True)
    # this box has no NVLink and PCIe P2P hangs: multi-GPU runs need NCCL_P2P_DISABLE=1 and no
    # custom all-reduce (which also uses P2P)
    if a.tp > 1:
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_util, max_model_len=a.max_len,
              tensor_parallel_size=a.tp, disable_custom_all_reduce=a.tp > 1,
              enable_prefix_caching=True, seed=0)
    outs = llm.generate([{"prompt_token_ids": ids_of[s]} for s in U.prompt],
                        SamplingParams(max_tokens=1, temperature=0, logprobs=20))
    lp_of = dict(zip(U.prompt, (o.outputs[0].logprobs[0] for o in outs)))
    top_of = dict(zip(U.prompt, (o.outputs[0].text for o in outs)))
    res = [readout(lp_of[s], json.loads(t)) for s, t in zip(P.prompt, P.target)]
    P["p"], P["mass"] = [x[0] for x in res], [x[1] for x in res]
    P["top_token"] = P.prompt.map(top_of)
    P["n_tokens"] = n_tok.values
    out = os.path.join(RESULTS, a.out)
    os.makedirs(out, exist_ok=True)
    P.drop(columns="prompt").to_parquet(os.path.join(out, "agent_scores.parquet"))
    P.groupby("idx").first().prompt.head(3).to_frame().to_json(os.path.join(out, "example_prompts.json"), indent=1)
    print(P.groupby(["protocol", "condition"]).agg(p=("p", "mean"), missing=("p", lambda s: s.isna().mean()),
                                                   mass=("mass", "mean")).round(3))
    print("wrote", out)


if __name__ == "__main__":
    main()
