"""Does fine-tuning on observational outcomes make agents faithful, or only calibrated? (2026-11-02)

train  LoRA on Qwen3-8B. One example per train-split agent (60% of communities): the intent
       prompt with the replies the user actually received -> the user's real 90-day return
       ("Yes"/"No"). Loss on the answer token only.
eval   held-out communities (val + test). The fine-tuned agent sees the same probe conditions as
       probes.py (reply ladder, negative-control community, observed replies), so its base rate,
       AUROC and intervention responses can be set against the base model on the same agents.

    CUDA_VISIBLE_DEVICES=9 python sft.py train --out sft-qwen3-8b
    CUDA_VISIBLE_DEVICES=9 python sft.py eval --out sft-qwen3-8b"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd

from agent_sim import SYSTEM, entry, persona, question, readout, replies
from common import BENCH, RESULTS
from probes import LADDER, ask_other, control_sub, reply_block

MODEL = "Qwen/Qwen3-8B"


def observed_prompt(r, tok, k):
    q, _ = question(r, "intent", random.Random(10_000 + k))
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{persona(r, random.Random(k))}\n\n{entry(r)}\n\n{replies(r, 'observed', k)}\n\n# Question\n{q}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def train(a):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    B = pd.read_parquet(BENCH)
    T = B[B.split == "train"]
    yes, no = tok("Yes", add_special_tokens=False).input_ids, tok("No", add_special_tokens=False).input_ids
    assert len(yes) == len(no) == 1
    data = [(tok(observed_prompt(r, tok, int(i)), add_special_tokens=False).input_ids, yes[0] if r.return_90d else no[0])
            for i, r in T.iterrows()]
    print(f"{len(data)} training agents, P(Yes) = {T.return_90d.mean():.3f}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps = a.epochs * len(data) // a.accum
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / steps))
    rng = random.Random(0)
    model.train()
    step, run_loss = 0, []
    for ep in range(a.epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        for n, j in enumerate(order):
            ids, ans = data[j]
            x = torch.tensor([ids], device="cuda")
            logits = model(input_ids=x, logits_to_keep=1).logits[0, -1].float()   # next-token distribution after the prompt
            loss = torch.nn.functional.cross_entropy(logits[None], torch.tensor([ans], device="cuda")) / a.accum
            loss.backward()
            run_loss.append(loss.item() * a.accum)
            if (n + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if step % 10 == 0:
                    print(f"epoch {ep} step {step}/{steps} loss {np.mean(run_loss[-10 * a.accum:]):.4f}", flush=True)
    out = os.path.join(RESULTS, "probes", a.out, "adapter")
    model.save_pretrained(out)
    print("saved", out)


def evaluate(a):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(MODEL)
    B = pd.read_parquet(BENCH)
    H = B[B.split != "train"]
    subs = sorted(B.subreddit.unique())
    conds = ["none", "ack", "disagree", "praise", "praise_welcome", "praise_invite", "praise_welcome_invite", "template"]
    rows = []
    for i, r in H.iterrows():
        k = int(i)
        p, e = persona(r, random.Random(k)), entry(r)
        q, _ = question(r, "intent", random.Random(10_000 + k))
        y = control_sub(r, subs)
        specs = [("ladder", c, reply_block(c, r.subreddit, k), q, r.subreddit) for c in conds]
        specs += [("observed", "observed", replies(r, "observed", k), q, r.subreddit)]
        specs += [("nc", c, reply_block(c, r.subreddit, k), ask_other(y), y) for c in ("none", "praise_welcome_invite", "template")]
        for exp, c, rep, qq, target in specs:
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{p}\n\n{e}\n\n{rep}\n\n# Question\n{qq}"}]
            rows.append({"idx": i, "exp": exp, "condition": c, "outcome_sub": target,
                         "prompt": tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)})
    P = pd.DataFrame(rows)
    print(f"{len(P)} prompts for {len(H)} held-out agents", flush=True)
    llm = LLM(model=MODEL, gpu_memory_utilization=a.gpu_util, max_model_len=8192, enable_prefix_caching=True, seed=0,
              enable_lora=True, max_lora_rank=a.rank)
    ids = [{"prompt_token_ids": tok(s, add_special_tokens=False).input_ids} for s in P.prompt]
    sp = SamplingParams(max_tokens=1, temperature=0, logprobs=20)
    for tag, lora in (("sft", LoRARequest("sft", 1, os.path.join(RESULTS, "probes", a.out, "adapter"))), ("base", None)):
        outs = llm.generate(ids, sp, lora_request=lora)
        res = [readout(o.outputs[0].logprobs[0], {"Yes": 1, "No": 0}) for o in outs]
        P[f"p_{tag}"], P[f"mass_{tag}"] = [x[0] for x in res], [x[1] for x in res]
    P.drop(columns="prompt").to_parquet(os.path.join(RESULTS, "probes", a.out, "scores.parquet"))
    print(P.groupby(["exp", "condition"])[["p_base", "p_sft"]].mean().round(4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--out", default="sft-qwen3-8b")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    a = ap.parse_args()
    train(a) if a.mode == "train" else evaluate(a)
