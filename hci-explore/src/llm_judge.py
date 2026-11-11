"""LLM-judge baselines through OpenRouter (key: hci-explore/.env OPENROUTER_API_KEY).

Input jsonl rows (same format as src/qwen_judge.py): {"id", "user", "system"?, "options": ["Yes","No"] | ["1",...,"5"]}
Output jsonl rows: {"id", "probs": {option: p}, "mass": mass of the options among the first-token top logprobs}
Probabilities come from the first output token's top-20 logprobs (G-Eval style), renormalised over the options.

Cost safety: hard cap OPENROUTER_MAX_USD (default $4.00) over results/openrouter_ledger.jsonl, checked before every
uncached request with a conservative estimate; every response is cached in data/or_cache/, so re-runs are free.
Actual cost is taken from OpenRouter's usage.cost field.

usage: python3 src/llm_judge.py MODEL in.jsonl out.jsonl [--limit N] [--workers 16]
"""
import argparse
import hashlib
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "or_cache")
LEDGER = os.path.join(ROOT, "results", "openrouter_ledger.jsonl")
BUDGET = float(os.environ.get("OPENROUTER_MAX_USD", "4.0"))
PRICE_IN = {"openai/gpt-4o-mini": 0.15, "openai/gpt-4o": 2.5, "qwen/qwen3-32b": 0.08, "qwen/qwen3-14b": 0.12}  # $/M
QW = {"reasoning": {"enabled": False}, "provider": {"require_parameters": True}}  # route to a provider that returns logprobs
EXTRA = {"qwen/qwen3-32b": QW, "qwen/qwen3-14b": QW}
# open-weight models are run locally (src/local_judge.py), never through the API (user instruction, 2026-11-03)
OPEN_WEIGHTS = ("qwen/", "meta-llama/", "mistralai/", "deepseek/", "google/gemma", "microsoft/phi", "nvidia/", "allenai/")
os.makedirs(CACHE, exist_ok=True)
KEY = [l.split("=", 1)[1].strip().strip("\"'") for l in open(os.path.join(ROOT, ".env")) if l.startswith("OPENROUTER_API_KEY=")][0]


class BudgetExceeded(RuntimeError):
    pass


def spent():
    if not os.path.exists(LEDGER):
        return 0.0
    return sum(json.loads(l).get("cost", 0.0) for l in open(LEDGER) if l.strip())


_lock = threading.Lock()
_state = {"spent": spent(), "reserved": 0.0}
_local = threading.local()


def _session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update({"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    return _local.s


def call(model, row, retries=6):
    msgs = ([{"role": "system", "content": row["system"]}] if row.get("system") else []) + [
        {"role": "user", "content": row["user"]}]
    body = {"model": model, "messages": msgs, "max_tokens": 1, "temperature": 0, "logprobs": True,
            "top_logprobs": 20, "usage": {"include": True}, **EXTRA.get(model, {})}
    raw = json.dumps(body, sort_keys=True, ensure_ascii=False)
    path = os.path.join(CACHE, hashlib.sha1(raw.encode()).hexdigest() + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    est = (len(raw) / 3 + 50) * PRICE_IN.get(model, 3.0) / 1e6 * 1.5
    with _lock:
        if _state["spent"] + _state["reserved"] + est > BUDGET:
            raise BudgetExceeded(f"spent ${_state['spent']:.3f} + reserved ${_state['reserved']:.3f} > ${BUDGET}")
        _state["reserved"] += est
    try:
        delay = 2
        for attempt in range(retries):
            try:
                t0 = time.time()
                r = _session().post("https://openrouter.ai/api/v1/chat/completions", data=raw.encode(), timeout=90)
                d = r.json()
                if r.status_code != 200 or "error" in d or not d.get("choices"):
                    raise RuntimeError(f"{r.status_code}: {str(d)[:300]}")
                lat = time.time() - t0
                break
            except Exception as e:  # noqa: BLE001
                if attempt == retries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30)
        u = d.get("usage", {})
        cost = float(u.get("cost") or 0.0)
        with _lock:
            _state["spent"] += cost
            with open(LEDGER, "a") as f:
                f.write(json.dumps({"t": time.time(), "model": model, "id": row["id"], "prompt_tokens": u.get("prompt_tokens"),
                                    "cost": cost, "latency_s": round(lat, 3), "provider": d.get("provider")}) + "\n")
        out = {"choices": d["choices"], "usage": u, "provider": d.get("provider"), "latency_s": lat}
        json.dump(out, open(path, "w"))
        return out
    finally:
        with _lock:
            _state["reserved"] -= est


def option_probs(resp, options):
    ch = resp["choices"][0]
    content = ((ch.get("logprobs") or {}).get("content") or [])
    cand = {}
    if content:
        for t in content[0].get("top_logprobs", []):
            s = t["token"].strip().lower()
            for o in options:
                if s == o.lower():
                    cand[o] = cand.get(o, 0.0) + math.exp(t["logprob"])
    mass = sum(cand.values())
    if mass == 0:  # no logprobs: fall back to the generated text
        txt = (ch["message"].get("content") or "").strip().lower()
        hit = [o for o in options if txt.startswith(o.lower())]
        return ({o: float(o in hit[:1]) for o in options} if hit else {o: 1 / len(options) for o in options}), 0.0
    return {o: cand.get(o, 0.0) / mass for o in options}, mass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.model.startswith(OPEN_WEIGHTS):
        raise SystemExit(f"{a.model} is an open-weight model: run it locally with src/local_judge.py, not through the API")
    rows = [json.loads(l) for l in open(a.inp)]
    if a.limit:
        rows = rows[:a.limit]
    before = spent()
    with ThreadPoolExecutor(a.workers) as pool:
        resps = list(pool.map(lambda r: call(a.model, r), rows))
    n_nolp = 0
    with open(a.out, "w") as f:
        for r, resp in zip(rows, resps):
            probs, mass = option_probs(resp, r["options"])
            n_nolp += mass == 0
            f.write(json.dumps({"id": r["id"], "probs": probs, "mass": mass, "latency_s": resp["latency_s"],
                                "prompt_tokens": resp["usage"].get("prompt_tokens")}) + "\n")
    lat = sorted(x["latency_s"] for x in resps)
    print(f"{a.model}: n={len(rows)} no-logprob={n_nolp} cost this run ${spent() - before:.4f} total ${spent():.4f} "
          f"median latency {lat[len(lat) // 2]:.2f}s")
