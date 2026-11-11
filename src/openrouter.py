"""Budgeted, cached OpenRouter client: chat models (first-token logprobs) and Jev 1.14 (decisions API).
(2026-11-03)

Key OPENROUTER_API_KEY and cap OPENROUTER_MAX_USD (set to $10) come from this repo's .env. The cap is
checked before every uncached request against the ledger data/openrouter_ledger.jsonl plus the
requests in flight (conservative estimate); the ledger records the actual cost from usage.cost.
Responses are cached in data/or_cache/ keyed by the request body, so re-runs are free.

  chat_yes(model, system, user) -> (P(Yes) over {Yes, No} in the first-token top-20 logprobs, mass);
                                   pinned to the OpenAI provider so every condition hits the same backend
  jev(state, questions)         -> System One answers from typesafe/jev-1.14. A request already answered
                                   by the direct TypeSafe API (src/jev.py cache) with a jev-1.14.x model
                                   is reused (hci-explore measured mean |diff| 0.0045 between backends).
"""
import hashlib
import json
import math
import os
import threading
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "or_cache")
LEDGER = os.path.join(ROOT, "data", "openrouter_ledger.jsonl")
DIRECT_JEV_CACHE = os.path.join(ROOT, "data", "jev_cache")
JEV_MODEL = "typesafe/jev-1.14"
PRICE_IN = {"openai/gpt-4o-mini": 0.15, "openai/gpt-4.1-nano": 0.10, "openai/gpt-4.1-mini": 0.40,
            JEV_MODEL: 0.042}  # $ per 1M input tokens, for the pre-request estimate only
os.makedirs(CACHE, exist_ok=True)


def _env():
    out = {}
    for line in open(os.path.join(ROOT, ".env"), encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


_E = _env()
BUDGET = float(os.environ.get("OPENROUTER_MAX_USD", _E.get("OPENROUTER_MAX_USD", "10")))


class BudgetExceeded(RuntimeError):
    pass


def spent():
    if not os.path.exists(LEDGER):
        return 0.0
    return sum(json.loads(l)["cost"] for l in open(LEDGER) if l.strip())


_lock = threading.Lock()
_state = {"spent": spent(), "reserved": 0.0}
_local = threading.local()


def _session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update({"Authorization": "Bearer " + _E["OPENROUTER_API_KEY"],
                                 "Content-Type": "application/json"})
    return _local.s


def _cached(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _post(url, body, model, tag, retries=6):
    raw = json.dumps(body, sort_keys=True, ensure_ascii=False)
    path = os.path.join(CACHE, hashlib.sha1(raw.encode()).hexdigest() + ".json")
    hit = _cached(path)
    if hit is not None:
        return hit
    est = (len(raw) / 2.5 + 50) * PRICE_IN.get(model, 3.0) / 1e6 * 1.5
    with _lock:
        if _state["spent"] + _state["reserved"] + est > BUDGET:
            raise BudgetExceeded(f"spent ${_state['spent']:.3f} + in flight ${_state['reserved']:.3f} "
                                 f"+ ${est:.5f} > cap ${BUDGET}")
        _state["reserved"] += est
    try:
        delay = 2
        for attempt in range(retries):
            try:
                t0 = time.time()
                r = _session().post(url, data=raw.encode(), timeout=120)
                if r.status_code in (429, 529) or r.status_code >= 500:
                    raise requests.HTTPError(f"{r.status_code}: {r.text[:200]}")
                d = r.json()
                if r.status_code != 200 or "error" in d:
                    raise ValueError(f"{r.status_code}: {str(d)[:400]}")
                lat = time.time() - t0
                break
            except requests.exceptions.JSONDecodeError as e:  # truncated / keep-alive body: retry
                if attempt == retries - 1:
                    raise RuntimeError(f"openrouter {tag}: bad JSON {e}") from e
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            except ValueError:
                raise
            except Exception as e:  # noqa: BLE001
                if attempt == retries - 1:
                    raise RuntimeError(f"openrouter {tag}: {e}") from e
                time.sleep(delay)
                delay = min(delay * 2, 30)
        u = d.get("usage", {})
        cost = float(u.get("cost") or 0.0)
        with _lock:
            _state["spent"] += cost
            with open(LEDGER, "a") as f:
                f.write(json.dumps({"t": time.time(), "tag": tag, "model": model, "resolved": d.get("model"),
                                    "provider": d.get("provider"), "usage": u, "cost": cost,
                                    "latency_s": round(lat, 3)}) + "\n")
        tmp = f"{path}.{threading.get_ident()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f)
        os.replace(tmp, path)  # atomic: concurrent readers never see a partial file
        return d
    finally:
        with _lock:
            _state["reserved"] -= est


def chat_yes(model, system, user, tag=""):
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
            "usage": {"include": True},
            "provider": {"order": ["OpenAI"], "allow_fallbacks": False, "require_parameters": True}}
    d = _post("https://openrouter.ai/api/v1/chat/completions", body, model, tag)
    ch = d["choices"][0]
    content = (ch.get("logprobs") or {}).get("content") or []
    m = {"yes": 0.0, "no": 0.0}
    for t in (content[0].get("top_logprobs", []) if content else []):
        s = t["token"].strip().rstrip(").:").strip().lower()
        if s in m:
            m[s] += math.exp(t["logprob"])
    tot = m["yes"] + m["no"]
    return (m["yes"] / tot if tot else float("nan")), tot


def jev(state, questions, tag=""):
    raw = json.dumps({"model": "jev-latest", "state": state, "questions": questions}, sort_keys=True, ensure_ascii=False)
    old = _cached(os.path.join(DIRECT_JEV_CACHE, hashlib.sha1(raw.encode()).hexdigest() + ".json"))
    if old is not None and str(old.get("model", "")).startswith("jev-1.14"):
        return old
    return _post("https://openrouter.ai/api/alpha/decisions",
                 {"model": JEV_MODEL, "state": state, "questions": questions}, JEV_MODEL, tag)


if __name__ == "__main__":
    print(chat_yes("openai/gpt-4o-mini", "Answer Yes or No.", "Is water wet?", tag="smoke"))
    o = jev({"text": "Water boils at 100 degrees Celsius at sea level."},
            {"q": {"type": "noul", "instructions": "Is `text` factually correct?"}}, tag="smoke")
    print(o.get("model"), o["answers"], o.get("usage"))
    print(f"spent ${spent():.4f} of ${BUDGET}")
