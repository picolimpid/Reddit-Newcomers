"""Budgeted, cached client for Jev System One, for hci-explore.

Backend (JEV_BACKEND):
- "openrouter" (default since 2026-11-03): POST https://openrouter.ai/api/alpha/decisions, model typesafe/jev-1.14,
  key OPENROUTER_API_KEY from hci-explore/.env; billed to the OpenRouter credit; cost read from usage.cost.
- "typesafe": the original direct API (https://api.typesafe.ai/v1/systemone, model jev-1.14.0, key from
  ../speculative-jev/.env). All pilot results before 2026-11-03 came from this backend.
Request/response format is identical on both. A 30-comment check (180 probabilities) gave mean |diff| 0.0045,
max 0.05, 70% identical, so cached direct responses are reused when the same request has no OpenRouter cache entry
(the returned dict says which backend produced it in "_backend").

Hard budget: JEV_MAX_USD (default $0.50) over this project's JEV ledger (both backends), checked before every
uncached request with a conservative estimate. Every response is cached on disk keyed by the request body.
Pricing: $0.042 per 1M input tokens, output free.
"""
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.environ.get("JEV_BACKEND", "openrouter")
CFG = {
    "openrouter": {"url": "https://openrouter.ai/api/alpha/decisions", "model": "typesafe/jev-1.14",
                   "env": os.path.join(ROOT, ".env"), "key": "OPENROUTER_API_KEY"},
    "typesafe": {"url": "https://api.typesafe.ai/v1/systemone", "model": "jev-1.14.0",
                 "env": None, "key": "TYPESAFE_API_KEY"},
}


def _find_up(rel):
    """First <ancestor of ROOT>/<rel> that exists (the project may sit one level deeper, e.g. inside Reddit Newcomers/)."""
    d = os.path.dirname(ROOT)
    while True:
        p = os.path.join(d, rel)
        if os.path.exists(p) or d == os.path.dirname(d):
            return p
        d = os.path.dirname(d)


CFG["typesafe"]["env"] = os.environ.get("TYPESAFE_ENV") or _find_up(os.path.join("speculative-jev", ".env"))
USD_PER_TOKEN = 0.042 / 1e6
CACHE = os.path.join(ROOT, "data", "jev_cache")
LEDGER = os.path.join(ROOT, "results", "jev_ledger.jsonl")
MODEL = os.environ.get("JEV_PIN", CFG[BACKEND]["model"])
BUDGET = float(os.environ.get("JEV_MAX_USD", "0.50"))
os.makedirs(CACHE, exist_ok=True)
os.makedirs(os.path.dirname(LEDGER), exist_ok=True)


def _key(backend):
    c = CFG[backend]
    if os.path.exists(c["env"]):
        for line in open(c["env"], encoding="utf-8"):
            if line.strip().startswith(c["key"] + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get(c["key"])


class BudgetExceeded(RuntimeError):
    pass


def spent_usd():
    if not os.path.exists(LEDGER):
        return 0.0
    tot = 0.0
    for l in open(LEDGER):
        if l.strip():
            r = json.loads(l)
            tot += r.get("cost_usd", r.get("input_tokens", 0) * USD_PER_TOKEN)
    return tot


_lock = threading.Lock()
_state = {"spent": spent_usd(), "reserved": 0.0}
_local = threading.local()


def _session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update({"Authorization": "Bearer " + _key(BACKEND), "Content-Type": "application/json"})
    return _local.s


def _path(model, state, questions):
    raw = json.dumps({"model": model, "state": state, "questions": questions}, sort_keys=True, ensure_ascii=False)
    return raw, os.path.join(CACHE, hashlib.sha1(raw.encode()).hexdigest() + ".json")


def ask(state, questions, tag="", model=None, retries=6, allow_legacy_cache=True):
    """One System One request -> dict with 'answers', 'usage', 'model', '_cached', '_backend'."""
    model = model or MODEL
    raw, path = _path(model, state, questions)
    candidates = [(path, BACKEND)]
    if allow_legacy_cache and model == CFG["openrouter"]["model"]:
        candidates.append((_path(CFG["typesafe"]["model"], state, questions)[1], "typesafe"))
    for p, b in candidates:
        if os.path.exists(p):
            out = json.load(open(p, encoding="utf-8"))
            out["_cached"], out["_backend"] = True, b
            return out
    est = (len(raw) / 2.5 + 200) * USD_PER_TOKEN  # conservative: ~2.5 chars/token
    with _lock:
        if _state["spent"] + _state["reserved"] + est > BUDGET:
            raise BudgetExceeded(f"spent ${_state['spent']:.4f} + reserved ${_state['reserved']:.4f} "
                                 f"+ est ${est:.5f} > ${BUDGET}")
        _state["reserved"] += est
    try:
        delay = 1
        for attempt in range(retries):
            try:
                t0 = time.time()
                r = _session().post(CFG[BACKEND]["url"], data=raw.encode(), timeout=120)
                if r.status_code in (429, 529) or r.status_code >= 500:
                    raise requests.HTTPError(f"{r.status_code}: {r.text[:200]}")
                if r.status_code != 200:
                    raise ValueError(f"{r.status_code}: {r.text[:500]}")
                out = r.json()
                lat = time.time() - t0
                break
            except ValueError:
                raise
            except Exception as e:  # noqa: BLE001
                if attempt == retries - 1:
                    raise RuntimeError(f"jev {tag}: {e}") from e
                time.sleep(delay)
                delay = min(delay * 2, 30)
        n = out["usage"]["input_tokens"]
        cost = float(out["usage"].get("cost") or n * USD_PER_TOKEN)
        with _lock:
            _state["spent"] += cost
            with open(LEDGER, "a") as f:
                f.write(json.dumps({"t": time.time(), "tag": tag, "backend": BACKEND, "model": out.get("model"),
                                    "input_tokens": n, "cost_usd": cost, "latency_s": round(lat, 3)}) + "\n")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f)
        out["_cached"], out["_backend"] = False, BACKEND
        return out
    finally:
        with _lock:
            _state["reserved"] -= est


def ask_many(items, workers=8):
    """items: list of dicts with keys state, questions, tag. Order preserved."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda it: ask(**it), items))


def p_yes(out, name):
    return float(out["answers"][name]["noul"])


def probs(out, name):
    return {k: float(v) for k, v in out["answers"][name]["probabilities"].items()}


def exp_score(out, name):
    """Expected score level normalised to [0, 1]."""
    pr = out["answers"][name]["probabilities"]
    n = len(pr)
    return sum(int(k) * float(v) for k, v in pr.items()) / max(n - 1, 1)


if __name__ == "__main__":
    o = ask({"text": "Water boils at 100 degrees Celsius at sea level."},
            {"q": {"type": "noul", "instructions": "Is `text` factually correct?"},
             "s": {"type": "score", "instructions": "How formal is `text`?",
                   "criteria": ["very informal", "informal", "neutral", "formal", "very formal"]}}, tag="smoke-or")
    print(BACKEND, o.get("model"), o["_backend"], o["_cached"], json.dumps(o["answers"])[:300], o["usage"])
    print("JEV spent (all backends) $%.4f" % spent_usd())
