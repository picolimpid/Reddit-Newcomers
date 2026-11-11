"""Thin cached client for the TypeSafe Jev System One API (POST /v1/systemone).

Every successful response is cached on disk (keyed by model + request body), so re-runs
cost nothing. Each billed call is appended to a ledger; once the cumulative cost reaches
JEV_MAX_BUDGET_USD the client refuses further uncached calls.

Pricing (docs.typesafe.ai/models, Oct 2026): $0.042 per 1M input tokens, output free.
"""
import hashlib
import json
import os
import threading
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://api.typesafe.ai/v1/systemone"
USD_PER_INPUT_TOKEN = 0.042 / 1e6
CACHE = os.path.join(ROOT, "data", "jev_cache")
LEDGER = os.path.join(ROOT, "data", "jev_ledger.jsonl")
os.makedirs(CACHE, exist_ok=True)


def _load_env():
    path = os.path.join(ROOT, ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()
MODEL = os.environ.get("JEV_MODEL", "jev-latest")
BUDGET = float(os.environ.get("JEV_MAX_BUDGET_USD", "3.0"))
_lock = threading.Lock()
_local = threading.local()


class BudgetExceeded(RuntimeError):
    pass


def spent():
    """Cumulative (input_tokens, usd) over every billed call in the ledger."""
    tok = 0
    if os.path.exists(LEDGER):
        for line in open(LEDGER, encoding="utf-8"):
            tok += json.loads(line)["input_tokens"]
    return tok, tok * USD_PER_INPUT_TOKEN


_spent_tok = [spent()[0]]


def _session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers["Authorization"] = "Bearer " + os.environ["TYPESAFE_API_KEY"]
    return _local.s


def ask(state, questions, assets=None, tag="", model=MODEL, retries=8):
    """One System One request. Returns the parsed response (answers, usage, model)."""
    body = {"model": model, "state": state, "questions": questions}
    headers = {}
    if assets:
        body["assets"] = assets
        headers["TypeSafe-Danger-Zone-Images"] = "true"
    raw = json.dumps(body, sort_keys=True, ensure_ascii=False)
    path = os.path.join(CACHE, hashlib.sha1(raw.encode()).hexdigest() + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    with _lock:
        if _spent_tok[0] * USD_PER_INPUT_TOKEN >= BUDGET:
            raise BudgetExceeded(f"spent ${_spent_tok[0] * USD_PER_INPUT_TOKEN:.3f} >= budget ${BUDGET}")
    delay = 1
    for attempt in range(retries):
        try:
            r = _session().post(BASE, data=raw.encode(), timeout=120,
                                headers={"Content-Type": "application/json", **headers})
            if r.status_code in (429, 529) or r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code}: {r.text[:200]}")
            if r.status_code != 200:  # 4xx other than 429: a request bug, do not retry
                raise ValueError(f"{r.status_code}: {r.text[:500]}")
            out = r.json()
            n = out["usage"]["input_tokens"]
            with _lock:
                _spent_tok[0] += n
                with open(LEDGER, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"t": time.time(), "tag": tag, "model": out.get("model"),
                                        "input_tokens": n, "output_tokens": out["usage"].get("output_tokens", 0)}) + "\n")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f)
            return out
        except ValueError:
            raise
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise RuntimeError(f"jev {tag}: {e}") from e
            time.sleep(delay)
            delay = min(delay * 2, 60)
