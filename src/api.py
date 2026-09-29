"""Thin cached client for the Arctic Shift Reddit archive API.

Every successful response is cached on disk (keyed by URL + params) so the
pipeline can be resumed and re-run without re-querying the server.
"""
import hashlib
import json
import os
import threading
import time

import requests

BASE = "https://arctic-shift.photon-reddit.com/api"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache")
os.makedirs(CACHE, exist_ok=True)

# STUDY selects a sub-folder for data/results so several target-post sets can coexist
# (empty = the original four-subreddit pilot; e.g. STUDY=global for the 2025 top posts).
STUDY = os.environ.get("STUDY", "")
DATA = os.path.join(ROOT, "data", STUDY)
RESULTS = os.path.join(ROOT, "results", STUDY)
os.makedirs(DATA, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

_local = threading.local()


def _session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers["User-Agent"] = "newcomer-alignment-study/0.1 (academic research)"
    return _local.s


def get(endpoint, retries=9, **params):
    key = endpoint + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
    path = os.path.join(CACHE, hashlib.sha1(key.encode()).hexdigest() + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    delay = 2
    for attempt in range(retries):
        try:
            r = _session().get(f"{BASE}/{endpoint}", params=params, timeout=90)
            if r.status_code == 429:
                reset = float(r.headers.get("X-RateLimit-Reset", delay))
                time.sleep(max(reset, 1) + 1)
                continue
            if r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code}: {r.text[:200]}")
            body = r.json()
            if body.get("error"):
                # Timeouts on very active users come back as errors; retry a few times
                raise RuntimeError(body["error"])
            data = body["data"]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            return data
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise RuntimeError(f"{key}: {e}") from e
            time.sleep(delay)
            delay = min(delay * 2, 120)


def paginate(endpoint, start, end, page_cap=None, **params):
    """Fetch everything in [start, end) sorted ascending, paging on created_utc."""
    out, seen, after, pages = [], set(), start, 0
    while True:
        batch = get(endpoint, after=after, before=end, sort="asc", limit=100, **params)
        pages += 1
        new = [x for x in batch if x["id"] not in seen]
        for x in new:
            seen.add(x["id"])
        out.extend(new)
        if len(batch) < 100 or not new or (page_cap and pages >= page_cap):
            break
        # step back one second so same-second items aren't skipped; dedup handles overlap
        after = int(batch[-1]["created_utc"]) - 1 if batch[-1]["created_utc"] - 1 > after else int(batch[-1]["created_utc"])
    return out
