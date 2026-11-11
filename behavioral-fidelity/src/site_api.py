"""API agents (gpt-4o-mini via OpenRouter, Jev) for the new sites (PREREG_sites.md §4), 2026-11-06.
Same prompts as panel_agents.py (base variant); a random 800 agents (seed 0) in all five conditions.

    python3 site_api.py ffxiv2019 openai/gpt-4o-mini [--dry]
    python3 site_api.py ffxiv2019 jev"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import panel_agents as PA
from agent_sim import SYSTEM

sys.path.insert(0, "/data/hanqim2/Reddit Newcomers/src")
import openrouter as orc  # noqa: E402


def main(a):
    R, sub, sticky, conds, prefix = PA.site_spec(a.site)
    R = R[R.n_hist > 0]
    R = R.sample(min(a.n, len(R)), random_state=0)
    T = PA.texts(R, sub, sticky, conds)
    q = PA.question(sub, "base")
    name = "jev" if a.api == "jev" else a.api.split("/")[-1]
    out = prefix + name
    tag = f"{a.site}-{name}"
    if a.api == "jev":
        reqs = [({"your_reddit_activity": x.persona, "what_just_happened": x.entry, "replies_to_your_comment": x.replies},
                 {"ret": {"type": "noul", "instructions": f"{SYSTEM}\n\n# Question\n{q}"}}) for x in T.itertuples()]
        f = lambda r: (orc.jev(*r, tag=tag)["answers"]["ret"]["noul"], np.nan)  # noqa: E731
        chars = sum(len(str(r[0])) + len(str(r[1])) for r in reqs)
    else:
        reqs = [(SYSTEM, f"{x.persona}\n\n{x.entry}\n\n{x.replies}\n\n# Question\n{q}") for x in T.itertuples()]
        f = lambda r: orc.chat_yes(a.api, *r, tag=tag)  # noqa: E731
        chars = sum(len(r[0]) + len(r[1]) for r in reqs)
    price = orc.PRICE_IN.get(orc.JEV_MODEL if a.api == "jev" else a.api, 0.15)
    est = chars / 3.5 / 1e6 * price
    print(f"{a.site} {name}: {len(reqs)} requests, ~{chars / 3.5 / 1e6:.2f}M input tokens, est ${est:.2f}; "
          f"spent ${orc.spent():.3f} of ${orc.BUDGET}", flush=True)
    if a.dry:
        return
    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(f, reqs))
    T["p"], T["mass"] = [x[0] for x in res], [x[1] for x in res]
    S = T.drop(columns=["persona", "entry", "replies"])
    os.makedirs(out, exist_ok=True)
    S.to_parquet(os.path.join(out, "scores.parquet"))
    print(S.groupby("condition").p.mean().round(4).to_string(), f"\nspent ${orc.spent():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("site")
    ap.add_argument("api")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--dry", action="store_true")
    main(ap.parse_args())
