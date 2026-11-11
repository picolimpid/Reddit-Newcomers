"""Human-side checks on the Study 2 newcomer findings (2026-11-02), same post-FE logit, controls
and clustered SEs as fingerprints.py.

  1 equivalence  TOST for the reply effect: which effect sizes the data can rule out
  2 outcomes     stricter return outcomes from the in-community items (insub_items): ordinary
                 threads only (< 500 comments), viral threads only, >= 2 distinct months
  3 event study  community fit in 4-week bins over the pre-entry year (one bin per model) and a
                 placebo fit to a random unvisited target community; 2-week bins for the last 12 weeks
  4 news         FP1 / FP3 without the news_politics posts (shared-news-event explanation)

    python human_checks.py   -> results/human_checks/*.csv, summary printed"""
import json
import os
import random
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

import c2v
from common import BENCH, DAY, RESULTS, SOURCE
from fingerprints import estimate

warnings.filterwarnings("ignore")
OUT = os.path.join(RESULTS, "human_checks")
os.makedirs(OUT, exist_ok=True)
B = pd.read_parquet(BENCH)


def row(tag, y, x, d, **kw):
    t, idx, m = estimate(d, y, [x] if isinstance(x, str) else x, **kw)
    t.insert(0, "outcome", y)
    t.insert(0, "check", tag)
    t["base_rate"] = d.loc[idx, y].mean()
    return t


# ---------------------------------------------------------------- 1 equivalence (TOST)
E = []
for y in ("return_90d", "return_180d"):
    t, idx, m = estimate(B, y, ["reply_received"])
    b, se = t.beta.iloc[0], t.se.iloc[0]
    fit_b = estimate(B, y, ["c2v_big"])[0].beta.iloc[0]
    for lo_or, hi_or in ((0.8, 1.25), (0.67, 1.5), (0.5, 2.0)):
        p_lo = norm.sf((b - np.log(lo_or)) / se)     # H0: beta <= log(lo)
        p_hi = norm.cdf((b - np.log(hi_or)) / se)    # H0: beta >= log(hi)
        E.append({"outcome": y, "OR": np.exp(b), "ci90_lo": np.exp(b - 1.645 * se), "ci90_hi": np.exp(b + 1.645 * se),
                  "bounds": f"[{lo_or}, {hi_or}]", "p_tost": max(p_lo, p_hi), "equivalent_at_05": max(p_lo, p_hi) < .05,
                  "mde80_OR": np.exp((1.96 + 0.84) * se), "fit_OR_per_sd": np.exp(fit_b), "n": t.n.iloc[0]})
E = pd.DataFrame(E)
E.to_csv(os.path.join(OUT, "equivalence.csv"), index=False)

# ---------------------------------------------------------------- 2 stricter outcomes
I = pd.read_parquet(os.path.join(SOURCE, "insub_items.parquet"))
I["dt"] = (I.t - I.idx.map(B.first_utc)) / DAY
I["ordinary"] = I.thread_comments < 500
I["viral"] = I.thread_comments >= 1000
g = I.groupby("idx")
for w in (90, 180):
    win = I[I.dt < w].groupby("idx")
    B[f"ret_ordinary_{w}d"] = B.index.isin(win.ordinary.any()[lambda s: s].index).astype(float)
    B[f"ret_viral_only_{w}d"] = B.index.isin((win.viral.all() & ~win.ordinary.any())[lambda s: s].index).astype(float)
B["ret_2months_180d"] = B.index.isin(g.dt.agg(lambda s: (s // 30).nunique())[lambda s: s >= 2].index).astype(float)
B["ret_ordinary_3plus_180d"] = B.index.isin(I[I.ordinary].groupby("idx").size()[lambda s: s >= 3].index).astype(float)
O = []
for y in ("return_90d", "ret_ordinary_90d", "ret_viral_only_90d", "return_180d", "ret_ordinary_180d",
          "ret_viral_only_180d", "ret_2months_180d", "ret_ordinary_3plus_180d", "regular_180d"):
    for x in ("c2v_big", "reply_received", ["fit_0_30", "fit_30_90", "fit_90_365"]):
        try:
            O.append(row("outcomes", y, x, B))
        except Exception as e:  # noqa: BLE001
            print("skip", y, x, str(e)[:60])
O = pd.concat(O, ignore_index=True)
O.to_csv(os.path.join(OUT, "outcomes.csv"), index=False)

# ---------------------------------------------------------------- 3 event study
users = pd.read_parquet(os.path.join(SOURCE, "users.parquet"))
hu = users[users.history.notna()].drop_duplicates(["post_id", "author"])
extra = [json.loads(s) for s in pd.read_parquet(os.path.join(SOURCE, "extra_users.parquet")).subs.dropna()]
vid, Emb = c2v.train([{x["sub"]: 1 for x in json.loads(h)} for h in hu.history] + [c for c in extra if c])
subs = sorted(B.subreddit.unique())


def placebo(r):
    seen = {x["sub"].lower() for x in json.loads(r.history)} | {r.subreddit.lower()}
    return random.Random(20_000 + int(r.name)).choice([s for s in subs if s.lower() not in seen])


bins = [(lo, lo + 28) for lo in range(0, 364, 28)] + [(lo, lo + 14) for lo in range(0, 84, 14)]
F = {}
for i, r in B.iterrows():
    hist = json.loads(r.history)
    pl = placebo(r)
    for lo, hi in bins:
        c = {}
        for x in hist:
            if lo <= (r.post_utc - x["t"]) / DAY < hi:
                c[x["sub"]] = c.get(x["sub"], 0) + 1
        ok = sum(c.values()) >= 3
        F[(i, lo, hi)] = (c2v.fit(c, r.subreddit, vid, Emb) if ok else np.nan,
                          c2v.fit(c, pl, vid, Emb) if ok else np.nan)
ES = []
for lo, hi in bins:
    B["_fit"] = [F[(i, lo, hi)][0] for i in B.index]
    B["_placebo"] = [F[(i, lo, hi)][1] for i in B.index]
    for y in ("return_180d", "ret_ordinary_180d"):
        for x in ("_fit", "_placebo"):
            try:
                t = row("event", y, x, B)
            except Exception as e:  # noqa: BLE001
                print("skip bin", lo, y, x, str(e)[:60])
                continue
            t["days_before_lo"], t["days_before_hi"], t["width"] = lo, hi, hi - lo
            ES.append(t)
ES = pd.concat(ES, ignore_index=True)
ES.to_csv(os.path.join(OUT, "event_study.csv"), index=False)

# ---------------------------------------------------------------- 4 without news posts
N = []
for tag, d in (("all", B), ("no_news", B[B.genre != "news_politics"]), ("news_only", B[B.genre == "news_politics"])):
    for y in ("return_180d", "ret_ordinary_180d"):
        for x in ("c2v_big", ["fit_0_30", "fit_30_90", "fit_90_365"], ["jw_d0_30", "jw_d30_90", "jw_d90_365"]):
            try:
                N.append(row(tag, y, x, d))
            except Exception as e:  # noqa: BLE001
                print("skip", tag, y, x, str(e)[:60])
N = pd.concat(N, ignore_index=True)
N.to_csv(os.path.join(OUT, "news.csv"), index=False)

pd.set_option("display.width", 220)
cols = ["check", "outcome", "predictor", "OR", "ci_lo", "ci_hi", "p", "n", "base_rate"]
print("== 1 equivalence\n", E.round(3).to_string(index=False))
print("== 2 outcomes\n", O[cols].round(3).to_string(index=False))
print("== 3 event study\n", ES[cols + ["days_before_lo", "days_before_hi"]].round(3).to_string(index=False))
print("== 4 news\n", N[cols].round(3).to_string(index=False))
