"""Panel median vs the human effect (2026-11-06; exploratory on r/science 2016, proposed as H10 for the sealed sites).

Single models disagree in sign. Does the median across the model panel track the human arm effect anyway?
Estimand per model = agent arm effect (logit of pooled P in treated − control, as in rscience_analyze.paired_gap);
panel estimate = median over models; gap = panel median − human arm effect. One post-cluster bootstrap resample is
shared by all models and the human outcome, so the CI includes the human sampling error. Equivalence: 90% CI
inside ±0.25 (TOST at α = .05, the PREREG margin).

    python3 panel_median.py                       -> results/panel/median.md (r/science 2016, exploratory)
    python3 panel_median.py <site> --open-only    -> results/panel/median-<site>.md (H10, PREREG_addendum_H10.md;
                                                     sealed sites: refuses to run until outcomes.parquet exists)
"""
import os

import numpy as np
import pandas as pd

from common import RESULTS
from panel_summary import ROOT, inputs, runs

EQ = 0.25
API = ("gpt-4o-mini", "jev")
OUTCOMES = {"futurology": os.path.join(ROOT, "data", "futurology2020", "outcomes.parquet")}


def with_outcomes(site):
    R = inputs(site)
    if "return_90d" in R.columns:
        return R
    f = OUTCOMES.get(site, os.path.join(ROOT, "data", "sites", site, "outcomes.parquet"))
    if not os.path.exists(f):
        raise SystemExit(f"{site}: outcomes not fetched yet (sealed); nothing to compare")
    O = pd.read_parquet(f)
    return pd.concat([R, O[["return_90d"]]], axis=1)


def lo(m):
    return np.log(m / (1 - m))


def arm(t, a, n):
    return lo(a[t == 1].sum() / n[t == 1].sum()) - lo(a[t == 0].sum() / n[t == 0].sum())


def main(site="rscience", B=2000, drop=(), open_only=False):
    R = with_outcomes(site)
    P = {}
    for s, name, f in runs():
        if s != site or name in drop or (open_only and name in API):
            continue
        W = pd.read_parquet(f).pivot_table(index="idx", columns="condition", values="p")
        if {"off", "full"} <= set(W.columns):
            P[name] = W
    idx = sorted(set.intersection(*[set(W.dropna(subset=["off", "full"]).index) for W in P.values()]))
    H = R.loc[idx].copy()
    for m, W in P.items():
        H[m] = np.where(H.treat == 1, W.loc[idx, "full"], W.loc[idx, "off"])
    g = H.groupby("post_id").agg(t=("treat", "first"), y=("return_90d", "sum"), n=("treat", "size"),
                                 **{m: (m, "sum") for m in P})
    models = list(P)

    def est(g):
        t, n = g.t.values, g.n.values
        ag = np.array([arm(t, g[m].values, n) for m in models])
        hu = arm(t, g.y.values, n)
        return np.median(ag), hu, np.median(ag) - hu, ag
    med, hu, gap, ag = est(g)
    rng = np.random.default_rng(0)
    bs = [est(g.iloc[rng.integers(0, len(g), len(g))]) for _ in range(B)]
    gb = np.array([b[2] for b in bs])
    mb = np.array([b[0] for b in bs])
    ci95, ci90 = np.percentile(gb, [2.5, 97.5]), np.percentile(gb, [5, 95])
    return {"site": site, "models": len(models), "n": len(H), "human": hu, "panel median": med,
            "median CI": np.percentile(mb, [2.5, 97.5]), "gap": gap, "gap 95% CI": ci95, "gap 90% CI": ci90,
            "equivalent (±0.25)": bool(ci90[0] > -EQ and ci90[1] < EQ),
            "per model": dict(zip(models, np.round(ag, 2))), "spread (max−min)": ag.max() - ag.min()}


def fmt(r):
    return (f"- models = {r['models']}, agents = {r['n']}; human arm effect {r['human']:+.2f}; panel median "
            f"{r['panel median']:+.2f} [{r['median CI'][0]:+.2f}, {r['median CI'][1]:+.2f}]; gap {r['gap']:+.2f} "
            f"[{r['gap 95% CI'][0]:+.2f}, {r['gap 95% CI'][1]:+.2f}] (90% CI [{r['gap 90% CI'][0]:+.2f}, "
            f"{r['gap 90% CI'][1]:+.2f}]); equivalent within ±{EQ}: {r['equivalent (±0.25)']}; "
            f"range of single-model arm effects {r['spread (max−min)']:.2f}")


if __name__ == "__main__":
    import sys
    site = next((x for x in sys.argv[1:] if not x.startswith("--")), "rscience")
    oo = "--open-only" in sys.argv
    r = main(site, open_only=oo)
    L = [f"# Panel median vs the human arm effect ({site}; {'open-weight models only' if oo else 'all models'})", "",
         fmt(r), "", f"H10 (90% CI of the panel-median gap within ±{EQ}): "
         + ("supported" if r["equivalent (±0.25)"] else "not supported"), "",
         "Single-model agent arm effects: " + ", ".join(f"{m} {v:+.2f}" for m, v in sorted(r["per model"].items(), key=lambda x: x[1])), "",
         "## Leave one model out", ""]
    for m in r["per model"]:
        L.append(f"- without {m}: " + fmt(main(site, drop=(m,), B=500, open_only=oo)).split("; ", 1)[1])
    out = "median.md" if site == "rscience" and not oo else f"median-{site}{'-open' if oo else ''}.md"
    open(os.path.join(RESULTS, "panel", out), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
