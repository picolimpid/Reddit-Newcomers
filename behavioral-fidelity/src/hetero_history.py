"""For whom? (2026-11-07, exploratory, not preregistered.) r/science 2016 sticky effect on 90-day return, split by how
much Reddit history the newcomer had (tertiles of n_hist, the past-year items the persona is built from).

Human: arm effect within tertile (logit of pooled return rates, treat − control, post bootstrap).
Agent: within-agent Δlogit full − off within tertile (post bootstrap).
Slope = tertile 3 − tertile 1 (experienced minus thin-history newcomers), same bootstrap draws for both tertiles.
The question is whether models invent heterogeneity the humans do not show (or miss one they do).

    python3 hetero_history.py [site]  -> results/panel/hetero_history[_<site>].md, .csv
At sites without fetched outcomes (Futurology, ffxiv2019, science2020 before their reveal) only agent rows are written."""
import sys
import os

import numpy as np
import pandas as pd

from analyze_probes import logit
from common import RESULTS
from panel_summary import inputs, runs

N = 1000


def lo(m):
    return np.log(m / (1 - m))


def human(R, rng):
    g = R.groupby(["post_id", "tert"]).agg(t=("treat", "first"), y=("return_90d", "sum"), n=("return_90d", "size"))
    g = g.reset_index()
    posts = g.post_id.unique()

    def est(G):
        out = {}
        for k in (0, 1, 2):
            s = G[G.tert == k]
            t, c = s[s.t == 1], s[s.t == 0]
            out[k] = lo(t.y.sum() / t.n.sum()) - lo(c.y.sum() / c.n.sum())
        return out
    gi = {p: d for p, d in g.groupby("post_id")}
    b = [est(pd.concat([gi[p] for p in rng.choice(posts, len(posts))])) for _ in range(N)]
    return est(g), b


def agent(W, R, rng):
    d = pd.DataFrame({"v": logit(W["full"]) - logit(W["off"]), "post": R.loc[W.index, "post_id"].values,
                      "tert": R.loc[W.index, "tert"].values}).dropna()
    g = d.groupby(["post", "tert"]).v.agg(["sum", "count"]).unstack(fill_value=0)
    k = rng.integers(0, len(g), (N, len(g)))
    pt = {t: d[d.tert == t].v.mean() for t in (0, 1, 2)}
    b = [{t: g[("sum", t)].values[r].sum() / g[("count", t)].values[r].sum() for t in (0, 1, 2)} for r in k]
    return pt, b


def ci(b, f):
    return np.percentile([f(x) for x in b], [2.5, 97.5])


def main(site="rscience"):
    R = inputs(site).copy()
    R["tert"] = pd.qcut(R.n_hist.rank(method="first"), 3, labels=False)
    cuts = R.groupby("tert").n_hist.agg(["min", "max"]).values.tolist()
    rows = []
    if "return_90d" in R:
        pt, b = human(R, np.random.default_rng(0))
        rows.append(("human", pt, b))
    for s, name, f in runs():
        if s != site:
            continue
        W = pd.read_parquet(f).pivot_table(index="idx", columns="condition", values="p")
        if not {"off", "full"} <= set(W.columns):
            continue
        pt, b = agent(W, R, np.random.default_rng(0))
        rows.append((name, pt, b))
    out = []
    for name, pt, b in rows:
        r = {"model": name}
        for t in (0, 1, 2):
            l, h = ci(b, lambda x: x[t])
            r[f"T{t + 1}"] = f"{pt[t]:+.2f} [{l:+.2f}, {h:+.2f}]"
        l, h = ci(b, lambda x: x[2] - x[0])
        r["slope T3−T1"] = f"{pt[2] - pt[0]:+.2f} [{l:+.2f}, {h:+.2f}]"
        r["slope_est"], r["slope_lo"], r["slope_hi"] = pt[2] - pt[0], l, h
        out.append(r)
    D = pd.DataFrame(out)
    tag = "" if site == "rscience" else f"_{site}"
    D.to_csv(os.path.join(RESULTS, "panel", f"hetero_history{tag}.csv"), index=False)
    A = D[D.model != "human"]
    nz = A[(A.slope_lo > 0) | (A.slope_hi < 0)]
    txt = (f"# For whom? Sticky effect on 90-day return by newcomer history tertile ({site}; exploratory)\n\n"
           f"Tertiles of n_hist (past-year items): T1 {cuts[0]}, T2 {cuts[1]}, T3 {cuts[2]}. Human = arm effect in the RCT "
           "(logit of pooled rates); agents = within-agent Δlogit full − off. 95% CIs by post bootstrap (1,000).\n\n"
           + D.drop(columns=["slope_est", "slope_lo", "slope_hi"]).to_markdown(index=False)
           + f"\n\nModels whose T3−T1 slope CI excludes 0: {len(nz)}/{len(A)} "
           + (f"({', '.join(nz.model)})" if len(nz) else "") + ".\n")
    open(os.path.join(RESULTS, "panel", f"hetero_history{tag}.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main(*sys.argv[1:2])
