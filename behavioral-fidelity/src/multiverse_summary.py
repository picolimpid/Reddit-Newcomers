"""P3 and P4 summaries (2026-11-06): prompt multiverse and sentence ablation.

    python3 multiverse_summary.py   -> results/panel/multiverse.md, results/panel/ablation.md
Multiverse: for each model with variant runs, within-agent Δlogit per (site, variant), base restricted to
the same agents; how often each model keeps its base sign; variance of the full-sticky Δ explained by
model vs variant vs site. Ablation: Δlogit of every sentence combination, and the knowledge/expert probes.
Swap (P8, swap_agents.py): Δlogit of each site's sticky text and a placebo pin, by model and population; variance of
the text effects explained by model vs population vs text -> results/panel/swap.md, swap.csv."""
import glob
import json
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from analyze_probes import boot, fmt, logit
from common import RESULTS
from panel_summary import inputs

VARIANTS = ["base", "q2", "nosys", "order", "observer"]
SITES = {"rscience": f"{RESULTS}/probes/rscience-", "futurology": f"{RESULTS}/futurology/futurology-"}
EQ = 0.25


def within(S, R, conds=("full", "rules", "welcome", "off+tmpl"), idx=None):
    W = S.pivot_table(index="idx", columns="condition", values="p")
    if idx is not None:
        W = W.loc[W.index.intersection(idx)]
    pid = R.loc[W.index, "post_id"]
    off = logit(W["off"])
    return {c: boot(logit(W[c]) - off, pid) for c in conds if c in W}, W.index


def sign(v):
    return "+" if v[1] > 0 and v[0] > EQ else ("−" if v[2] < 0 and v[0] < -EQ else "0")


def multiverse():
    rows = []
    for site, pre in SITES.items():
        R = inputs(site)
        for f in sorted(glob.glob(pre + "*-observer/scores.parquet")):
            name = os.path.basename(os.path.dirname(f))[len(os.path.basename(pre)):-len("-observer")]
            idx = None
            for v in VARIANTS[1:]:
                p = f"{pre}{name}-{v}/scores.parquet"
                if os.path.exists(p):
                    w, ix = within(pd.read_parquet(p), R)
                    idx = ix if idx is None else idx.intersection(ix)
                    rows += [{"site": site, "model": name, "variant": v, "condition": c, "est": x[0], "lo": x[1],
                              "hi": x[2]} for c, x in w.items()]
            base = f"{pre}{name}/scores.parquet"
            if os.path.exists(base) and idx is not None:
                w, _ = within(pd.read_parquet(base), R, idx=idx)
                rows += [{"site": site, "model": name, "variant": "base", "condition": c, "est": x[0], "lo": x[1],
                          "hi": x[2]} for c, x in w.items()]
    if not rows:
        return ["(no multiverse runs yet)"]
    D = pd.DataFrame(rows)
    D["sign"] = [sign((e, lo, hi)) for e, lo, hi in zip(D.est, D.lo, D.hi)]
    D.to_csv(f"{RESULTS}/panel/multiverse.csv", index=False)
    L = ["# P3: prompt multiverse (only the measurement instrument changes)", ""]
    for c in ["full", "rules", "welcome", "off+tmpl"]:
        T = D[D.condition == c].copy()
        T["cell"] = [f"{s} {e:+.2f}" for s, e in zip(T.sign, T.est)]
        L += [f"## {c} (sign, Δlogit)", "",
              T.pivot_table(index=["site", "model"], columns="variant", values="cell", aggfunc="first")
              .reindex(columns=VARIANTS).to_markdown(), ""]
    F = D[D.condition == "full"]
    keep = F.merge(F[F.variant == "base"][["site", "model", "sign"]], on=["site", "model"], suffixes=("", "_base"))
    keep = keep[keep.variant != "base"]
    L += [f"Share of variants that keep the base sign of the full sticky: {(keep.sign == keep.sign_base).mean():.2f} "
          f"({(keep.sign == keep.sign_base).sum()}/{len(keep)})", ""]
    if F.model.nunique() > 1 and F.variant.nunique() > 1:
        m = smf.ols("est ~ C(model) + C(variant) + C(site)", F).fit()
        import statsmodels.api as sm
        a = sm.stats.anova_lm(m, typ=2)
        a["share of SS"] = a.sum_sq / a.sum_sq.sum()
        L += ["Variance of the full-sticky Δlogit (type-II sums of squares):", "", a.round(3).to_markdown(), ""]
    return L


def ablation():
    rows, pr = [], []
    for d in sorted(glob.glob(f"{RESULTS}/ablation/*/")):
        site, name = os.path.basename(d.rstrip("/")).split("-", 1)
        if os.path.exists(d + "probes.json"):
            p = json.load(open(d + "probes.json"))
            pr.append({"site": site, "model": name,
                       **{f"knows: {k}": round(v, 2) for k, v in p["knowledge"].items() if k != "mass"},
                       **{f"expert: {k}": round(v, 2) for k, v in p["expert"].items() if k != "mass"}})
        if not os.path.exists(d + "scores.parquet"):
            continue
        S, R = pd.read_parquet(d + "scores.parquet"), inputs(site)
        W = S.pivot_table(index="idx", columns="condition", values="p")
        off, pid = logit(W["off"]), R.loc[W.index, "post_id"]
        for c in W.columns:
            if c != "off":
                v = boot(logit(W[c]) - off, pid)
                rows.append({"site": site, "model": name, "condition": c, "cell": f"{sign(v)} {v[0]:+.2f}"})
    if not rows:
        return ["(no ablation runs yet)"]
    D = pd.DataFrame(rows)
    L = ["# P4: sentence ablation (within agent, Δlogit vs no sticky; W welcome, T removal threat, M moderators, "
         "P purpose clause, N descriptive norm, A auto-removed replies)", ""]
    for s, g in D.groupby("site"):
        L += [f"## {s}", "", g.pivot_table(index="model", columns="condition", values="cell", aggfunc="first").to_markdown(), ""]
    if pr:
        L += ["## Model-level probes (no persona): does the model know the study, and what does it predict as an expert?",
              "", pd.DataFrame(pr).to_markdown(index=False), ""]
    return L


def swap():
    rows = []
    for f in sorted(glob.glob(f"{RESULTS}/swap/*/scores.parquet")):
        pop, name = os.path.basename(os.path.dirname(f)).split("-", 1)
        S, R = pd.read_parquet(f), inputs(pop)
        conds = [c for c in S.condition.unique() if c != "off"]
        w, _ = within(S, R, conds=conds)
        rows += [{"pop": pop, "model": name, "condition": c, "est": x[0], "lo": x[1], "hi": x[2]} for c, x in w.items()]
    if not rows:
        return ["(no swap runs yet)"]
    D = pd.DataFrame(rows)
    D["sign"] = [sign((e, lo, hi)) for e, lo, hi in zip(D.est, D.lo, D.hi)]
    D.to_csv(f"{RESULTS}/panel/swap.csv", index=False)
    D["cell"] = [f"{s} {e:+.2f}" for s, e in zip(D.sign, D.est)]
    L = ["# P8: text x population swap (within agent, Δlogit vs no sticky; text_<site> = that site's sticky with the "
         "subreddit name set to the agent's own community; placebo = content-free pinned comment)", "",
         D.pivot_table(index=["model", "pop"], columns="condition", values="cell", aggfunc="first").to_markdown(), ""]
    T = D[D.condition.str.startswith("text_")]
    if T.model.nunique() > 1:
        import statsmodels.api as sm
        m = smf.ols("est ~ C(model) + C(pop) + C(condition)", T).fit()
        a = sm.stats.anova_lm(m, typ=2)
        a["share of SS"] = a.sum_sq / a.sum_sq.sum()
        L += ["Variance of the text effects (type-II SS; model vs population vs text):", "", a.round(3).to_markdown(), ""]
        g = T.groupby("model").sign.agg(lambda x: x.value_counts().iloc[0] / len(x))
        L += ["Share of (population, text) cells with the model's modal sign:", "", g.round(2).to_frame("share").to_markdown(), ""]
    return L


if __name__ == "__main__":
    os.makedirs(f"{RESULTS}/panel", exist_ok=True)
    for fn, out in [(multiverse, "multiverse.md"), (ablation, "ablation.md"), (swap, "swap.md")]:
        txt = "\n".join(fn()) + "\n"
        open(f"{RESULTS}/panel/{out}", "w").write(txt)
        print(txt)
