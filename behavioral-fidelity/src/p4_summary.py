"""P4 summary (2026-11-08): sentence ablation across models.

For each results/ablation/<site>-<name>/scores.parquet: within-agent Δlogit(condition − off), post-cluster bootstrap.
Also the two probes (knowledge of the Matias result; the model's own 'expert' prediction).
Exploratory (P4 was never preregistered).

    python3 p4_summary.py  -> results/ablation/summary.md, summary.csv"""
import glob
import json
import os

import pandas as pd

from analyze_probes import boot, logit
import panel_summary as PS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "ablation")


def main():
    rows, probes = [], []
    for f in sorted(glob.glob(os.path.join(OUT, "*", "scores.parquet"))):
        run = os.path.basename(os.path.dirname(f))
        site, name = run.split("-", 1)
        S = pd.read_parquet(f)
        W = S.pivot_table(index="idx", columns="condition", values="p")
        R = PS.inputs(site)
        pid = R.loc[W.index, "post_id"].values
        for c in W.columns:
            if c == "off":
                continue
            m, lo, hi = boot((logit(W[c]) - logit(W["off"])).values, pid)
            rows.append(dict(site=site, model=name, condition=c, est=m, lo=lo, hi=hi, n=len(W),
                             mass=S.loc[S.condition == c, "mass"].mean()))
        pj = os.path.join(os.path.dirname(f), "probes.json")
        if os.path.exists(pj):
            P = json.load(open(pj))
            probes.append(dict(site=site, model=name, know_inc=P["knowledge"]["Increased"],
                               know_unch=P["knowledge"]["Unchanged"], exp_higher=P["expert"]["Higher"],
                               exp_lower=P["expert"]["Lower"]))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    L = ["# P4 sentence ablation across models (exploratory)", "",
         "Within-agent Δlogit(condition − off), post-cluster bootstrap 95% CI. 'all' = full sticky.", ""]
    for site in D.site.unique():
        T = D[D.site == site].assign(cell=lambda x: x.apply(
            lambda r: f"{r.est:+.2f} [{r.lo:+.2f},{r.hi:+.2f}]", axis=1))
        L += [f"## {site}", "", T.pivot(index="condition", columns="model", values="cell").to_markdown(), ""]
    if probes:
        L += ["## Probes", "", pd.DataFrame(probes).round(3).to_markdown(index=False), ""]
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
