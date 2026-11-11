"""H14/H15 (PREREG_addendum_H14.md, 2026-11-07): does the sticky's effect on newcomer return depend on how much Reddit
history the newcomer has, and do agents predict that? Discovery: r/science 2016 (exploratory). Tests: Futurology 2020
(primary), ffxiv2019 and science2020 (secondary), each only once its outcomes are fetched under its own freeze.

Human model, per site, newcomers with n_hist > 0 (the frozen futurology_analyze population):
    logit P(return_90d) ~ treat * z,  z = standardised log1p(n_hist) within site, SE clustered by post.
H14: the treat:z coefficient > 0 at Futurology, one-sided p < .05.
H15: agent tertile slopes (T3 − T1 of within-agent Δlogit full − off, hetero_history.py, frozen before any outcome);
     a model "matches" when its slope CI excludes 0 with the sign of the human treat:z estimate. Predicted: at most
     a third of the models with a scored run match at Futurology.

    python3 h14_analyze.py rscience futurology [ffxiv2019 science2020]  -> results/panel/h14.md"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import RESULTS
from panel_summary import ROOT, inputs

DIRS = {"futurology": os.path.join(ROOT, "data", "futurology2020")}


def load(site):
    R = inputs(site)
    if site != "rscience":
        p = os.path.join(DIRS.get(site, os.path.join(ROOT, "data", "sites", site)), "outcomes.parquet")
        if not os.path.exists(p):
            return None
        O = pd.read_parquet(p)
        assert (R.author.values == O.author.values).all()
        R = pd.concat([R, O[["return_90d"]]], axis=1)
    R = R[R.n_hist > 0].copy()
    lh = np.log1p(R.n_hist)
    R["z"] = (lh - lh.mean()) / lh.std()
    return R


def fit(R):
    m = smf.logit("return_90d ~ treat * z", R).fit(disp=0, cov_type="cluster",
                                                     cov_kwds={"groups": pd.factorize(R.post_id)[0]})
    b, se = m.params["treat:z"], m.bse["treat:z"]
    from scipy import stats
    return b, b - 1.96 * se, b + 1.96 * se, 1 - stats.norm.cdf(b / se), len(R)


def main(sites):
    L = ["# H14/H15: sticky × newcomer history (PREREG_addendum_H14.md)", "",
         "treat:z = change in the sticky's logit effect on 90-day return per SD of log(1 + past-year items).", "",
         "| site | role | n | treat:z [95% CI] | one-sided p | H14 | agent matches (H15) |", "|---|---|---|---|---|---|---|"]
    for s in sites:
        R = load(s)
        role = "discovery" if s == "rscience" else ("primary" if s == "futurology" else "secondary")
        if R is None:
            L.append(f"| {s} | {role} | – | outcomes not fetched | – | – | – |")
            continue
        b, lo, hi, p, n = fit(R)
        tag = "" if s == "rscience" else f"_{s}"
        A = pd.read_csv(os.path.join(RESULTS, "panel", f"hetero_history{tag}.csv"))
        A = A[A.model != "human"]
        hit = A[(A.slope_lo > 0) if b > 0 else (A.slope_hi < 0)]
        h14 = "–" if s == "rscience" else ("supported" if (p < .05) else "not supported")
        L.append(f"| {s} | {role} | {n} | {b:+.3f} [{lo:+.3f}, {hi:+.3f}] | {p:.3f} | {h14} | "
                 f"{len(hit)}/{len(A)} ({', '.join(hit.model) or 'none'}) |")
    L += ["", "H15 is supported at a site when matches ≤ one third of the scored models."]
    open(os.path.join(RESULTS, "panel", "h14.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1:] or ["rscience", "futurology", "ffxiv2019", "science2020"])
