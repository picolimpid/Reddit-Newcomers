"""P9 analysis (plan frozen in PREREG_addendum_P9.md). Human target: negative-binomial IRR of newcomer comments per
non-AMA post, sticky vs control (CivilServant posts file). Agent: ratio of mean P(Yes) full / off over the same
agent × post cells, bootstrap over the shown posts; gap = log agent ratio − log human IRR, with the human
uncertainty added by a normal draw on the log scale. Also the within-agent Δlogit and the welcome / rules parts.

    python3 p9_analyze.py -> results/p9/summary.md, summary.csv"""
import glob
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from common import RESULTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def human():
    P = pd.read_csv(os.path.join(ROOT, "data", "rscience2016", "civilservant", "r_science_experiment_1_posts.09.26.2016.csv"))
    N = P[~P["post.ama"]].rename(columns={"newcomer.comments": "y"})
    m = smf.glm("y ~ TREAT", N, family=sm.families.NegativeBinomial(alpha=1)).fit()
    b, (lo, hi) = m.params.TREAT, m.conf_int().loc["TREAT"].values
    return b, (hi - lo) / 3.92, len(N)


def main(n=2000):
    hb, hse, hn = human()
    rng = np.random.default_rng(0)
    rows = []
    for f in sorted(glob.glob(os.path.join(RESULTS, "p9", "rscience-*", "scores.parquet"))):
        name = os.path.basename(os.path.dirname(f)).split("-", 1)[1]
        S = pd.read_parquet(f)
        W = S.pivot_table(index=["idx", "post_id"], columns="condition", values="p").reset_index()
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        g = W.groupby("post_id")[["off", "full", "welcome", "rules"]].sum()
        W["d"] = lg(W.full) - lg(W.off)
        gd = W.groupby("post_id").d.agg(["sum", "count"])
        B = rng.integers(0, len(g), (n, len(g)))
        tot = g.values[B].sum(1)  # n × 4
        cols = list(g.columns)
        lr = lambda num, den: np.log(tot[:, cols.index(num)] / tot[:, cols.index(den)])
        est = {c: np.log(g[c].sum() / g["off"].sum()) for c in ["full", "welcome", "rules"]}
        bl = {c: lr(c, "off") for c in ["full", "welcome", "rules"]}
        gap = bl["full"] - rng.normal(hb, hse, n)
        dw = gd["sum"].values[B].sum(1) / gd["count"].values[B].sum(1)
        q = lambda x: np.percentile(x, [2.5, 97.5])
        rows.append({"model": name, "mean P off": W.off.mean(), "ratio full/off": np.exp(est["full"]),
                     "ratio CI": "[%.2f, %.2f]" % tuple(np.exp(q(bl["full"]))),
                     "ratio welcome/off": np.exp(est["welcome"]), "ratio rules/off": np.exp(est["rules"]),
                     "log gap vs human": est["full"] - hb, "gap CI": "[%+.2f, %+.2f]" % tuple(q(gap)),
                     "matches": bool(q(gap)[0] <= 0 <= q(gap)[1]),
                     "within Δlogit": "%+.2f [%+.2f, %+.2f]" % (W.d.mean(), *q(dw)), "n cells": len(W)})
    D = pd.DataFrame(rows)
    os.makedirs(os.path.join(RESULTS, "p9"), exist_ok=True)
    D.to_csv(os.path.join(RESULTS, "p9", "summary.csv"), index=False)
    txt = (f"# P9: entry (would you comment on this post?) vs the real newcomer-participation effect\n\n"
           f"Human: IRR {np.exp(hb):.2f} [{np.exp(hb - 1.96 * hse):.2f}, {np.exp(hb + 1.96 * hse):.2f}] newcomer comments per "
           f"non-AMA post (negative binomial, n = {hn} posts). Agent ratio = mean P(Yes | sticky) / mean P(Yes | off), "
           "bootstrap over shown posts; gap on the log scale includes the human uncertainty.\n\n"
           + (D.round(3).to_markdown(index=False) if len(D) else "(no P9 runs yet)") + "\n")
    open(os.path.join(RESULTS, "p9", "summary.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
