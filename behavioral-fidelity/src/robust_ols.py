"""Robustness of agent fingerprints to weighting. The primary spec (fractional logit) weights
agents by p(1-p), so when P(return) is near 0 for most agents (Qwen2.5-72B: median ~0) a few
dozen agents drive the estimates. Here every agent counts equally: OLS of logit(P) on the same
predictors, controls and post fixed effects (SEs clustered by post). Scale-free ratios are
comparable with the primary verdicts.

    python robust_ols.py qwen3-8b qwen3-32b qwen2.5-72b   -> results/robust_ols.csv"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import BENCH, CONTROLS, RESULTS

warnings.filterwarnings("ignore")
SPECS = {"c2v": ["fit_0_30", "fit_30_90", "fit_90_365"], "jev": ["jw_d0_30", "jw_d30_90", "jw_d90_365"],
         "reply": ["reply_received"], "fit": ["c2v_big"]}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main(runs):
    B = pd.read_parquet(BENCH)
    rows = []
    for run in runs:
        S = pd.read_parquet(os.path.join(RESULTS, run, "agent_scores.parquet"))
        for proto in ("intent", "choice"):
            s_ = S[(S.protocol == proto) & (S.condition == "observed")].set_index("idx")
            p = s_.p
            d = B.join(logit(p).rename("lp"))
            # choice: the randomized position of the r/X option is a control, as in evaluate.py
            d["pos"] = s_.target.map(lambda t: next(k for k, v in json.loads(t).items() if v == 1)) \
                if proto == "choice" else "-"
            extra = " + C(pos)" if proto == "choice" else ""
            b = {}
            for xs in SPECS.values():
                dd = d.dropna(subset=xs + CONTROLS + ["lp"]).copy()
                for x in xs:
                    if x != "reply_received":
                        dd[x] = (dd[x] - dd[x].mean()) / dd[x].std()
                m = smf.ols(f"lp ~ {' + '.join(xs + CONTROLS)}{extra} + C(post_id)", dd).fit(
                    cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.post_id)[0]})
                b |= {x: m.params[x] for x in xs}
            rows.append({"run": run, "protocol": proto, "median_P": p.median(), "share_P_gt_0.01": (p > .01).mean(),
                         "old/recent (c2v)": b["fit_90_365"] / b["fit_0_30"],
                         "old/recent (Jev)": b["jw_d90_365"] / b["jw_d0_30"],
                         "reply/fit": b["reply_received"] / b["c2v_big"], **{f"b_{k}": v for k, v in b.items()}})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(RESULTS, "robust_ols.csv"), index=False)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or ["qwen3-8b", "qwen3-14b", "qwen3-32b", "qwen2.5-72b"])
