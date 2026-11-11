"""Within-agent time-reversal test: same history content with mirrored ages (--history reverse).
If an agent uses the pre-entry trajectory, reversing it should lower P(return) most for users who
were drifting towards the community (recent fit - 3-12-month fit > 0).

    python reversal.py qwen3-8b qwen3-8b-reverse   -> results/<reverse run>/reversal.csv"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import BENCH, RESULTS
from fingerprints import estimate


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def main(real, rev):
    B = pd.read_parquet(BENCH)
    B["drift"] = B.fit_0_30 - B.fit_90_365
    def get(run, proto):
        S = pd.read_parquet(os.path.join(RESULTS, run, "agent_scores.parquet"))
        return S[(S.protocol == proto) & (S.condition == "observed")].set_index("idx").p

    rows = []
    for proto in ("intent", "choice"):
        d = B.join(get(real, proto).rename("p_real")).join(get(rev, proto).rename("p_rev"))
        d["dlogit"] = logit(d.p_rev) - logit(d.p_real)
        dd = d.dropna(subset=["drift", "dlogit"]).copy()
        dd["z"] = (dd.drift - dd.drift.mean()) / dd.drift.std()
        m = smf.ols("dlogit ~ z", dd).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.post_id)[0]})
        lo, hi = m.conf_int().loc["z"]
        rows.append({"protocol": proto, "spearman_real_rev": d.p_real.corr(d.p_rev, method="spearman"),
                     "mean_dlogit": d.dlogit.mean(), "dlogit_per_sd_drift": m.params.z, "ci_lo": lo, "ci_hi": hi,
                     "p": m.pvalues.z, "n": int(m.nobs)})
    t, _, _ = estimate(B, "return_90d", ["drift"])
    rows.append({"protocol": "human: return_90d ~ drift (log OR per SD)", "dlogit_per_sd_drift": t.beta[0],
                 "ci_lo": np.log(t.ci_lo[0]), "ci_hi": np.log(t.ci_hi[0]), "p": t.p[0], "n": t.n[0]})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(RESULTS, rev, "reversal.csv"), index=False)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main(*(sys.argv[1:3] if len(sys.argv) > 2 else ("qwen3-8b", "qwen3-8b-reverse")))
