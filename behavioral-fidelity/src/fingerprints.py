"""Population-level behavioral fingerprints, estimated identically for humans and agents.

Every fingerprint is a post-fixed-effects logit of a return outcome on pre-entry and
first-contact variables, with the Reddit Newcomers controls and SEs clustered by post:
  humans   y = observed return (0/1), binomial GLM = logit; posts without outcome variation
           are dropped, as in the original analysis
  agents   y = the agent's return probability, fractional logit (binomial quasi-likelihood),
           on exactly the rows the human model used
Continuous predictors are z-scored on the estimation rows, so effects are per +1 SD.

    python fingerprints.py      -> results/human_fingerprints.csv (+ check against REPORT)"""
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from common import BENCH, CONTROLS, RESULTS

warnings.filterwarnings("ignore")

# name -> (predictors, sample restriction)
SPECS = {
    "FP1_traj_c2v": (["fit_0_30", "fit_30_90", "fit_90_365"], None),
    "FP1_traj_jev": (["jw_d0_30", "jw_d30_90", "jw_d90_365"], None),
    "FP2_reply": (["reply_received"], None),
    "FP2_welcoming": (["got_welcoming"], "reply_received == 1"),
    "FP2_gatekeeping": (["got_gatekeeping"], "reply_received == 1"),
    "FP3_fit": (["c2v_big"], None),
    "FP3_joint": (["c2v_big", "reply_received"], None),
}
BINARY = {"reply_received", "top_level"}


def estimate(d, y, xs, rows=None, frac=False, extra=()):
    """Returns (table of per-predictor effects, index of rows used, fitted result).
    extra: categorical columns added as controls (e.g. the randomized option position)."""
    d = d.dropna(subset=[y] + xs + CONTROLS).copy()
    if rows is not None:
        d = d.loc[d.index.intersection(rows)]
    elif not frac:
        d = d[d.groupby("post_id")[y].transform("nunique") > 1]
    for x in xs:
        if x not in BINARY:
            d[x] = (d[x] - d[x].mean()) / d[x].std()
    ctrl = [c for c in CONTROLS if d[c].nunique() > 1]
    f = f"{y} ~ {' + '.join(xs + ctrl + [f'C({c})' for c in extra])} + C(post_id)"
    m = smf.glm(f, d, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.post_id)[0]})
    out = []
    for x in xs:
        b, se = m.params[x], m.bse[x]
        out.append({"predictor": x, "beta": b, "se": se, "OR": np.exp(b), "ci_lo": np.exp(b - 1.96 * se),
                    "ci_hi": np.exp(b + 1.96 * se), "p": m.pvalues[x], "n": int(m.nobs)})
    return pd.DataFrame(out), d.index, m


def contrast(m, a, b):
    """beta_a - beta_b with its clustered SE and p."""
    from scipy.stats import norm
    V = m.cov_params().loc[[a, b], [a, b]].values
    diff = m.params[a] - m.params[b]
    se = np.sqrt(V[0, 0] + V[1, 1] - 2 * V[0, 1])
    return diff, se, 2 * norm.sf(abs(diff / se))


def all_fingerprints(B, y, rows_by_spec=None, frac=False, tag="", extra=()):
    """All SPECS for one outcome column. rows_by_spec pins the estimation rows (for agents)."""
    res, rows = [], {}
    for name, (xs, where) in SPECS.items():
        d = B.query(where) if where else B
        try:
            t, idx, m = estimate(d, y, xs, rows=None if rows_by_spec is None else rows_by_spec[name], frac=frac,
                                extra=extra)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {name} ({y}): {str(e)[:80]}")
            continue
        t.insert(0, "spec", name)
        if name.startswith("FP1"):
            diff, se, p = contrast(m, xs[0], xs[-1])
            t["recent_minus_oldest"], t["recent_minus_oldest_p"] = diff, p
        res.append(t)
        rows[name] = idx
    r = pd.concat(res, ignore_index=True)
    r.insert(0, "outcome", y)
    r.insert(0, "who", tag)
    return r, rows


if __name__ == "__main__":
    B = pd.read_parquet(BENCH)
    out = []
    for y in ("return_90d", "return_180d"):
        r, _ = all_fingerprints(B, y, tag="human")
        out.append(r)
    H = pd.concat(out, ignore_index=True)
    os.makedirs(RESULTS, exist_ok=True)
    H.to_csv(os.path.join(RESULTS, "human_fingerprints.csv"), index=False)
    pd.set_option("display.width", 200)
    print(H[["outcome", "spec", "predictor", "OR", "ci_lo", "ci_hi", "p", "n", "recent_minus_oldest_p"]].round(3).to_string())
    print("\nbase rates:", B[["return_30d", "return_90d", "return_180d", "regular_180d"]].mean().round(3).to_dict())
    print("REPORT check: c2v windows 90d 1.80/1.03/1.00, 180d 2.06/1.24/0.91 (n=1,130); Jev windows 180d 1.57/0.98/0.83;"
          " c2v fit 90d 1.52 (base controls in §4.10 differ slightly), 180d 1.64; reply 90d 1.07")
