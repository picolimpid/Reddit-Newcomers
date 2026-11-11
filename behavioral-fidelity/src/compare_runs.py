"""One row per (agent run, protocol): the scale-free fingerprint ratios, verdicts, RQ2 effects,
base rate and AUROC, next to the human reference.

    python compare_runs.py qwen3-1.7b qwen3-8b qwen3-14b ...   -> results/compare.{csv,md}"""
import os
import sys

import pandas as pd

from common import RESULTS
from evaluate import verdicts


def ratios(F, who):
    b = F[F.who == who].set_index(["spec", "predictor"]).beta
    return {"old/recent (c2v)": b[("FP1_traj_c2v", "fit_90_365")] / b[("FP1_traj_c2v", "fit_0_30")],
            "old/recent (Jev)": b[("FP1_traj_jev", "jw_d90_365")] / b[("FP1_traj_jev", "jw_d0_30")],
            "reply/fit": b[("FP2_reply", "reply_received")] / b[("FP3_fit", "c2v_big")],
            "fit OR": float(pd.Series(b[("FP3_fit", "c2v_big")]).rpow(2.718281828).iloc[0])}


def main(runs):
    rows = []
    for run in runs:
        d = os.path.join(RESULTS, run)
        F = pd.read_csv(os.path.join(d, "fingerprints.csv"))
        I = pd.read_csv(os.path.join(d, "interventions.csv"))
        R = pd.read_csv(os.path.join(d, "base_rates.csv")).set_index("who")
        if not rows:
            v = verdicts(F, "human")
            rows.append({"run": "human", "protocol": "-", **ratios(F, "human"), "FP passed": f"{sum(v.values())}/4",
                         "welcome (fit SD)": I.query("protocol == 'human (observational)'").in_fit_sd_units.iloc[0],
                         "base rate": R.loc["human", "mean_return_90d"], "AUROC 90d": R.loc["c2v fit alone", "auroc_90d"]})
        for proto in ("intent", "choice"):
            who = f"agent_{proto}"
            if who not in set(F.who):
                continue
            v = verdicts(F, who)
            g = I[I.protocol == proto].set_index("contrast")
            get = lambda c, k: g.loc[c, k] if c in g.index else float("nan")
            rows.append({"run": run, "protocol": proto, **ratios(F, who), "FP passed": f"{sum(v.values())}/4",
                         "welcome dlogit": get("welcoming - none", "dlogit"),
                         "welcome (fit SD)": get("welcoming - none", "in_fit_sd_units"),
                         "hostile dlogit": get("hostile - none", "dlogit"),
                         "real reply dlogit": get("observed - none", "dlogit"),
                         "base rate": R.loc[who, "mean_return_90d"], "AUROC 90d": R.loc[who, "auroc_90d"]})
    C = pd.DataFrame(rows)
    C.to_csv(os.path.join(RESULTS, "compare.csv"), index=False)
    md = ("Human row: ratios from the 90-day return model; 'welcome (fit SD)' is the observational reply "
          "association; AUROC is c2v fit alone.\n\n" + C.round(2).to_markdown(index=False))
    open(os.path.join(RESULTS, "compare.md"), "w").write(md + "\n")
    print(md)


if __name__ == "__main__":
    main(sys.argv[1:] or ["qwen3-1.7b", "qwen3-8b"])
