"""Fidelity scorecard (2026-11-07): every lever we can check on the same r/science 2016 newcomers, per model.

Rows: human, then each model with r/science panel scores. Columns:
  sticky (RCT)      within-agent Δlogit of the full sticky; agent − human arm-effect gap (paired post bootstrap)
  template reply    within-agent Δlogit of adding a generic welcoming reply (no human counterpart; the real-reply
                    association in the same people is shown in the human row, observational)
  upvotes (P5)      within-agent Δlogit 50 vs 1 point; agent − human ATT gap (same quasi-experimental design)
  who returns       AUC of P(no sticky) for the real 90-day return (counts-only logistic baseline in the human row)
  matches           number of gaps whose 95% CI includes 0, out of the gaps available
  sensitivity       mean |within-agent Δ| over sticky, template reply and upvotes (human: all three ≈ 0)

    python3 scorecard.py   -> results/panel/scorecard.md, scorecard.csv"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import p5_analyze as P5
from analyze_probes import boot, logit
from common import RESULTS
from panel_summary import inputs, runs
from rscience_analyze import paired_gap


def f(v):
    return f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]"


def main():
    R = inputs("rscience")
    W0 = pd.read_csv(os.path.join(RESULTS, "panel", "within.csv"))
    W0 = W0[W0.site == "rscience"].set_index(["model", "condition"])
    R5, _, _, _ = P5.design()
    rows = []
    g = R.groupby(R.n_replies > 0).return_90d.mean()
    rows.append({"model": "human", "sticky within": "–", "sticky gap": "−0.03 [−0.23, +0.17] (effect)",
                 "template reply within": f"real reply: {logit(pd.Series([g[True]])).iloc[0] - logit(pd.Series([g[False]])).iloc[0]:+.2f} (obs.)",
                 "upvotes within": "–", "upvotes gap": "−0.14 [−0.40, +0.15] (ATT)", "AUC": "0.662 (counts)",
                 "matches": "–", "sensitivity": 0.0})
    for site, name, fn in runs():
        if site != "rscience":
            continue
        S = pd.read_parquet(fn)
        W = S.pivot_table(index="idx", columns="condition", values="p")
        if not {"off", "full"} <= set(W.columns):
            continue
        H = R.loc[W.index].copy()
        H["p_arm"] = np.where(H.treat == 1, W["full"], W["off"])
        gap = paired_gap(H, n=1000)
        sw = (W0.loc[(name, "full"), ["est", "lo", "hi"]].values if (name, "full") in W0.index else [np.nan] * 3)
        tw = (W0.loc[(name, "off+tmpl"), ["est", "lo", "hi"]].values if (name, "off+tmpl") in W0.index else [np.nan] * 3)
        row = {"model": name, "sticky within": f(sw), "sticky gap": f(gap), "template reply within": f(tw)}
        gaps = [gap]
        sens = [abs(sw[0]), abs(tw[0])]
        p5 = os.path.join(RESULTS, "p5", f"rscience2016-{name}", "scores.parquet")
        if os.path.exists(p5):
            V = pd.read_parquet(p5).pivot_table(index="idx", columns="condition", values="p")
            uw = boot(logit(V["high"]) - logit(V["low"]), R.loc[V.index, "post_id"])
            Q = R5.copy()
            Q["p"] = V["real"].reindex(Q.index)
            Q = Q.dropna(subset=["p"])
            est, ci, dci = P5.boot_att(Q, ["return_90d", "p"], n=500)
            ug = (est[1] - est[0], *dci[:, 1])
            row.update({"upvotes within": f(uw), "upvotes gap": f(ug)})
            gaps.append(ug)
            sens.append(abs(uw[0]))
        else:
            row.update({"upvotes within": "(pending)", "upvotes gap": "(pending)"})
        ok = W["off"].notna()
        row["AUC"] = f"{roc_auc_score(R.loc[W.index[ok], 'return_90d'], W['off'][ok]):.3f}"
        row["matches"] = f"{sum(lo <= 0 <= hi for _, lo, hi in gaps)}/{len(gaps)}"
        row["sensitivity"] = round(float(np.nanmean(sens)), 2)
        rows.append(row)
    D = pd.DataFrame(rows)
    D = pd.concat([D.iloc[:1], D.iloc[1:].sort_values("sensitivity")])
    D.to_csv(os.path.join(RESULTS, "panel", "scorecard.csv"), index=False)
    txt = ("# Fidelity scorecard: r/science 2016 newcomers (same people for every lever)\n\n"
           "Within = the agent's own causal response (Δlogit, post bootstrap). Gap = agent − human on the same design. "
           "Matches = gaps whose 95% CI includes 0. Sensitivity = mean |within Δ| over the levers (the human effects of all "
           "three levers on 90-day return are ≈ 0). A model can match by being insensitive to everything.\n\n"
           + D.to_markdown(index=False) + "\n")
    open(os.path.join(RESULTS, "panel", "scorecard.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
