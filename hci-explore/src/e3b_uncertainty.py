"""E3b (free, cached): does JEV's score-distribution uncertainty flag the summaries where it disagrees with experts?
Re-reads cached JEV responses; error = |JEV expected score - expert mean| (both 1-5)."""
import os, sys, json, hashlib
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
import jev
from e3_checklist_eval import CRIT, questions, OUT
s = pd.read_parquet(os.path.join(jev.ROOT, "data", "ext", "summeval", "summeval.parquet"))
rows = []
for di, r in s.iterrows():
    for si, summ in enumerate(r.machine_summaries):
        o = jev.ask({"source_article": r.text, "summary": summ}, questions(), tag="e3")
        assert o["_cached"]
        rec = {"doc": di, "sys": si}
        for c in CRIT:
            pr = np.array([float(v) for _, v in sorted(o["answers"][c]["probabilities"].items())])
            rec[c + "_ent"] = -(pr[pr > 0] * np.log(pr[pr > 0])).sum()
            rec[c + "_conf"] = float(o["answers"][c].get("confidence", np.nan))
            rec[c + "_pred"] = 1 + float((pr * np.arange(5)).sum())
            rec[c + "_gold"] = float(r[c][si])
        rows.append(rec)
d = pd.DataFrame(rows)
out = []
for c in CRIT:
    err = (d[c + "_pred"] - d[c + "_gold"]).abs()
    big = err > 1.0
    out.append({"criterion": c, "mean_abs_err": err.mean(), "share_err>1": big.mean(),
                "spearman(entropy, err)": spearmanr(d[c + "_ent"], err).statistic,
                "auroc(entropy -> err>1)": roc_auc_score(big, d[c + "_ent"]),
                "auroc(1-confidence -> err>1)": roc_auc_score(big, -d[c + "_conf"]),
                # review budget: experts check the 20% most uncertain summaries -> share of big errors caught
                "big errors caught @20% review": big[d[c + "_ent"] >= d[c + "_ent"].quantile(.8)].sum() / big.sum()})
r = pd.DataFrame(out).set_index("criterion").round(3)
print(r.to_string()); r.to_csv(os.path.join(OUT, "uncertainty.csv"))
print("spent $%.4f (should be unchanged)" % jev.spent_usd())
