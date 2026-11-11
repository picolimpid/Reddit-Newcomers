"""E1b: can a handful of human-coded items calibrate JEV to the codebook? Platt scaling on logit(p) fit on k labelled
items (repeated random splits), evaluated on the rest: accuracy vs majority, Brier vs soft label, disagreement AUROC."""
import os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
import jev
from e1_perspectivist import CODES, OUT, ent, supervised_baseline, add_llm_judges
s = pd.read_parquet(os.path.join(OUT, "sample.parquet")).merge(pd.read_parquet(os.path.join(OUT, "jev.parquet")))
s = s.merge(supervised_baseline(s), on="comment_id")
s, JUDGES = add_llm_judges(s, {"j": "JEV", "s": "TFIDF+LR"})
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
rows = []
for c in CODES:
    h = s["h_" + c].values; maj = (h > .5).astype(int); cont = (h > 0) & (h < 1)
    print(c, "mean JEV p %.2f | mean human soft %.2f | JEV p>0.5 rate %.2f | majority rate %.2f" % (
        s["j_" + c].mean(), h.mean(), (s["j_" + c] > .5).mean(), maj.mean()))
    for k in (20, 40, 80):
        for judge, jname in JUDGES.items():
            accs, briers, dis = [], [], []
            for seed in range(30):
                idx = np.random.default_rng(seed).permutation(len(s)); tr, te = idx[:k], idx[k:]
                x = lg(s[f"{judge}_{c}"].values).reshape(-1, 1)
                # fit on soft labels: replicate each item as yes/no weighted by the human fraction
                X = np.vstack([x[tr], x[tr]]); y = np.r_[np.ones(k), np.zeros(k)]; w = np.r_[h[tr], 1 - h[tr]]
                if w[:k].sum() == 0 or w[k:].sum() == 0: continue
                cal = LogisticRegression(C=10).fit(X, y, sample_weight=w + 1e-6)
                p = cal.predict_proba(x[te])[:, 1]
                accs.append(((p > .5) == maj[te]).mean()); briers.append(np.mean((p - h[te]) ** 2))
                if cont[te].any() and (~cont[te]).any(): dis.append(roc_auc_score(cont[te], ent(p)))
            rows.append({"code": c, "k": k, "judge": jname + "+Platt",
                         "acc": np.mean(accs), "brier": np.mean(briers), "auroc_contested": np.mean(dis)})
    for judge, jname in JUDGES.items():
        p = s[f"{judge}_{c}"].values
        rows.append({"code": c, "k": 0, "judge": jname + " raw", "acc": ((p > .5) == maj).mean(),
                     "brier": np.mean((p - h) ** 2), "auroc_contested": roc_auc_score(cont, ent(p))})
r = pd.DataFrame(rows); r.to_csv(os.path.join(OUT, "calibration.csv"), index=False)
print(r.groupby(["judge", "k"])[["acc", "brier", "auroc_contested"]].mean().round(3))
print(pd.read_csv(os.path.join(OUT, "triage.csv")).groupby(["judge", "deferred"]).acc.mean().unstack().round(3).to_string())
