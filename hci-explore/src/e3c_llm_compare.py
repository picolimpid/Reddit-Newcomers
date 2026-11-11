"""E3c: JEV vs LLM judges (OpenRouter, G-Eval-style prompt, probability-weighted 1-5 score from first-token logprobs)
on SummEval.  Also: does each judge's score-distribution entropy flag its own large errors?"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import jev  # noqa: E402
from e3_checklist_eval import CRIT, OUT, doc_corr, questions  # noqa: E402

NAMES = {"gpt-4o-mini": "GPT-4o-mini", "qwen3-32b": "Qwen3-32B (API)", "qwen3-32b-local": "Qwen3-32B (local)", "gpt-4o": "GPT-4o"}
s = pd.read_parquet(os.path.join(jev.ROOT, "data", "ext", "summeval", "summeval.parquet"))

# JEV: expected score + entropy from cached responses (free)
rows = []
for di, r in s.iterrows():
    for si, summ in enumerate(r.machine_summaries):
        o = jev.ask({"source_article": r.text, "summary": summ}, questions(), tag="e3")
        for c in CRIT:
            pr = np.array([float(v) for _, v in sorted(o["answers"][c]["probabilities"].items())])
            rows.append({"doc": di, "sys": si, "crit": c, "judge": "JEV", "pred": 1 + (pr * np.arange(5)).sum(),
                         "ent": -(pr[pr > 0] * np.log(pr[pr > 0])).sum(), "gold": float(r[c][si])})
for f in sorted(f for f in glob.glob(os.path.join(OUT, "llm_*.jsonl")) if not os.path.basename(f).startswith("llm_in")):
    tag = os.path.basename(f)[4:-6]
    for line in open(f):
        x = json.loads(line)
        di, si, c = x["id"].split("|")
        pr = np.array([x["probs"][k] for k in "12345"])
        rows.append({"doc": int(di), "sys": int(si), "crit": c, "judge": NAMES.get(tag, tag), "mass": x["mass"],
                     "pred": 1 + (pr * np.arange(5)).sum(), "ent": -(pr[pr > 0] * np.log(pr[pr > 0])).sum(),
                     "gold": float(s.loc[int(di), c][int(si)]), "lat": x.get("latency_s")})
d = pd.DataFrame(rows)
judges = list(dict.fromkeys(d.judge))


def table(sub, label):
    out = []
    for j in judges:
        for c in CRIT:
            g = sub[(sub.judge == j) & (sub.crit == c)]
            if g.doc.nunique() < 5:
                continue
            sp, kt = doc_corr(g, "pred", "gold")
            out.append({"judge": j, "crit": c, "spearman": sp, "kendall": kt, "n_docs": g.doc.nunique()})
    t = pd.DataFrame(out)
    if t.empty:
        return t
    sp = t.pivot(index="judge", columns="crit", values="spearman")[CRIT]; sp["avg"] = sp.mean(axis=1)
    kt = t.pivot(index="judge", columns="crit", values="kendall")[CRIT]; kt["avg"] = kt.mean(axis=1)
    print(f"\n== {label}: Spearman (n docs per judge: {t.groupby('judge').n_docs.max().to_dict()})")
    print(sp.loc[[j for j in judges if j in sp.index]].round(3).to_string())
    print("Kendall avg:", kt["avg"].round(3).to_dict())
    return sp


full = table(d, "all docs")
common = d.groupby("doc").judge.nunique()
sub_docs = common[common == len(judges)].index
sub = table(d[d.doc.isin(sub_docs)], f"common subset ({len(sub_docs)} docs)")
pd.concat({"all": full, "common": sub}).to_csv(os.path.join(OUT, "llm_compare.csv"))

print("\n== entropy -> |error|>1 (AUROC), share of big errors caught when reviewing the 20% most uncertain")
unc = []
for j in judges:
    for c in CRIT:
        g = d[(d.judge == j) & (d.crit == c)]
        big = (g.pred - g.gold).abs() > 1
        if big.sum() < 5 or big.all():
            continue
        top = g.ent >= g.ent.quantile(.8)
        unc.append({"judge": j, "crit": c, "big_err_share": big.mean(), "auroc": roc_auc_score(big, g.ent),
                    "caught@20%": (big & top).sum() / big.sum(), "mean_abs_err": (g.pred - g.gold).abs().mean()})
u = pd.DataFrame(unc)
print(u.pivot(index="judge", columns="crit", values="auroc").round(3).to_string())
print(u.pivot(index="judge", columns="crit", values="caught@20%").round(3).to_string())
print(u.pivot(index="judge", columns="crit", values="mean_abs_err").round(2).to_string())
u.to_csv(os.path.join(OUT, "llm_uncertainty.csv"), index=False)
if "mass" in d:
    print("\nmedian probability mass on 1-5 tokens:", d.groupby("judge")["mass"].median().round(3).to_dict())
    print("median latency per call (s):", d.groupby("judge")["lat"].median().round(2).to_dict())
