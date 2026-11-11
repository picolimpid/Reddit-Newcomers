"""Paired bootstrap 95% CIs for the pilot comparisons, from saved outputs and the JEV cache only (no API calls).

E1  resample the 400 comments; each metric per code, then averaged over the 6 codes (as in the report).
    Calibrated (k=20 anchors): each replicate draws 20 anchors shared by all judges, fits Platt per judge and code, and
    scores a bootstrap resample of the other 380 comments, so the CI covers anchor choice and test sampling.
E2  resample the 120 moments (the 60 with a proposed task come along).
E3  resample the 100 documents; per-document Spearman over the 16 summaries, averaged over documents and criteria;
    entropy -> |error| > 1 AUROC on consistency.
Writes results/ci/{e1,e1_cal,e2,e3}.csv; prints point estimate [2.5%, 97.5%] for each judge and for JEV minus judge.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
os.environ["JEV_MAX_USD"] = "0"  # any uncached JEV request raises BudgetExceeded: this script must not spend
import jev  # noqa: E402
from e1_perspectivist import CODES, OUT as E1, add_llm_judges, ent, supervised_baseline  # noqa: E402
from e3_checklist_eval import CRIT, OUT as E3, questions  # noqa: E402

CI = os.path.join(jev.ROOT, "results", "ci")
os.makedirs(CI, exist_ok=True)
B = int(os.environ.get("B", "2000"))


def auc(y, s):
    y = np.asarray(y, bool)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def spear(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return np.nan if den == 0 else (ra * rb).sum() / den


def summarize(point, reps, ref="JEV"):
    """point: {judge: {metric: v}}, reps: list of same -> long table with CIs, plus ref-minus-judge differences."""
    out = []
    metrics = list(next(iter(point.values())).keys())
    for m in metrics:
        for j in point:
            v = np.array([r[j][m] for r in reps], float)
            out.append({"metric": m, "judge": j, "est": point[j][m], "lo": np.nanpercentile(v, 2.5),
                        "hi": np.nanpercentile(v, 97.5)})
            if j != ref:
                d = np.array([r[ref][m] - r[j][m] for r in reps], float)
                out.append({"metric": m, "judge": f"{ref} - {j}", "est": point[ref][m] - point[j][m],
                            "lo": np.nanpercentile(d, 2.5), "hi": np.nanpercentile(d, 97.5),
                            "p_le_0": np.nanmean(d <= 0)})
    return pd.DataFrame(out)


def show(t, title):
    print(f"\n== {title}")
    for m, g in t.groupby("metric", sort=False):
        cells = [f"{r.judge}: {r.est:.3f} [{r.lo:.3f}, {r.hi:.3f}]" + ("*" if " - " in r.judge and (r.lo > 0 or r.hi < 0) else "")
                 for r in g.itertuples()]
        print(f"  {m}\n    " + "\n    ".join(cells))


# ---------------------------------------------------------------- E1
s = pd.read_parquet(os.path.join(E1, "sample.parquet")).merge(pd.read_parquet(os.path.join(E1, "jev.parquet")))
s = s.merge(supervised_baseline(s), on="comment_id")
s, J1 = add_llm_judges(s, {"j": "JEV", "s": "TF-IDF+LR"})
H = {c: s["h_" + c].values for c in CODES}
P = {(k, c): s[f"{k}_{c}"].values for k in J1 for c in CODES}


def e1_metrics(idx):
    res = {}
    for k, name in J1.items():
        acc = {"spearman_soft": [], "auroc_majority": [], "brier_soft": [], "auroc_contested": []}
        for c in CODES:
            h, p = H[c][idx], P[(k, c)][idx]
            acc["spearman_soft"].append(spear(p, h))
            acc["auroc_majority"].append(auc(h > .5, p))
            acc["brier_soft"].append(np.mean((p - h) ** 2))
            acc["auroc_contested"].append(auc((h > 0) & (h < 1), ent(p)))
        res[name] = {m: np.nanmean(v) for m, v in acc.items()}
    return res


n1 = len(s)
rng = np.random.default_rng(0)
t1 = summarize(e1_metrics(np.arange(n1)), [e1_metrics(rng.integers(0, n1, n1)) for _ in range(B)])
t1.to_csv(os.path.join(CI, "e1.csv"), index=False)
show(t1, f"E1 raw, zero-shot (n={n1} comments, B={B})")

lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
LG = {key: lg(v) for key, v in P.items()}


def e1_cal(seed, k=20, boot=True):
    r = np.random.default_rng(seed)
    perm = r.permutation(n1)
    tr, rest = perm[:k], perm[k:]
    te = rest[r.integers(0, len(rest), len(rest))] if boot else rest
    res = {name: {"acc": [], "brier": [], "auroc_contested": []} for name in J1.values()}
    for c in CODES:
        h = H[c]
        if h[tr].sum() == 0 or (1 - h[tr]).sum() == 0:
            continue  # cannot fit on these anchors: skip this code for every judge (paired)
        y = np.r_[np.ones(k), np.zeros(k)]; w = np.r_[h[tr], 1 - h[tr]] + 1e-6
        for key, name in J1.items():
            x = LG[(key, c)].reshape(-1, 1)
            p = LogisticRegression(C=10).fit(np.vstack([x[tr], x[tr]]), y, sample_weight=w).predict_proba(x[te])[:, 1]
            res[name]["acc"].append(((p > .5) == (h[te] > .5)).mean())
            res[name]["brier"].append(np.mean((p - h[te]) ** 2))
            res[name]["auroc_contested"].append(auc((h[te] > 0) & (h[te] < 1), ent(p)))
    return {n: {m: np.nanmean(v) for m, v in d.items()} for n, d in res.items()}


B_CAL = max(B // 2, 200)
cal_point = [e1_cal(sd, boot=False) for sd in range(30)]  # same 30 anchor draws as e1b_calibrate (no resampling)
pt = {n: {m: np.mean([r[n][m] for r in cal_point]) for m in cal_point[0][n]} for n in cal_point[0]}
t1c = summarize(pt, [e1_cal(10_000 + b) for b in range(B_CAL)])
t1c.to_csv(os.path.join(CI, "e1_cal.csv"), index=False)
show(t1c, f"E1 after Platt calibration on 20 anchors (B={B_CAL}; point = mean of 30 anchor draws)")

# ---------------------------------------------------------------- E2
E2 = os.path.join(jev.ROOT, "results", "e2")
d2 = pd.read_parquet(os.path.join(E2, "jev.parquet"))
names2 = {"gpt-4o-mini": "GPT-4o-mini", "qwen3-32b": "Qwen3-32B (API)", "gpt-4o": "GPT-4o",
          "qwen3-32b-local": "Qwen3-32B (local)"}
J2 = {"JEV": "j_"}
for f in sorted(glob.glob(os.path.join(E2, "llm_*.jsonl"))):
    tag = os.path.basename(f)[4:-6]
    pre = "L" + tag.replace("-", "").replace(".", "") + "_"
    q = pd.read_json(f, lines=True)
    q[["id", "kind"]] = q.id.astype(str).str.split("|", expand=True)
    q["p"] = q.probs.apply(lambda x: x["Yes"])
    d2 = d2.merge(q.pivot(index="id", columns="kind", values="p").add_prefix(pre).reset_index(), on="id", how="left")
    J2[names2.get(tag, tag)] = pre


def e2_metrics(idx):
    d = d2.iloc[idx]
    t = d[d.has_task]
    res = {}
    for name, pre in J2.items():
        acc_t = t[pre + "accept"].values
        cd, fa = acc_t[t.valid.values], acc_t[~t.valid.values]
        pred_valid = np.where(d.has_task, d[pre + "accept"] > .5, d[pre + "need"] < .5)
        res[name] = {"need_auroc": auc(d.help_needed.values, d[pre + "need"].values),
                     "accept_auroc": auc(t.valid.values, acc_t),
                     "reward_acc_all120": (pred_valid == d.valid.values).mean(),
                     "FA_blocked@keep90%CD": (fa < np.quantile(cd, .1)).mean() if len(cd) and len(fa) else np.nan}
    return res


n2 = len(d2)
rng = np.random.default_rng(1)
t2 = summarize(e2_metrics(np.arange(n2)), [e2_metrics(rng.integers(0, n2, n2)) for _ in range(B)])
t2.to_csv(os.path.join(CI, "e2.csv"), index=False)
show(t2, f"E2 ProactiveBench (n={n2} moments, B={B})")

# ---------------------------------------------------------------- E3
se = pd.read_parquet(os.path.join(jev.ROOT, "data", "ext", "summeval", "summeval.parquet"))
names3 = {"gpt-4o-mini": "GPT-4o-mini", "qwen3-32b": "Qwen3-32B (API)", "gpt-4o": "GPT-4o",
          "qwen3-32b-local": "Qwen3-32B (local)"}
rows = []
for di, r in se.iterrows():
    for si, summ in enumerate(r.machine_summaries):
        o = jev.ask({"source_article": r.text, "summary": summ}, questions(), tag="e3")
        assert o["_cached"]
        for c in CRIT:
            pr = np.array([float(v) for _, v in sorted(o["answers"][c]["probabilities"].items())])
            rows.append({"doc": di, "crit": c, "judge": "JEV", "pred": 1 + (pr * np.arange(5)).sum(),
                         "ent": -(pr[pr > 0] * np.log(pr[pr > 0])).sum(), "gold": float(r[c][si])})
for f in sorted(f for f in glob.glob(os.path.join(E3, "llm_*.jsonl")) if not os.path.basename(f).startswith("llm_in")):
    tag = os.path.basename(f)[4:-6]
    for line in open(f):
        x = json.loads(line)
        di, si, c = x["id"].split("|")
        pr = np.array([x["probs"][k] for k in "12345"])
        rows.append({"doc": int(di), "crit": c, "judge": names3.get(tag, tag), "pred": 1 + (pr * np.arange(5)).sum(),
                     "ent": -(pr[pr > 0] * np.log(pr[pr > 0])).sum(), "gold": float(se.loc[int(di), c][int(si)])})
d3 = pd.DataFrame(rows)
J3 = list(dict.fromkeys(d3.judge))
docs_of = {j: sorted(d3[d3.judge == j].doc.unique()) for j in J3}
SP, ER = {}, {}  # per (judge, crit, doc): Spearman over the 16 summaries; (big-error flags, entropies)
for (j, c, doc), g in d3.groupby(["judge", "crit", "doc"]):
    SP[(j, c, doc)] = spear(g.pred.values, g.gold.values) if g.gold.nunique() > 1 and g.pred.nunique() > 1 else np.nan
    ER[(j, c, doc)] = ((g.pred - g.gold).abs().values > 1, g.ent.values)


def e3_metrics(doc_sample, judges):
    res = {}
    for j in judges:
        per_c = [np.nanmean([SP[(j, c, d)] for d in doc_sample]) for c in CRIT]
        big = np.concatenate([ER[(j, "consistency", d)][0] for d in doc_sample])
        en = np.concatenate([ER[(j, "consistency", d)][1] for d in doc_sample])
        res[j] = {"spearman_avg": np.mean(per_c), **{f"spearman_{c}": v for c, v in zip(CRIT, per_c)},
                  "consistency_entropy_auroc": auc(big, en)}
    return res


out3 = []
for label, judges in [("all 100 docs", [j for j in J3 if len(docs_of[j]) == 100]),
                      ("GPT-4o's 10 docs", [j for j in J3 if set(docs_of["GPT-4o"]) <= set(docs_of[j])] if "GPT-4o" in J3 else [])]:
    if len(judges) < 2:
        continue
    docs = np.array(docs_of["GPT-4o"] if label.startswith("GPT") else docs_of["JEV"])
    rng = np.random.default_rng(2)
    t3 = summarize(e3_metrics(docs, judges), [e3_metrics(docs[rng.integers(0, len(docs), len(docs))], judges)
                                               for _ in range(B)])
    t3["subset"] = label
    out3.append(t3)
    show(t3[t3.metric.isin(["spearman_avg", "consistency_entropy_auroc"])], f"E3 SummEval, {label} (B={B})")
pd.concat(out3).to_csv(os.path.join(CI, "e3.csv"), index=False)
print(f"\n* = 95% CI of the difference excludes 0.  JEV spent (unchanged) ${jev.spent_usd():.4f}")
