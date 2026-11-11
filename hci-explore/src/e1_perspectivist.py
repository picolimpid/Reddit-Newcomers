"""E1 (Idea 1): do JEV's probabilities track *human disagreement* in content coding, and can they triage which
items need human review?  Data: Measuring Hate Speech (Kennedy et al. 2020; ucberkeley-dlab), comments with >= 4
annotators.  Six codes are asked in ONE JEV request per comment (state billed once); Qwen3-8B gets one prompt per code.

  python3 src/e1_perspectivist.py prep      # sample + JEV run + write Qwen prompts
  CUDA_VISIBLE_DEVICES=0 $PY src/qwen_judge.py results/e1/qwen_in.jsonl results/e1/qwen_out.jsonl
  python3 src/e1_perspectivist.py analyze
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import jev  # noqa: E402

OUT = os.path.join(jev.ROOT, "results", "e1")
os.makedirs(OUT, exist_ok=True)
N = 400
CODES = {  # name: (human rule, question)
    "hatespeech": (lambda d: d.hatespeech == 2,
                   "Is `comment` hate speech, i.e. does it attack or demean people on the basis of an identity such as "
                   "race, religion, ethnicity, gender, sexual orientation or disability?"),
    "insult": (lambda d: d.insult >= 3, "Is `comment` insulting toward someone?"),
    "humiliate": (lambda d: d.humiliate >= 3, "Does `comment` humiliate, mock or belittle someone?"),
    "dehumanize": (lambda d: d.dehumanize >= 3,
                   "Does `comment` dehumanize a person or group, e.g. by comparing them to animals, vermin, disease "
                   "or objects?"),
    "violence": (lambda d: d.violence >= 3, "Does `comment` call for, threaten or endorse violence against someone?"),
    "disrespect": (lambda d: d.respect >= 3, "Is `comment` disrespectful toward someone?"),
}


def load():
    m = pd.read_parquet(os.path.join(jev.ROOT, "data", "ext", "mhs", "mhs.parquet"),
                        columns=["comment_id", "annotator_id", "text", "hate_speech_score", "hatespeech", "insult",
                                 "humiliate", "dehumanize", "violence", "respect"])
    for c, (rule, _) in CODES.items():
        m["y_" + c] = rule(m).astype(float)
    g = m.groupby("comment_id")
    agg = g.agg(text=("text", "first"), n=("annotator_id", "size"), hs=("hate_speech_score", "mean"),
                **{"h_" + c: ("y_" + c, "mean") for c in CODES})
    agg = agg[(agg.n >= 4) & (agg.text.str.len().between(20, 1200))]
    # stratify on the continuous hate score so all levels (incl. ambiguous middle) are represented
    agg["bin"] = pd.qcut(agg.hs, 5, labels=False)
    return agg.groupby("bin", group_keys=False).apply(lambda x: x.sample(N // 5, random_state=0)).reset_index()


def prep():
    s = load()
    s.to_parquet(os.path.join(OUT, "sample.parquet"))
    qs = {c: {"type": "noul", "instructions": q} for c, (_, q) in CODES.items()}
    outs = jev.ask_many([{"state": {"comment": t}, "questions": qs, "tag": "e1"} for t in s.text])
    pd.DataFrame([{"comment_id": cid, **{"j_" + c: jev.p_yes(o, c) for c in CODES}}
                  for cid, o in zip(s.comment_id, outs)]).to_parquet(os.path.join(OUT, "jev.parquet"))
    with open(os.path.join(OUT, "qwen_in.jsonl"), "w") as f:
        for cid, t in zip(s.comment_id, s.text):
            for c, (_, q) in CODES.items():
                f.write(json.dumps({"id": f"{cid}|{c}", "options": ["Yes", "No"],
                                    "user": f"Comment:\n\"\"\"\n{t}\n\"\"\"\n\nQuestion: {q.replace('`comment`', 'the comment')}\n"
                                            "Answer with Yes or No only."}) + "\n")
    print("n", len(s), "spent $%.4f" % jev.spent_usd())


def supervised_baseline(s):
    """Strong cheap baseline: TF-IDF + logistic regression trained on the per-annotator labels of all OTHER
    comments (no overlap with the sample), predicting P(an annotator says yes) per code."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    m = pd.read_parquet(os.path.join(jev.ROOT, "data", "ext", "mhs", "mhs.parquet"),
                        columns=["comment_id", "text", "hatespeech", "insult", "humiliate", "dehumanize", "violence", "respect"])
    m = m[~m.comment_id.isin(set(s.comment_id))]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True)
    X = vec.fit_transform(m.text)
    Xs = vec.transform(s.text)
    out = pd.DataFrame({"comment_id": s.comment_id.values})
    for c, (rule, _) in CODES.items():
        clf = LogisticRegression(C=4.0, max_iter=2000).fit(X, rule(m).astype(int))
        out["s_" + c] = clf.predict_proba(Xs)[:, 1]
    return out


def ent(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


LLM_NAMES = {"gpt-4o-mini": "GPT-4o-mini", "qwen3-32b": "Qwen3-32B (API)", "qwen3-32b-local": "Qwen3-32B (local)", "gpt-4o": "GPT-4o", "qwen_out": "Qwen3-8B (local)"}


def add_llm_judges(s, judges):
    """Merge every results/e1/llm_<model>.jsonl (and a local qwen_out.jsonl if present); one prompt per code."""
    import glob
    files = sorted(glob.glob(os.path.join(OUT, "llm_*.jsonl"))) + [f for f in [os.path.join(OUT, "qwen_out.jsonl")]
                                                                     if os.path.exists(f) and os.path.getsize(f) > 0]
    for f in files:
        tag = os.path.basename(f)[:-6].replace("llm_", "")
        pre = "L" + tag.replace("-", "").replace(".", "").replace("_", "")
        q = pd.read_json(f, lines=True)
        q[["comment_id", "code"]] = q.id.str.split("|", expand=True)
        q["p"] = q.probs.apply(lambda d: d["Yes"])
        q = q.pivot(index="comment_id", columns="code", values="p").add_prefix(pre + "_").reset_index()
        q["comment_id"] = q.comment_id.astype(s.comment_id.dtype)
        s = s.merge(q)
        judges[pre] = LLM_NAMES.get(tag, tag)
    return s, judges


def analyze():
    s = pd.read_parquet(os.path.join(OUT, "sample.parquet")).merge(pd.read_parquet(os.path.join(OUT, "jev.parquet")))
    judges = {"j": "JEV", "s": "TF-IDF+LR (supervised, 129k labels on 39k comments)"}
    s, judges = add_llm_judges(s, judges)
    s = s.merge(supervised_baseline(s), on="comment_id")
    rows, tri = [], []
    for c in CODES:
        h = s["h_" + c].values
        maj = (h > 0.5).astype(int)
        contested = (h > 0) & (h < 1)
        for judge, jname in judges.items():
            p = s[f"{judge}_{c}"].values
            r = {"code": c, "judge": jname, "base_rate": maj.mean().round(3),
                 "spearman_soft": spearmanr(p, h).statistic, "auroc_majority": roc_auc_score(maj, p),
                 "brier_soft": np.mean((p - h) ** 2),
                 # does the judge's uncertainty find the items humans disagree on?
                 "auroc_contested": roc_auc_score(contested, ent(p)),
                 "spearman_disagree": spearmanr(ent(p), ent(h)).statistic}
            rows.append(r)
            # triage: send the k% most uncertain items to humans (majority label), keep model label otherwise
            order = np.argsort(-ent(p))
            for frac in (0, .1, .2, .3, .4, .5):
                k = int(frac * len(p))
                pred = (p > 0.5).astype(int)
                pred[order[:k]] = maj[order[:k]]
                tri.append({"code": c, "judge": r["judge"], "deferred": frac, "acc": (pred == maj).mean()})
            rnd = []
            for seed in range(50):
                o = np.random.default_rng(seed).permutation(len(p))
                for frac in (.2,):
                    k = int(frac * len(p)); pred = (p > 0.5).astype(int); pred[o[:k]] = maj[o[:k]]
                    rnd.append((pred == maj).mean())
            tri.append({"code": c, "judge": r["judge"] + " (random 20%)", "deferred": .2, "acc": np.mean(rnd)})
    res = pd.DataFrame(rows)
    tr = pd.DataFrame(tri)
    res.to_csv(os.path.join(OUT, "metrics.csv"), index=False)
    tr.to_csv(os.path.join(OUT, "triage.csv"), index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print(res.groupby("judge")[["spearman_soft", "auroc_majority", "brier_soft", "auroc_contested", "spearman_disagree"]].mean().round(3))
    print(tr.groupby(["judge", "deferred"]).acc.mean().unstack().round(3))
    print("contested share per code:", {c: round(((s['h_' + c] > 0) & (s['h_' + c] < 1)).mean(), 2) for c in CODES})
    print("median annotators per comment:", s.n.median())


if __name__ == "__main__":
    {"prep": prep, "analyze": analyze}[sys.argv[1]]()
