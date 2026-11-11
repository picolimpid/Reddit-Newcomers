"""E3 (Idea 3): JEV as a multi-criteria, checklist-style evaluator whose atomic checks can be aligned to humans
with a few labels (EvalGen / EvalLM / InteractEval style).  Data: SummEval (Fabbri et al. 2021; mteb/summeval):
100 CNN/DM articles x 16 system summaries, expert ratings (1-5) on coherence, consistency, fluency, relevance.
ONE JEV request per summary carries 4 direct score questions + 12 atomic yes/no checks (article billed once).

  LIMIT_DOCS=10 python3 src/e3_checklist_eval.py run   # pilot, prints cost
  python3 src/e3_checklist_eval.py run                  # all docs
  python3 src/e3_checklist_eval.py analyze
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, os.path.dirname(__file__))
import jev  # noqa: E402

OUT = os.path.join(jev.ROOT, "results", "e3")
os.makedirs(OUT, exist_ok=True)
CRIT = ["coherence", "consistency", "fluency", "relevance"]
LV = lambda name: [f"1: very poor {name}", f"2: poor {name}", f"3: fair {name}", f"4: good {name}", f"5: excellent {name}"]
DIRECT = {  # G-Eval-style definitions
    "coherence": "Rate the coherence of `summary`: the collective quality of all sentences. It should be well-structured "
                 "and well-organized, building from sentence to sentence into a coherent body of information about a topic, "
                 "not a heap of related information.",
    "consistency": "Rate the factual consistency of `summary` with `source_article`: the summary should contain only "
                   "statements that are entailed by the article. Penalize hallucinated facts.",
    "fluency": "Rate the fluency of `summary`: the quality of its sentences in grammar, spelling, punctuation, word "
               "choice and sentence structure. Ignore lowercasing and spaces before punctuation, which come from tokenization.",
    "relevance": "Rate the relevance of `summary`: it should include only the important information from `source_article`. "
                 "Penalize redundancies and excess information.",
}
CHECKS = {
    "coh_order": ("coherence", "Do the sentences of `summary` follow a logical order?"),
    "coh_whole": ("coherence", "Does `summary` read as one connected account rather than a list of loosely related facts?"),
    "coh_refs": ("coherence", "Is `summary` free of abrupt topic jumps and of pronouns or references with no clear referent?"),
    "con_supported": ("consistency", "Is every factual statement in `summary` supported by `source_article`?"),
    "con_nocontra": ("consistency", "Is `summary` free of names, numbers or events that contradict `source_article`?"),
    "con_attrib": ("consistency", "Does `summary` attribute every statement and action to the right person or entity?"),
    "flu_grammar": ("fluency", "Is `summary` free of grammatical errors? (Ignore lowercasing and spaces before punctuation.)"),
    "flu_garbled": ("fluency", "Is `summary` free of garbled, repeated or cut-off phrases?"),
    "flu_natural": ("fluency", "Does `summary` read like natural, well-formed English sentences?"),
    "rel_main": ("relevance", "Does `summary` capture the main point of `source_article`?"),
    "rel_lean": ("relevance", "Does `summary` avoid unimportant details and redundant repetition?"),
    "rel_reader": ("relevance", "Would someone who reads only `summary` learn the most important facts of `source_article`?"),
}


def questions():
    q = {c: {"type": "score", "instructions": DIRECT[c], "criteria": LV(c)} for c in CRIT}
    q.update({k: {"type": "noul", "instructions": v[1]} for k, v in CHECKS.items()})
    return q


def run():
    s = pd.read_parquet(os.path.join(jev.ROOT, "data", "ext", "summeval", "summeval.parquet"))
    lim = int(os.environ.get("LIMIT_DOCS", "0")) or len(s)
    items, meta = [], []
    for di, r in s.head(lim).iterrows():
        for si, summ in enumerate(r.machine_summaries):
            items.append({"state": {"source_article": r.text, "summary": summ}, "questions": questions(), "tag": "e3"})
            meta.append({"doc": di, "sys": si, **{"h_" + c: float(r[c][si]) for c in CRIT}})
    before = jev.spent_usd()
    outs = jev.ask_many(items)
    recs = []
    for m, o in zip(meta, outs):
        rec = dict(m)
        for c in CRIT:
            rec["d_" + c] = 1 + 4 * jev.exp_score(o, c)
        for k in CHECKS:
            rec["k_" + k] = jev.p_yes(o, k)
        rec["n_tok"] = o["usage"]["input_tokens"]
        recs.append(rec)
    pd.DataFrame(recs).to_parquet(os.path.join(OUT, "jev.parquet"))
    print("summaries", len(recs), "this run $%.4f" % (jev.spent_usd() - before), "total $%.4f" % jev.spent_usd())


GEVAL = ("You will be given one summary written for a news article.\n\nYour task is to rate the summary on one metric.\n\n"
         "Evaluation Criteria:\n\n{name} (1-5) - {definition}\n\nSource Text:\n\n{article}\n\nSummary:\n\n{summary}\n\n"
         "Evaluation Form (scores ONLY):\n\n- {name} (1-5):")


def prep_llm():
    """G-Eval-style prompts (one per summary x criterion) for LLM-judge baselines -> results/e3/llm_in.jsonl"""
    s = pd.read_parquet(os.path.join(jev.ROOT, "data", "ext", "summeval", "summeval.parquet"))
    with open(os.path.join(OUT, "llm_in.jsonl"), "w") as f:
        for di, r in s.iterrows():
            for si, summ in enumerate(r.machine_summaries):
                for c in CRIT:
                    dfn = DIRECT[c].split(": ", 1)[1].replace("`summary`", "the summary").replace("`source_article`", "the source text")
                    f.write(json.dumps({"id": f"{di}|{si}|{c}", "options": ["1", "2", "3", "4", "5"],
                                        "user": GEVAL.format(name=c.capitalize(), definition=dfn, article=r.text, summary=summ)}) + "\n")


def doc_corr(d, pred, gold):
    """G-Eval protocol: correlation over the 16 summaries of each document, averaged over documents."""
    sp, kt = [], []
    for _, g in d.groupby("doc"):
        if g[gold].nunique() > 1 and g[pred].nunique() > 1:
            sp.append(spearmanr(g[pred], g[gold]).statistic)
            kt.append(kendalltau(g[pred], g[gold]).statistic)
    return np.mean(sp), np.mean(kt)


def analyze():
    d = pd.read_parquet(os.path.join(OUT, "jev.parquet"))
    for c in CRIT:
        ks = [k for k, v in CHECKS.items() if v[0] == c]
        d["c_" + c] = d[["k_" + k for k in ks]].mean(axis=1)
        d["m_" + c] = (d["d_" + c] - 1) / 4 + d["c_" + c]  # direct + checklist
    rows = []
    for c in CRIT:
        for name, col in (("JEV direct score", "d_"), ("JEV checklist (mean of 3 checks)", "c_"), ("JEV direct + checklist", "m_")):
            sp, kt = doc_corr(d, col + c, "h_" + c)
            rows.append({"criterion": c, "method": name, "spearman": sp, "kendall": kt})
    # few-label alignment: per criterion, ridge on all 16 JEV features fit on k labelled documents, tested on the rest
    from sklearn.linear_model import RidgeCV
    feats = ["d_" + c for c in CRIT] + ["k_" + k for k in CHECKS]
    docs = d.doc.unique()
    for k in (5, 10, 20):
        for c in CRIT:
            sps, kts = [], []
            for seed in range(20):
                perm = np.random.default_rng(seed).permutation(docs)
                tr = d[d.doc.isin(perm[:k])]; te = d[d.doc.isin(perm[k:])].copy()
                mdl = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(tr[feats], tr["h_" + c])
                te["a_" + c] = mdl.predict(te[feats])
                sp, kt = doc_corr(te, "a_" + c, "h_" + c); sps.append(sp); kts.append(kt)
            rows.append({"criterion": c, "method": f"JEV 16 features + ridge on {k} labelled docs", "spearman": np.mean(sps), "kendall": np.mean(kts)})
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, "metrics.csv"), index=False)
    tab = r.pivot(index="method", columns="criterion", values="spearman")
    tab["avg"] = tab.mean(axis=1)
    order = ["JEV direct score", "JEV checklist (mean of 3 checks)", "JEV direct + checklist"] + [
        f"JEV 16 features + ridge on {k} labelled docs" for k in (5, 10, 20)]
    print("Spearman (summary-level, averaged over docs); n docs =", d.doc.nunique())
    print(tab.loc[order].round(3).to_string())
    tk = r.pivot(index="method", columns="criterion", values="kendall"); tk["avg"] = tk.mean(axis=1)
    print("Kendall"); print(tk.loc[order].round(3).to_string())
    print("median input tokens per request:", d.n_tok.median(), "| cost per summary (16 judgments): $%.6f" % (d.n_tok.median() * jev.USD_PER_TOKEN))
    lat = pd.read_json(jev.LEDGER, lines=True).query("tag == 'e3'").latency_s
    print("JEV latency median %.3fs p90 %.3fs" % (lat.median(), lat.quantile(.9)))


if __name__ == "__main__":
    {"run": run, "analyze": analyze, "prep_llm": prep_llm}[sys.argv[1]]()
