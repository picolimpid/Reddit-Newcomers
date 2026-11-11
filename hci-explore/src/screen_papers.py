"""Stage 1: JEV screens candidate abstracts for 'framework with a model-as-module that contains a judgment step'.
Output: data/screened.jsonl (candidate fields + JEV probabilities)."""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
import jev

ROOT = jev.ROOT
ROLES = {
    "annotate": "Classifying or coding content (qualitative coding, labeling, tagging, extracting categories).",
    "evaluate": "Evaluating or scoring the quality of outputs (LLM-as-judge, feedback, grading, assessment).",
    "detect_state": "Detecting a user's state or context to decide when or how the system should intervene.",
    "rank_select": "Ranking, filtering or selecting among candidates (suggestions, generations, items, actions).",
    "simulate": "Simulating human responses, personas, users or social agents.",
    "moderate": "Detecting harm, misinformation, toxicity, privacy or safety risks.",
    "generate_only": "The model mainly generates content; there is no separate judgment step.",
}
Q_BASE = {
    "system": {"type": "noul", "instructions": "Does the paper build a system, tool, framework or computational pipeline (rather than only reporting an interview, survey or observational study)?"},
    "model_module": {"type": "noul", "instructions": "Is an LLM or another machine-learning model used as one module inside that system or pipeline?"},
    "judgment": {"type": "noul", "instructions": "Does the pipeline include a step where a model makes a judgment about something (classifies, scores, rates, verifies, filters, ranks, detects, or decides whether or when to act), as opposed to only generating text?"},
    "measurable": {"type": "noul", "instructions": "Does the paper evaluate that model step against labels, human ratings, or task outcomes that someone could reuse to compare a different model?"},
}
def questions(key):
    ks = list(ROLES); random.Random(key).shuffle(ks)
    return {**Q_BASE, "role": {"type": "choice", "instructions": "Which role does the model play that is most central to the paper's contribution?", "criteria": {k: ROLES[k] for k in ks}}}

rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "candidates.jsonl"))]
rows = [r for r in rows if len(r["abstract"]) > 200]
items = [{"state": {"paper": {"title": r["title"], "venue": r["venue"], "abstract": r["abstract"][:2500]}},
          "questions": questions(r["id"]), "tag": "screen"} for r in rows]
outs = jev.ask_many(items, workers=8)
with open(os.path.join(ROOT, "data", "screened.jsonl"), "w") as f:
    for r, o in zip(rows, outs):
        r = dict(r)
        for q in Q_BASE: r["p_" + q] = jev.p_yes(o, q)
        pr = jev.probs(o, "role"); r["role"] = max(pr, key=pr.get); r["role_p"] = pr[r["role"]]
        r["fit"] = r["p_system"] * r["p_model_module"] * r["p_judgment"]
        f.write(json.dumps(r) + "\n")
print("n", len(rows), "spent $%.4f" % jev.spent_usd())
