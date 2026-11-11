"""E2 (Idea 2): JEV as a fast, calibrated interruption gate for proactive assistants.
Data: ProactiveBench reward test set (Lu et al., "Proactive Agent", 2024; thunlp/ProactiveAgent), 120 human-annotated
moments of computer activity.  Labels: help_needed (all 120); valid = the agent's proposal was accepted
(60 items with a proposed task: 30 Correct-Detection, 30 False-Alarm).

  python3 src/e2_proactive_gate.py prep
  CUDA_VISIBLE_DEVICES=0 $PY src/qwen_judge.py results/e2/qwen_in.jsonl results/e2/qwen_out.jsonl
  python3 src/e2_proactive_gate.py analyze
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import jev  # noqa: E402

OUT = os.path.join(jev.ROOT, "results", "e2")
os.makedirs(OUT, exist_ok=True)
Q_NEED = ("Based on `recent_activity`, does the user need help right now that a proactive assistant could usefully "
          "offer without being asked (e.g. they are stuck, searching repeatedly, or about to do something tedious)?")
Q_ACCEPT = ("If the assistant interrupted the user right now to offer `proposed_task`, would the user accept the "
            "offer as helpful (rather than dismiss it as unnecessary or off-target)?")
Q_INTERRUPT = ("Would interrupting the user right now be more disruptive than helpful?")


TEST = os.path.join(jev.ROOT, "data", "ext", "proactive", "test_data.jsonl")


def load(path=TEST, prefix=""):
    rows = [json.loads(l) for l in open(path)]
    for i, r in enumerate(rows):
        r["id"] = prefix + str(i)
        try:
            t0 = float(r["obs"][0]["time"])
            r["activity"] = [f"[+{float(o['time']) - t0:.0f}s] {o['event']}" for o in r["obs"]]
        except ValueError:  # some items carry human-readable times ("Day 1, 8:00 PM")
            r["activity"] = [f"[{o['time']}] {o['event']}" for o in r["obs"]]
    return rows


def jev_item(r, tag="e2"):
    st = {"recent_activity": r["activity"]}
    qs = {"need": {"type": "noul", "instructions": Q_NEED},
          "disrupt": {"type": "noul", "instructions": Q_INTERRUPT}}
    if r["pred_task"]:
        st["proposed_task"] = r["pred_task"]
        qs["accept"] = {"type": "noul", "instructions": Q_ACCEPT}
    return {"state": st, "questions": qs, "tag": tag}


def llm_lines(r):
    """Yes/No prompts (same wording as the JEV questions) for token-probability judges."""
    act = "\n".join(r["activity"])
    base = f"Recent activity of a computer user (oldest first):\n{act}\n\n"
    out = [json.dumps({"id": r["id"] + "|need", "options": ["Yes", "No"], "user": base + "Question: " +
                       Q_NEED.replace("`recent_activity`", "this activity") + "\nAnswer with Yes or No only."})]
    if r["pred_task"]:
        out.append(json.dumps({"id": r["id"] + "|accept", "options": ["Yes", "No"], "user": base +
                               f"Proposed task: {r['pred_task']}\n\nQuestion: " +
                               Q_ACCEPT.replace("`proposed_task`", "the proposed task") +
                               "\nAnswer with Yes or No only."}))
    return out


def prep():
    rows = load()
    outs = jev.ask_many([jev_item(r) for r in rows])
    recs = []
    for r, o in zip(rows, outs):
        recs.append({"id": r["id"], "category": r["category"], "help_needed": r["help_needed"], "valid": r["valid"],
                     "has_task": bool(r["pred_task"]), "j_need": jev.p_yes(o, "need"),
                     "j_disrupt": jev.p_yes(o, "disrupt"),
                     "j_accept": jev.p_yes(o, "accept") if r["pred_task"] else np.nan,
                     "n_tok": o["usage"]["input_tokens"]})
    pd.DataFrame(recs).to_parquet(os.path.join(OUT, "jev.parquet"))
    with open(os.path.join(OUT, "qwen_in.jsonl"), "w") as f:
        for r in rows:
            for line in llm_lines(r):
                f.write(line + "\n")
    lat = pd.read_json(jev.LEDGER, lines=True).query("tag == 'e2'").latency_s
    print("n", len(rows), "spent $%.4f" % jev.spent_usd(), "jev latency median %.3f p90 %.3f" % (lat.median(), lat.quantile(.9)))


def analyze():
    d = pd.read_parquet(os.path.join(OUT, "jev.parquet"))
    import glob
    names = {"gpt-4o-mini": "GPT-4o-mini", "qwen3-32b": "Qwen3-32B (API)", "qwen3-32b-local": "Qwen3-32B (local)", "gpt-4o": "GPT-4o", "qwen_out": "Qwen3-8B (local)"}
    judges = [("JEV", "j_")]
    lat_rows = []
    for qf in sorted(glob.glob(os.path.join(OUT, "llm_*.jsonl"))) + [f for f in [os.path.join(OUT, "qwen_out.jsonl")]
                                                                       if os.path.exists(f) and os.path.getsize(f) > 0]:
        tag = os.path.basename(qf)[:-6].replace("llm_", "")
        pre = "L" + tag.replace("-", "").replace(".", "").replace("_", "") + "_"
        q = pd.read_json(qf, lines=True)
        if "latency_s" in q:
            lat_rows.append((names.get(tag, tag), q.latency_s.median(), q.latency_s.quantile(.9)))
        q[["id", "kind"]] = q.id.astype(str).str.split("|", expand=True)
        q["p"] = q.probs.apply(lambda x: x["Yes"])
        q = q.pivot(index="id", columns="kind", values="p").add_prefix(pre).reset_index()
        d = d.merge(q, on="id", how="left")
        judges.append((names.get(tag, tag), pre))
    res = {}
    for judge, pre in judges:
        r = {}
        r["need_auroc"] = roc_auc_score(d.help_needed, d[pre + "need"])
        r["need_acc@0.5"] = ((d[pre + "need"] > .5) == d.help_needed).mean()
        t = d[d.has_task]
        r["accept_auroc"] = roc_auc_score(t.valid, t[pre + "accept"])
        r["accept_acc@0.5"] = ((t[pre + "accept"] > .5) == t.valid).mean()
        # reward-model view over all 120: was the agent's action (stay silent / propose) correct?
        pred_valid = np.where(d.has_task, d[pre + "accept"] > .5, d[pre + "need"] < .5)
        r["reward_acc_all120"] = (pred_valid == d.valid).mean()
        # gate: how many false alarms are blocked while keeping >= 90% / 80% of correct detections?
        for keep in (.9, .8):
            score = t[pre + "accept"].values
            cd = score[t.valid.values]
            tau = np.quantile(cd, 1 - keep)
            r[f"FA_blocked@keep{int(keep * 100)}%CD"] = (score[~t.valid.values] < tau).mean()
        if pre == "j_":
            r["disrupt_auroc_vs_not_needed"] = roc_auc_score(~d.help_needed, d.j_disrupt)
        res[judge] = r
    out = pd.DataFrame(res).round(3)
    print(out.to_string())
    out.to_csv(os.path.join(OUT, "metrics.csv"))
    lat = pd.read_json(jev.LEDGER, lines=True).query("tag == 'e2'")
    print("JEV latency median %.3fs p90 %.3fs; input tokens median %d" % (lat.latency_s.median(), lat.latency_s.quantile(.9), d.n_tok.median()))
    for name, med, p90 in lat_rows:
        print(f"{name} latency via OpenRouter: median {med:.2f}s p90 {p90:.2f}s")
    print(d.groupby("category")[[c for c in d.columns if c.endswith(("_need", "_accept"))]].mean().round(2).T)


if __name__ == "__main__":
    {"prep": prep, "analyze": analyze}[sys.argv[1]]()
