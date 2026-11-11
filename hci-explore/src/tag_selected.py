"""Stage 2: final 100 papers + JEV tags on dimensions where JEV's properties (calibrated probabilities, ~0.1 s latency,
$0.04/M tokens, many questions per state billed once, image input) matter. Output: data/selected100.jsonl"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import jev, pandas as pd
FINAL = {"annotate":[7,12,48,49,60,78,86,117,144,153,156,168,172,185,199,202,206],
"detect_state":[9,10,11,15,19,28,33,38,40,50,53,58,69,70,73,75,81,85,110,118,131,143,152,105],
"evaluate":[0,1,30,42,43,47,55,63,77,104,107,111,112,115,121,124,130,151,155,159,160,174,186,204],
"moderate":[3,14,27,103,109,133,285],
"rank_select":[2,5,8,36,41,62,76,102,108,122,164,345],
"simulate":[39,57,80,87,116,129,140,177,187,188,192,196,203,218,221,170]}
Q = {
 "realtime": {"type": "noul", "instructions": "Must the model's judgment be made in real time while a user is interacting (latency matters, e.g. under a second or a few seconds)?"},
 "high_volume": {"type": "noul", "instructions": "Does the system need the model to make a very large number of judgments (e.g. over every item in a large corpus, every frame, or every message), so per-call cost matters?"},
 "threshold": {"type": "noul", "instructions": "Does the system act on a decision threshold or trigger (e.g. intervene / alert / defer to a human / filter only when the model is confident enough), so calibrated probabilities would be useful?"},
 "multi_criteria": {"type": "noul", "instructions": "Does the model judge the same item on several criteria, codes, or dimensions at once?"},
 "visual": {"type": "noul", "instructions": "Does the judgment depend on images, screenshots, or video frames?"},
 "human_labels": {"type": "noul", "instructions": "Does the paper collect or use human labels or ratings that could serve as ground truth for the judgment step?"},
}
d = pd.read_json(os.path.join(jev.ROOT, "data", "pool.jsonl"), lines=True).set_index("k")
items, meta = [], []
for role, ks in FINAL.items():
    for k in ks:
        r = d.loc[k]
        items.append({"state": {"paper": {"title": r.title, "abstract": r.abstract[:2500]}}, "questions": Q, "tag": "tag100"})
        meta.append((role, k, r))
outs = jev.ask_many(items)
with open(os.path.join(jev.ROOT, "data", "selected100.jsonl"), "w") as f:
    for (role, k, r), o in zip(meta, outs):
        rec = {"k": int(k), "role": role, "title": r.title, "venue": r.venue, "year": int(r.year), "doi": r.doi,
               "cites": int(r.cites), **{q: round(jev.p_yes(o, q), 3) for q in Q}}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(len(meta), "spent $%.4f" % jev.spent_usd())
