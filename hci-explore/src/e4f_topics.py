"""E4f (descriptive, PLAN.md D7): which DICES-990 topics need per-person calibration most?  Per conversation:
disagreement = probability that two random raters disagree; personal-calibration value = within-conversation AUROC of
the judge + own intercept (k = 20 random anchors; E1i's out-of-fold margins, pairs formed within a rater fold).
Averaged over the conversations of each harm_type (a conversation can have several).

  python3 -I src/e4f_topics.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import jev  # noqa: E402
import e1i_dices990 as e1i  # noqa: E402

OUT = os.path.join(jev.ROOT, "results", "e4f")
os.makedirs(OUT, exist_ok=True)


def main():
    z = np.load(os.path.join(jev.ROOT, "results", "e1i", "oof_filtered.npz"))
    Y, rfold, M = z["Y"], z["rfold"], z["0|20|0"]
    _, items = e1i.load()
    df = pd.read_csv(e1i.CSV, usecols=["item_id", "harm_type"]).drop_duplicates("item_id").set_index("item_id")
    ht = df.loc[items.index].harm_type.fillna("").values
    rows = []
    for i in range(Y.shape[0]):
        lab = ~np.isnan(Y[i])
        p = Y[i, lab].mean()
        num = den = 0.
        for g in range(5):
            sel = lab & (rfold == g)
            pos, neg = np.flatnonzero(sel & (Y[i] == 1)), np.flatnonzero(sel & (Y[i] == 0))
            if len(pos) and len(neg):
                d = M[i, pos][:, None] - M[i, neg][None, :]
                num += ((d > 0) + .5 * (d == 0)).sum()
                den += d.size
        for t in [x.strip() for x in ht[i].split(",") if x.strip()] or ["(none)"]:
            rows.append(dict(topic=t, item=i, disagree=2 * p * (1 - p), yes_rate=p, num=num, den=den))
    r = pd.DataFrame(rows)
    g = r.groupby("topic").agg(conversations=("item", "nunique"), disagreement=("disagree", "mean"),
                               yes_rate=("yes_rate", "mean"), num=("num", "sum"), den=("den", "sum"))
    g["within_auroc_k20"] = g.num / g.den
    g = g[g.conversations >= 10].drop(columns=["num", "den"]).sort_values("disagreement", ascending=False)
    allv = r.drop_duplicates("item")
    print(f"all conversations: disagreement {allv.disagree.mean():.3f}, within-conversation AUROC (k = 20) "
          f"{allv.num.sum() / allv.den.sum():.3f}")
    print(g.round(3).to_string())
    g.to_csv(os.path.join(OUT, "topics.csv"))


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
