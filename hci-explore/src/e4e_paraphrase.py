"""E4e: test-retest reliability of the fixed judge itself -- does JEV's answer survive a rewording of the question?
(PLAN.md D6; responds to LLM-Mod's observation that GPT-3.5 reverses its decision when questioned.)  300 Jigsaw
community-rules rows from E4a's evaluation cells (stratified by cell), asked with two new wordings; the original
wording's answers come from E4a's cache.

Pre-registered: Spearman correlation between the original and each rewording >= 0.8 AND decision flip rate
(p >= 0.5) <= 10% for both rewordings.

  python3 -I src/e4e_paraphrase.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
import jev  # noqa: E402
import e4a_community_calib as e4a  # noqa: E402

OUT = os.path.join(jev.ROOT, "results", "e4e")
os.makedirs(OUT, exist_ok=True)
WORDINGS = {
    "original": e4a.Q_GENERIC,
    "violate": "Is `comment` a violation of `rule`? `comment` is a Reddit comment; `rule` is one of the rules of the "
               "subreddit it was posted in.",
    "moderator": "Imagine a careful moderator who enforces `rule`. Would that moderator consider the Reddit comment "
                 "`comment` to break `rule`?",
}


def main():
    d = pd.read_parquet(os.path.join(jev.ROOT, "results", "e4a", "jev.parquet")).reset_index(drop=True)
    n_cell = d.cell.value_counts()
    rng = np.random.default_rng(0)
    take = []
    for c, g in d.groupby("cell"):
        k = max(3, int(round(300 * len(g) / len(d))))
        take += list(rng.choice(g.index.values, min(k, len(g)), replace=False))
    s = d.loc[sorted(take)].head(300) if len(take) > 300 else d.loc[sorted(take)]
    print(f"{len(s)} rows from {s.cell.nunique()} cells")
    reqs = []
    for name, q in WORDINGS.items():
        for r in s.itertuples():
            reqs.append({"state": {"comment": r.body, "rule": r.rule},
                         "questions": {"violates": {"type": "noul", "instructions": q}}, "tag": f"e4e-{name}"})
    outs = jev.ask_many(reqs, workers=8)
    n = len(s)
    P = {name: np.array([jev.p_yes(o, "violates") for o in outs[i * n:(i + 1) * n]]) for i, name in enumerate(WORDINGS)}
    print(f"original wording reproduced from cache: max |diff| vs E4a = {np.abs(P['original'] - s.p_generic.values).max():.4f}")
    y = s.rule_violation.values
    from sklearn.metrics import roc_auc_score
    rows = []
    for name in WORDINGS:
        rows.append(dict(wording=name, mean_p=P[name].mean(), auroc=roc_auc_score(y, P[name]),
                         spearman_vs_original=spearmanr(P[name], P["original"]).correlation,
                         flip_rate_vs_original=np.mean((P[name] >= .5) != (P["original"] >= .5)),
                         mean_abs_diff=np.mean(np.abs(P[name] - P["original"]))))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))
    ok = all((t.spearman_vs_original[1:] >= .8) & (t.flip_rate_vs_original[1:] <= .10))
    print(f"pre-registered criterion (rho >= 0.8 and flips <= 10% for both rewordings): {'MET' if ok else 'NOT met'}")
    # bootstrap CIs for the two rewordings
    brng = np.random.default_rng(1)
    for name in list(WORDINGS)[1:]:
        bs = []
        for _ in range(2000):
            i = brng.integers(0, n, n)
            bs.append((spearmanr(P[name][i], P["original"][i]).correlation,
                       np.mean((P[name][i] >= .5) != (P["original"][i] >= .5))))
        bs = np.array(bs)
        print(f"  {name}: rho 95% CI [{np.quantile(bs[:, 0], .025):.3f}, {np.quantile(bs[:, 0], .975):.3f}], "
              f"flip rate 95% CI [{np.quantile(bs[:, 1], .025):.3f}, {np.quantile(bs[:, 1], .975):.3f}]")
    out = s[["row_id", "cell", "rule_violation"]].copy()
    for name in WORDINGS:
        out["p_" + name] = P[name]
    out.to_csv(os.path.join(OUT, "answers.csv"), index=False)
    print(f"JEV spent so far ${jev.spent_usd():.4f}")


if __name__ == "__main__":
    main()
