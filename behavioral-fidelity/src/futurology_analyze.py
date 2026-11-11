"""Pre-registered retrodiction, step 3 (run only after registration): the analysis of PREREG_futurology.md §3-4.
Same estimators as rscience_analyze.py. (2026-11-03)

    python3 futurology_analyze.py futurology-qwen3-8b futurology-gpt-4o-mini futurology-jev [futurology-qwen3-32b]
    -> results/futurology/summary.md"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from analyze_probes import boot, fmt, logit
from common import RESULTS
from rscience_analyze import arm_fit, paired_gap

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "futurology2020")
OUT = os.path.join(RESULTS, "futurology")
PRIMARY = "futurology-qwen3-8b"
EQ = 0.25  # H4 equivalence bound (logit)
SIGNS = {"qwen3-8b": {"rules": -1, "welcome": +1}, "qwen3-32b": {"rules": +1, "welcome": -1},
         "gpt-4o-mini": {"welcome": +1}}  # H6: r/science signs where |Δlogit| > 0.25 (addendum §3.1)


def ci_text(b):
    return f"{b[0]:+.3f} [{b[1]:+.3f}, {b[2]:+.3f}]"


def main(runs):
    P = pd.read_parquet(os.path.join(D, "posts.parquet"))
    P = P[P.treat.notna()]
    F = pd.read_parquet(os.path.join(D, "commenters.parquet"))
    C = pd.read_parquet(os.path.join(D, "thread_comments.parquet"))
    R = pd.read_parquet(os.path.join(D, "inputs.parquet"))
    O = pd.read_parquet(os.path.join(D, "outcomes.parquet"))
    assert (R.author.values == O.author.values).all()
    R = pd.concat([R, O[["return_90d", "return_180d", "days_to_return", "n_later"]]], axis=1)
    R = R[R.n_hist > 0]
    L = ["# Pre-registered retrodiction: r/Futurology 2020 sticky-comment experiment", "",
         f"Experiment posts kept {len(P)} (treated share {P.treat.mean():.3f}); sampled commenters {len(F)}; "
         f"newcomers {int(F.newcomer.sum())}; agent-eligible newcomers {len(R)}.", ""]
    C["treat"] = C.post_id.map(P.set_index("id").treat)
    att = C.assign(deleted=C.author.eq("[deleted]")).groupby("treat").agg(comments=("id", "size"),
                                                                           deleted_author=("deleted", "mean"))
    b = arm_fit(F.assign(nc=F.newcomer.astype(int)), "nc")
    L += ["## Attrition and composition", "", att.round(3).to_markdown(), "",
          f"Newcomer share by arm: control {F[F.treat == 0].newcomer.mean():.3f}, treated {F[F.treat == 1].newcomer.mean():.3f}; "
          f"logit Δ {ci_text(b)}, p = {b[3]:.3f}.", ""]

    # human effect + H4 equivalence (TOST from the cluster-robust SE)
    rows, H4 = [], None
    big = C.post_id.value_counts().index[:10]
    for y in ("return_90d", "return_180d"):
        b, f, x = arm_fit(R, y), arm_fit(R, y, fe=True), arm_fit(R[~R.post_id.isin(big)], y)
        se = (b[2] - b[0]) / 1.96
        p_tost = max(1 - stats.norm.cdf((b[0] + EQ) / se), 1 - stats.norm.cdf((EQ - b[0]) / se))
        if y == "return_90d":
            H4 = (b, p_tost)
        rows.append({"outcome": y, "control": R[R.treat == 0][y].mean(), "treated": R[R.treat == 1][y].mean(),
                     "Δlogit [95% CI]": ci_text(b), "p": b[3], f"TOST p (±{EQ})": p_tost,
                     "block FE": ci_text(f), "excl. 10 largest threads": ci_text(x), "n": b[4]})
    L += ["## Human effect of the sticky on newcomer return (conditional on commenting)", "",
          pd.DataFrame(rows).round(3).to_markdown(index=False), ""]

    verdicts = [f"- **H4** (human |effect| < {EQ}, TOST): Δlogit {ci_text(H4[0])}, TOST p = {H4[1]:.3f} -> "
                + ("supported" if H4[1] < .05 else "inconclusive")]
    full = {}
    for run in runs:
        S = pd.read_parquet(os.path.join(OUT, run, "scores.parquet"))
        W = S.pivot_table(index="idx", columns="condition", values="p")
        H = R.loc[R.index.intersection(W.index)].copy()
        W = W.loc[H.index]
        H["p_arm"] = np.where(H.treat == 1, W["full"], W["off"])
        b, hb = arm_fit(H, "p_arm"), arm_fit(H, "return_90d")
        g = paired_gap(H)
        off = logit(W["off"])
        within = {c: boot(logit(W[c]) - off, H.post_id) for c in W.columns if c != "off"}
        pop = np.log(W["full"].mean() / (1 - W["full"].mean())) - np.log(W["off"].mean() / (1 - W["off"].mean()))
        L += [f"## Agents: {run} ({len(H)} newcomers)", "",
              "Mean P: " + ", ".join(f"{c} {W[c].mean():.3f}" for c in W.columns)
              + f" | human 90-day return {H.return_90d.mean():.3f}", "",
              f"AUROC of P(arm) for the real 90-day return: {roc_auc_score(H.return_90d, H.p_arm):.3f}", "",
              f"**Arm effect, same regression as the human outcome:** agent {ci_text(b)} vs human {ci_text(hb)}.", "",
              f"Agent − human gap (paired post-cluster bootstrap): {fmt(*g, nd=3)}", "",
              f"Population counterfactual (logit of mean P, full vs off, all agents): {pop:+.3f}", "",
              pd.DataFrame([{"condition": c, "Δlogit vs off": fmt(*v), "n agents": int(W[c].notna().sum())}
                            for c, v in within.items()]).to_markdown(index=False), ""]
        name = run.replace("futurology-", "")
        full[name] = within["full"]
        for comp, sign in SIGNS.get(name, {}).items():  # H6 (addendum §3.1)
            v = within[comp]
            ok = v[1] > 0 if sign > 0 else v[2] < 0
            verdicts.append(f"- **H6** ({name} {comp} {'+' if sign > 0 else '−'}): {fmt(*v)} -> "
                            + ("supported" if ok else "not supported"))
        if run == PRIMARY:
            verdicts.append(f"- **H1** ({name} gap < 0): {fmt(*g, nd=3)} -> " + ("supported" if g[2] < 0 else "not supported"))
            r = within["rules"]
            d = boot(logit(W["rules"]) - logit(W["welcome"]), H.post_id)
            verdicts.append(f"- **H2** ({name} rules < 0 and rules < welcome): rules {fmt(*r)}, rules − welcome {fmt(*d)} -> "
                            + ("supported" if r[2] < 0 and d[2] < 0 else "not supported"))
        elif "gpt" in run:
            w, t, r = within["welcome"], within["off+tmpl"], within["rules"]
            d = boot(logit(W["off+tmpl"]) - logit(W["welcome"]), H.post_id)
            ok = w[1] > 0 and d[1] > 0 and abs(r[0]) < 0.25
            verdicts.append(f"- **H3** ({name} welcome > 0, template > welcome, |rules| < 0.25): welcome {fmt(*w)}, "
                            f"template − welcome {fmt(*d)}, rules {fmt(*r)} -> " + ("supported" if ok else "not supported")
                            + f"; gap (two-sided, exploratory) {fmt(*g, nd=3)}")
        elif "jev" in run:
            ok = abs(b[0]) < 0.10 and all(abs(v[0]) < 0.15 for v in within.values())
            verdicts.append(f"- **H5** (Jev flat): arm effect {ci_text(b)}, max |within| "
                            f"{max(abs(v[0]) for v in within.values()):.2f} -> " + ("supported" if ok else "not supported"))
        else:
            verdicts.append(f"- exploratory {name}: gap {fmt(*g, nd=3)}")
    if {"qwen3-8b", "qwen3-32b"} <= set(full):  # H7 (addendum §3.1)
        a8, a32 = full["qwen3-8b"], full["qwen3-32b"]
        verdicts.append(f"- **H7** (full sticky: 8B < 0, 32B > 0): 8B {fmt(*a8)}, 32B {fmt(*a32)} -> "
                        + ("supported" if a8[2] < 0 and a32[1] > 0 else "not supported"))
    L += ["## Pre-registered hypotheses", ""] + verdicts + [""]
    txt = "\n".join(L) + "\n"
    open(os.path.join(OUT, "summary.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main(sys.argv[1:])
