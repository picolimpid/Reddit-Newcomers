"""P8 exploratory (2026-11-07, not preregistered): content effect = text_<site> − placebo, within agent.
The frozen swap() table contrasts every condition with 'off', which mixes 'a moderator pinned something' with what it
says. Here the placebo (a content-free pinned comment) is the baseline, so the estimate is what the words add.

    python3 swap_content.py -> results/panel/swap_content.md"""
import glob
import os

import pandas as pd

from analyze_probes import boot, logit
from common import RESULTS
from panel_summary import inputs


def main():
    rows = []
    for f in sorted(glob.glob(f"{RESULTS}/swap/*/scores.parquet")):
        pop, name = os.path.basename(os.path.dirname(f)).split("-", 1)
        W = pd.read_parquet(f).pivot_table(index="idx", columns="condition", values="p")
        post = inputs(pop).loc[W.index, "post_id"]
        rows.append({"model": name, "pop": pop, "contrast": "placebo − off",
                     **dict(zip(["est", "lo", "hi"], boot(logit(W.placebo) - logit(W.off), post)))})
        for c in [c for c in W.columns if c.startswith("text_")]:
            rows.append({"model": name, "pop": pop, "contrast": f"{c} − placebo",
                         **dict(zip(["est", "lo", "hi"], boot(logit(W[c]) - logit(W.placebo), post)))})
    D = pd.DataFrame(rows)
    D["cell"] = [f"{e:+.2f} [{lo:+.2f}, {hi:+.2f}]" for e, lo, hi in zip(D.est, D.lo, D.hi)]
    txt = ("# P8 exploratory: what the words add over a content-free pinned comment (within agent, Δlogit, post bootstrap)\n\n"
           + D.pivot_table(index=["model", "pop"], columns="contrast", values="cell", aggfunc="first").to_markdown() + "\n")
    open(f"{RESULTS}/panel/swap_content.md", "w").write(txt)
    D.to_csv(f"{RESULTS}/panel/swap_content.csv", index=False)
    print(txt)


if __name__ == "__main__":
    main()
