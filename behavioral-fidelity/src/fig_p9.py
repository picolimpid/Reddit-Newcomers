"""Figure (2026-11-07): one disposition, two decisions. x = within-agent Δlogit of P(return) under the full sticky
(r/science 2016); y = log ratio of P(would comment) full/off (P9). Humans: arm effect on return −0.03 [−0.23, +0.17]
and newcomer-comment IRR 1.67 [1.53, 1.82]. Every model lies on the diagonal-ish band; humans lie off it.

    python3 fig_p9.py -> results/p9/fig_entry_vs_return.png"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import RESULTS  # noqa: E402
from fig_panel import GRID, INK, INK2, NEG, POS, SURF  # noqa: E402


OFF = {"ministral3-8b": (-6, 4, "right"), "llama3.1-8b": (-2, 7, "center"), "qwen3-32b": (6, -10, "left")}


def main():
    D = pd.read_csv(os.path.join(RESULTS, "p9", "summary.csv"))
    W = pd.read_csv(os.path.join(RESULTS, "panel", "within.csv"))
    W = W[(W.site == "rscience") & (W.condition == "full")].set_index("model")
    D = D.join(W[["est", "lo", "hi"]], on="model").dropna(subset=["est"])
    D["y"] = np.log(D["ratio full/off"])
    D[["ylo", "yhi"]] = D["ratio CI"].str.strip("[]").str.split(",", expand=True).astype(float).apply(np.log)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(6.2, 4.4), facecolor=SURF)
    ax.set_facecolor(SURF)
    ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.axvline(0, color=GRID, lw=1, zorder=0)
    for r in D.itertuples():
        ax.plot([r.lo, r.hi], [r.y, r.y], color=INK2, lw=1, alpha=0.5, zorder=1)
        ax.plot([r.est, r.est], [r.ylo, r.yhi], color=INK2, lw=1, alpha=0.5, zorder=1)
        ax.plot(r.est, r.y, "o", ms=7, color=POS if r.y > 0 else NEG, mec=SURF, mew=1.5, zorder=3)
        dx, dy, ha = OFF.get(r.model, (5, 4, "left"))
        ax.annotate(r.model, (r.est, r.y), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color=INK2, ha=ha)
    hy, hlo, hhi = np.log(1.67), np.log(1.53), np.log(1.82)
    ax.plot([-0.23, 0.17], [hy, hy], color=INK, lw=2, solid_capstyle="round", zorder=4)
    ax.plot([-0.03, -0.03], [hlo, hhi], color=INK, lw=2, solid_capstyle="round", zorder=4)
    ax.plot(-0.03, hy, "D", ms=8, color=INK, mec=SURF, mew=1.5, zorder=5)
    ax.annotate("real newcomers\n(no return effect, +67% comments)", (-0.03, hy), xytext=(-12, 0),
                textcoords="offset points", fontsize=8, color=INK, fontweight="bold", va="center", ha="right")
    ax.set_xlabel("Sticky effect on P(return in 90 days), within-agent Δlogit")
    ax.set_ylabel("Sticky effect on P(would comment), log ratio")
    ax.set_ylim(-1.25, 0.72)
    ax.set_title("No model reaches the human point",
                 fontsize=10, color=INK, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(color=GRID, lw=0.5, zorder=0)
    fig.text(0.01, 0.005, "10 open-weight models, r/science 2016 newcomers. Across models: Spearman ρ = +0.67 (H12, p = .02). "
             "95% CIs.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = os.path.join(RESULTS, "p9", "fig_entry_vs_return.png")
    fig.savefig(out, dpi=200, facecolor=SURF)
    print(out)


if __name__ == "__main__":
    main()
