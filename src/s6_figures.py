"""Stage 6: static figures for the report."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from api import DATA, RESULTS

OUT = RESULTS
BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF})
YLAB = {"reply_received": "Reply received (odds ratio)", "n_replies": "Number of replies (rate ratio)",
        "reengaged": "Re-engaged in thread (odds ratio)"}
MLAB = {"mini_mean": "MiniLM · mean", "mini_top5": "MiniLM · top-5", "mini_recency": "MiniLM · recency-wtd",
        "mini_top10": "MiniLM · top-10", "mpnet_mean": "MPNet · mean", "mpnet_top5": "MPNet · top-5"}

# ---- Figure 1: forest plot, effect per +1 SD alignment, by measure & outcome
main = pd.read_csv(os.path.join(OUT, "main_models.csv"))
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
order = list(MLAB)[::-1]
for ax, (y, lab) in zip(axes, YLAB.items()):
    d = main[main.outcome == y].set_index("measure").loc[order]
    ypos = np.arange(len(order))
    ax.axvline(1, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.hlines(ypos, d.ci_lo, d.ci_hi, color=BLUE, lw=2)
    ax.plot(d.effect, ypos, "o", color=BLUE, ms=7, mec=SURF, mew=2)
    ax.set_title(lab, fontsize=10, color=INK, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
axes[0].set_yticks(np.arange(len(order)), [MLAB[m] for m in order])
fig.supxlabel("Effect of +1 SD prior-interest alignment (95% CI, post fixed effects, clustered by post)",
              fontsize=9, color=MUTED)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_forest.png"), dpi=160)

# ---- Figure 2: outcome rate by within-post alignment quartile
q = pd.read_csv(os.path.join(OUT, "within_post_quartiles.csv"))
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, y in zip(axes, YLAB):
    for m, c, lab in (("mini_mean", BLUE, "mean similarity"), ("mini_top5", ORANGE, "top-5 similarity")):
        d = q[(q.measure == m) & (q.outcome == y)]
        ax.errorbar(d.q + 1 + (0.06 if c == ORANGE else -0.06), d["mean"], yerr=1.96 * d["sem"],
                    color=c, lw=2, marker="o", ms=7, mec=SURF, mew=2, capsize=0, label=lab)
    ax.set_xticks([1, 2, 3, 4], ["Q1\nlow", "Q2", "Q3", "Q4\nhigh"])
    ax.set_title(YLAB[y].split(" (")[0] + (" (mean count)" if y == "n_replies" else " (share)"),
                 fontsize=10, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
axes[0].legend(frameon=False, fontsize=9)
fig.supxlabel("Alignment quartile, computed within each target post (MiniLM)", fontsize=9, color=MUTED)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_quartiles.png"), dpi=160)

# ---- Figure 3: validity - alignment with the entry post vs. a random post elsewhere
df = pd.read_parquet(os.path.join(DATA, "analysis_table.parquet"))
nc = df[df.newcomer_12m == 1]
fig, ax = plt.subplots(figsize=(6.2, 3.4))
bins = np.linspace(min(nc.mini_mean.min(), nc.mini_mean_othersub.min()),
                   max(nc.mini_mean.max(), nc.mini_mean_othersub.max()), 40)
ax.hist(nc.mini_mean_othersub, bins=bins, color=ORANGE, alpha=0.55, label="random post, other subreddit")
ax.hist(nc.mini_mean, bins=bins, color=BLUE, alpha=0.55, label="the post they actually entered through")
ax.set_xlabel("Mean-history cosine similarity (MiniLM)", color=MUTED)
ax.set_ylabel("Newcomers", color=MUTED)
ax.legend(frameon=False, fontsize=9)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_validity.png"), dpi=160)
print("figures saved")
