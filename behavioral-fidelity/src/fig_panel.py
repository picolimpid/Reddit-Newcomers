"""Figure 1 (2026-11-06): within-agent effect of the full sticky by model (rows) and site (facets).

    python3 fig_panel.py   (after panel_summary.py)  -> results/panel/fig1_full_sticky.png
Dot + 95% CI; polarity uses the diverging pair (blue +, red −) with a neutral gray for |Δ| <= 0.25 or a CI
that includes 0, plus marker shape as a second encoding. Gray band = human r/science 2016 arm effect;
dashed lines = ±0.25 equivalence bounds (PREREG H4)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from common import RESULTS  # noqa: E402

EQ = 0.25
POS, NEG, NEU = "#2a78d6", "#e34948", "#8a8984"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SITES = [("rscience", "r/science 2016\n(human outcome known)"), ("futurology", "r/Futurology 2020\n(human outcome sealed)"),
         ("ffxiv2019", "r/ffxiv 2019\n(sealed)"), ("science2020", "r/science 2020\n(sealed)")]
HUMAN = {"rscience": (-0.03, -0.23, 0.17)}


def main(cond="full"):
    D = pd.read_csv(os.path.join(RESULTS, "panel", "within.csv"))
    D = D[D.condition == cond]
    sites = [(s, t) for s, t in SITES if s in set(D.site)]
    order = (D[D.site == "rscience"].set_index("model").est.reindex(D.model.unique())
             .fillna(D.groupby("model").est.mean()).sort_values().index.tolist())
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, len(sites), figsize=(2.9 * len(sites) + 1.2, 0.32 * len(order) + 1.4),
                             sharey=True, facecolor=SURF)
    axes = [axes] if len(sites) == 1 else axes
    lim = max(abs(D.lo.min()), abs(D.hi.max()), 1.0) * 1.08
    for ax, (s, title) in zip(axes, sites):
        ax.set_facecolor(SURF)
        if s in HUMAN:
            e, lo, hi = HUMAN[s]
            ax.axvspan(lo, hi, color=GRID, zorder=0, lw=0)
            ax.text(-EQ - 0.06, -0.75, "human effect (95% CI)", ha="right", va="center", color=INK2, fontsize=7.5)
        for x in (-EQ, EQ):
            ax.axvline(x, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
        ax.axvline(0, color=INK2, lw=1, zorder=1)
        G = D[D.site == s].set_index("model")
        for y, m in enumerate(order):
            if m not in G.index:
                continue
            r = G.loc[m]
            pos, neg = r.lo > 0 and r.est > EQ, r.hi < 0 and r.est < -EQ
            c, mk = (POS, "o") if pos else (NEG, "s") if neg else (NEU, "D")
            ax.plot([r.lo, r.hi], [y, y], color=c, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(r.est, y, marker=mk, ms=6, color=c, mec=SURF, mew=1.5, zorder=3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-1.3, len(order) - 0.5)
        ax.set_title(title, color=INK, fontsize=9, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.tick_params(length=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel("Δlogit P(return), sticky vs none")
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order)
    h = [plt.Line2D([], [], color=POS, marker="o", lw=2, label="predicts more return"),
         plt.Line2D([], [], color=NEG, marker="s", lw=2, label="predicts less return"),
         plt.Line2D([], [], color=NEU, marker="D", lw=2, label=f"|Δ| ≤ {EQ} or CI includes 0")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Same sticky comment, different advice: within-agent effect of the {cond} sticky by model",
                 x=0.01, ha="left", color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    out = os.path.join(RESULTS, "panel", f"fig1_{cond}_sticky.png")
    fig.savefig(out, dpi=200, facecolor=SURF)
    print(out)


if __name__ == "__main__":
    main()
