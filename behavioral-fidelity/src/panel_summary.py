"""Many-model panel summary (2026-11-06). Agent-only for sealed sites; for r/science 2016 (outcomes public
since 2019) it also reports the arm-effect gap against the real newcomers.

    python3 panel_summary.py            -> results/panel/summary.md, results/panel/within.csv
Reads every scores.parquet under results/probes/rscience-*, results/futurology/futurology-*,
results/sites/<site>/<site>-*. Never reads data/futurology2020/outcomes.parquet or a site's outcomes."""
import glob
import os
import warnings

import numpy as np
import pandas as pd

from analyze_probes import boot, fmt, logit
from common import RESULTS

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(RESULTS, "panel")
EQ = 0.25
SKIP = ("-tmpl", "-limit", "-q2", "-nosys", "-order", "-observer")  # probe runs and prompt variants (multiverse_summary.py)


def runs():
    pats = {"rscience": f"{RESULTS}/probes/rscience-*/scores.parquet",
            "futurology": f"{RESULTS}/futurology/futurology-*/scores.parquet"}
    for d in glob.glob(f"{RESULTS}/sites/*/"):
        s = os.path.basename(d.rstrip("/"))
        pats[s] = f"{d}{s}-*/scores.parquet"
    for site, pat in pats.items():
        for f in sorted(glob.glob(pat)):
            name = os.path.basename(os.path.dirname(f))[len(site) + 1:]
            if not any(x in name for x in SKIP):
                yield site, name, f


def inputs(site):
    if site == "rscience":
        return pd.read_parquet(os.path.join(ROOT, "data", "rscience2016", "retro.parquet"))
    if site == "futurology":
        return pd.read_parquet(os.path.join(ROOT, "data", "futurology2020", "inputs.parquet"))
    return pd.read_parquet(os.path.join(ROOT, "data", "sites", site, "inputs.parquet"))


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, gaps, cache = [], [], {}
    for site, name, f in runs():
        R = cache.setdefault(site, inputs(site))
        S = pd.read_parquet(f)
        if "variant" in S.columns:
            S = S[S.variant == "base"]
        W = S.pivot_table(index="idx", columns="condition", values="p")
        if "off" not in W:
            continue
        pid = R.loc[W.index, "post_id"]
        off = logit(W["off"])
        for c in W.columns:
            if c == "off":
                continue
            m = W[c].notna() & W["off"].notna()
            v = boot((logit(W[c]) - off)[m], pid[m])
            rows.append({"site": site, "model": name, "condition": c, "est": v[0], "lo": v[1], "hi": v[2],
                         "n": int(m.sum()), "mean_p_off": W["off"].mean()})
        if site == "rscience" and "return_90d" in R.columns and "full" in W:
            from rscience_analyze import arm_fit, paired_gap
            H = R.loc[W.index].copy()
            H = H[W["full"].notna() & W["off"].notna()]
            H["p_arm"] = np.where(H.treat == 1, W.loc[H.index, "full"], W.loc[H.index, "off"])
            g = paired_gap(H, n=2000)
            gaps.append({"model": name, "agent arm effect": fmt(*arm_fit(H, "p_arm")[:3]),
                         "gap vs human": fmt(*g), "gap excludes 0": g[1] > 0 or g[2] < 0, "n": len(H)})
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, "within.csv"), index=False)
    L = ["# Many-model panel: within-agent Δlogit of each sticky component vs no sticky", "",
         "Cell = mean Δlogit [post-cluster bootstrap 95% CI]. **Bold** = CI excludes 0. "
         "Human r/science 2016 arm effect: −0.03 [−0.23, +0.17].", ""]
    for c in ["full", "rules", "welcome", "off+tmpl"]:
        T = D[D.condition == c].copy()
        T["cell"] = [("**" if (lo > 0 or hi < 0) else "") + fmt(e, lo, hi) + ("**" if (lo > 0 or hi < 0) else "")
                     for e, lo, hi in zip(T.est, T.lo, T.hi)]
        P = T.pivot_table(index="model", columns="site", values="cell", aggfunc="first")
        L += [f"## {c}", "", P.to_markdown(), ""]
    F = D[D.condition == "full"]
    sg = F.assign(sign=np.where((F.lo > 0) & (F.est > EQ), "+", np.where((F.hi < 0) & (F.est < -EQ), "−", "0")))
    L += [f"## Sign of the full-sticky prediction (+ / −: CI excludes 0 and |Δ| > {EQ}, the PREREG_futurology H6 "
          "threshold; 0 otherwise)", "",
          sg.pivot_table(index="model", columns="site", values="sign", aggfunc="first").to_markdown(), "",
          "Count by site: " + "; ".join(f"{s}: " + ", ".join(f"{k} {v}" for k, v in g.sign.value_counts().items())
                                        for s, g in sg.groupby("site")), ""]
    if len(F.site.unique()) > 1:  # do models keep their sign across communities?
        P = F.pivot_table(index="model", columns="site", values="est")
        L += ["Correlation of the full-sticky Δlogit across sites (models as units):", "",
              P.corr().round(2).to_markdown(), ""]
    if gaps:
        L += ["## r/science 2016: agent arm effect vs the real newcomers (same people, same arms)", "",
              pd.DataFrame(gaps).to_markdown(index=False), ""]
    txt = "\n".join(L) + "\n"
    open(os.path.join(OUT, "summary.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
