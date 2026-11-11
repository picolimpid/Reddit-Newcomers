"""Formal H8 / H9 (PREREG_sites.md §3), computed after the 'scores <site>' freezes of 2026-11-11.
Reads results/panel/within.csv (panel_summary.py). Agent side only; no outcome is used.
    python3 h8_h9_formal.py -> results/sites/h8_h9_formal.md"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = pd.read_csv(os.path.join(ROOT, "results", "panel", "within.csv"))
F = W[W.condition == "full"].set_index(["model", "site"])


def sign(m, s):
    if (m, s) not in F.index:
        return None
    r = F.loc[(m, s)]
    if abs(r.est) > 0.25 and (r.lo > 0 or r.hi < 0):
        return 1 if r.est > 0 else -1
    return 0


models = sorted(W.model.unique())
pred = [m for m in models if sign(m, "rscience") and sign(m, "rscience") == sign(m, "futurology")]
rows = []
for m in pred:
    for s in ("ffxiv2019", "science2020"):
        g = sign(m, s)
        r = F.loc[(m, s)]
        rows.append(dict(model=m, site=s, predicted="+" if sign(m, "rscience") > 0 else "−",
                         observed=f"{r.est:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]", hit=g == sign(m, "rscience")))
D = pd.DataFrame(rows)
k, n = int(D.hit.sum()), len(D)
L = ["# Formal H8 / H9 (PREREG_sites.md; computed 2026-11-11 after freezes 'scores ffxiv2019' and 'scores science2020')", "",
     f"Predicted set (same sign at r/science 2016 and Futurology, |Δ| > 0.25, CI excludes 0): {', '.join(pred)} ({len(pred)} models).", "",
     D.to_markdown(index=False), "",
     f"**H8: {k}/{n} = {k / n:.1%} of predictions hold (threshold 80%) -> {'SUPPORTED' if k / n >= 0.8 else 'NOT SUPPORTED'}.**", ""]
for s in ("ffxiv2019", "science2020"):
    T = W[(W.condition == "full") & (W.site == s)]
    hi, lo = T.loc[T.est.idxmax()], T.loc[T.est.idxmin()]
    rng = hi.est - lo.est
    L.append(f"**H9 {s}:** range of full-sticky Δlogit over {len(T)} panel models = {rng:.2f} ({hi.model} {hi.est:+.2f} to "
             f"{lo.model} {lo.est:+.2f}); threshold 1.0 -> {'SUPPORTED' if rng > 1 else 'NOT SUPPORTED'}.")
L += ["", "Deviations: Qwen2.5-72B dropped (online fp8 readout mass 0; AWQ diagnostic OOM at load); gpt-4o-mini not scored on "
      "science2020 (OpenRouter cap). Neither is in the H8 predicted set."]
open(os.path.join(ROOT, "results", "sites", "h8_h9_formal.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
