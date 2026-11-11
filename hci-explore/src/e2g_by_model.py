"""E2g, part 3: is "JEV beats the suggesting model's own token log-probabilities" true for each suggesting model?
Uses E2g's out-of-fold scores (results/e2g/oof.parquet); programmer bootstrap within each model's programmers.

  python3 -I src/e2g_by_model.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import e2e_realhumaneval as e2e  # noqa: E402
import e2f_cdhf as e2f  # noqa: E402

OUT = os.path.join(e2e.jev.ROOT, "results", "e2g")


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    a = e2e.load()
    o = pd.read_parquet(os.path.join(OUT, "oof.parquet"))
    assert (o.row.values == a.row.values).all()
    names = ["JEV only", "LR | text", "LR | text + conf", "LR | text + JEV", "LR | all (text + conf + history)",
             "LR | all (text + conf + history) + JEV"]
    pairs = [("LR | text + JEV", "LR | text"), ("LR | text + JEV", "LR | text + conf"),
             ("LR | text + conf", "LR | text"),
             ("LR | all (text + conf + history) + JEV", "LR | all (text + conf + history)")]
    rows = []
    for mdl, g in a.groupby("model"):
        sel = (a.model == mdl).values
        y, uid = o.accepted.values[sel], o.programmer_id.values[sel]
        tab, pt, bt = e2f.boot_table(y, uid, {n: o[n].values[sel] for n in names}, names, 2000, np.random.default_rng(0))
        d = e2f.diffs(pairs, pt, bt)
        print(f"\n== suggesting model {mdl}: {sel.sum()} suggestions, {len(np.unique(uid))} programmers, "
              f"accept rate {y.mean():.3f}\n" + tab[["auroc", "within_user_auroc"]].to_string() + "\n"
              + d[["pair", "auroc", "within_user_auroc"]].to_string(index=False))
        rows.append(d.assign(model=mdl, n=sel.sum(), programmers=len(np.unique(uid))))
    pd.concat(rows).to_csv(os.path.join(OUT, "by_model.csv"), index=False)


if __name__ == "__main__":
    main()
