"""Follow-up: is the alignment -> re-engagement association just a by-product of
receiving replies? Re-fit with reply_received as a control, and separately among
newcomers who did / did not receive a reply."""
import os

import pandas as pd

import s5_analysis as A  # re-uses covariates and fit(); re-runs stage 5 on import

rows = []
for m in ["mini_mean", "mini_top5", "mpnet_mean", "mpnet_top5"]:
    for name, d, extra in [("+ reply_received control", A.nc, " + reply_received"),
                           ("received a reply", A.nc[A.nc.reply_received == 1], ""),
                           ("received no reply", A.nc[A.nc.reply_received == 0], "")]:
        try:
            r = A.fit(d, "reengaged", m, "logit", extra)
        except Exception as e:  # noqa: BLE001
            r = {"measure": m, "error": str(e)[:100]}
        r["spec"] = name
        rows.append(r)
out = pd.DataFrame(rows)
out.to_csv(os.path.join(A.OUT, "reengagement_check.csv"), index=False)
print(pd.crosstab(A.nc.reply_received, A.nc.reengaged, margins=True))
print(out[["spec", "measure", "n", "n_posts", "effect", "ci_lo", "ci_hi", "p"]].round(4).to_string())
