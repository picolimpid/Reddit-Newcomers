"""P5 (2026-11-06): final scores of each newcomer's first comment (Arctic Shift retrieves 2016/2020 comments
weeks to months after creation), the 'positive feedback' treatment of a Lambert et al. (CHI 2025)-style design.
The score is post-entry information but is not the outcome (return); for sealed sites it is still fetched only
after their reveal.

    python3 p5_scores.py rscience2016   -> data/rscience2016/first_scores.parquet"""
import os
import sys

import pandas as pd

sys.path.insert(0, "/data/hanqim2/Reddit Newcomers/src")
from api import get  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHS = {"rscience2016": ("rscience2016", "retro.parquet")}


def main(site):
    d, f = PATHS[site]
    R = pd.read_parquet(os.path.join(ROOT, "data", d, f))
    ids = list(R.comment_id)
    rows = []
    for i in range(0, len(ids), 100):
        rows += get("comments/ids", ids=",".join(ids[i:i + 100]), fields="id,score,created_utc,retrieved_on")
    S = pd.DataFrame(rows)
    S["age_days_at_retrieval"] = (S.retrieved_on - S.created_utc) / 86400
    S.to_parquet(os.path.join(ROOT, "data", d, "first_scores.parquet"))
    print(len(R), len(S), S.score.describe().round(1).to_dict(), S.age_days_at_retrieval.describe().round(1).to_dict())


if __name__ == "__main__":
    main(sys.argv[1])
