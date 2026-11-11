"""Sites 3-4 (PREREG_sites.md): the frozen Futurology analysis, unchanged, pointed at another site (2026-11-06).

    python3 site_analyze.py ffxiv2019 qwen3-8b gpt-4o-mini jev qwen3-32b
    -> results/sites/<site>/summary.md
Only paths and the run-name prefix change; every estimator and verdict rule is futurology_analyze.main."""
import os
import sys

import futurology_analyze as FA
from common import RESULTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(site, models):
    FA.D = os.path.join(ROOT, "data", "sites", site)
    FA.OUT = os.path.join(RESULTS, "sites", site)
    runs = [f"{site}-{m}" for m in models]
    # futurology_analyze maps run names to SIGNS keys by stripping "futurology-"; give it that prefix via symlinks
    for r, m in zip(runs, models):
        link = os.path.join(FA.OUT, f"futurology-{m}")
        if not os.path.exists(link):
            os.symlink(os.path.join(FA.OUT, r), link)
    FA.PRIMARY = "futurology-qwen3-8b"
    FA.main([f"futurology-{m}" for m in models])
    p = os.path.join(FA.OUT, "summary.md")
    txt = open(p).read().replace("r/Futurology 2020 sticky-comment experiment", f"{site} sticky-comment experiment (PREREG_sites.md)")
    open(p, "w").write(txt.replace("## Agents: futurology-", f"## Agents: {site}-"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
