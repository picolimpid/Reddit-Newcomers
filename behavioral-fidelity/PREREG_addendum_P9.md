# Addendum P9: entry as a calibration target (2026-11-07, written and hashed before any P9 score exists)

**Why.** Every agent experiment so far asks "will you return?", whose human effect is ≈ 0, so a model can match by
being insensitive. The r/science 2016 sticky has one large, well-identified human effect on newcomers' *entry*:
newcomer comments per non-AMA post rose from 6.34 to 10.57, negative-binomial IRR 1.67 [1.53, 1.82]
(CivilServant posts file, n = 2214; Matias 2019 PNAS H3). This addendum asks whether agents reproduce a real
non-zero effect, as called for by Goyal, Wang, Chandrasekharan & Sundaram (2026, "From Plausible to Causal").

**Design.** `src/p9_entry.py`. The same 1,000 r/science 2016 newcomer personas as P7. Each sees 2 posts drawn
(seeded by agent) from the 415 experiment posts that had a newcomer, never the post they actually commented on, in
4 conditions (off, full, welcome, rules), and answers "Would you write a comment on this post?" (first-token
P(Yes)). The sticky is only visible inside an opened post, so the human count ≈ views × P(comment | opened) with views
independent of the arm, and the agent analogue of the IRR is mean P(Yes | full) / mean P(Yes | off).

**Estimands** (`src/p9_analyze.py`). Agent ratio full/off with a bootstrap over shown posts; gap = log agent ratio −
log human IRR (human uncertainty added); within-agent Δlogit; welcome/off and rules/off ratios (exploratory).

**Hypotheses** (secondary, no multiplicity correction; models = every open-weight P1 panel model run on P9).
- **H11 (calibration fails).** For at least 2/3 of models, the 95% CI of the gap excludes 0. Expected: supported.
- **H12 (one disposition, two questions).** Across models, the Spearman correlation between the P9 log ratio
  (full/off) and the r/science full-sticky return Δlogit (results/panel/within.csv) is > 0. A model that reads the
  sticky as a deterrent should do so for both decisions. Expected: supported; reported with its permutation p.
- **H13 (direction of the miss).** Among models whose gap CI excludes 0, the majority under-predict (gap < 0).
  Expected: supported, because most panel models read the removal threat as discouraging (E16).

**Reported regardless:** every model run, the welcome/rules split, mean P(Yes | off) (a base-rate sanity check),
and readout mass. Deviations get a dated line here.
