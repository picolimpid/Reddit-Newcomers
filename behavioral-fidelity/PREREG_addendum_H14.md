# Pre-registration addendum H14/H15: for whom does the sticky work? (2026-11-07)

Addendum to PREREG_futurology.md and PREREG_sites.md. Written and hashed before any Futurology 2020, ffxiv2019 or
science2020 outcome has been fetched (`data/futurology2020/outcomes.parquet` and `data/sites/*/outcomes.parquet` do not
exist at the time of hashing).

## Where the hypothesis comes from (not confirmatory)

On r/science 2016 the sticky has no average effect on newcomer 90-day return (−0.03 logit). Splitting newcomers by how
much Reddit history they had in the past year (n_hist, the items their persona is built from) was named as a follow-up
before it was run, but it is still one look at data that was already open, so r/science is the discovery sample:

- Tertiles of n_hist: T1 −0.30 [−0.67, +0.04], T2 −0.04, T3 +0.18; slope T3 − T1 +0.48 [+0.02, +0.98].
- Continuous, newcomers with n_hist > 0: treat:z = +0.275 [+0.074, +0.477], one-sided p = .004 (n = 3,540).
- Reading: the null average may hide a sticky that discourages thin-history newcomers and helps experienced ones.

Only 3 of 13 agent models show a positive slope with a CI excluding 0 (gemma3-27b, gpt-4o-mini, qwen3-8b); 6 show a
significant negative slope.

## Hypotheses

- **H14 (human, primary at Futurology).** In `logit P(return_90d) ~ treat * z` (z = standardised log1p(n_hist) within
  site; newcomers with n_hist > 0; SE clustered by post), the treat:z coefficient is > 0 at Futurology 2020, one-sided
  p < .05. Reported, not counted, at ffxiv2019 and science2020 (secondary).
- **H15 (agents).** A model "matches" a site when its T3 − T1 tertile slope (`hetero_history.py`, within-agent Δlogit
  full − off, post bootstrap) has a CI excluding 0 with the sign of that site's human treat:z estimate. Predicted: at
  most one third of the scored models match at Futurology.
  - Models: every panel model whose Futurology scores are frozen in results/futurology/FREEZE.txt when the outcomes are
    fetched. `hetero_history_futurology.csv` is regenerated and hashed at that moment, before the outcome fetch.
  - At the new sites, the models in the "scores <site>" freeze.

## Analysis

`src/h14_analyze.py` exactly as hashed. No covariates, no alternative splits for the test. If H14 fails at Futurology,
the r/science result is reported as an unreplicated exploratory finding.

## Known weaknesses

- n_hist is capped at 200 (100 comments + 100 posts fetched), so it mixes activity level with the cap.
- n_hist is measured before the experiment, so the moderation is not caused by treatment, but it is not randomised:
  it may proxy account age, karma, or community overlap.
- Futurology power: about 4,997 agent-eligible newcomers, more than r/science, but the base rate there is unknown.
- H15 depends on the human sign; if H14 fails and the Futurology estimate is near 0, "matches" is computed against
  a noisy sign and is reported with that caveat.
