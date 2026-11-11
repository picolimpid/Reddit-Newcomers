# Pre-registration addendum: H10, does the median of a model panel track the human effect? (2026-11-06)

Applies to the three sealed sites: r/Futurology 2020 (PREREG_futurology.md), r/ffxiv 2019 and r/science 2020
(PREREG_sites.md). Written and hashed before any outcome of these sites is fetched. It adds one secondary hypothesis and
changes nothing in H1–H9.

## Why

Single models disagree in sign (ROADMAP R1). A common response is to average several models ("wisdom of the silicon
crowd", Schoenegger et al., Science Advances 2024). This tests whether that rescues the design advice.

What we already know (exploratory, r/science 2016; `src/panel_median.py`, `results/panel/median*.md`):

- **All 9 models, including gpt-4o-mini and Jev:** the panel median of the agent arm effects is −0.08; its gap to the human
  effect is −0.06 [−0.33, +0.15]. The 90% CI is [−0.27, +0.11], so the gap is not equivalent within ±0.25.
- **Leave one model out:** the median moves between −0.05 and −0.31 depending on which model is dropped.
- **The 7 open-weight models only:** the median is −0.53, and the gap is −0.50 [−0.67, +0.01].

## H10

- **Panel.** Every open-weight model whose site scores (conditions off and full, all agents) are hashed in the
  "scores <site>" freeze line. For Futurology, this is the panel models' Futurology scores hashed before
  `futurology_collect.py outcomes` runs. API models (gpt-4o-mini, Jev) are left out: they score only 800 agents at the new
  sites, and they are not reproducible.
- **Estimand.**
  - Per model: the agent arm effect, defined as the logit of pooled P(arm) in treated minus the logit of pooled P(arm) in
    control, computed as in `rscience_analyze.paired_gap`.
  - Panel estimate: the median over models.
  - Gap: the panel median minus the human arm effect on 90-day return.
- **Inference.** A post-cluster bootstrap with 2,000 resamples and seed 0, where one resample is shared by all models and
  the human outcome (`src/panel_median.py <site> --open-only`).
- **H10 (equivalence).** The 90% CI of the gap lies inside ±0.25, which is TOST at α = .05 with the H4 margin.
  - Expectation, stated before the reveal: **not supported**. On r/science 2016 the open-weight median is −0.53.
  - H10 is reported either way, together with the gap estimate and the leave-one-model-out table.
- **Status.** Secondary, with no correction. H1 remains the single primary test per site.

## Disclosure

- **Timing.** This addendum is written after the Futurology plan was frozen (freezes 1–4). It is also after the agent-only
  hypotheses H2, H3 and H5–H7 were evaluated, which OSF_FILING.md discloses.
- **What has been seen.** No human outcome of any sealed site has been fetched.
- **What has not been seen.** Futurology scores of the newer panel models are frozen together with this file. The new-site
  scores do not exist yet.
