# Pre-registration: do LLM-simulated newcomers predict the effect of a posted-rules sticky comment? A prospective retrodiction of the r/Futurology 2020 field experiment

Draft written 2026-11-03, before any post-entry activity of the r/Futurology commenters was fetched. Status lines at the end record what had and had not been done at each freeze.

## 1. Background and what we already know

- **Prior retrodiction (r/science 2016, Matias 2019 PNAS; `PROBES.md` §9, `results/probes/rscience_summary.md`).** We used the same newcomers and their real random assignment.
  - Sticky effect on 90-day return among newcomer commenters: human Δlogit −0.03 [−0.23, +0.17]. Qwen3-8B agents predicted −0.53 [−0.76, −0.30]; agent − human gap −0.50 [−0.81, −0.20].
  - Within agent, the removal-warning clause drove the drop (Qwen3-8B rules-only −0.78; welcome-only +0.54).
  - The hypotheses below were formed from that result and are tested here on a new community.
- **Published r/Futurology result.** The CAT Lab results post (reddit.com/r/Futurology/comments/x0p5lx, 2022-08-29) reports two things:
  - no detectable effect of the sticky on newcomer rule compliance or newcomer participation in r/Futurology;
  - 26% of posts were dropped from their study after a software glitch.

  It reports nothing on later return to the community, which is our outcome.
- **What we have not looked at.** We have not fetched or examined any activity of r/Futurology commenters after their first experiment comment. That data is fetched only by `futurology_collect.py outcomes`, which runs after this document is registered.

## 2. Design

### 2.1 Field experiment

- **Intervention.** CivilServant randomized new r/Futurology posts (starting January 2020) to one of two arms:
  - **control:** no sticky comment;
  - **treatment:** a sticky comment by CivilServantBot.
- **Treatment text** (as observed in Arctic Shift; 6,769 bot comments, 2020-01-07 to 2020-10-04):

  > Welcome to /r/Futurology! To maintain a healthy, vibrant community, **comments will be removed if they are disrespectful, off-topic, or spread misinformation** ([rules]). While thousands of people comment daily and follow the rules, mods do remove a few hundred comments per day. Replies to this announcement are auto-removed.

- **Randomization.** Blocks of 6, 50/50 within each block. The sequence is public at `CivilServant-Analysis/research-designs/reddit/randomizations/outputs/sticky_comment_futurology_0.randomizations.csv`.

### 2.2 Treatment status (`futurology_collect.py assign`)

No post-level assignment file is public.

**Attempted and abandoned: exact reconstruction.** We tried aligning the bot's stickies, in post-creation order, one-to-one with the public randomization sequence.
- It fails. Only 49% of stretches between consecutive stickied posts are consistent at a single global offset.
- The best offset jumps every few blocks (5, 7, −19, −20, …).
- Fewer than 8% of 25-stretch windows match the sequence anywhere.
- This is consistent with the software glitch the study team reported (26% of posts dropped).

**Design used: as-treated, with an eligibility restriction.** This uses assignment data only; no outcome was involved.
1. Start from all r/Futurology posts created between 10 minutes before the first bot sticky and the last one: 20,319 posts, 6,768 with a sticky.
2. Keep posts that were not removed or deleted, in both arms (10,354).
3. Drop runs of more than 6 consecutive unstickied posts. Under blocks of 6 with 3 treated these runs are impossible, so they mark periods when the bot was not processing posts. This removes 10 runs and 420 posts.
4. `treat` = the post carries the CivilServantBot sticky, which is also what each commenter saw.

**Checks on the kept 9,934 posts.**
- The treated share is 0.495 (design: 0.5).
- Pre-treatment covariates are balanced: hour, weekday, title length, self post, top-10 domain, no flair, prolific author. Every |standardized difference| ≤ 0.05; the largest is self post, −0.049 (`data/futurology2020/post_balance.csv`).
- The treated share by month (0.49–0.54) and by flair (0.47–0.51) is flat.
- Commenter-level pre-treatment balance is reported in the results: newcomer share, history size and attrition by arm.

**Remaining threat.** Bot failures could be correlated with unobserved post features. The as-treated contrast is then not a randomized comparison. The agent − human comparison (H1–H3) still uses the same people under the same condition. Month strata replace randomization blocks in the block-FE robustness column.

### 2.3 Units and outcomes (identical to the r/science retrodiction)

- **Sample.** Commenters on kept experiment posts; each author's first experiment comment. Exclude `[deleted]`, AutoModerator and CivilServantBot. Draw a random sample of 8,000 authors (shuffle seed 0).
- **Newcomer.** No r/Futurology comment or post in the 365 days before the post.
- **Agent eligibility.** At least one comment or post anywhere in that year (n_hist > 0), so a persona can be built.
- **Primary outcome.** `return_90d`: any r/Futurology comment outside the entry thread, or any r/Futurology post, within 90 days after the first experiment comment.
- **Secondary outcome.** `return_180d`, the same within 180 days.

### 2.4 Agents (fixed before registration; no prompt changes after r/science)

- **Agent construction.** Each newcomer becomes an agent exactly as in `rscience_agents.py`:
  - the same persona, system prompt, entry block, observed replies, and question with the subreddit changed;
  - first-token P(Yes) (Jev: the "noul" score).
- **Sticky conditions** (within agent):
  - `off`: no pinned comment.
  - `full`: the treatment text above, with markdown removed.
  - `welcome`: only "Welcome to /r/Futurology!".
  - `rules`: everything after it.
  - `off+tmpl`: no sticky, plus the benchmark's template welcome reply.
- **`arm` condition.** The condition each person actually saw: `full` if treated, `off` if control.
- **Models.**
  - **Primary:** Qwen3-8B (bf16, vLLM, temperature 0); gpt-4o-mini (OpenRouter, provider pinned to OpenAI).
  - **Secondary:** Jev 1.14 (typesafe/jev-1.14 via OpenRouter); Qwen3-32B (fp8 weight-only), included only if its scores are frozen before the reveal.
- **Sample per condition.** Qwen models: all eligible newcomers in every condition. API models: all newcomers in `off`/`full`, and a random 1,000 (seed 0) in `welcome`/`rules`/`off+tmpl` (budget).

## 3. Hypotheses

The hypotheses come from the r/science 2016 retrodiction (same newcomers, real arms, `results/probes/rscience_summary.md`). In that experiment, model families erred in different directions:

| r/science 2016 | arm-effect gap vs human | within: full | rules only | welcome only | template reply |
|---|---|---|---|---|---|
| Qwen3-8B | −0.50 [−0.81, −0.20] | −1.25 | −0.78 | +0.54 | +2.09 |
| gpt-4o-mini | +0.02 [−0.33, +0.38] | +0.47 | +0.05 | +1.54 | +5.63 |
| Jev 1.14 | +0.04 [−0.16, +0.23] | +0.01 | +0.03 | −0.06 | +0.09 |

- **H1 (primary, confirmatory).** For Qwen3-8B, the agent − human gap in the arm effect on `return_90d` is negative.
  - Gap = agent arm effect − human arm effect, both estimated by the same regression on the same newcomers (§4).
  - Supported if the 95% paired post-cluster bootstrap CI lies entirely below 0.
- **H2 (mechanism, Qwen3-8B).** Both of the following hold, each with its 95% CI entirely below 0:
  - within-agent Δlogit(rules − off) < 0;
  - Δlogit(rules − welcome) < 0.
- **H3 (gpt-4o-mini: over-sensitivity to welcoming text, not to the warning).** All three hold:
  - within-agent Δlogit(welcome − off) > 0 (CI lower bound > 0);
  - Δlogit(template − welcome) > 0 (CI lower bound > 0);
  - |Δlogit(rules − off)| < 0.25.

  Its arm-effect gap is reported two-sided, with no directional prediction, because the r/science gap was ≈ 0.
- **H4 (human effect, equivalence test).** The human arm effect on `return_90d` lies within ±0.25 logit.
  - TOST, α = .05, using the cluster-robust SE.
  - If the CI is wider than the bounds, H4 is reported as inconclusive and not as support.
- **H5 (Jev insensitivity).** Jev's arm effect is within ±0.10 logit, and |within-agent Δlogit| < 0.15 for every condition.
- **Qwen3-32B.** Exploratory: all estimates are reported, with no hypothesis.
- **Multiplicity.** H1 is the single primary test. H2–H5 are secondary and reported without correction.

What we would report as a failure of H1: the gap CI includes or exceeds 0. This would mean Qwen3-8B's predicted arm effect is statistically indistinguishable from the human one, for example because r/Futurology newcomers really are deterred by the warning.

### 3.1 Addendum (2026-11-04, after the r/science Qwen3-32B run; before any r/Futurology agent score or outcome existed)

The r/science Qwen3-32B run finished after freeze 1. Within agent, it reverses Qwen3-8B on both components:

| r/science 2016, within agent | full | rules only | welcome only | template reply |
|---|---|---|---|---|
| Qwen3-32B | +0.77 | +1.03 | −0.74 | +1.25 |

Its arm-effect gap was +0.15 [−0.14, +0.44]. Two hypotheses about agents only are added (no human data involved):

- **H6 (model-specific signs are stable across communities).** For each component where the model's r/science |Δlogit| > 0.25, the r/Futurology within-agent Δlogit has the same sign, with the 95% CI excluding 0. The predicted signs:
  - Qwen3-8B: rules −, welcome +;
  - Qwen3-32B: rules +, welcome −;
  - gpt-4o-mini: welcome +.

  Each of the five predictions is reported separately.
- **H7 (models disagree on the design recommendation).** In r/Futurology, the full-sticky within-agent Δlogit is negative for Qwen3-8B and positive for Qwen3-32B (each CI excludes 0).

Qwen3-32B remains exploratory for H1-type human comparisons: its gap is reported, with no directional hypothesis.

## 4. Analysis (code frozen as `futurology_analyze.py`, written from `rscience_analyze.py`)

- **Arm effect.** `glm(y ~ treat, Binomial)` with SEs clustered by post. y is the human `return_90d`, or the agent's P under the `arm` condition (fractional logit).
  - Primary specification: no block fixed effects. Assignment is 50/50 within every block, so the plain contrast is unbiased; block FE is reported as a robustness column.
- **Gap.** Agent − human difference of the logit of pooled rates. Post-cluster bootstrap, 4,000 replicates, seed 0.
- **Within-agent Δlogit.** Mean of logit P(condition) − logit P(off), with a post-cluster bootstrap 95% CI (2,000 replicates, seed 0).
- **Also reported:**
  - attrition (`[deleted]` author share by arm);
  - newcomer share by arm;
  - mean P by condition;
  - AUROC of the `arm` P for `return_90d`.
- **Robustness (reported, not used for the hypotheses):**
  - `return_180d`;
  - block FE;
  - excluding the 10 largest threads.

## 5. Interpretation limits stated in advance

- **Conditional on commenting.** The sticky can change who comments, so the human contrast is not a pure causal effect on return. The agent − human comparison uses the same people and the same arms, so H1–H3 are unaffected.
- **Missing authors.** Removed comments lose their author in Arctic Shift, so newcomers whose first comment was removed are missing. We report this by arm.
- **As-treated, not intention-to-treat.** The human contrast compares posts with and without a sticky among posts the bot was processing (§2.2). It is a randomized comparison only to the extent that bot failures were unrelated to post features. Balance checks bound this; they do not prove it.
- **Removed posts excluded.** Posts removed or deleted after the bot processed them are excluded in both arms, so the human estimate is for posts that stayed up.

## 6. Freeze protocol

1. This file and `futurology_collect.py`, `futurology_agents.py` and `futurology_analyze.py` are frozen. Their SHA-256 hashes are recorded in `results/futurology/FREEZE.txt`.
2. Inputs are collected (`collect`, no outcomes). Agents are run, and the agent score files are hashed into `FREEZE.txt`.
3. The user files this document, with `FREEZE.txt`, as an OSF registration.
4. Only then are `futurology_collect.py outcomes` and `futurology_analyze.py` run. Every deviation from this plan is listed in the results.

## 7. Reconstruction and freeze record

(filled in below by the assign / freeze steps)
