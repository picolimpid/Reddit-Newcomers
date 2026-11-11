# Pre-registration addendum: two more CivilServant sticky-comment sites (r/ffxiv 2019, r/science 2020)

Written 2026-11-06, before any agent score or any post-entry activity existed for either site. It extends `PREREG_futurology.md`; everything not stated here is identical to that document.

## 1. Sites

The CivilServantBot comment history in Arctic Shift shows two further newcomer sticky-comment experiments. Randomization sequences are public in CivilServant-Analysis (`sticky_comment_ffxiv_0`, `sticky_comment_science_1`): 2 arms, blocks of 6.

| Site | Window (bot comments) | Bot stickies | Sticky text (markdown removed) |
|---|---|---|---|
| `ffxiv2019` | 2019-07-06 → 2019-09 | 6,043 | "Welcome to /r/ffxiv! Gyorin the Gunbreaker asks you to keep in mind Reddiquette, follow our community rules, and be civil with your fellow Warriors of Light. Threads on bad experiences with other players (even anonymous) as well as hate-based comments such as personal attacks, bigotry, hate speech, and name shaming are subject to removal by the moderator team under rule 1. Please report any rule violations; the moderator team will review them as soon as possible." |
| `science2020` | 2020-02-21 → 2020-07 | 4,128 | "Welcome to r/science! Our team of 1,500+ moderators will remove comments if they are jokes, anecdotes, memes, off-topic or medical advice (rules). We encourage respectful discussion about the science of the post." |

For both sites, `welcome` = the first sentence and `rules` = the rest.

## 2. Treatment status (`site_collect.py assign`)

The as-treated rule of PREREG_futurology §2.2 is used, with one change: the definition of removed or deleted posts. 2019 Arctic Shift posts have no `removed_by_category`, so a post counts as removed or deleted if any of the following holds:

- `removed_by_category` is set;
- the author is `[deleted]`;
- the body is `[removed]` or `[deleted]`;
- the moderators' flair contains "Removed".

The proxies were added after a first pass on ffxiv showed `[deleted]` posts concentrated in the control arm (treated share 0.36, against 0.48 overall). The same rule is then applied to both sites. No outcome was involved.

| Site | Posts in window | Removed/deleted (control / treated) | Downtime runs dropped | Kept | Treated share | Max \|std diff\| |
|---|---|---|---|---|---|---|
| ffxiv2019 | 15,587 | 0.297 / 0.200 | 119 runs, 1,448 posts | 10,097 | 0.479 | 0.079 (no flair; base rate 0.6%) |
| science2020 | 16,859 | 0.827 / 0.526 | 14 runs, 193 posts | 3,966 | 0.493 | 0.163 (no flair; base rate 1.3%); next largest is self post, −0.099 |

**Stated limitation.** In both sites, removed posts are much more common among unstickied posts. This is consistent with removal before the bot processed a post. Excluding removed posts in both arms keeps the comparison among posts that stayed up, but it does not make the comparison randomized. science2020 keeps only 24% of window posts.

## 3. Hypotheses (per site; agent scores are scored only after this file is frozen)

**Per-site hypotheses (identical to PREREG_futurology).** H1–H7 apply to each site separately, with the same models, thresholds and code (`site_analyze.py`, which calls the frozen `futurology_analyze.main`). In short:

- H1: the Qwen3-8B gap < 0;
- H2: Qwen3-8B rules < 0, and rules < welcome;
- H3: gpt-4o-mini criteria;
- H4: human TOST within ±0.25;
- H5: Jev flat;
- H6: sign stability;
- H7: 8B full < 0 and 32B full > 0.

**Cross-site panel hypotheses (new; agent-only, evaluated by `panel_summary.py`).**

- **H8 (model identity predicts direction at new sites).** The predicted set is every panel model whose full-sticky within-agent Δlogit has the same sign at r/science 2016 and Futurology 2020, with |Δ| > 0.25 and a CI that excludes 0 at both. For each such model and each new site, predict the same sign, with |Δ| > 0.25 and the CI excluding 0.
  - Supported if ≥ 80% of these model × site predictions hold.
  - The set is determined mechanically from P1 runs that finish before the new sites are scored.
- **H9 (between-model spread exceeds the human effect).** At each new site, the range of full-sticky within-agent Δlogit across panel models is larger than 1.0 logit. This is four times the H4 equivalence bound.

H1 remains the single primary test per site. H2–H9 are secondary and reported without correction.

## 4. Models

- **Agents.** Qwen3-8B, Qwen3-32B (fp8), gpt-4o-mini and Jev, as in PREREG_futurology.
  - API models score a random 800 agents (seed 0) in all conditions, because the remaining OpenRouter budget is about $2.3.
- **Panel.** Every P1 panel model that has finished (ROADMAP.md) is also scored on both sites before the freeze. These runs are exploratory except for H8 and H9.

## 5. Freeze and reveal

1. This file and `site_fetch.py`, `site_collect.py`, `site_analyze.py`, `panel_agents.py` and `panel_summary.py` are hashed into `results/sites/FREEZE.txt` (freeze S1) before any new-site agent score exists.
2. Inputs are collected, then all agents are scored, then the inputs and scores are hashed (freeze "scores <site>").
3. `site_collect.py outcomes <site>` refuses to run until that line exists.
   - Ideally, the user files this addendum on OSF together with PREREG_futurology.md before the outcomes are fetched.
   - If it is not filed, the timestamps in FREEZE.txt are the only (self-attested) record, and the paper says so.
