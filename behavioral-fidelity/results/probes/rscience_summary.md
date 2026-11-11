# Retrodicting the r/science sticky-comment experiment (Matias 2019)

## Attrition and composition

Comments fetched from Arctic Shift on non-AMA experiment posts, by arm (author `[deleted]` = removed or deleted, not identifiable):

|   treat |   comments |   deleted_author |
|--------:|-----------:|-----------------:|
|       0 |      29641 |            0.438 |
|       1 |      43968 |            0.448 |

CivilServant newcomer comments per post (all newcomers, including removed; paper's data): arm 0: 6.34, arm 1: 10.57

Sampled identifiable commenters: 8000; newcomers (no r/science activity in the prior year): 3565 (44.6%).

Newcomer share by arm: control 0.442, treated 0.448; logit Δ +0.026 [-0.133, +0.185], p = 0.751.

## Human effect of the sticky on newcomer return (conditional on commenting)

| outcome     |   control |   treated | Δlogit [95% CI]         |    OR |     p | block FE Δlogit         |    n |
|:------------|----------:|----------:|:------------------------|------:|------:|:------------------------|-----:|
| return_90d  |     0.154 |     0.15  | -0.029 [-0.230, +0.172] | 0.971 | 0.775 | -0.086 [-0.496, +0.325] | 3540 |
| return_180d |     0.23  |     0.222 | -0.043 [-0.228, +0.142] | 0.958 | 0.647 | +0.129 [-0.226, +0.485] | 3540 |

## Agents: rscience-qwen3-1.7b (3540 newcomers)

Mean P: full 0.979, off 0.997, off+tmpl 0.998, rules 0.975, welcome 0.995 | human 90-day return 0.152

AUROC of P(arm) for the real 90-day return: 0.509

**Arm effect, same regression as the human outcome:** agent Δlogit -1.816 [-2.135, -1.497] vs human -0.029 [-0.230, +0.172] (same 3540 newcomers).

Agent − human gap (paired post-cluster bootstrap): -1.787 [-2.189, -1.383]

Within-agent Δlogit vs no sticky (all agents):

| condition   | Δlogit               |   n agents |
|:------------|:---------------------|-----------:|
| full        | -1.97 [-2.06, -1.88] |       3540 |
| off+tmpl    | +1.16 [+1.09, +1.24] |       3540 |
| rules       | -2.16 [-2.25, -2.07] |       3540 |
| welcome     | -0.33 [-0.42, -0.23] |       3540 |

## Agents: rscience-qwen3-8b (3540 newcomers)

Mean P: full 0.036, off 0.071, off+tmpl 0.222, rules 0.043, welcome 0.114 | human 90-day return 0.152

AUROC of P(arm) for the real 90-day return: 0.576

**Arm effect, same regression as the human outcome:** agent Δlogit -0.528 [-0.760, -0.296] vs human -0.029 [-0.230, +0.172] (same 3540 newcomers).

Agent − human gap (paired post-cluster bootstrap): -0.499 [-0.810, -0.199]

Within-agent Δlogit vs no sticky (all agents):

| condition   | Δlogit               |   n agents |
|:------------|:---------------------|-----------:|
| full        | -1.25 [-1.35, -1.16] |       3540 |
| off+tmpl    | +2.09 [+2.01, +2.17] |       3540 |
| rules       | -0.78 [-0.86, -0.69] |       3540 |
| welcome     | +0.54 [+0.46, +0.61] |       3540 |

## Agents: rscience-qwen3-32b (3540 newcomers)

Mean P: full 0.066, off 0.053, off+tmpl 0.124, rules 0.076, welcome 0.036 | human 90-day return 0.152

AUROC of P(arm) for the real 90-day return: 0.615

**Arm effect, same regression as the human outcome:** agent Δlogit +0.121 [-0.166, +0.409] vs human -0.029 [-0.230, +0.172] (same 3540 newcomers).

Agent − human gap (paired post-cluster bootstrap): +0.151 [-0.141, +0.440]

Within-agent Δlogit vs no sticky (all agents):

| condition   | Δlogit               |   n agents |
|:------------|:---------------------|-----------:|
| full        | +0.77 [+0.67, +0.88] |       3540 |
| off+tmpl    | +1.25 [+1.12, +1.37] |       3540 |
| rules       | +1.03 [+0.92, +1.15] |       3540 |
| welcome     | -0.74 [-0.83, -0.65] |       3540 |

## Agents: rscience-gpt-4o-mini (3540 newcomers)

Mean P: full 0.080, off 0.072, off+tmpl 0.393, rules 0.061, welcome 0.123 | human 90-day return 0.152

AUROC of P(arm) for the real 90-day return: 0.546

**Arm effect, same regression as the human outcome:** agent Δlogit -0.010 [-0.312, +0.291] vs human -0.029 [-0.230, +0.172] (same 3540 newcomers).

Agent − human gap (paired post-cluster bootstrap): +0.019 [-0.327, +0.375]

Within-agent Δlogit vs no sticky (all agents):

| condition   | Δlogit               |   n agents |
|:------------|:---------------------|-----------:|
| full        | +0.47 [+0.32, +0.60] |       3540 |
| off+tmpl    | +5.63 [+5.37, +5.92] |       1000 |
| rules       | +0.05 [-0.15, +0.24] |       1000 |
| welcome     | +1.54 [+1.41, +1.69] |       1000 |

## Agents: rscience-jev (3540 newcomers)

Mean P: full 0.284, off 0.282, off+tmpl 0.299, rules 0.286, welcome 0.271 | human 90-day return 0.152

AUROC of P(arm) for the real 90-day return: 0.596

**Arm effect, same regression as the human outcome:** agent Δlogit +0.006 [-0.040, +0.051] vs human -0.029 [-0.230, +0.172] (same 3540 newcomers).

Agent − human gap (paired post-cluster bootstrap): +0.035 [-0.160, +0.231]

Within-agent Δlogit vs no sticky (all agents):

| condition   | Δlogit               |   n agents |
|:------------|:---------------------|-----------:|
| full        | +0.01 [+0.00, +0.02] |       3540 |
| off+tmpl    | +0.09 [+0.08, +0.10] |       1000 |
| rules       | +0.03 [+0.02, +0.04] |       1000 |
| welcome     | -0.06 [-0.06, -0.05] |       1000 |

