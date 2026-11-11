# Probe experiments: qwen3-32b-fp8

Agents: 2413. Intent protocol, first-token P(Yes). Effects are within-agent mean differences vs the no-reply condition, post-cluster bootstrap 95% CI. Fit-SD units use the agent's own 1-SD c2v fit effect from `qwen3-32b` (beta = 0.35).

## 1. Reply ladder: which part of the welcome moves the agent?

| condition             |   mean P |   median P | Δlogit vs none       | ΔP vs none              | fit-SD units   |
|:----------------------|---------:|-----------:|:---------------------|:------------------------|:---------------|
| none                  |   0.047  |     0.0052 | –                    | –                       | –              |
| ack                   |   0.0896 |     0.018  | +1.05 [+0.98, +1.12] | +0.043 [+0.038, +0.048] | 3.0            |
| disagree              |   0.1157 |     0.0293 | +1.65 [+1.58, +1.73] | +0.069 [+0.062, +0.076] | 4.7            |
| praise                |   0.1132 |     0.0293 | +1.44 [+1.36, +1.52] | +0.066 [+0.060, +0.073] | 4.1            |
| praise_invite         |   0.1643 |     0.0474 | +2.18 [+2.09, +2.27] | +0.117 [+0.107, +0.128] | 6.2            |
| praise_welcome        |   0.2928 |     0.1824 | +3.44 [+3.29, +3.59] | +0.246 [+0.226, +0.267] | 9.8            |
| praise_welcome_invite |   0.3599 |     0.2689 | +4.16 [+4.02, +4.31] | +0.313 [+0.291, +0.337] | 11.9           |
| template              |   0.3602 |     0.2689 | +4.19 [+4.05, +4.33] | +0.313 [+0.291, +0.337] | 12.0           |

Decomposition (Δlogit):

| component                                      | Δlogit               |
|:-----------------------------------------------|:---------------------|
| welcome cue (praise_welcome − praise)          | +1.99 [+1.88, +2.11] |
| invitation (praise_invite − praise)            | +0.74 [+0.67, +0.80] |
| both (praise_welcome_invite − praise)          | +2.72 [+2.60, +2.84] |
| interaction (both − welcome − invite + praise) | -0.01 [-0.08, +0.06] |
| valence (praise − disagree)                    | -0.21 [-0.25, -0.17] |
| reply presence (ack − none)                    | +1.05 [+0.98, +1.12] |

Reproducibility vs `qwen3-32b` (same prompts; differs only if weights/numerics differ):

| condition   |   corr(logit) |   mean P probe |   mean P qwen3-32b |   max |ΔP| |
|:------------|--------------:|---------------:|-------------------:|-----------:|
| none        |        0.9895 |         0.047  |             0.0424 |     0.1792 |
| template    |        0.99   |         0.3602 |             0.3331 |     0.3395 |

## 2. Negative-control outcome (r/Y: a random target community the user never visited)

Human base rate of any activity in r/Y within 90 days: 5.926% (r/X: 20.3%). Human r/Y rate with vs without a reply in r/X: 6.783% vs 5.693%.

| condition             |   mean P(r/Y) | Δlogit r/Y vs none   | Δlogit r/X vs none (ladder)   | r/Y effect as share of r/X effect   |
|:----------------------|--------------:|:---------------------|:------------------------------|:------------------------------------|
| ack                   |        0.1166 | +0.48 [+0.44, +0.52] | +1.05 [+0.98, +1.12]          | 0.46                                |
| none                  |        0.0854 | –                    | –                             | –                                   |
| praise_welcome_invite |        0.1133 | +0.31 [+0.26, +0.36] | +4.16 [+4.02, +4.31]          | 0.07                                |
| template              |        0.1151 | +0.37 [+0.32, +0.43] | +4.19 [+4.05, +4.33]          | 0.09                                |

## 3. Outsider line removed ("You had not posted or commented in r/X ... past year.")

| condition   |   mean P with line |   mean P without line | Δlogit (without − with)   |
|:------------|-------------------:|----------------------:|:--------------------------|
| none        |             0.047  |                0.2731 | +3.37 [+3.23, +3.51]      |
| template    |             0.3602 |                0.5946 | +1.73 [+1.62, +1.84]      |

Template welcome effect with the line +4.19 [+4.05, +4.33], without +2.55 [+2.40, +2.72].

