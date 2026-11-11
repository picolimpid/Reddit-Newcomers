# Probe experiments: qwen3-8b

Agents: 2413. Intent protocol, first-token P(Yes). Effects are within-agent mean differences vs the no-reply condition, post-cluster bootstrap 95% CI. Fit-SD units use the agent's own 1-SD c2v fit effect from `qwen3-8b` (beta = 0.29).

## 1. Reply ladder: which part of the welcome moves the agent?

| condition             |   mean P |   median P | Δlogit vs none       | ΔP vs none              | fit-SD units   |
|:----------------------|---------:|-----------:|:---------------------|:------------------------|:---------------|
| none                  |   0.0365 |     0.0052 | –                    | –                       | –              |
| ack                   |   0.0813 |     0.018  | +1.43 [+1.37, +1.51] | +0.045 [+0.040, +0.050] | 5.0            |
| disagree              |   0.0545 |     0.0141 | +0.97 [+0.90, +1.04] | +0.018 [+0.015, +0.021] | 3.4            |
| praise                |   0.172  |     0.0601 | +2.64 [+2.55, +2.74] | +0.135 [+0.125, +0.147] | 9.2            |
| praise_invite         |   0.3107 |     0.1824 | +3.88 [+3.76, +4.02] | +0.274 [+0.258, +0.292] | 13.5           |
| praise_welcome        |   0.3077 |     0.2227 | +3.99 [+3.87, +4.11] | +0.271 [+0.256, +0.288] | 13.8           |
| praise_welcome_invite |   0.5065 |     0.5    | +5.37 [+5.24, +5.52] | +0.470 [+0.451, +0.491] | 18.6           |
| template              |   0.382  |     0.3208 | +4.50 [+4.38, +4.64] | +0.345 [+0.328, +0.364] | 15.6           |

Decomposition (Δlogit):

| component                                      | Δlogit               |
|:-----------------------------------------------|:---------------------|
| welcome cue (praise_welcome − praise)          | +1.34 [+1.24, +1.45] |
| invitation (praise_invite − praise)            | +1.24 [+1.16, +1.33] |
| both (praise_welcome_invite − praise)          | +2.73 [+2.61, +2.85] |
| interaction (both − welcome − invite + praise) | +0.14 [+0.06, +0.23] |
| valence (praise − disagree)                    | +1.67 [+1.61, +1.73] |
| reply presence (ack − none)                    | +1.43 [+1.37, +1.51] |

Reproducibility vs `qwen3-8b` (same prompts; differs only if weights/numerics differ):

| condition   |   corr(logit) |   mean P probe |   mean P qwen3-8b |   max |ΔP| |
|:------------|--------------:|---------------:|------------------:|-----------:|
| none        |             1 |         0.0365 |            0.0365 |      0.005 |
| template    |             1 |         0.382  |            0.382  |      0     |

## 2. Negative-control outcome (r/Y: a random target community the user never visited)

Human base rate of any activity in r/Y within 90 days: 5.926% (r/X: 20.3%). Human r/Y rate with vs without a reply in r/X: 6.783% vs 5.693%.

| condition             |   mean P(r/Y) | Δlogit r/Y vs none   | Δlogit r/X vs none (ladder)   | r/Y effect as share of r/X effect   |
|:----------------------|--------------:|:---------------------|:------------------------------|:------------------------------------|
| ack                   |        0.0295 | -0.01 [-0.06, +0.05] | +1.43 [+1.37, +1.51]          | -0.01                               |
| none                  |        0.0287 | –                    | –                             | –                                   |
| praise_welcome_invite |        0.0425 | +0.64 [+0.56, +0.72] | +5.37 [+5.24, +5.52]          | 0.12                                |
| template              |        0.0362 | +0.53 [+0.45, +0.61] | +4.50 [+4.38, +4.64]          | 0.12                                |

## 3. Outsider line removed ("You had not posted or commented in r/X ... past year.")

| condition   |   mean P with line |   mean P without line | Δlogit (without − with)   |
|:------------|-------------------:|----------------------:|:--------------------------|
| none        |             0.0365 |                0.1432 | +2.38 [+2.25, +2.52]      |
| template    |             0.382  |                0.6539 | +1.98 [+1.86, +2.09]      |

Template welcome effect with the line +4.50 [+4.38, +4.64], without +4.11 [+3.95, +4.27].

## 4. Readout: first-token P vs verbalized probability vs thinking-mode answer

400 agents (random subsample). Human 90-day return on this subsample: 20.0%. Thinking: 4 samples per prompt, T = 0.6, median 355 generated tokens.

| readout     |   mean (none) |   mean (ack) |   mean (template) | ΔP ack − none           | ΔP template − none      |   unparsed |   AUROC vs human 90d (none) |   Spearman with first_token (none) |
|:------------|--------------:|-------------:|------------------:|:------------------------|:------------------------|-----------:|----------------------------:|-----------------------------------:|
| first_token |         0.034 |        0.083 |             0.395 | +0.049 [+0.040, +0.060] | +0.361 [+0.329, +0.395] |          0 |                       0.561 |                              1     |
| verbalized  |         0.605 |        0.718 |             0.835 | +0.114 [+0.087, +0.142] | +0.230 [+0.193, +0.266] |          0 |                       0.479 |                              0.517 |
| thinking    |         0.013 |        0.034 |             0.149 | +0.021 [+0.010, +0.033] | +0.136 [+0.113, +0.159] |          0 |                       0.502 |                              0.222 |

