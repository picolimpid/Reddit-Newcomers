# Observational SFT: sft-qwen3-8b

Held-out agents (val + test communities, never seen in training): 1225. Human 90-day return on them: 20.6%.

## Calibration, discrimination and fingerprints (observed replies)

| who   |   mean P / rate |   AUROC 90d |   FP3 fit OR | FP1 Jev windows    |   FP2 reply OR |
|:------|----------------:|------------:|-------------:|:-------------------|---------------:|
| human |           0.206 |     nan     |        1.294 | 1.26 / 0.96 / 0.91 |          0.987 |
| base  |           0.048 |       0.522 |        1.289 | 1.25 / 0.73 / 1.41 |          2.625 |
| sft   |           0.172 |       0.554 |        1.055 | 1.07 / 0.98 / 1.06 |          1.19  |

Reference: c2v fit alone AUROC 0.561. Human fingerprints on the held-out half are noisy (n ≈ 987).

## Reply interventions (Δlogit vs no reply, within agent)

|                                     | base                 | sft                  |
|:------------------------------------|:---------------------|:---------------------|
| P none                              | 0.037135145338393565 | 0.1658912441716      |
| ack                                 | +1.51 [+1.42, +1.61] | +0.15 [+0.13, +0.16] |
| disagree                            | +1.02 [+0.93, +1.11] | +0.14 [+0.12, +0.16] |
| praise                              | +2.72 [+2.59, +2.84] | +0.26 [+0.24, +0.28] |
| praise_welcome                      | +4.00 [+3.83, +4.17] | +0.55 [+0.52, +0.58] |
| praise_invite                       | +3.97 [+3.81, +4.14] | +0.50 [+0.47, +0.53] |
| praise_welcome_invite               | +5.37 [+5.17, +5.56] | +0.74 [+0.70, +0.77] |
| template                            | +4.53 [+4.35, +4.70] | +0.64 [+0.61, +0.67] |
| real reply (observed − none, n=288) | +1.46 [+1.26, +1.66] | +0.18 [+0.15, +0.21] |
| NC r/Y: template                    | +0.61 [+0.48, +0.74] | +0.15 [+0.14, +0.17] |

Human reference (observational, full sample): reply received OR 1.07 → Δlogit +0.07; TOST rules out |Δlogit| > 0.41 at 90 days.


## SFT vs Platt scaling of the base model

Platt fit on 1188 train-split agents: logit P' = -1.22 + 0.032 · logit P_base. Affine fit of logit P_sft on logit P_base over all 14700 held-out prompts: slope 0.122, R² 0.683.

| model   |   mean P |   AUROC 90d |   Brier |   FP3 fit OR (human 1.29) |   FP2 reply OR (human 0.99) |
|:--------|---------:|------------:|--------:|--------------------------:|----------------------------:|
| base    |    0.048 |       0.522 |   0.201 |                     1.289 |                       2.625 |
| platt   |    0.201 |       0.522 |   0.163 |                     1.012 |                       1.043 |
| sft     |    0.172 |       0.554 |   0.165 |                     1.055 |                       1.19  |

Human base-rate Brier (predict 0.206 for everyone): 0.163.

Within-agent Δlogit vs no reply (Platt column = b × base):

| condition             | base                 | platt                | sft                  |   sft / base |
|:----------------------|:---------------------|:---------------------|:---------------------|-------------:|
| ack                   | +1.51 [+1.42, +1.61] | +0.05 [+0.05, +0.05] | +0.15 [+0.13, +0.16] |        0.097 |
| disagree              | +1.02 [+0.93, +1.11] | +0.03 [+0.03, +0.04] | +0.14 [+0.12, +0.16] |        0.137 |
| praise                | +2.72 [+2.59, +2.84] | +0.09 [+0.08, +0.09] | +0.26 [+0.24, +0.28] |        0.095 |
| praise_welcome        | +4.00 [+3.83, +4.17] | +0.13 [+0.12, +0.13] | +0.55 [+0.52, +0.58] |        0.138 |
| praise_invite         | +3.97 [+3.81, +4.14] | +0.13 [+0.12, +0.13] | +0.50 [+0.47, +0.53] |        0.125 |
| praise_welcome_invite | +5.37 [+5.17, +5.56] | +0.17 [+0.16, +0.18] | +0.74 [+0.70, +0.77] |        0.137 |
| template              | +4.53 [+4.35, +4.70] | +0.14 [+0.14, +0.15] | +0.64 [+0.61, +0.67] |        0.141 |
| real reply            | +1.46 [+1.26, +1.66] | +0.05 [+0.04, +0.05] | +0.18 [+0.15, +0.21] |        0.124 |
| NC r/Y: template      | +0.61 [+0.48, +0.74] | +0.02 [+0.02, +0.02] | +0.15 [+0.14, +0.17] |        0.253 |

Each effect as a share of the template effect:

|      |   ack |   disagree |   praise |   praise_welcome |   praise_invite |   praise_welcome_invite |   template |   real reply |   NC r/Y: template |
|:-----|------:|-----------:|---------:|-----------------:|----------------:|------------------------:|-----------:|-------------:|-------------------:|
| base |  0.33 |       0.22 |      0.6 |             0.88 |            0.88 |                    1.19 |          1 |         0.32 |               0.13 |
| sft  |  0.23 |       0.22 |      0.4 |             0.87 |            0.78 |                    1.15 |          1 |         0.28 |               0.24 |

