# P9: entry (would you comment on this post?) vs the real newcomer-participation effect

Human: IRR 1.67 [1.53, 1.82] newcomer comments per non-AMA post (negative binomial, n = 2214 posts). Agent ratio = mean P(Yes | sticky) / mean P(Yes | off), bootstrap over shown posts; gap on the log scale includes the human uncertainty.

| model         |   mean P off |   ratio full/off | ratio CI     |   ratio welcome/off |   ratio rules/off |   log gap vs human | gap CI         | matches   | within Δlogit        |   n cells |
|:--------------|-------------:|-----------------:|:-------------|--------------------:|------------------:|-------------------:|:---------------|:----------|:---------------------|----------:|
| gemma3-12b    |        0.684 |            0.813 | [0.79, 0.83] |               1.083 |             0.917 |             -0.718 | [-0.81, -0.63] | False     | -1.67 [-1.84, -1.51] |      2000 |
| gemma3-4b     |        0.927 |            0.593 | [0.57, 0.61] |               0.659 |             0.724 |             -1.033 | [-1.13, -0.94] | False     | -8.43 [-8.68, -8.18] |      2000 |
| llama3.1-8b   |        0.357 |            1.32  | [1.30, 1.34] |               1.059 |             1.364 |             -0.233 | [-0.32, -0.15] | False     | +0.55 [+0.52, +0.57] |      2000 |
| ministral3-8b |        0.141 |            1.217 | [1.19, 1.24] |               0.936 |             1.495 |             -0.314 | [-0.41, -0.23] | False     | +0.30 [+0.28, +0.33] |      2000 |
| phi-4         |        0.321 |            0.455 | [0.44, 0.47] |               0.535 |             0.449 |             -1.299 | [-1.39, -1.21] | False     | -1.31 [-1.36, -1.27] |      2000 |
| qwen3-1.7b    |        0.985 |            0.92  | [0.91, 0.93] |               1.015 |             0.829 |             -0.594 | [-0.68, -0.50] | False     | -4.41 [-4.60, -4.21] |      2000 |
| qwen3-14b     |        0.375 |            1.091 | [1.04, 1.14] |               1.305 |             1.486 |             -0.424 | [-0.53, -0.32] | False     | +1.22 [+0.94, +1.49] |      2000 |
| qwen3-32b     |        0.135 |            1.282 | [1.22, 1.35] |               0.697 |             1.2   |             -0.262 | [-0.37, -0.16] | False     | +0.97 [+0.89, +1.06] |      2000 |
| qwen3-4b      |        0.384 |            0.459 | [0.41, 0.51] |               0.261 |             0.519 |             -1.29  | [-1.42, -1.16] | False     | -5.85 [-6.36, -5.32] |      2000 |
| qwen3-8b      |        0.426 |            0.34  | [0.32, 0.36] |               1.296 |             0.199 |             -1.588 | [-1.70, -1.48] | False     | -2.94 [-3.06, -2.82] |      2000 |

## Hypotheses (PREREG_addendum_P9.md)

- H11 (calibration fails for >= 2/3): 10/10 gap CIs exclude 0 -> supported.
- H12 (entry log ratio vs r/science return Δlogit, Spearman > 0): rho = +0.67, one-sided permutation p = 0.0207 (10,000 permutations, n = 10 models).
- H13 (misses under-predict): 10/10 -> supported.
