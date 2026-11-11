# P5: positive feedback (highly upvoted first comment) and newcomer return, r/science 2016

Treated: first-comment score ≥ 4 (top quartile); control: ≤ 1 (median); n treated 975, control 1795.

Weighted standardized differences after ATT weighting: delay +0.019, len -0.001, top -0.024, hist -0.010, sticky +0.007, thread +0.010, hour -0.023, nsubs -0.000

Unweighted 90-day return: treated 0.151, control 0.156

## Same design, same people: human return vs agent P(return) with the real score shown

| who           | ATT Δlogit [95% CI]   | agent − human [95% CI]   |
|:--------------|:----------------------|:-------------------------|
| human         | -0.14 [-0.40, +0.15]  |                          |
| qwen3-8b      | +2.26 [+2.05, +2.46]  | +2.39 [+2.00, +2.75]     |
| qwen3-32b     | +1.42 [+1.14, +1.70]  | +1.56 [+1.19, +1.91]     |
| llama3.1-8b   | +0.04 [-0.01, +0.08]  | +0.18 [-0.12, +0.45]     |
| ministral3-8b | +0.57 [+0.51, +0.62]  | +0.71 [+0.42, +0.97]     |
| gemma3-12b    | +1.41 [+0.80, +2.26]  | +1.55 [+0.85, +2.45]     |
| qwen3-14b     | +3.09 [+2.48, +3.87]  | +3.23 [+2.57, +4.03]     |
| phi-4         | +1.76 [+1.52, +1.99]  | +1.90 [+1.52, +2.26]     |
| gemma3-4b     | +0.33 [+0.10, +0.57]  | +0.47 [+0.09, +0.84]     |
| qwen3-4b      | +2.56 [+2.32, +2.82]  | +2.70 [+2.36, +3.04]     |

## The agent's own causal response to feedback (within agent)

| model         | within: 50 pts vs 1 pt   |   mean P real |
|:--------------|:-------------------------|--------------:|
| qwen3-8b      | +2.13 [+2.03, +2.24]     |         0.117 |
| qwen3-32b     | +1.10 [+0.99, +1.21]     |         0.031 |
| llama3.1-8b   | +0.14 [+0.13, +0.15]     |         0.291 |
| ministral3-8b | +0.34 [+0.33, +0.35]     |         0.041 |
| gemma3-12b    | +0.88 [+0.84, +0.91]     |         0.013 |
| qwen3-14b     | +4.63 [+4.50, +4.77]     |         0.031 |
| phi-4         | +1.91 [+1.84, +1.98]     |         0.075 |
| gemma3-4b     | +1.10 [+1.04, +1.18]     |         0.227 |
| qwen3-4b      | +9.01 [+8.76, +9.27]     |         0.32  |

Human anchor from the literature: Lambert et al. CHI 2025, highly upvoted posts -> +2.24% daily post rate (largest for newer accounts); a different outcome, so only an order-of-magnitude comparison.

