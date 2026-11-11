# P7: compliance by generation (r/science 2016)

Human (CivilServant, 805 non-AMA posts with newcomer comments): removed share of newcomer comments 0.553 (control) → 0.492 (sticky); Δ -6.2 pp [-17.5, +3.7], Δ logit -0.25 [-0.71, +0.15] (pooled, no FE).
Primary, block FE binomial GLM: Δ logit(removed) -0.44 [-0.80, -0.07], p = 0.018. An agent matches the human direction if its within-agent Δ logit(P remove) for the full sticky is < 0 with a CI excluding 0.

Agents: within-agent change in judged P(remove) vs no sticky (mean over 2 samples per agent; post-cluster bootstrap).

| generator     | judge      | condition   |   P(remove) off |   P(remove) cond | Δ pp                | Δ logit              | tokens off → cond   | match          |
|:--------------|:-----------|:------------|----------------:|-----------------:|:--------------------|:---------------------|:--------------------|:---------------|
| gemma3-12b    | gemma3-27b | full        |           0.484 |            0.509 | +2.5 [-0.2, +5.0]   | +0.10 [-0.01, +0.20] | 31 → 29             | CI includes 0  |
| gemma3-12b    | gemma3-27b | rules       |           0.484 |            0.496 | +1.2 [-1.1, +3.6]   | +0.05 [-0.04, +0.15] | 31 → 30             | CI includes 0  |
| gemma3-12b    | qwen3-32b  | full        |           0.211 |            0.215 | +0.4 [-1.3, +2.1]   | +0.02 [-0.07, +0.13] | 31 → 29             | CI includes 0  |
| gemma3-12b    | qwen3-32b  | rules       |           0.211 |            0.209 | -0.2 [-1.7, +1.3]   | -0.01 [-0.11, +0.08] | 31 → 30             | CI includes 0  |
| llama3.1-8b   | gemma3-27b | full        |           0.501 |            0.506 | +0.5 [-2.3, +3.0]   | +0.02 [-0.09, +0.12] | 62 → 61             | CI includes 0  |
| llama3.1-8b   | gemma3-27b | rules       |           0.501 |            0.505 | +0.4 [-1.9, +2.6]   | +0.02 [-0.08, +0.10] | 62 → 61             | CI includes 0  |
| llama3.1-8b   | qwen3-32b  | full        |           0.301 |            0.315 | +1.4 [-0.3, +3.0]   | +0.07 [-0.01, +0.14] | 62 → 61             | CI includes 0  |
| llama3.1-8b   | qwen3-32b  | rules       |           0.301 |            0.309 | +0.8 [-1.1, +2.6]   | +0.04 [-0.05, +0.13] | 62 → 61             | CI includes 0  |
| ministral3-8b | gemma3-27b | full        |           0.751 |            0.73  | -2.1 [-4.0, -0.1]   | -0.11 [-0.21, -0.01] | 126 → 128           | same direction |
| ministral3-8b | gemma3-27b | rules       |           0.751 |            0.786 | +3.5 [+1.4, +5.7]   | +0.20 [+0.08, +0.32] | 126 → 124           | opposite       |
| ministral3-8b | qwen3-32b  | full        |           0.488 |            0.491 | +0.3 [-1.7, +2.2]   | +0.01 [-0.07, +0.09] | 126 → 128           | CI includes 0  |
| ministral3-8b | qwen3-32b  | rules       |           0.488 |            0.568 | +8.0 [+5.6, +10.2]  | +0.32 [+0.23, +0.41] | 126 → 124           | opposite       |
| qwen3-32b     | gemma3-27b | full        |           0.421 |            0.349 | -7.2 [-9.8, -4.5]   | -0.31 [-0.42, -0.19] | 51 → 49             | same direction |
| qwen3-32b     | gemma3-27b | rules       |           0.421 |            0.298 | -12.3 [-15.1, -9.5] | -0.54 [-0.66, -0.42] | 51 → 51             | same direction |
| qwen3-32b     | qwen3-32b  | full        |           0.232 |            0.191 | -4.0 [-6.1, -2.1]   | -0.24 [-0.36, -0.13] | 51 → 49             | same direction |
| qwen3-32b     | qwen3-32b  | rules       |           0.232 |            0.167 | -6.5 [-8.5, -4.6]   | -0.41 [-0.52, -0.30] | 51 → 51             | same direction |
| qwen3-8b      | gemma3-27b | full        |           0.288 |            0.189 | -9.9 [-12.5, -7.0]  | -0.55 [-0.70, -0.40] | 63 → 56             | same direction |
| qwen3-8b      | gemma3-27b | rules       |           0.288 |            0.264 | -2.4 [-4.6, -0.1]   | -0.12 [-0.23, -0.00] | 63 → 57             | same direction |
| qwen3-8b      | qwen3-32b  | full        |           0.13  |            0.087 | -4.3 [-5.8, -2.7]   | -0.45 [-0.60, -0.30] | 63 → 56             | same direction |
| qwen3-8b      | qwen3-32b  | rules       |           0.13  |            0.127 | -0.3 [-1.6, +1.0]   | -0.03 [-0.14, +0.09] | 63 → 57             | CI includes 0  |

Judge gemma3-27b on the agents' real first comments (all survived moderation): mean P(remove) 0.422, share > 0.5 0.421.
Judge qwen3-32b on the agents' real first comments (all survived moderation): mean P(remove) 0.241, share > 0.5 0.229.
