# Formal H8 / H9 (PREREG_sites.md; computed 2026-11-11 after freezes 'scores ffxiv2019' and 'scores science2020')

Predicted set (same sign at r/science 2016 and Futurology, |Δ| > 0.25, CI excludes 0): gemma3-12b, gemma3-27b, gemma3-4b, llama3.1-8b, qwen3-14b, qwen3-32b, qwen3-4b, qwen3-8b (8 models).

| model       | site        | predicted   | observed             | hit   |
|:------------|:------------|:------------|:---------------------|:------|
| gemma3-12b  | ffxiv2019   | −           | -0.03 [-0.11, +0.05] | False |
| gemma3-12b  | science2020 | −           | -0.71 [-0.76, -0.67] | True  |
| gemma3-27b  | ffxiv2019   | −           | -2.53 [-2.76, -2.30] | True  |
| gemma3-27b  | science2020 | −           | -0.29 [-0.35, -0.23] | True  |
| gemma3-4b   | ffxiv2019   | −           | -0.93 [-1.18, -0.70] | True  |
| gemma3-4b   | science2020 | −           | -1.85 [-2.04, -1.64] | True  |
| llama3.1-8b | ffxiv2019   | +           | +0.31 [+0.30, +0.33] | True  |
| llama3.1-8b | science2020 | +           | +0.42 [+0.40, +0.43] | True  |
| qwen3-14b   | ffxiv2019   | +           | +0.06 [-0.02, +0.14] | False |
| qwen3-14b   | science2020 | +           | +0.38 [+0.32, +0.44] | True  |
| qwen3-32b   | ffxiv2019   | +           | +1.69 [+1.59, +1.78] | True  |
| qwen3-32b   | science2020 | +           | +0.59 [+0.51, +0.67] | True  |
| qwen3-4b    | ffxiv2019   | −           | -0.97 [-1.27, -0.70] | True  |
| qwen3-4b    | science2020 | −           | -0.75 [-0.95, -0.57] | True  |
| qwen3-8b    | ffxiv2019   | −           | -1.18 [-1.28, -1.08] | True  |
| qwen3-8b    | science2020 | −           | -0.82 [-0.90, -0.75] | True  |

**H8: 14/16 = 87.5% of predictions hold (threshold 80%) -> SUPPORTED.**

**H9 ffxiv2019:** range of full-sticky Δlogit over 14 panel models = 4.22 (qwen3-32b +1.69 to gemma3-27b -2.53); threshold 1.0 -> SUPPORTED.
**H9 science2020:** range of full-sticky Δlogit over 13 panel models = 2.44 (qwen3-32b +0.59 to gemma3-4b -1.85); threshold 1.0 -> SUPPORTED.

Deviations: Qwen2.5-72B dropped (online fp8 readout mass 0; AWQ diagnostic OOM at load); gpt-4o-mini not scored on science2020 (OpenRouter cap). Neither is in the H8 predicted set.
