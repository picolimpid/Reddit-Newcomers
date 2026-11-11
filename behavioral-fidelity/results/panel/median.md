# Panel median vs the human arm effect (r/science 2016; exploratory)

- models = 9, agents = 3540; human arm effect -0.03; panel median -0.08 [-0.31, +0.02]; gap -0.06 [-0.33, +0.15] (90% CI [-0.27, +0.11]); equivalent within ±0.25: False; range of single-model arm effects 2.17

Single-model agent arm effects: qwen3-1.7b -1.82, gemma3-4b -0.55, gemma3-12b -0.54, qwen3-8b -0.53, ministral3-8b -0.08, gpt-4o-mini -0.01, jev +0.01, qwen3-32b +0.12, llama3.1-8b +0.36

## Leave one model out

- without gemma3-12b: human arm effect -0.03; panel median -0.05 [-0.22, +0.01]; gap -0.02 [-0.29, +0.15] (90% CI [-0.25, +0.12]); equivalent within ±0.25: True; range of single-model arm effects 2.17
- without gemma3-4b: human arm effect -0.03; panel median -0.05 [-0.22, +0.04]; gap -0.02 [-0.29, +0.15] (90% CI [-0.25, +0.13]); equivalent within ±0.25: True; range of single-model arm effects 2.17
- without gpt-4o-mini: human arm effect -0.03; panel median -0.31 [-0.39, -0.00]; gap -0.28 [-0.43, +0.04] (90% CI [-0.40, -0.01]); equivalent within ±0.25: False; range of single-model arm effects 2.17
- without jev: human arm effect -0.03; panel median -0.31 [-0.45, +0.00]; gap -0.28 [-0.49, +0.06] (90% CI [-0.45, +0.01]); equivalent within ±0.25: False; range of single-model arm effects 2.17
- without llama3.1-8b: human arm effect -0.03; panel median -0.31 [-0.45, -0.01]; gap -0.28 [-0.49, +0.04] (90% CI [-0.45, -0.01]); equivalent within ±0.25: False; range of single-model arm effects 1.94
- without ministral3-8b: human arm effect -0.03; panel median -0.27 [-0.45, +0.04]; gap -0.24 [-0.49, +0.08] (90% CI [-0.45, +0.03]); equivalent within ±0.25: False; range of single-model arm effects 2.17
- without qwen3-1.7b: human arm effect -0.03; panel median -0.05 [-0.22, +0.04]; gap -0.02 [-0.29, +0.15] (90% CI [-0.25, +0.13]); equivalent within ±0.25: True; range of single-model arm effects 0.90
- without qwen3-32b: human arm effect -0.03; panel median -0.31 [-0.45, -0.01]; gap -0.28 [-0.49, +0.04] (90% CI [-0.45, -0.01]); equivalent within ±0.25: False; range of single-model arm effects 2.17
- without qwen3-8b: human arm effect -0.03; panel median -0.05 [-0.22, +0.04]; gap -0.02 [-0.29, +0.15] (90% CI [-0.25, +0.13]); equivalent within ±0.25: True; range of single-model arm effects 2.17
