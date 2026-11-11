# Panel median vs the human arm effect (rscience; open-weight models only)

- models = 7, agents = 3540; human arm effect -0.03; panel median -0.53 [-0.65, -0.03]; gap -0.50 [-0.67, +0.01] (90% CI [-0.62, -0.04]); equivalent within ±0.25: False; range of single-model arm effects 2.17

H10 (|gap| CI90 within ±0.25): not supported
H10b (panel median |gap| < median single-model |gap|): True

Single-model agent arm effects: qwen3-1.7b -1.82, gemma3-4b -0.55, gemma3-12b -0.54, qwen3-8b -0.53, ministral3-8b -0.08, qwen3-32b +0.12, llama3.1-8b +0.36

## Leave one model out

- without gemma3-12b: human arm effect -0.03; panel median -0.31 [-0.40, -0.15]; gap -0.28 [-0.45, -0.04] (90% CI [-0.43, -0.07]); equivalent within ±0.25: False; range of single-model arm effects 2.17
- without gemma3-4b: human arm effect -0.03; panel median -0.31 [-0.40, +0.02]; gap -0.28 [-0.46, +0.07] (90% CI [-0.42, +0.02]); equivalent within ±0.25: False; range of single-model arm effects 2.17
- without llama3.1-8b: human arm effect -0.03; panel median -0.53 [-0.68, -0.18]; gap -0.50 [-0.70, -0.15] (90% CI [-0.65, -0.20]); equivalent within ±0.25: False; range of single-model arm effects 1.94
- without ministral3-8b: human arm effect -0.03; panel median -0.53 [-0.68, -0.14]; gap -0.50 [-0.70, -0.13] (90% CI [-0.65, -0.18]); equivalent within ±0.25: False; range of single-model arm effects 2.17
- without qwen3-1.7b: human arm effect -0.03; panel median -0.31 [-0.39, +0.02]; gap -0.28 [-0.43, +0.07] (90% CI [-0.40, +0.02]); equivalent within ±0.25: False; range of single-model arm effects 0.90
- without qwen3-32b: human arm effect -0.03; panel median -0.53 [-0.68, -0.18]; gap -0.50 [-0.70, -0.15] (90% CI [-0.65, -0.20]); equivalent within ±0.25: False; range of single-model arm effects 2.17
- without qwen3-8b: human arm effect -0.03; panel median -0.31 [-0.46, +0.02]; gap -0.28 [-0.46, +0.06] (90% CI [-0.43, +0.01]); equivalent within ±0.25: False; range of single-model arm effects 2.17
