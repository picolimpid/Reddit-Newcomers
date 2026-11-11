# P4: sentence ablation (within agent, Δlogit vs no sticky; W welcome, T removal threat, M moderators, P purpose clause, N descriptive norm, A auto-removed replies)

## futurology

| model    | all     | noA     | noN     | noP     | noT     | noW     | onlyA   | onlyN   | onlyT   | onlyW   |
|:---------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|
| qwen3-8b | − -0.54 | − -0.50 | − -0.72 | − -0.58 | − -0.61 | − -0.98 | − -0.91 | − -0.75 | − -0.60 | + +0.98 |

## rscience

| model       | all     | noM     | noT     | noW     | onlyM   | onlyT   | onlyW   |
|:------------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|
| llama3.1-8b | + +0.40 | + +0.39 | + +0.30 | + +0.38 | + +0.33 | + +0.38 | 0 +0.19 |
| qwen3-8b    | − -1.26 | − -1.03 | 0 -0.12 | − -0.75 | 0 -0.06 | − -0.69 | + +0.54 |

## Model-level probes (no persona): does the model know the study, and what does it predict as an expert?

| site       | model       |   knows: Increased |   knows: Decreased |   knows: Unchanged |   expert: Higher |   expert: Lower |   expert: Same |
|:-----------|:------------|-------------------:|-------------------:|-------------------:|-----------------:|----------------:|---------------:|
| futurology | llama3.1-8b |               0.66 |               0.31 |               0.03 |             0.1  |            0.87 |           0.02 |
| futurology | qwen3-8b    |               1    |               0    |               0    |             0    |            1    |           0    |
| rscience   | llama3.1-8b |               0.75 |               0.22 |               0.03 |             0.6  |            0.37 |           0.03 |
| rscience   | qwen3-8b    |               1    |               0    |               0    |             0.08 |            0.92 |           0    |

