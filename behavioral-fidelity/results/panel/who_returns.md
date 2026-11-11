# Who returns? r/science 2016 newcomers, 90-day return (exploratory; E12)

n = 3565, base rate 0.152. 5-fold CV grouped by post, averaged over 5 repeats; AUC CI by post bootstrap.

| predictor                           |   AUC |    lo |    hi |   Δ vs counts |   Δ vs counts + text |
|:------------------------------------|------:|------:|------:|--------------:|---------------------:|
| counts                              | 0.662 | 0.64  | 0.685 |       nan     |              nan     |
| counts + text                       | 0.666 | 0.642 | 0.688 |       nan     |              nan     |
| agent gemma3-12b alone              | 0.563 | 0.537 | 0.587 |       nan     |              nan     |
| counts + agent gemma3-12b           | 0.668 | 0.646 | 0.689 |         0.007 |              nan     |
| counts + text + agent gemma3-12b    | 0.668 | 0.645 | 0.69  |       nan     |                0.001 |
| agent gemma3-4b alone               | 0.529 | 0.499 | 0.558 |       nan     |              nan     |
| counts + agent gemma3-4b            | 0.664 | 0.642 | 0.683 |         0.003 |              nan     |
| counts + text + agent gemma3-4b     | 0.669 | 0.646 | 0.69  |       nan     |                0.002 |
| agent gpt-4o-mini alone             | 0.547 | 0.518 | 0.575 |       nan     |              nan     |
| counts + agent gpt-4o-mini          | 0.668 | 0.647 | 0.688 |         0.007 |              nan     |
| counts + text + agent gpt-4o-mini   | 0.668 | 0.645 | 0.69  |       nan     |                0.001 |
| agent jev alone                     | 0.602 | 0.577 | 0.625 |       nan     |              nan     |
| counts + agent jev                  | 0.683 | 0.663 | 0.702 |         0.022 |              nan     |
| counts + text + agent jev           | 0.672 | 0.651 | 0.694 |       nan     |                0.005 |
| agent llama3.1-8b alone             | 0.557 | 0.531 | 0.583 |       nan     |              nan     |
| counts + agent llama3.1-8b          | 0.67  | 0.648 | 0.691 |         0.009 |              nan     |
| counts + text + agent llama3.1-8b   | 0.669 | 0.645 | 0.693 |       nan     |                0.003 |
| agent ministral3-8b alone           | 0.539 | 0.515 | 0.562 |       nan     |              nan     |
| counts + agent ministral3-8b        | 0.664 | 0.644 | 0.684 |         0.003 |              nan     |
| counts + text + agent ministral3-8b | 0.666 | 0.643 | 0.688 |       nan     |               -0     |
| agent phi-4 alone                   | 0.531 | 0.502 | 0.561 |       nan     |              nan     |
| counts + agent phi-4                | 0.664 | 0.643 | 0.685 |         0.003 |              nan     |
| counts + text + agent phi-4         | 0.666 | 0.642 | 0.689 |       nan     |               -0     |
| agent qwen3-1.7b alone              | 0.489 | 0.461 | 0.516 |       nan     |              nan     |
| counts + agent qwen3-1.7b           | 0.66  | 0.638 | 0.68  |        -0.001 |              nan     |
| counts + text + agent qwen3-1.7b    | 0.667 | 0.643 | 0.69  |       nan     |                0     |
| agent qwen3-14b alone               | 0.557 | 0.53  | 0.584 |       nan     |              nan     |
| counts + agent qwen3-14b            | 0.666 | 0.646 | 0.686 |         0.005 |              nan     |
| counts + text + agent qwen3-14b     | 0.668 | 0.645 | 0.69  |       nan     |                0.001 |
| agent qwen3-32b alone               | 0.622 | 0.599 | 0.648 |       nan     |              nan     |
| counts + agent qwen3-32b            | 0.687 | 0.666 | 0.706 |         0.026 |              nan     |
| counts + text + agent qwen3-32b     | 0.678 | 0.656 | 0.7   |       nan     |                0.011 |
| agent qwen3-4b alone                | 0.518 | 0.494 | 0.543 |       nan     |              nan     |
| counts + agent qwen3-4b             | 0.665 | 0.644 | 0.685 |         0.004 |              nan     |
| counts + text + agent qwen3-4b      | 0.667 | 0.644 | 0.689 |       nan     |                0     |
| agent qwen3-8b alone                | 0.566 | 0.541 | 0.592 |       nan     |              nan     |
| counts + agent qwen3-8b             | 0.665 | 0.644 | 0.685 |         0.004 |              nan     |
| counts + text + agent qwen3-8b      | 0.667 | 0.645 | 0.688 |       nan     |                0     |

Paired post-bootstrap ΔAUC (1,000 resamples):

- qwen3-32b: + agent over counts: ΔAUC +0.026 [+0.012, +0.039]
- qwen3-32b: + agent over counts + text: ΔAUC +0.011 [+0.004, +0.019]
- jev: + agent over counts: ΔAUC +0.022 [+0.008, +0.036]
- jev: + agent over counts + text: ΔAUC +0.005 [-0.003, +0.013]
