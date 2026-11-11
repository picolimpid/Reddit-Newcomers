# Jev System One vs Qwen3-8B (and API chat models), same 400 agents

Human 90-day return on these agents: 0.212; r/Y human base rate ≈ 0.059 (full sample).

Δlogit vs no reply (within agent, post-cluster bootstrap 95% CI):

|                       | jev                  | qwen3-8b             | gpt-4o-mini          |
|:----------------------|:---------------------|:---------------------|:---------------------|
| P none                | 0.299                | 0.040                | 0.028                |
| P observed            | 0.302                | 0.051                | 0.050                |
| AUROC (observed)      | 0.557                | 0.534                | 0.545                |
| ack                   | +0.08 [+0.06, +0.09] | +1.41 [+1.26, +1.56] | +1.91 [+1.67, +2.14] |
| disagree              | +0.06 [+0.05, +0.07] | +0.92 [+0.79, +1.06] | +1.44 [+1.22, +1.65] |
| praise                | +0.08 [+0.07, +0.09] | +2.68 [+2.50, +2.86] | +2.91 [+2.66, +3.17] |
| praise_welcome        | +0.05 [+0.03, +0.07] | +3.98 [+3.76, +4.19] | +6.25 [+5.96, +6.56] |
| praise_invite         | -0.00 [-0.02, +0.01] | +3.90 [+3.66, +4.13] | +4.83 [+4.51, +5.15] |
| praise_welcome_invite | +0.01 [-0.01, +0.03] | +5.34 [+5.09, +5.58] | +6.79 [+6.44, +7.16] |
| template              | -0.05 [-0.06, -0.03] | +4.45 [+4.22, +4.67] | +7.29 [+6.96, +7.63] |
| NC r/Y template       | +0.01 [+0.01, +0.02] | +0.54 [+0.38, +0.69] | -0.25 [-0.34, -0.16] |
| P(r/Y) none           | 0.185                | 0.027                | 0.000                |
| real reply (n=88)     | +0.06 [+0.03, +0.09] | +1.57 [+1.23, +1.94] | +2.33 [+1.83, +2.85] |

Between-agent SD of logit P(none): jev 0.35, qwen3-8b 2.50, gpt-4o-mini 3.04

- jev: Spearman(logit P none, c2v fit) = 0.348
- qwen3-8b: Spearman(logit P none, c2v fit) = 0.251
- gpt-4o-mini: Spearman(logit P none, c2v fit) = 0.290
- human: AUROC of c2v fit on these agents = 0.605
