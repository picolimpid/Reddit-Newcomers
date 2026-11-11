# P8: text x population swap (within agent, Δlogit vs no sticky; text_<site> = that site's sticky with the subreddit name set to the agent's own community; placebo = content-free pinned comment)

|                                 | placebo   | text_ffxiv2019   | text_futurology   | text_rscience   | text_science2020   |
|:--------------------------------|:----------|:-----------------|:------------------|:----------------|:-------------------|
| ('gemma3-12b', 'futurology')    | − -0.78   | 0 +0.01          | − -0.75           | 0 -0.21         | − -0.52            |
| ('gemma3-12b', 'rscience')      | − -0.79   | 0 -0.05          | − -0.54           | − -0.40         | − -0.59            |
| ('llama3.1-8b', 'futurology')   | 0 +0.16   | + +0.28          | + +0.28           | + +0.32         | + +0.28            |
| ('llama3.1-8b', 'rscience')     | 0 +0.17   | + +0.38          | + +0.41           | + +0.40         | + +0.40            |
| ('ministral3-8b', 'futurology') | 0 -0.08   | + +0.26          | 0 +0.16           | 0 +0.17         | 0 +0.18            |
| ('ministral3-8b', 'rscience')   | − -0.30   | 0 +0.06          | 0 -0.03           | 0 -0.04         | 0 +0.02            |
| ('qwen3-32b', 'futurology')     | 0 -0.08   | + +1.54          | + +0.58           | + +0.80         | + +0.49            |
| ('qwen3-32b', 'rscience')       | 0 -0.20   | + +1.38          | + +0.39           | + +0.78         | + +0.37            |
| ('qwen3-8b', 'futurology')      | − -0.35   | − -1.33          | − -0.56           | − -0.96         | − -0.83            |
| ('qwen3-8b', 'rscience')        | 0 -0.15   | − -1.56          | − -0.62           | − -1.26         | − -0.67            |

Variance of the text effects (type-II SS; model vs population vs text):

|              |   sum_sq |   df |       F |   PR(>F) |   share of SS |
|:-------------|---------:|-----:|--------:|---------:|--------------:|
| C(model)     |   14.706 |    4 |  41.895 |    0     |         0.832 |
| C(pop)       |    0.047 |    1 |   0.532 |    0.471 |         0.003 |
| C(condition) |    0.208 |    3 |   0.791 |    0.508 |         0.012 |
| Residual     |    2.72  |   31 | nan     |  nan     |         0.154 |

Share of (population, text) cells with the model's modal sign:

| model         |   share |
|:--------------|--------:|
| gemma3-12b    |    0.62 |
| llama3.1-8b   |    1    |
| ministral3-8b |    0.88 |
| qwen3-32b     |    1    |
| qwen3-8b      |    1    |

