# Behavioral Fidelity of LLM Social Agents

用 Reddit Newcomers 研究中真实用户的行为规律（behavioral fingerprints）来检验 LLM agent 能否像真人一样行为，而不只是给出看起来合理的社交反应。

| 文件 | 内容 |
|---|---|
| [PROPOSAL.md](PROPOSAL.md) | 项目 proposal（RQ1–RQ3） |
| [FINGERPRINTS.md](FINGERPRINTS.md) | 人类参照：从 Reddit Newcomers Study 2 中摘出的 fingerprint，以及 agent 的判定标准 |
| [PLAN.md](PLAN.md) | 第一轮测试的设计：benchmark、agent 协议、指标、局限 |
| [RESULTS.md](RESULTS.md) | **结果**：Qwen3 1.7B → 32B 和 Qwen2.5-72B。规模大了能复现进入前轨迹（FP1），但对欢迎语仍然过度敏感，基线比率失真 |
| `reference/` | Reddit Newcomers 的 REPORT.md 和 RQ.md 副本（所有数字的出处） |
| `data/source` | 指向 `/data/hanqim2/Reddit Newcomers/data/global` 的 symlink，只读 |
| `data/benchmark.parquet` | 2,413 名 newcomer 的 agent 输入和人类参照 |
| `results/` | `human_fingerprints.csv`；每个 agent run 一个子目录 |

## 运行

```bash
cd src
python3 build_benchmark.py      # ~3.5 min，CPU；训练 c2v 并计算窗口 fit
python3 fingerprints.py         # 人类 fingerprint，和 REPORT 对照

PY=/data/hanqim2/.conda/envs/vllm311/bin/python
$PY agent_sim.py --dry-run                                        # 只渲染 prompt，不用 GPU
CUDA_VISIBLE_DEVICES=<gpu> $PY agent_sim.py --model Qwen/Qwen3-8B --out qwen3-8b --limit 50
CUDA_VISIBLE_DEVICES=<gpu> $PY agent_sim.py --model Qwen/Qwen3-8B --out qwen3-8b
python3 evaluate.py qwen3-8b    # -> results/qwen3-8b/summary.md
python3 reversal.py qwen3-8b qwen3-8b-reverse   # 时间反转机制检验（需先跑 --history reverse）
```

| `src/` | 作用 |
|---|---|
| `build_benchmark.py` | 从 Study 2 数据构造 benchmark，含 c2v 窗口 fit 和按版块的切分 |
| `fingerprints.py` | 人类和 agent 共用的估计器（post FE GLM，按帖子聚类） |
| `agent_sim.py` | vLLM agent：intent / choice 两种协议 × 4 种回复条件；`--history reverse`、`--persona timeline` |
| `evaluate.py` | agent vs 人类 fingerprint、判定、RQ2 干预效应、基线比率、AUROC |
| `reversal.py` | 同一个 agent 在原始历史和时间反转历史下的倾向对比 |
| `robust_ols.py` | 每个 agent 等权（对 logit(P) 做 OLS）的稳健性检验 |
| `compare_runs.py` | 跨模型汇总表 → `results/compare.md` |
| `run_scale.sh`、`run_followup.sh`、`run_resume.sh` | 规模扩展和 32B 机制检验的运行脚本 |
| `c2v.py` | 从 Reddit Newcomers 复制的 community2vec 实现 |

- 环境：数据处理和统计用系统 Python（`/usr/local/anaconda3`，有 pyarrow / statsmodels），vLLM 用 `vllm311` 环境。
- 不需要任何 API key，也不调用 Jev。
