# hci-explore：判官固定，校准人（HCI 论文 × JEV）

> 状态（2026-11-11）：探索 + 试点，已做 **30 组实验**（其中 16 项有事先写定的判定标准）。论文还没写，方向待两位老师（Koustuv Saha、Eshwar Chandrasekharan）决定。
> 本目录是 Reddit-Newcomers 仓库的子项目 C（总说明见 [../HANDOFF.md](../HANDOFF.md)）；它在服务器上的工作目录是 `/data/hanqim2/hci-explore`（现为指向本目录的软链接）。

## 一句话

一个便宜、快、直接输出概率的判断模型（JEV，TypeSafe System One；或本地 Qwen3-32B）负责“内容本身怎么样”；人与人的差异交给**每人一个截距**，用这个人 5–20 条判断就能估出来，不用微调判官。

这个机制在 4 份有真实个人差异的数据上都成立（MHS、DICES-350、DICES-990、POPQUORN），并且：
- **稳定**：截距的分半信度 0.92–0.99；
- **便宜**：训练日志每条只要 1 个人的标签（少 50 倍）就够；
- **判官可换**：换成本地开源模型，个人校准的效果不变；
- **更公平**：按种族的校准差距从 0.101 降到 0.004（POPQUORN，事先注册）；
- **有用**：能提前指出 3 人小组会在哪些条目上分歧（AUROC +0.07 到 +0.10）。

另一个场景（代码补全的接受门，RealHumanEval）：没有遥测时，只看文本 + JEV 就达到 CDHF（Mozannar et al., AAAI 2024）的水平。

## 先看哪几个文件

| 想知道 | 看 |
|---|---|
| 所有实验的一张总表（数据、问题、结果、是否事先注册、脚本、结果目录） | [RESULTS.md](RESULTS.md) |
| 这些结果说明了什么、哪些不成立、局限、下一步 | [ANALYSIS.md](ANALYSIS.md) |
| 按时间写的完整报告（v2–v9，含 CI 表和复现命令 §6） | [REPORT.md](REPORT.md) |
| 实验计划、事先写定的判定标准、执行记录（含作废的错误版本）、下一轮队列 | [PLAN.md](PLAN.md) |
| 研究主线、主张与证据强度、三种论文写法、方法论文提纲 | [RESEARCH_NOTES.md](RESEARCH_NOTES.md) |
| 三个 idea 的完整设计（原论文、缺口、系统、研究设计、风险） | [IDEAS_README.md](IDEAS_README.md) |
| 以两位老师的视角推演的组会（**根据他们的论文推出，不代表本人意见**） | [GROUP_MEETING.md](GROUP_MEETING.md)、读书笔记 [PROF_READING.md](PROF_READING.md) |
| 文献调研：100 篇 CHI/CSCW/UIST/IUI 论文表、读过的全部论文 | [PAPERS_100.md](PAPERS_100.md)、[PAPERS_READ.md](PAPERS_READ.md) |

周报（Claude Docs，中文，含图）：https://claude.ai/artifact/3qBjAZjo15sPpiQKEyN8tg （私有，作者分享后才能看）。

## 目录结构

```
hci-explore/
├── README.md  RESULTS.md  ANALYSIS.md      # 入口、结果总表、分析
├── REPORT.md  PLAN.md  RESEARCH_NOTES.md  IDEAS_README.md
├── GROUP_MEETING.md  PROF_READING.md  PAPERS_100.md  PAPERS_READ.md
├── src/                 # 全部脚本（一个实验一个文件，文件名 = 实验编号）
│   ├── jev.py           # JEV 客户端：缓存、账本、$0.50 上限（默认走 OpenRouter）
│   ├── llm_judge.py     # OpenRouter 闭源 LLM 判官（GPT-4o / 4o-mini）
│   ├── local_judge.py   # 本地 vLLM 判官（开源模型只在本地跑）
│   ├── e1*.py           # Idea 1：多标注者内容编码、每人校准（MHS、DICES）
│   ├── e2*.py           # Idea 2：主动助手 / 代码补全的接受门（ProactiveBench、RealHumanEval）
│   ├── e3*.py           # Idea 3：多准则评估（SummEval，已暂缓）
│   └── e4*.py           # D 线：组会推演后的实验（Jigsaw、DICES、POPQUORN）
├── results/<实验编号>/   # 每个实验的日志（log*.txt）、表（*.csv）、判官输出（llm_*.jsonl、jev*.parquet）
│   ├── jev_ledger.jsonl, openrouter_ledger.jsonl   # 每次付费请求的花费账本
│   └── e*/api/, e3/invalid/                         # 被本地结果替换的 API 版、作废的运行
├── archive/             # 旧版报告备份（REPORT v2–v7、IDEAS_README v5–v7）
├── logs/                # 早期运行日志
└── data/                # 原始数据和缓存（不进 git，见下）
```

## 不在 git 里的东西

为了避免泄露密钥、重新分发第三方数据和超过 GitHub 的文件大小限制，以下内容只在服务器上（见 [.gitignore](.gitignore)）：

- `.env`（OpenRouter key）；TypeSafe key 在 `../speculative-jev/.env`。
- `data/`（约 380 MB）：下载的数据集、论文 PDF、JEV 缓存。
- 判官的输入文件（`*_in.jsonl`、`emb_in.jsonl` 等，含数据集原文）和带原文的 `results/e1/sample.parquet`、`results/e2b/jev_all.parquet`、`results/e4a/jev.parquet`。
- `*.npz`（样本外预测和句向量，单个最大 62 MB；脚本能重新生成）。

所以 clone 下来的仓库可以看全部结果和代码，但要重跑需要先把数据下到 `data/ext/`（下表），JEV 结果没有缓存时会重新花钱。

## 数据

| 数据 | 用在 | 来源 / 许可 |
|---|---|---|
| Measuring Hate Speech（MHS） | E1、E1b–E1d | HF `ucberkeley-dlab/measuring-hate-speech`（CC BY 4.0） |
| ProactiveBench | E2、E2b–E2d | thunlp/ProactiveAgent 的 `dataset/reward_data/` |
| SummEval | E3 | Fabbri et al. 2021 |
| DICES-350 / DICES-990 | E1e–E1j、E4b–E4g、E4i、E4k | google-research-datasets/dices-dataset（CC BY 4.0） |
| RealHumanEval | E2e–E2h、E4b | HF `hsseinmz/realhumaneval`（只读 CSV，没有运行仓库代码） |
| Jigsaw Agile Community Rules | E4a、E4e | Kaggle 比赛数据的镜像（许可待核对） |
| POPQUORN | E4h、E4j | Pei & Jurgens 2023（许可待核对） |

## 复现

全部 JEV 和 LLM 响应在服务器上已缓存，重跑分析不花钱。命令见 [REPORT.md](REPORT.md) §6；D 线的脚本：

```bash
cd /data/hanqim2/hci-explore      # 或 "/data/hanqim2/Reddit Newcomers/hci-explore"
python3 -I src/e4c_single_label.py          # 先跑：E4d、E4i、E4k 用它的样本外预测
python3 -I src/e4b_construct.py             # 加 --posthoc、--replicate350 跑事后分析和复现
python3 -I src/e4d_venire_sim.py
python3 -I src/e4f_topics.py; python3 -I src/e4g_local_judge.py
python3 -I src/e4h_popquorn.py; python3 -I src/e4j_popquorn_more.py
python3 -I src/e4i_placebo.py; python3 -I src/e4k_prior_sensitivity.py
python3 -I src/e4a_community_calib.py analyze; python3 -I src/e4e_paraphrase.py analyze   # 只读 JEV 缓存
```

本地 Qwen3-32B（不要用 0 号 GPU，它故障，会让 CUDA 初始化失败）：

```bash
CUDA_VISIBLE_DEVICES=7,9 NCCL_P2P_DISABLE=1 /data/hanqim2/.conda/envs/vllm311/bin/python src/local_judge.py \
    --model /data/hanqim2/models/Qwen3-32B --tp 2 --gpu-mem 0.80 --max-model-len 2048 \
    --tag qwen3-32b-local results/e4h/qwen_in.jsonl
```

环境：系统 `python3`（numpy、scipy、pandas、scikit-learn）；E1h 需要 torch（`/data/hanqim2/.conda/envs/specjev/bin/python`，CPU）；本地判官用 `vllm311` 环境。

## 花费

| 项 | 用了 | 上限 |
|---|---|---|
| JEV（`results/jev_ledger.jsonl`） | **$0.4381**（14,843 个请求、约 1,043 万 token） | $0.50 |
| OpenRouter 上的 LLM 对照（`results/openrouter_ledger.jsonl`） | 约 $2.99（JEV 也走 OpenRouter，合计约 $3.20） | 总额度 $20 |
| 本地 Qwen3-32B | 0（GPU） | — |

## 工作规则（写在 PLAN.md §0）

1. 先探索、后确认：新方法先在 DICES-350 / RealHumanEval 上探索，站住的结论要在没碰过的数据上用事先写好的分析复现。
2. 每个实验在跑之前写下主要比较和判定标准；95% CI 不含 0（或非劣效界）才算“有利”。
3. 结果好坏都记，包括作废的错误版本（旧日志保留）。
4. 文献由人自己读，不用 JEV 或其他模型筛选；开源模型只在本地跑。
