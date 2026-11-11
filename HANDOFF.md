# 项目交接 / 重启说明（2026-11-11 暂停）

这个仓库里有三个相关项目：

| 部分 | 位置（本仓库） | 服务器上的工作目录 | 状态 |
|---|---|---|---|
| **A. Reddit Newcomers（信号研究）** | 仓库根目录：`src/`、`results/`、`REPORT.md`、`RQ.md`、`PROPOSAL.MD` | `/data/hanqim2/Reddit Newcomers` | 主假设为零结果；剩下的信号是探索性的，还没复制 |
| **B. Behavioral fidelity（LLM 模拟 newcomer）** | `behavioral-fidelity/` | `/data/hanqim2/behavioral-fidelity` | 大部分实验已跑完；预注册的 outcome 还**封存**着 |
| **C. hci-explore（判官固定，校准人：JEV + 每人一个截距）** | `hci-explore/` | `/data/hanqim2/Reddit Newcomers/hci-explore`（旧路径 `/data/hanqim2/hci-explore` 是指向它的软链接） | 30 组实验已跑完并整理；入口 `hci-explore/README.md`，结果总表 `RESULTS.md`，分析 `ANALYSIS.md`；论文方向待老师决定 |

论文和 thesis 的主线是 B，A 是 B 的人类参照来源。B 最完整的叙述在 `behavioral-fidelity/NARRATIVE.md`（证据表 E1–E23，§5 机制，§7 论文结构，§9 弱点），执行细节和每次运行的流水在 `behavioral-fidelity/ROADMAP.md`。

---

## 1. 结论（截至 11-11；除非标了"预注册"，都是探索性的）

**一句话：** LLM 模拟的 newcomer 对同一段治理文字只有「一种读法」，这种读法由模型决定，而不是由社区或人群决定。真实的 newcomer 则因决策（回来 / 参与 / 合规）和人（新手 / 老手）而异。没有一个模型能同时对上人类的多个杠杆。

### 1.1 人类端（r/science 2016 sticky RCT，CivilServant / Matias 2019）
- 90 天回访：sticky 效应 ≈ 0（−0.03 [−0.23, +0.17]）。
- 参与：newcomer 评论数 IRR 1.67 [1.53, 1.82]。
- 合规：评论被删的 logit −0.44 [−0.80, −0.07]（p = .018）。
- 对谁：按历史长度分三等分，T1 −0.30、T2 −0.04、T3 +0.18，斜率 +0.48 [+0.02, +0.98]；treat × z(log n_hist) = +0.275 [+0.074, +0.477]，p = .004（H14 在 r/science 上的探索版本）。对历史少的新手是负的，对有经验的人是正的。

### 1.2 Agent 端（16 个模型：Qwen3 1.7B–32B、Gemma-3 4B–27B、Llama-3.1-8B、Ministral-8B、Mistral-Small-24B、Phi-4、GLM-4-32B、gpt-4o-mini、Jev；另有 OLMo 和 Qwen2.5-72B 因 readout 失败被剔除）
| 编号 | 结论 | 关键数字 | 出处 |
|---|---|---|---|
| E18 / **H8（预注册）** | 模型身份能预测新站点上的方向 | **14/16 = 87.5%，成立**（阈值 80%）；两个未中的都缩到 ≈0，没有反号 | `behavioral-fidelity/results/sites/h8_h9_formal.md` |
| **H9（预注册）** | 模型之间的分歧远大于人类效应 | ffxiv 极差 4.22，science2020 极差 2.44（阈值 1.0），**成立** | 同上 |
| E10 / P8 | 文本 × 人群互换：方向是模型的 | 方差：模型 83%，人群 0.3%，文本 1.2% | `results/panel/swap*.md` |
| E9 / P3 | prompt 多元宇宙 | 模型 53%，prompt 变体 8%，站点 0.2%；40 个变体里 31 个保持符号 | `results/panel/multiverse.md` |
| E19 / P9（预注册 10 模型 + 3 个扩展） | 进入（会不会评论）全部低估 | 13/13 的比率都低于人类 IRR 下限 1.53；同一模型的 return 和 entry 方向 ρ = 0.96 | `results/p9/summary*.md` |
| E17 / P5 | 在一个杠杆上对上 ≠ 有保真度 | 9 个模型跑了 upvote 准实验，8 个高估 upvote（唯一对上的 Llama 几乎不反应），**没有一个模型 2/2** | `results/panel/scorecard.md`、`results/p5/summary.md` |
| E21 | 对谁：多数模型预测不了人类的异质性 | 13 个模型里只有 3 个和人类同号（Gemma-27B、gpt-4o-mini、Qwen3-8B），而且跨站点不稳 | `results/panel/hetero_history*.md`、`h14.md` |
| E22 / E23 / P4 | 四种「读法」 | A 一律负（Gemma、Mistral）；B 一律正（Llama）；C 无视治理文字、只对回复反应（GLM、Jev）；D 读内容但语义相反（Qwen3-8B：welcome +、删除 −；Qwen3-32B：删除 +、welcome −）。消融网格的方差：模型 47–72%，句子 1–6% | `results/ablation/summary.md` |
| E23 | 对版主：模拟器挑出来的措辞不可移植 | 各措辞的排序在模型之间平均 Spearman 只有 0.15–0.21 | 同上 |
| E23 | 不是训练数据污染 | 「知道」Matias 结论的模型方向有正有负 | 同上 |
| E23（事先写下的预测） | Mistral 属于读法 A | 17/17 负、极差 0.48，**预测成立** | ROADMAP 11-08 18:05 |

### 1.3 A 部分（Reddit Newcomers 信号研究）
- 主假设（兴趣对齐 → 第一次参与）为零结果；试点里 re-engagement 的关联没有复制。
- 唯一通过 FDR 的主要结局：对齐度 +1 SD → 90 天回访 OR 1.26 [1.05, 1.51]（只在 Study 2）。c2v network fit 的信号是探索性的，**还没复制**（缺预注册的 2024 复制和人工验证）。
- 细节见 `REPORT.md`、`RQ.md`。

### 1.4 弱点（直接列出，详见 NARRATIVE §9）
- Futurology、ffxiv、science2020 的人类端结果**还没取**（封存），所以 H1–H7、H14/H15 都还没有检验。
- 只有 r/science 2016 是真正的随机分组 RCT，另外几个站点是 as-treated。
- 预注册**还没有在 OSF 上提交**；目前唯一的时间记录是 FREEZE.txt 的哈希和时间戳（自证）。
- agent 人群是「留下了作者名」的幸存者；历史上限 n_hist ≤ 200；结局只有 return / 参与 / 合规。
- 每个模型都只有一种 readout（首 token P(Yes)）。GLM 需要补一个换行才能读（tp_wrap 的 GEN_SUFFIX），这是一处模型特定的偏离。

---

## 2. 预注册和封存状态（重启时最容易出错的地方）

- 冻结记录：`behavioral-fidelity/results/futurology/FREEZE.txt`（freeze 1–10）和 `behavioral-fidelity/results/sites/FREEZE.txt`（S1–S3，以及 11-11 的 `scores ffxiv2019`、`scores science2020`）。
- 预注册文件：`PREREG_futurology.md`、`PREREG_sites.md`、`PREREG_addendum_H10.md`、`PREREG_addendum_P9.md`、`PREREG_addendum_H14.md`。
- **还没取过的 outcome**：Futurology 2020、ffxiv2019、science2020。
- **重启后第一步：** 先把这 5 份文件提交到 OSF（或者明确决定不提交，并在论文里说明），然后才能：
  - `python3 src/futurology_collect.py outcomes` → `python3 src/futurology_analyze.py`
  - `python3 src/site_collect.py outcomes ffxiv2019` / `science2020` → `site_analyze.py`（脚本会检查 FREEZE 里有没有 `scores <site>`，现在已经有了）
  - H14/H15：取到 Futurology outcome 之前，先重新生成 `results/panel/hetero_history_futurology.csv` 并做哈希（PREREG_addendum_H14.md），然后运行 `h14_analyze.py`。
- **冻结后不能改的文件：** futurology_collect / agents / analyze、panel_agents、panel_summary、panel_median、multiverse_summary、swap_agents、p9_entry、p9_analyze、h14_analyze、hetero_history、所有 PREREG 文件，以及 `results/panel/hetero_history*.csv`。新模型的 for-whom 斜率要临时另算，不能覆盖这些 csv。
- 新模型加进来时：先冻结它的 Futurology 分数，再看任何 outcome。

---

## 3. 没跑完 / 待决定

| 项目 | 状态 | 重启方法 |
|---|---|---|
| P7 合规扩展（GLM、Mistral-Small、Gemma-27B） | GLM 生成完成；另外两个生成任务处于 `# HOLD(2gpu first)`；judge3 还没跑 | 去掉 jobs.txt 里的 HOLD；两个 gen 都完成后，judge3-* 会自动开始；最后运行 `p7_analyze.py` |
| P4-glm4-32b（检验读法 C 的预测） | failed：RPC 超时 | 换一个新任务名（failed 是终态），并在命令前加 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600` |
| Qwen2.5-72B AWQ | DBG 加载时 OOM（当时 GPU 被别人占用） | 换新名字重跑 DBG；首 token 正常后 `touch results/queue/.ok_qwen72awq`；进 H8 已经来不及（站点已冻结），只能作为探索性结果 |
| gpt-4o-mini science2020 | 没跑：预计 $1.18，OpenRouter 只剩约 $0.91（上限 $10） | 需要你决定提高上限（例如 $11），或者接受只跑一个子集并在论文里说明 |
| P6 版主研究 | 设计见 `MODERATOR_STUDY.md` | 需要 IRB |
| A 部分：2024 预注册复制、人工验证 | 未开始 | 投稿 A 部分结果之前必须先做 |

---

## 4. 怎么重启

**环境**
- GPU 机器：GPU 0 坏了；GPU 9 是别人的。队列只用 GPU 4–8，`MAX_GPUS=3`。nvidia-smi 一律套 `timeout 20`，它卡死过一次。
- vLLM 环境：`/data/hanqim2/.conda/envs/vllm311/bin/python`（vLLM 0.19.1）。统计用系统的 `python3`（有 statsmodels 和 sklearn）。
- 双卡（TP=2）一律通过 `src/tp_wrap.py` 运行（NCCL_P2P_DISABLE=1 + disable_custom_all_reduce），长任务再加 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600`。
- 在线 fp8 在 OLMo-3.1-32B、GLM-4-32B、Qwen2.5-72B 上 readout mass = 0。**每个新模型都要先检查 mass。**
- `.env`（不在 git 里）需要 TYPESAFE_API_KEY、JEV_*、OPENROUTER_API_KEY、OPENROUTER_MAX_USD=10。支出记录在 `data/openrouter_ledger.jsonl`，目前约 $9.09。

**数据（不在 git 里，公开仓库里不能放 Reddit 用户数据）**
- 服务器上：`/data/hanqim2/Reddit Newcomers/data`（约 4.1 GB）、`/data/hanqim2/behavioral-fidelity/data`（约 520 MB）。`/mnt/disk1/hanqim2` 只读，不要动。
- 重建：A 部分按 `src/s1_*` → `s15c_*` 的顺序；B 部分用 `build_benchmark.py`、`rscience_collect.py` / `rscience_retro_collect.py`、`futurology_collect.py inputs`、`site_fetch.py` / `site_collect.py assign|collect`。
- 为了不在公开仓库里放用户内容，**下面这些没有推送**：`results/*/example_prompts.json`（含真实用户历史）；`results/p7/*/gen.parquet` 和 `results/probes/qwen3-8b/readout.parquet`（以真实历史为条件生成的文本）；`results/**/*.log`（日志里打印了 prompt 示例；runner.log 保留了）；SFT adapter（175 MB，在用户数据上训练）。这些都还在服务器上。

**队列**
```bash
cd /data/hanqim2/behavioral-fidelity/src
setsid nohup python3 -u queue_runner.py >> ../results/queue/runner.log 2>&1 < /dev/null &
```
- 任务的格式：`results/queue/jobs.txt` 中每行一个 `name | n 或 n:G | 依赖文件 | 命令`。`# ` 开头的行会被跳过；`HOLD(...)` 只是一种注释写法。
- 状态记录在 `state.tsv`。`failed(...)` 是终态，要重跑就换一个任务名。
- 单卡任务会占满 3 张卡，让双卡任务一直排不上。所以有双卡任务排队时，先把后面的单卡任务暂时 HOLD 起来。
- 结束任务只按明确的 PID 来，不要用 pkill -f。TP worker 可能变成孤儿进程（11-08 出现过，占着 45 GB），要检查。

**跑完新结果后的汇总顺序**
`panel_summary.py` → `fig_panel.py` → `scorecard.py`；`multiverse_summary.py`、`swap_content.py`（P3 / P8）；`p5_analyze.py <models...>`；`p9_analyze.py`；`p4_summary.py`；`h8_h9_formal.py`（已完成）。

---

## 5. 论文和 thesis 方向（详见 NARRATIVE §7–§8）

- 暂定题目：*Same Sticky, Opposite Advice: Calibrating LLM-Simulated Newcomers Against Online Field Experiments*。目标 CSCW 或 ICWSM，截稿日期待核实。
- RQ1：方向是否由模型决定、是否跨站点稳定（H8 / H9、P8）。RQ2：校准到真实的非零效应（P9 参与、P7 合规、scorecard）。RQ3：能否预测对谁有效（E21、H14 / H15）。
- 给版主的讨论：报告多个模型的分布，不要只给一个数；先锚定到真实效应再谈预测；不要用模拟器挑措辞；把新手保护当作「对谁有效」来检验。
- 参考的研究方法：Eshwar Chandrasekharan（UIUC）组——校准到真实干预效应、按 unit 拆分 RQ、关注对谁有效以及对版主的含义。
