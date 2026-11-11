# 实验设计：第一轮测试（RQ1 + RQ2，base model）

*Proposal 见 [PROPOSAL.md](PROPOSAL.md)，人类参照见 [FINGERPRINTS.md](FINGERPRINTS.md)。*

## 1. 这一轮要回答什么

- **RQ1：** 用真实 newcomer 的历史构造 agent，让它经历和真人一样的进入事件，agent 的回访决策在群体层面是否复现 FP1–FP3？
- **RQ2：** 在其他条件完全相同的情况下，只换掉 agent 看到的回复（无回复 / 欢迎 / 排斥），agent 的回访倾向变多少？和 1 SD community fit 的效应相比是大是小？
- **RQ3 本轮不做**，但 benchmark 已经按版块切好 train / val / test（49 / 17 / 17 个版块；1,188 / 608 / 617 名 newcomer），之后的 SFT 只在 train 版块上训练，在没见过的 test 版块上检验 fingerprint 是否出现。

## 2. Benchmark（`src/build_benchmark.py` → `data/benchmark.parquet`）

一行是一名真实 newcomer（Study 2，2,413 人）。

| 用途 | 内容 |
|---|---|
| agent 的输入 | 进入前 365 天的历史（带日期、版块、正文）；按 0–30 / 30–90 / 90–365 天统计的版块活动；进入的帖子（标题、正文、是否视频）；自己的首条评论及其回复对象；真实收到的直接回复 |
| 人类参照 | 30 / 90 / 180 天回访、成为常驻；整体 c2v fit 和三个窗口的 c2v fit；Jev 窗口分数；Jev 对回复的编码；模型控制变量 |
| 切分 | 按版块的 train / val / test（seed 0） |

检查结果：
- 三个窗口都有 ≥ 5 条的 newcomer 正好 1,130 人，和 REPORT §4.12 一致；
- 回复条数和原始 `n_replies` 完全一致（2,413 / 2,413），楼主回复 17 / 17；
- 用 `src/fingerprints.py` 重算的人类 fingerprint 和 REPORT 的数字一致到小数点后两位（c2v 窗口 90 天 1.79 / 1.03 / 1.01，180 天 2.05 / 1.23 / 0.91；Jev 窗口 1.57 / 0.98 / 0.83；收到回复 1.07；fit 1.51 / 1.61）。

## 3. Agent 协议（`src/agent_sim.py`）

每个 agent 的 prompt 由四部分组成：

1. **记忆：** 这个用户自己过去一年的活动，包括三个时间窗口内的版块分布，以及 40 条带日期的评论 / 帖子（最近 25 条 + 更早的随机 15 条）。
2. **进入事件：** 帖子上了首页；"你过去一年没在 r/X 发过言"；你在发帖 N 小时后留下了这条评论（如果是回复别人，附上被回复的评论）。
3. **回复条件**（四选一）：
   - `observed`：真人实际收到的回复（79% 的人没有回复）；
   - `none`：无回复；
   - `welcoming`：模板欢迎语（"Welcome to r/X! Stick around…"，3 个变体轮换）；
   - `hostile`：模板排斥语（"Go back to r/popular, this sub isn't for tourists"，3 个变体轮换）。
4. **决策问题**（两种协议）：
   - `intent`：未来 3 个月你会再在 r/X 发言吗？读 P(Yes)。对应人类 90 天回访。
   - `choice`：两周后你打开 Reddit，先做什么？选项包括 r/X、这个用户自己最常去的 3 个版块、r/popular 和"关掉 Reddit"，每个 agent 的选项顺序随机打乱。读 P(选 r/X)。这是行为选择，不是自我陈述，可以减轻 LLM 的"Yes 偏好"。

读出方式是第一个 token 的概率（temperature 0、top-20 logprobs、在合法选项上归一化），一次前向就得到连续的倾向值，不需要采样。

每个 agent 有 2 种协议 × 4 种条件 = 8 个 prompt。没收到回复的 agent，observed 和 none 是同一个 prompt，只算一次。总计 19,304 个 prompt，token 中位数 1,990。

## 4. 指标

- **Fingerprint：** agent 和人类用同一个模型设定（post FE、相同控制变量、按帖子聚类）、同一批样本行估计。判定用无量纲比值，标准在 FINGERPRINTS.md 末尾。合成数据检验过：一个"窗口等权、对回复敏感"的假 agent 四项全部不通过，人类四项全部通过。
- **RQ2 干预效应：** 同一个 agent 在 welcoming 和 none 下的 logit 差，再除以这个 agent 群体自己的 1 SD fit 效应，得到"一句欢迎相当于多少 SD 的 community fit"。人类只有观察值，约 0.18 SD（90 天）。
- **次要指标：** 基线比率（agent 平均倾向 vs 20.3%）；个体层面 AUROC（参照：c2v fit 单独的 AUROC 为 0.58 / 0.59）。

## 5. 模型和算力

- 主模型：Qwen3-8B（本地已缓存），关闭 thinking mode。规模对照：Qwen3-1.7B。
- 显存：Qwen3-8B bf16 约 16 GB 权重，再加 KV cache。`gpu_memory_utilization` 0.45 就够用。
- 预计耗时：一张 A6000 上 2,413 个 agent 全量约 20–40 分钟（开 prefix caching，同一个 agent 的 8 个 prompt 共享前缀）。
- 先用 `--limit 50` 跑通，再跑全量。

## 6. 已知局限（写结果时要说明）

1. **记忆格式把时间窗口直接写了出来。** 这让 agent 更容易注意到"最近 30 天"。如果 agent 在这种情况下仍然不复现 FP1，结论更强；如果复现了，要用不分窗口的纯时间线格式再测一次。
2. **自我陈述 vs 行为。** 两种协议都是单步决策，不是 OASIS 那样多步的 feed 模拟。choice 协议更接近行为，但仍是一次性选择。
3. **训练数据污染。** 2025 年的 Top Posts 可能出现在 Qwen3 的训练数据里，但用户个人的回访行为基本不可能被记住。
4. **模板回复不真实。** welcoming / hostile 模板刻意做成明确的社交信号。这正是 RQ2 要测的"简单社交信号"，但它的效应不能直接等同于真实回复的效应。真实回复的效应用 observed − none 衡量（516 个收到回复的 agent）。
5. **人类一侧只有观察关联。** RQ2 的干预效应没有对应的人类因果估计，只能和人类的观察关联比较。

## 7. 下一步（视第一轮结果）

1. **历史消融：** 把最近 30 天的条目挪到更早的日期，或反过来，检验 agent 是否真的在用"轨迹"。
2. **多步 feed 模拟：** 类似 OASIS，12 周、每周一个 session、用真实帖子构造 feed，回访定义为在 r/X 上的任何行动。
3. **更多模型：** 更大的开源模型和 API 模型，看 fidelity 是否随规模提升。
4. **RQ3：** 在 train 版块上用真实回访标签做 SFT / behavior cloning，在 test 版块上检验 FP1–FP3。另设一个只拟合个体标签、不碰群体统计的对照，看 fingerprint 是否"自然出现"。
