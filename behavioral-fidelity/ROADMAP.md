# 持续执行计划（2026-11-06 起）

本文件既是计划，也是检查清单：每次回来先按第 3 节检查，再更新第 1 节的状态。GPU 任务由 `src/queue_runner.py` 自动排队执行（`results/queue/jobs.txt` 可以随时追加，状态写在 `state.tsv`）。研究思路和证据表在 [NARRATIVE.md](NARRATIVE.md)。

## 0. 先写定什么算"有利结果"

"跑到有利结果为止"如果理解成"一直跑，直到某个检验显著"，就是 p-hacking，审稿人一眼能看出来，预注册也会作废。所以这里把"有利"定义成**论文可以写出来的结论**，每条都事先写好判定标准：不管结果是否有利都照实报告，变化的只是论文怎么写。

| 编号 | 结论 | 判定标准（事先固定） | 当前状态 |
|---|---|---|---|
| R1 | 模拟给出的设计建议取决于模型 | 每个站点至少有 2 个模型方向相反（\|Δ\| > 0.25 且 CI 不含 0）；各模型效应的跨站点相关 ≥ 0.7 | **已满足**：2 个站点、11 个模型 r = 0.94；每站都有正有负（Futurology +3 / −5 / 0 3 个，r/science +4 / −5 / 0 3 个）；面板扩到约 18 个模型后再判一次 |
| R2 | 预注册的前瞻预测 | Futurology 的 H1（8B 的 gap < 0），在 OSF 提交之后揭晓 | 待揭晓。agent 部分的 H6 和 H7 全部成立 |
| R3 | 方向由模型决定，不由测量方式决定 | 同一个模型在 4 种 prompt 变体中至少 3 种方向一致，而且模型间方差 > prompt 间方差 | 部分满足（11-07 01:10）：Qwen3-8B 两站 4/4；Llama r/science 3/4、Futurology 1/4（掉到 0，没有反向）→ 按事先标准 Llama-Futurology 不满足；第二条（model 方差 67% > variant 11%）满足。还差 3 个模型 **11-07 12:40 更新：按事先标准不成立**。Gemma-3-12B：r/science 3/4 保持负号但 q2 反转为 +0.70；Futurology 3/4（q2 → +0.06）。Ministral 两站都只有 0/4 显著同号（base ≈ 0）。模型间方差 42% vs 变体 10%（第二条仍满足），但第一条对 Llama-Fut、Ministral 失败。 |
| R4 | 普遍对正向社交线索过度反应 | 面板中 ≥ 80% 的模型对模板欢迎语的反应 > +1 logit，比人类锚点大一个数量级（Lambert et al. CHI 2025：被高赞后发帖率约 +2%/天） | **按事先标准不成立**（11-06 20:00）：两站都 > +1 的只有 gpt、Qwen3-8B、Qwen3-32B；Llama-3.1-8B、Gemma-3-4B、Jev、Ministral-3-8B 都 ≤ +1。改写成「正向线索敏感度也是模型特有的」，不再声称普遍 |
| R5 | "通过检验的理由不对" | 每个 gap 不显著的模型，看它的组件结构是否也对（例如 rules 和 welcome 互相抵消） | r/science 已经看到 gpt 和 32B 是抵消；新站点还要再看 |
| R6 | 跨站点外推 | 新站点（ffxiv2019、science2020）的 agent 分数在揭晓前冻结，揭晓后的 gap 计算方式和 Futurology 完全一样 | Phase 2 |
| R7 | 多模型平均也救不回来 | H10：开源模型面板中位数的 gap，其 90% CI 不落在 ±0.25 内（3 个封存站点；PREREG_addendum_H10.md，事先写明预期不成立） | r/science 回溯：开源中位数 gap −0.50；全部 9 个模型 −0.06，但对选哪些模型很敏感 |
| R8 | 校准到真实的非零效应也失败，而且失败方式有结构 | P9 预注册 H11–H13（freeze 6）：10/10 模型低估 newcomer 参与效应（IRR 1.67），entry 与 return 方向跨模型正相关 ρ=+0.67 | **成立**（11-07，10 个开源模型） |
| R9 | 模型能否回答"对谁有效？"（异质性保真度） | r/science 发现：人类 sticky 效应随 newcomer 历史量上升（treat:z +0.28，p=.004）；只有 3/13 模型同号。预注册 H14（Futurology 人类复制）/H15（≤1/3 模型匹配），freeze 8 | 发现成立；**复制待 OSF 后取结局** |

**纪律：**
- 所有任务都记在 `state.tsv` 里，面板结果全部报告，不挑模型。
- 每个站点的 outcome 都在冻结之后才抓（`site_collect.py outcomes` 会检查 `results/sites/FREEZE.txt`）。
- Futurology 的 outcome 要等你的 OSF 链接。

## 1. 队列和阶段

| 阶段 | 内容 | 资源 | 状态 |
|---|---|---|---|
| P1 | 多模型面板，base prompt，Futurology 和 r/science 两个站点 | 约 15 个开源模型 × 2 个站点，GPU 4–8，同时最多占 3 块 | 运行中（`results/queue/`） |
| P2 | 新站点 r/ffxiv 2019 和 r/science 2020（CivilServant 的另外两次复制实验）：fetch → assign → collect → 面板 → 冻结 → outcome | Arctic Shift，几个小时；GPU | fetch 运行中 |
| P3 | Prompt 多元宇宙：`q2` / `nosys` / `order` / `observer` | 5 个模型 × 4 种变体 × 2 个站点 × 1,500 个 agent | 排在 P1 之后 |
| P4 | Sticky 文本消融：按句拆开（描述性规范 "thousands follow the rules"、删除威胁、版主人数、welcome），找出导致方向翻转的那一句 | 6 个模型 × 1,000 个 agent | 待写 `ablation_agents.py` |
| P5 | Eshwar 风格的第二种干预：正反馈。在 newcomer 第一条评论上，人类端用准实验（Lambert et al. 2025 的 PSM + DiD）估计 CTE，agent 端操纵 "你的评论得了 X 分 / 收到感谢回复" | Reddit Newcomers 数据加新抓的多社区样本 | 需要先确认 Arctic Shift 的 score 是否可用 |
| P8 | Text × population swap：同一批 agent 依次看 4 个站点的 sticky（社区名换成自己的）和一个无内容置顶（placebo）；`swap_agents.py`，结果写进 `results/panel/swap.md` | 5 个模型 × 2 个人群 × 6 个条件 × 1,000 agent | 已排队 |
| P9 | Entry 校准（11-07，Eshwar 视角）：人类锚点 IRR 1.67。10 个开源模型全部跑完：H11 10/10、H13 10/10、H12 ρ=+0.67 p=.021，三条预注册假设都成立。结果 `results/p9/summary.md`，图 `results/p9/fig_entry_vs_return.png`；NARRATIVE E19 | **完成** |
| P10 | "For whom"（11-07，Eshwar 视角，CPU 即可）：`hetero_history.py` 四站 agent 斜率 + r/science 人类；`h14_analyze.py` 在结局取回后跑。NARRATIVE E21 | agent 端完成；人类复制待结局 |
| P6 | 版主研究（论文的 HCI 部分）：拿 "StickyLab" 原型（同一段 sticky 文字给出多模型预测分布和人类锚点），对 10–15 位版主做半结构访谈 | IRB；招募 | 需要写 protocol，交给你提交 IRB |

## 2. API 预算（OpenRouter 上限 $10，已用 $9.09）

- 剩余约 $2.35，只给新站点用：
  - Jev：全量 off/full，约 $0.3 一个站点；
  - gpt-4o-mini：每个站点 800 个 agent × 5 个条件，约 $0.9 一个站点。
- 要加 Claude Haiku 或 Gemini Flash，需要你把上限调高。

## 3. 每次检查（可以手动做，也可以让我按这个顺序做）

1. `tail results/queue/runner.log; cat results/queue/state.tsv`：失败的任务看日志、修好之后，换个名字重新追加到 `jobs.txt`。队列 v2（11-06 19:54）：`1:G` 表示可以和别人的任务共用一块还剩 ≥ G GiB 的 GPU（≤ 8B 的模型用 `1:20`，{util} 自动填）；启动时显存被别人抢走记为 `busy`，自动重试最多 6 次；重启 runner 时会接管 `running.tsv` 里还在跑的任务。**11-06 21:10：** runner 自动避开你自己 `vllm serve` 所在的 GPU（目前是 3 和 6），也可以在 `results/queue/AVOID` 里手动写要避开的 GPU。`/mnt/disk1` 上的 Ministral-3 checkpoint 已经被删了，所以 Ministral-3-8B 重新下载到了 `/data` 的 HF cache（`download_ministral.sh`，同一个 repo，权重相同）；Ministral-3-3B 和 14B 从面板里去掉了（家族内的规模梯度已经有 Qwen 1.7/4/8/14/32B 和 Gemma 4/12/27B）。
2. `python3 src/panel_summary.py` 更新 `results/panel/summary.md`。
3. 新站点的 `inputs.parquet` 生成后，把 P2 的面板任务追加进 `jobs.txt`。全部完成后，用 `freeze.py` 把分数写进 `results/sites/FREEZE.txt`，记录名写成 `scores <site>`。
4. 把新数字写进 NARRATIVE.md 的证据表，每一条都标明是"前瞻"还是"回溯"。
5. 任何站点的 outcome 都不在冻结之前运行。

## 4. 时间线（需要按你的课程和投稿日期调整；会议日期待核实）

| 周 | 内容 |
|---|---|
| 11-06 → 11-13 | P1 和 P2 跑完；Futurology OSF 提交并揭晓；P3 开始 |
| 11-13 → 11-27 | P3、P4；四个站点的揭晓；论文 Results 初稿；P5 可行性检查 |
| 11-27 → 12-18 | P5；P6 的 IRB protocol 提交；Intro、Related Work |
| 12 月 → 1 月 | 版主访谈；投稿。ICWSM 2027 的 1 月轮（待核实），或者 CSCW 的下一轮（待核实）；thesis 章节同步写 |

## 5. 风险和备选

- **GPU 被占满。** P1 的 8B 级模型可以压到 1,500 个 agent 的子样本，方向判定只需要几百个 agent。
- **新站点抓取太慢。** 先只做 ffxiv（它的 sticky 文本和规则结构与另外两个站点差别最大，信息量最大）。
- **揭晓后 H1 不成立**（8B 的 gap 不显著）。这条照实报告，R1 和 R3 才是主结论。H1 不成立说明 8B 碰巧更接近人类，但方向依然取决于选了哪个模型。
- **P3 发现方向随 prompt 翻转。** 论文改写成"模拟设计评估连模型内部都不稳定"。结论更强但更负面，需要配一个可操作的建议，比如多 prompt、多模型的 ensemble，再加上用人类锚点校准。

**11-07 15:05 OLMo-3.1-32B 从面板中移除。** vLLM 0.19.1 + fp8 加载后只生成 "............"（`results/queue/DBG-olmo.log`：前 20 个首 token 都是噪声，日志里有 rope_parameters beta_fast/beta_slow 类型解析警告），Yes/No 读出质量为 0。bf16 需要两张空卡（tp=2），每次加载约 45 分钟，性价比低。无效分数移到 `results/futurology/_invalid_olmo3.1-32b_mass0/`，没有进入任何汇总或冻结。论文写明：面板里唯一的完全开放数据模型因推理栈问题没有跑成。


## 6. 11-07 晚：GLM-4-32B

Futurology 分数 mass = 0（同 OLMo），已移到 results/futurology/_invalid_glm4-32b_mass0/，r/science 部分按 PID 停掉；P2f/P2s-glm4 暂停（HOLD）。DBG-glm4 结果（17:34）：fp8 下贪心生成是乱码（"天下负重久…"、top-1 logprob 约 −3.3），**不是** readout 问题，和 OLMo 同类（vLLM 0.19.1 在线 fp8）。已排 DBG-glm4-bf16（TP=2，不量化）。若 bf16 正常：P1/P2 的 GLM 改成 `--tp 2` 不加 `--quant` 重跑，论文写明 GLM 用 bf16；若仍乱码：GLM 退出 panel，与 OLMo 一起在附录列为"无法在本栈运行"。

11-08 01:06 更新：DBG-glm4-bf16c（`tp_wrap.py`：NCCL_P2P_DISABLE + disable_custom_all_reduce）**正常生成**：bf16 的 GLM 输出 "\nNo"，首 token 为 "\n"（p≈1），其后 No/Yes。所以 GLM 有两个问题：fp8 在本栈损坏（同 OLMo）+ 先输出换行（读法问题）。修复：bf16、TP=2、`GEN_SUFFIX="\n"`（在 assistant 前缀后补一个换行，各条件相同），任务 P1/P2f/P2s-glm4-32b-bf16 已排队；Futurology 分数出来后先查 mass，再冻结，P2 才会启动。论文附录写明这是 GLM 专用的偏差。TP=2 修复同时用于 Qwen2.5-72B（`-w` 任务）。

11-08 05:15：Qwen2.5-72B（在线 fp8、TP=2）ffxiv 分数 mass = 0，移到 results/sites/_invalid/；P2s/P1 fp8 任务暂停。本栈上在线 fp8 已在 3 个模型上失败（OLMo-3.1-32B、GLM-4-32B、Qwen2.5-72B），在 Qwen3-32B、Gemma-3-27B、Phi-4、Mistral-Small-24B 上正常 → 论文写明"每个模型先过 readout mass 检查"。改用官方 Qwen2.5-72B-Instruct-AWQ（下载中，`download_qwen72awq.sh`）；DBG-qwen72awq 通过后手动 touch `.ok_qwen72awq` 才会跑 P1/P2。queue_runner 改为：多卡任务放不下时跳过而不是 break（避免 1 卡任务被挡住；代价是多卡任务要等 1 卡任务排空）。runner 新 PID 908334。Mistral-Small-24B 新站点（探索性，不在 H8 预测集）：ffxiv −0.27 [−0.29, −0.25]、science2020 −0.26 [−0.27, −0.24]。

11-08 15:50：P4 扩到 4 个模型（`src/p4_summary.py` → results/ablation/summary.md，探索性），见 NARRATIVE E23：3/4 模型方向与句子内容无关，措辞排序跨模型不可移植（平均 Spearman ≈ 0）。队列：P5 的 4 个单卡任务暂时 `# HOLD(2gpu first)`，因为单卡任务一直填满 MAX_GPUS=3，双卡任务（P2s-glm4、P9-glm4、DBG-qwen72awq）会饿死。P2s-glm4 跑完后才能做 sites 的 scores freeze 和正式 H8/H9。双卡任务启动后再取消 HOLD。

11-08 17:50：P1-mistral-small-24b 完成（mass 0.993），Futurology 分数已 freeze 10。完整 sticky 在 4 个站点都为负：r/science −0.17、Futurology −0.25、science2020 −0.26、ffxiv −0.27（符号和幅度都很稳，像 Gemma 的「一律负」读法）；模板回复 +0.62–0.85。scorecard：sticky gap −0.15 [−0.36, +0.04]，1/1 匹配（P5 未跑）。for-whom 斜率（临时计算，冻结的 csv 未覆盖）：r/science +0.09 [+0.06, +0.12]，和人类同号；ffxiv −0.09、science2020 +0.06，跨站不稳。P4 扩到 6 模型，见 E23。P5 单卡任务已恢复（双卡任务在文件中排在前面；1 单卡 + 1 双卡 = 3，不会饿死）。

**11-08 18:05 P4 读法预测（在 P4-mistral-small-24b / P4-glm4-32b 运行之前写下；探索性，不是 OSF 预注册，但时间戳在结果之前）：**
- Mistral-Small-24B 属于读法 A：两个站点的 17 个消融条件里，至少 15 个 CI 在 0 以下、0 个在 0 以上；消融极差 < 0.5。
- GLM-4-32B 属于读法 C：17 个条件中至少 14 个 |Δ| < 0.25；极差 < 0.5。
- 若不成立，§5 的分类表按结果改写，并在论文里报告这次失败的预测。

## 7. 11-11 暂停（项目停在这里；重启说明见 HANDOFF.md）

- 11-09 跑完：P2s-glm4-32b-bf16-r2（science2020，mass 0.999）、P4-mistral-small-24b（读法 A 预测成立）、P7-gen-glm4-32b（生成完成，judge 未跑）。
- 11-09 失败：DBG-qwen72awq（加载时 OOM，GPU 被别人占用）；P4-glm4-32b（RPC sample_tokens 超时，命令里没加 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600）。
- 11-11：freeze "scores ffxiv2019" 与 "scores science2020"；正式 H8 14/16 = 87.5% 成立，H9 两站都成立（results/sites/h8_h9_formal.md）。
- 11-11：queue runner（PID 908334）已停，GPU 全部释放。
