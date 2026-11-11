# 研究思路梳理（随结果持续更新；最后更新 2026-11-08）

执行计划和判定标准见 [ROADMAP.md](ROADMAP.md)，数字的出处见 `PROBES.md` §9–12 和 `results/panel/summary.md`。

## 1. 一句话

用真实 newcomer 的历史构建 LLM agent，再拿它们去预测 Reddit 上真实的 field experiment。结果是：**同一段版规 sticky，不同模型给出相反的设计建议。每个模型在不同社区之间都保持自己的方向（预注册跨站预测 14/16），而真实人类的效应接近 0。**

**11-08 的三层版本（按 Eshwar 的"校准到真实效应 + 对谁有效"）：** 人类对这个 sticky 的反应有三层——(a) 回不回来：平均 ≈ 0；(b) 这次说不说话、守不守规矩：真实的非零效应（评论数 IRR 1.67、违规 −0.44 logit）；(c) 对谁：平均 0 掩盖了"伤薄历史新手、帮老手"（treat×history p=.004，待 Futurology 预注册复制）。没有一个模型三层都对：10/10 低估 (b) 的参与效应（H11/H13），合规只有 Qwen3 对，(c) 只有 3/13 同号；而且模型在 (a) 上的错误方向能预测它在 (b) 上的错误方向（ρ=+0.67，H12）。**结论不是"模型不够好"，而是模型对一条治理文字只有一种总体读法（威胁/欢迎），人类则在不同决策、不同人群上分开反应。**

## 2. 问题和空位

**问题。** HCI/CSCW 越来越多地用 LLM agent 预先评估社区设计和治理策略，例如：
- Social Simulacra（Park et al., UIST 2022）用模拟的社区帮设计者预演规则；
- COSMOS（Fidone et al., AAAI 2026）用反事实模拟得出"个性化的治理消息更能降低毒性"；
- SynDisco 等工作用 LLM 用户加 LLM 版主模拟讨论。

这些结论几乎都没有和真实的因果效应对照过。COSMOS 自己承认：治理效果没有用真人或 field experiment 验证过。

**最接近的验证工作，以及它们和本项目的区别：**
- **Hewitt / Ashokkumar et al. 2024**：GPT-4 能预测问卷实验的效应，r 约 0.85。但那是问卷情境，预测的是总体平均效应。
- **Park et al. 2024**（1,000 人的 generative agents）：复现的也是问卷和经济博弈。
- 两者都不是平台上的真实行为，也不是同一批人在同一个分组下的反事实。
- **Goyal, Wang, Chandrasekharan & Sundaram 2026**（arXiv 2604.03920，"From Plausible to Causal"，✓ 11-07 读过 p.1–5）：立场论文，主张 "believability is not causality"，模拟得到的效应只是 "simulator-conditional causal estimates"，政策意义取决于模拟器保真度，并呼吁拿真实干预结果做校准。**本项目就是这件事的实证版**：同一个 RCT、同一批人，测出各模型的 simulator-conditional 效应离真实效应多远。这是导师组自己的论文，related work 里必须正面引用，并把本文定位成对它的回应（而不是只引 Park / Hewitt）。

**空位。** CivilServant 的 sticky comment 实验提供了一个现成的 benchmark：
- Matias 2019 PNAS 的 r/science 2016 实验，以及 2019–2020 年的三次复制（r/ffxiv、r/Futurology、r/science）；
- 有真实的 newcomer、真实的分组、公开的历史数据；
- 我们可以让同一批人、在同一个分组下，先由 agent 预测，再和真人对照。

## 3. 方法要点

- **Agent 构建。** 每个 agent 对应一名真实 newcomer：过去一年的历史、进入的帖子、自己的评论、收到的真实回复。问题是"未来 3 个月还会在 r/X 发帖或评论吗"，读第一个 token 的 P(Yes)。
- **三个估计量：**
  - **组间（arm effect）：** agent 用它所在的真实分组的 P，人类用真实回访，两边跑同一个回归。gap = agent − human，用按帖子聚类的配对 bootstrap。
  - **个体内：** 同一个 agent 在 off、full、welcome、rules、模板欢迎语五个条件下分别作答。这样可以把一段 sticky 拆成组件，看模型在对哪一部分起反应。
  - **前瞻：** Futurology 的 agent 分数在抓取人类 outcome 之前冻结（SHA-256 记录在 FREEZE.txt），并做 OSF 预注册。新站点照同样流程。
- **多模型面板：** 约 18 个模型，覆盖 Qwen、Llama、Gemma、Mistral、Phi、OLMo、GLM 七个开源家族，加上 gpt-4o-mini 和 Jev，参数规模从 1.7B 到 72B。

## 4. 证据表

| # | 结论 | 证据 | 性质 | 强度 |
|---|---|---|---|---|
| E1 | 人类：sticky 对 newcomer 的 90 天回访没有效应 | r/science 2016：Δlogit −0.03 [−0.23, +0.17]，TOST p = .016 | 回溯，真实随机分组 | 强 |
| E2 | 设计建议取决于模型 | r/science 的完整 sticky：8B −1.25、32B +0.77、gpt +0.47、1.7B −1.97、Jev +0.01 | 回溯 | 强（CI 很窄） |
| E3 | 每个模型在不同社区之间保持方向 | Futurology 上 H6 的 5 条全部成立，H7 成立（8B −0.60 vs 32B +0.61）。跨站点相关：4 个模型 r = 0.95 → 6 个 r = 0.83 → 8 个 r = 0.93 → **11 个 r = 0.94**（11-06 21:50）。8 个模型里 7 个在两个站点的符号一致（gpt-4o-mini 是例外：Futurology 0，r/science +）。Gemma-3-4B 两站都强负（−2.50 / −3.70），Llama-3.1-8B 两站都正（+0.39 / +0.28） | **前瞻**（分数揭晓前冻结） | 强；P8 swap 拆分人群和文字 |
| E4 | 方向翻转来自对不同组件的反应 | 8B：rules −0.78 / −1.03，welcome +0.54 / +0.90；32B 两者都反过来 | 回溯 + 前瞻 | 强 |
| E5 | 通过检验的理由不对；没通过的模型错的方向也不一样 | gpt 和 32B 在 r/science 上 gap 不显著，但个体内效应明显不为 0（组件互相抵消）；Jev 对什么都不敏感。9 个模型里 4 个 gap 的 CI 不含 0：Llama-3.1-8B **+0.38**（高估），Qwen3-8B −0.50、Gemma-3-4B −0.52、Qwen3-1.7B −1.79（低估） | 回溯 | 中 |
| E6 | 对模板欢迎语过度反应**只是部分模型的特点，不是普遍规律**（R4 按事先标准不成立） | 模板欢迎语 > +1 logit：gpt +5.6 / +5.8、Qwen3-8B +2.1 / +3.2、Qwen3-32B +1.3 / +2.6、Gemma-3-12B +0.5 / +2.2、Qwen3-1.7B +1.2；≤ +1：Ministral-3-8B +0.73 / +0.85、Jev ≈ 0、Llama-3.1-8B −0.21 / −0.06、Gemma-3-4B −0.81 / +0.08。人类锚点：被高赞之后发帖率约 +2%/天（Lambert et al. CHI 2025）；我们自己的 r/science 2016 P5 人类 ATT −0.14 [−0.40, +0.15] | 回溯 | 中；论文里改写成「正向线索敏感度也是模型特有的」 |
| E7 | 预注册主假设：8B 的 gap < 0 | 待 OSF 提交后揭晓 | 前瞻 | – |
| E8 | 第 3、4 个站点 | ffxiv2019、science2020 | 前瞻 | 进行中 |
| E9 | ~~方向不受测量方式影响（对效应大的模型成立；效应小的模型会掉到 0，但不反向）~~ **11-07 更正：测量方式能让方向反转**。Gemma-3-12B 在 r/science 上换问法（q2）后从 −0.39 变成 +0.70（都显著）；Futurology 上 q2 也把 −0.73 拉到 +0.06。4 个模型的方差分解：model 42%、variant 10%、site 0.2%、残差 47%（之前 2 个模型时 model 67%）。所以「方向由模型决定」要改成「方向由模型 × 测量工具决定」；observer 框架对 Gemma 是 −2.51 / −3.32，放大而不是缩小 | P3 prompt 多元宇宙（换问法 q2、不用 system、选项顺序、第三人称 observer）。Qwen3-8B 两站：8/8 个变体保持负号。Llama-3.1-8B：r/science 3/4 保持正号（observer → +0.09）；Futurology 只有 1/4（q2 +0.07、nosys +0.21、observer −0.05 掉到 0）。16 个变体里没有一个反向；按 0.25 阈值算保持率 12/16 = 0.75。方差：model 67%、variant 11%、site 0%。第三人称 observer 常让效应变小（Qwen3-8B r/science −1.27 → −0.26），角色扮演可能放大效应。**11-08 加入 Qwen3-32B（5 个模型 × 5 变体 × 2 站）**：方差模型 53%、变体 8%（p=.09）、站点 0.2%、残差 38%；保持基准符号的变体 31/40 = 78%。Qwen3-32B 的完整 sticky 在 10 个格子里 9 正 1 零（Futurology observer +0.15），是最稳健的"欢迎"读法 | 回溯 + 前瞻 | 中–强；还有 3 个模型 **11-07 Ministral-3-8B**：base 两站都 ≈0，换问法（q2）后两站都变成显著负（r/science −0.71，Futurology −0.28）——对接近零的模型，测量方式本身就能制造出一个「效应」。 |
| E10 | 方向是模型的、不是人群的；文字只调大小（有的模型连文字都不读）。**11-08 加入 Qwen3-32B（5 个模型）**：模型 83%、人群 0.3%、文本 1.2%、残差 15%。同族反向：Qwen3-32B 文本 − 占位 = +0.57 至 +1.62（r/science、Futurology 两个人群都是），Qwen3-8B = −0.20 至 −1.41；占位（空白置顶）两者都略负（−0.20 / −0.15）。即同一家族、同样四段文字，读法随规模翻转（与 E14/E19 的 8B→14B 翻号一致） | P8 swap（同一批人依次看 4 个站点的 sticky，社区名换成自己的，再加一个无内容置顶）。**Qwen3-8B**：4 段文字在两个人群里全部为负，ffxiv 文字 −1.33 / −1.56 最强、Futurology 文字 −0.56 / −0.62 最弱；无内容置顶 −0.35 / −0.15。**Llama-3.1-8B**：4 段文字全部 +0.28 到 +0.41，几乎一样；无内容置顶 +0.16 / +0.17。~~也就是说 Llama 只对「版主置顶了一条评论」起反应，不读内容~~（11-07 更正：探索性分析 `swap_content.py` 以无内容置顶为基线，Llama 的 4 段文字仍各加 +0.12 到 +0.24，CI 都不含 0；所以 Llama 读了文字，只是不区分站点）。**Ministral-3-8B**（11-07 加入）：r/science 人群里无内容置顶 −0.30，4 段文字都 ≈ 0（−0.04 到 +0.06）；Futurology 人群里文字 +0.16 到 +0.26，置顶 −0.08。也就是说对 Ministral 而言，文字内容相对置顶本身是 +0.3 左右，方向和 Qwen3-8B 相反。3 个模型 × 2 个人群 × 4 段文字的方差：model 88%，population 0.3%，text 3%。**11-07 加入 Gemma-3-12B**：无内容置顶 −0.78 / −0.79，文字相对置顶都为正（+0.03 到 +0.79），ffxiv 文字最正——而对 Qwen3-8B，ffxiv 文字最负。4 个模型的方差：model 84%，population 0.2%，text 0.1%。文字相对置顶本身（`results/panel/swap_content.md`，探索性）：Qwen3-8B −0.20 到 −1.41，Llama +0.12 到 +0.24，Ministral +0.24 到 +0.35；同一段文字的符号跟着模型走，三个模型的「置顶本身」和「文字内容」两个分量可以符号相反（Ministral：−0.30 vs +0.3）。自身站点格子复现了面板数字 | 回溯（只用 agent 端） | 强（3 个模型）；还有 2 个排队 |
| E11 | 人类对三种线索都没反应 | r/science 2016 的 newcomer：sticky（RCT）−0.03 [−0.23, +0.17]；第一条评论收到回复 vs 没收到，90 天回访 15.3% vs 15.1%，raw logit +0.02 [−0.20, +0.23]；评论得分（P5，倾向得分 ATT）−0.14 [−0.40, +0.15]。Reddit Newcomers 主项目的回复效应同样 ≈ 0（OR 1.07 [0.76, 1.51]） | 回溯；回复和得分是观察性的 | 中（三条一致） |
| E15 | 对正反馈（高分）过度反应：同一个准实验设计下 agent 和人差两个数量级 | P5，r/science 2016，第一条评论得分 ≥ 4 vs ≤ 1，ATT 加权后平衡（|SMD| ≤ 0.024）。人类 ATT −0.14 [−0.40, +0.15]；Qwen3-8B（看到真实得分）+2.26 [+2.05, +2.46]，gap **+2.39 [+2.00, +2.75]**；Qwen3-8B 个体内 50 分 vs 1 分 +2.13。**Llama-3.1-8B 几乎不反应**：ATT +0.04 [−0.01, +0.08]，gap +0.18 [−0.12, +0.45]（不显著），50 vs 1 分 +0.14。Llama 对所有线索都只有小的正反应（sticky 不论文字 +0.3、无内容置顶 +0.16、高分 +0.14、模板欢迎语 ≈ 0），像 Jev 一样「不敏感」，所以它在 P5 上「通过」不等于模拟得好。其余 3 个模型排队中 | 回溯；人类端是准实验（Lambert et al. 2025 的设计） | 强（Qwen3-8B）；模型间差异本身就是结论 |
| E16 | 机制（Qwen3-8B）：负向来自「删除」类句子；描述性规范假设不成立；不是训练数据污染 | P4 句子消融，1,000 个 agent。r/science：去掉删除威胁 T 后完整 sticky 从 −1.26 变成 −0.12；只留 T 是 −0.69，只留 welcome +0.54，版主人数 ≈ 0。Futurology：除 welcome 外每句单独都为负，包括描述性规范「thousands … follow the rules」（−0.75）和「Replies … auto-removed」（−0.91）；只留 welcome +0.98。知识探针：Qwen3-8B 认为 Matias 的研究报告了「Increased」（P ≈ 1）；专家探针：它自己预测「Lower」（0.92–1.0），和它的模拟 agent 一致，和它「知道」的文献结论相反 | 回溯（agent 端） | 中；32B 的探针是关键对照 |
| E17 | 在一个杠杆上和人类一致说明不了什么：通过 sticky 检验的模型，对别的线索照样过度反应 | 记分卡（`results/panel/scorecard.md`，r/science 2016 同一批人）：12 个模型里 7 个的 sticky gap CI 含 0，它们的平均敏感度从 Jev 0.05 到 gpt-4o-mini 3.05；gpt / Qwen3-14B / Qwen3-32B 通过 sticky 检验，但对一句通用欢迎回复是 +5.6 / +3.5 / +1.25 logit（人在同一批人里收到真实回复的关联 ≈ +0.02）；最不敏感的 Jev、Llama 部分是因为对什么都几乎不反应才通过。**11-08 加入 Qwen3-32B**：upvotes ATT +1.42 vs 人类 −0.14，gap +1.56 [+1.17, +1.92]（失败）；sticky gap +0.15 含 0（通过）→ 1/2。5 个有 P5 的模型：Llama、Ministral、Gemma-12B、Qwen3-32B 都是 1/2，Qwen3-8B 0/2；**仍然没有模型两项都过** **11-08 19:10 加入 Qwen3-14B：** within（50 分 vs 1 分）+4.63 [+4.50, +4.77]，ATT +3.09，与人类的差距 +3.23 [+2.57, +4.03]，不通过；sticky 匹配，所以 1/2。有 P5 的 6 个模型中，只有 Llama-8B 的 upvote 差距 CI 含 0（它对 upvote 几乎没有反应，within +0.14），其余 5 个都高估 upvote 的作用，且没有模型两项都过。人类 ATT −0.14 [−0.40, +0.15]。 **20:20 加入 Phi-4：** upvote within +1.91，差距 +1.90 [+1.52, +2.26]，不通过；sticky 差距 −0.40 也不通过，0/2。7 个模型中 6 个高估 upvote，仍然没有 2/2。 **21:50 加入 Gemma-4B：** upvote within +1.10，差距 +0.47 [+0.09, +0.84]，不通过；sticky 差距 −0.52 也不通过，0/2。8 个模型中 7 个高估 upvote（唯一通过的 Llama 几乎不反应），仍然没有 2/2。 **11-11 加入 Qwen3-4B：** upvote within +9.01，差距 +2.70 [+2.36, +3.04]，不通过；sticky 差距 −0.41 也不通过，0/2。9 个模型中 8 个高估 upvote，没有 2/2。 | 回溯 | 强；P5 列还在补  **11-07 更新（P5 3 个模型）**：有两个 gap 的模型里没有一个两项都通过——Llama 过 upvotes（+0.18 n.s.）不过 sticky（+0.38）；Ministral 过 sticky（−0.06）不过 upvotes（+0.71 [+0.41, +0.97]）；Qwen3-8B 两项都不过；Gemma-3-12B 过 sticky（−0.51 n.s.）不过 upvotes（+1.55 [+0.85, +2.45]）。4 个模型里 0 个两项都过。|
| E18 | **预注册的跨站预测成立（中期计数已锁定 ≥ 81%，只用 agent 端）** | H8：r/science 与 Futurology 同号且 |Δ|>0.25 的模型（预测集 8 个：Gemma-3-4B/12B/27B、Qwen3-4B/8B 为负；Llama、Qwen3-14B/32B 为正），在新站点预测同号。预测集 8 个模型 × 2 站 = 16 格全部完成，14 个成立（**14/16 = 87.5%**，门槛 80%）。两个失败都在 ffxiv2019，都是**缩到 0 而不是翻号**：Gemma-3-12B −0.03 [−0.11, +0.05]、Qwen3-14B +0.06 [−0.02, +0.14]。成立的：ffxiv2019 Gemma-3-27B −2.53 [−2.76, −2.30]、Gemma-3-4B −0.93、Qwen3-4B −0.97、Qwen3-8B −1.18、Llama +0.31、Qwen3-32B +1.69；science2020 Gemma-3-27B −0.29 [−0.35, −0.23]、Gemma-3-4B −1.85、Gemma-3-12B −0.71、Qwen3-14B +0.38、Qwen3-4B −0.75、Qwen3-8B −0.82、Llama +0.42、Qwen3-32B +0.59。H9：模型间极差 ffxiv2019 2.87、science2020 2.44，门槛 1.0。不在预测集里的：Ministral 两站 ≈0（+0.12 / +0.04）、Jev ≈0、gpt-4o-mini ffxiv +0.08。预测集已全部计入；Mistral-Small-24B 与 Qwen2.5-72B 的新站点分数跑完后与其它分数一起冻结（"scores <site>"），之后才算正式值。**反例要写进正文**：Phi-4（不在预测集，r/science −0.23 未过 0.25 门槛、Futurology −0.39）在两个新站点都转为正（ffxiv +0.26 [+0.21, +0.31]、science2020 +0.06 [+0.02, +0.10]）——弱效应的模型方向并不由模型决定，"模型决定方向"只适用于效应大的模型。跑完后冻结再算正式值，结局未取 **11-11 正式结果（freeze 'scores ffxiv2019' / 'scores science2020' 之后，`src/h8_h9_formal.py` → results/sites/h8_h9_formal.md）：H8 14/16 = 87.5%，成立（阈值 80%）**；两个未中都在 ffxiv（Gemma-12B −0.03、Qwen3-14B +0.06，缩到 ≈0，没有反号）。**H9 成立**：ffxiv 极差 4.22（Qwen3-32B +1.69 到 Gemma-27B −2.53），science2020 极差 2.44。偏离：Qwen2.5-72B 剔除（fp8 mass 0，AWQ 加载 OOM）；gpt-4o-mini 未在 science2020 打分（预算）。 | 前瞻（agent 端已定，人类结局封存） | 中期 |
| E19 | **预注册的校准检验：10/10 模型都对不上真实的非零效应，且全部低估；每个模型对 return 和 entry 用同一种「读法」** | P9 entry（freeze 6 在任何分数之前冻结）。人类：sticky 让每帖 newcomer 评论数 IRR 1.67 [1.53, 1.82]，但对 return ≈ 0。agent 比值 full/off：Llama 1.32、Qwen3-32B 1.28、Ministral 1.22、Qwen3-14B 1.09（方向对、偏小）；Qwen3-1.7B 0.92、Gemma-3-12B 0.81、Gemma-3-4B 0.59、Qwen3-4B 0.46、Phi-4 0.46、Qwen3-8B 0.34（方向反）。**H11** 10/10 gap CI 排除 0 → 成立；**H13** 10/10 低估 → 成立；**H12** entry 对数比值与 return Δlogit 跨模型 Spearman ρ = +0.67，单侧置换 p = .021 → 成立。Qwen3 家族再次在 8B→14B 之间翻号（同 E14）。图：`results/p9/fig_entry_vs_return.png`（人类点在所有模型之外：return 0、entry +0.51 log）。解释：人类把「回不回来」和「这次要不要说话」分开处理；模型对 sticky 只有一种总体倾向（威胁 or 欢迎），两个决策都套用它。**设计缺陷**：比值受基线上限约束（Gemma-3-4B P(Yes|off)=0.93、Qwen3-1.7B 0.985，最多 1.08 / 1.02）；基线 0.14–0.99 远高于真实评论率，只能比比值。**预注册之外的扩展（11-08，单独报告，预注册判定以 results/p9/summary_prereg10models.md 为准）**：Gemma-3-27B 比值 0.14 [0.10, 0.18]（P(off)=0.11，方向反且最极端，与其 return −0.26 同向）；Mistral-Small-24B 比值 0.98 [0.96, 0.99]（P(off)=0.40；近乎不敏感，与其新站点 return −0.27 / −0.26 同向，r/science return 在跑）；GLM-4-32B 在跑 **11-08 19:40 扩到 13 个模型（预注册的 10 个 + Gemma-27B、Mistral-Small、GLM-4-32B；预注册版保留在 summary_prereg10models.md）：** GLM ratio 1.26 [1.23, 1.29]、Mistral-Small 0.98、Gemma-27B 0.14，**13/13 都低于人类 IRR 1.67 的下限 1.53**；最接近的是 Llama 1.32、Qwen3-32B 1.28、GLM 1.26。「一种读法」更明确了：13 个模型的 return within Δ 和 entry within Δ 的 Spearman ρ = 0.96（Pearson 0.93），12/13 同号（唯一例外 Ministral：return −0.04 ≈ 0、entry +0.30）。GLM 是最接近人类「拆开」模式的模型：return 几乎不动（+0.06，低于 0.25 阈值），entry ×1.26（within +0.46）；人类是 return ≈ 0、entry ×1.67。但它的 entry 仍然低估，for-whom 斜率（−0.00）也对不上（P7 合规未跑，已排队），而且它对治理文字整体不敏感（E22），所以更像「对什么都反应弱」，而不是读懂了 sticky。 | RCT（预注册 addendum） | **强**（10 个模型，3/3 假设成立）；API 模型未跑（预算） |
| E20 | **保真度是「按杠杆」的：在 return / entry 上最差的模型，在合规上反而最像人**（两个不同家族的 judge 一致） | P7：agent 看到/没看到 sticky 后写评论，judge 按 sticky 列出的规则打 P(删除)。人类（block FE GLM）Δlogit(删除) −0.44 [−0.80, −0.07]。Qwen3-8B 生成：Qwen3-32B judge −0.45 [−0.60, −0.30]、Gemma-3-27B judge −0.55 [−0.70, −0.40]；Qwen3-32B 生成：−0.24 / −0.31（同向）；Gemma-3-12B +0.02 / +0.10、Llama +0.07 / +0.02（≈0）；Ministral +0.01 / −0.11（≈0 到很小），只看规则时两个 judge 都反向（+0.32 / +0.20）。同族偏好解释不成立：Gemma judge 下 Qwen 仍然合规，而 Gemma judge 并不偏好 Gemma 生成的评论。**剩下的弱点**：judge 对真实首评（全部未被删）的假阳性高（P(删除) 均值 Qwen 0.24、Gemma 0.42），只能比 within-agent 差值；agent 群体是在合规上被选择过的（被删者丢失作者），所以只比方向和大小。和 E17 / E19 合起来：Qwen3-8B 在 return（−1.25）和 entry（0.34×）上都错，在合规上对；Llama 在 entry 上方向对、合规上 ≈0。**没有一个模型在三个杠杆上都对** | RCT | **强**（5 个生成模型 × 2 个 judge） |
| E21 | **"对谁有效？" 平均为 0 的 sticky 在人类里可能是"伤新手、帮老手"；多数模型预测了相反的异质性**（Eshwar 式问题，已预注册复制） | r/science 2016，按 newcomer 过去一年的 Reddit 活动量（n_hist，最多 100 评论 + 100 帖）三分位：人类 T1 −0.30 [−0.67, +0.04]、T2 −0.04、T3 +0.18；斜率 T3−T1 +0.48 [+0.02, +0.98]；连续交互（n_hist>0，log 标准化，post 聚类）treat:z +0.275 [+0.074, +0.477]，单侧 p = .004；四分位单调（−0.34、−0.20、+0.02、+0.31）。agent：13 个模型中 9 个斜率 CI 排除 0，但只有 3 个与人类同号（Gemma-3-27B +0.17、gpt-4o-mini +0.39、Qwen3-8B +0.18），6 个反号（Phi-4 −0.41、Qwen3-4B −0.52、Qwen3-1.7B −0.29、Qwen3-14B −0.24、Ministral −0.15、Gemma-3-12B −0.10）。模型自己的斜率跨站也不稳定（gpt-4o-mini：r/science +0.39、Futurology +0.08、ffxiv +1.03）。**弱点**：发现于已打开的数据（一次切分，事先点名但不是预注册）；n_hist 有上限 200，可能代理账号年龄或社区重叠。**补救**：写成 H14/H15（PREREG_addendum_H14.md，freeze 8，任何新站点/Futurology 结局之前）；Futurology 是主检验 | 发现（RCT 内的调节，探索性）+ 预注册复制待做 | 中（发现）；复制待定 |
| E22 | **第三种读法："无视治理文字，只对社交回应反应"**（GLM-4-32B 加入后成形） | GLM-4-32B（bf16，补换行读法，freeze 9）：完整 sticky r/science +0.06 [+0.04, +0.08]、Futurology +0.03；"欢迎回复"模板 +0.68 / +0.59；history 斜率 −0.00 [−0.03, +0.02]。与 Jev 同型（sticky +0.01、模板 +0.09）。两者在 scorecard 的 sticky gap 上都"匹配"人类（+0.08、+0.03，CI 含 0），**但这是靠不敏感匹配的**，不是读对了：P9 entry 的人类效应 IRR 1.67，这类模型按构造给不出来（待 GLM 跑 P9 确认）。于是 panel 有三种读法：威胁（Qwen3-4B/8B、Gemma、Phi-4、Mistral-Small）、欢迎（Llama、Qwen3-14B/32B、gpt-4o-mini）、无视（Jev、GLM、Ministral 接近）。GLM 在 ffxiv2019 −0.23 [−0.25, −0.21]（略负、未过 0.25 门槛；探索性） | 回溯 | 中（2 个模型） |
| E23 | **多数模型不读 sticky 的内容；读内容的两个模型，同一句话读出相反方向**（P4，6 个模型，11-08 17:50） | 句子消融 × 6 模型（Qwen3-8B/32B、Gemma-12B/27B、Llama-8B、Ministral-8B），1,000 agent/站点，within-agent Δlogit。消融网格的方差：模型占 72%（Futurology）/ 47%（r/science），句子条件只占 6% / 2%。**4/6 对内容不敏感**：Gemma-27B 17/17 个条件都为负，极差只有 0.13–0.19；Gemma-12B 17/17 负；Llama-8B 17/17 正（只留删除威胁 +0.33 / +0.38）；Ministral 每格 |Δ|<0.3。**2/6 对内容敏感，都是 Qwen3，但机制相反**：Qwen3-8B 里 welcome 为正（+0.98 / +0.54）、删除威胁为负（−0.60 / −0.69）；Qwen3-32B 正好倒过来，只留删除威胁 +0.84 / +0.97，只留 welcome −0.08 / −0.74，去掉删除威胁后 r/science 从 +0.79 变成 −0.34。两者措辞排序的 Spearman 在 r/science 是 −0.77。所以「32B 把 sticky 读成欢迎」是错的：它把删除规则读成正面信号（「管得好的社区」）。知识探针不解释方向：「知道」Matias 结论（P(Increased) ≥ 0.66）的 4 个模型里，2 个正（Llama、Qwen3-32B）、2 个负（Qwen3-8B、Gemma-27B）。专家探针与模拟 agent 的符号在 11 个可判格中 7 个一致；Gemma-27B 在 Futurology 被直接问时预测 Higher（1.0），它的 agent 却是 −0.62。**对版主的含义**：用模拟器挑措辞不可移植。各措辞效果在模型两两之间的 Spearman 从 −0.94 到 +0.94，平均 0.15 / 0.21 **11-09 事先预测检验：** Mistral-Small 的消融 17/17 为负、0 个为正、极差 0.48，预测（≥15 负、0 正、极差 < 0.5）成立，归入读法 A。GLM 的 P4 因 TP=2 RPC 超时失败，读法 C 的预测未检验。 | 回溯（agent 端，探索性，未预注册） | 中：每个模型一次 prompt；措辞排序要在 P3 变体上复核 |
| E12 | agent 预测「谁会回来」不如几个计数特征；最好的两个模型带来一点计数和通用文本 embedding 都没有的信息 | 9 个计数特征的 logistic AUC 0.662 [0.640, 0.685]；再加 bge-small 的首条评论和历史 embedding 只到 0.666。单独用 agent 的 P：0.49–0.62。加进 counts：Qwen3-32B +0.026、Jev +0.022，其余 10 个模型 ≤ +0.009；加进 counts + text：Qwen3-32B +0.011，其余 ≤ +0.005（`results/panel/who_returns.md`） 。配对 bootstrap：Qwen3-32B 比 counts +0.026 [+0.012, +0.039]，比 counts + text +0.011 [+0.004, +0.019]；Jev 比 counts +0.022 [+0.008, +0.036]，比 counts + text +0.005 [−0.003, +0.013] | 回溯，**探索性**（没预注册） | 中：Qwen3-32B 的增益在 text baseline 之上仍然不含 0，但很小 |
| E13 | 多模型取中位数救不回来，取决于选了哪些模型 | r/science 2016：9 个模型的中位数 gap −0.06 [−0.33, +0.15]，90% CI [−0.27, +0.11]，不等价；去掉任意 1 个正向模型，中位数就跳到 −0.3；只用 7 个开源模型时中位数 −0.53，gap −0.50 [−0.67, +0.01]。已写成 H10 预注册（PREREG_addendum_H10.md，freeze 5 / S3），预期不成立 | 回溯 + 前瞻（3 个封存站点） | 中 |
| E14 | 同一家族里，规模越大越倾向「sticky 让人更想回来」；Qwen3 在 8B 和 14B 之间翻转符号 | 完整 sticky（r/science / Futurology）：Qwen3 1.7B −1.97 / –，4B −1.60 / −2.79，8B −1.25 / −0.60，**14B +0.27 / +0.66**，32B +0.77 / +0.61；Gemma-3 4B −2.50 / −3.70，12B −0.38 / −0.76（27B 待跑）。但跨家族不成立：Phi-4（14B）−0.23 / −0.39，Llama-3.1-8B 是正的。12 个模型时跨站点 r = 0.94 | 回溯 + 前瞻（Futurology 分数已冻结）  **11-07 检验预先写下的预测（Gemma-3-27B 比 12B 更正）**：Futurology 27B −0.60 [−0.66, −0.54] vs 12B −0.76，方向对、差距小（freeze 7 已冻结 27B 的 Futurology 分数）；r/science 27B −0.26 [−0.34, −0.19] vs 12B −0.38。**两站都比 12B 更正 → 预测成立**，但差距小（0.12–0.16）且仍是负号，所以 Gemma 家族里「更大更正」只是弱趋势，不像 Qwen3 那样翻号。27B 对模板回复 +1.96。 | 中；Gemma-3-27B 是事先可以写下的检验：预测比 12B 更正 |

## 5. 解释：为什么同一段文字，小模型读成"威胁"，大模型读成"好社区"

**工作假设。** Matias 的 sticky 是按社会规范理论写的，同时包含三样东西：
- 删除威胁："comments will be removed"；
- 描述性规范："thousands of people comment daily and follow the rules"；
- 版主人数："~1200 moderators"。

小模型可能只抓住了威胁，大模型可能把整段读成"这是一个被认真维护的社区"，所以更想回来。

**P4 的检验。** 按句消融，观察各个模型的方向翻转是不是集中在描述性规范那一句。

**11-07 更新（Qwen3-8B 的 P4）：** 8B 的负向主要来自删除威胁；Futurology 的描述性规范那句单独也是负的，所以「大模型把描述性规范读成好社区」至少在 8B 上看不到反面（8B 本来就是负的），要等 14B / 32B 的消融才能检验。

**11-08 结论（P4 扩到 6 个模型，E23）：原假设被否定。** Qwen3-32B 的正向不来自描述性规范或 welcome，而来自删除威胁本身：只留 T 时 +0.84 / +0.97，只留 welcome 时 −0.08 / −0.74。同一家族里，同一句「comments will be removed」，8B 读成「会被删，别来」，32B 读成「管得好，值得来」。另外 4 个模型根本不按句子反应。

**由此得到的「读法」分类（论文 5.1 可以直接用）：**
| 读法 | 模型 | 证据 |
|---|---|---|
| A. 一律负：任何置顶治理文字都降低回访 | Gemma-12B/27B、Mistral-Small（未消融但 4 站同号）；Gemma-4B、Qwen3-1.7B/4B 只有完整 sticky 为负，未消融，暂不归类 | 消融 17/17 同号；Mistral 4 站都 −0.17 到 −0.27 |
| B. 一律正：任何置顶文字都提高回访 | Llama-8B | 消融 17/17 正 |
| C. 无视治理文字，只对社交回应反应 | GLM-4-32B、Jev（Ministral-8B 接近） | sticky ≈ 0，模板回复 +0.6–0.7（E22） |
| D. 读内容，但语义因模型而异 | Qwen3-8B（welcome +、删除 −）；Qwen3-32B（删除 +、welcome −） | 消融极差 1.8–2.0，两者排序 ρ = −0.77 |

人类在这四种读法里都找不到：return ≈ 0，但参与和合规上升，而且对新手和老手方向相反（E21）。所以问题不是「哪个模型读对了」，而是每个模型对一段文字只有一种读法，人类则因决策和人而异。

**另一个解释：训练数据污染。** Matias 2019 报告过"sticky 提高了 newcomer 的参与"。大模型如果读过这篇论文，就可能把这个结论带进角色扮演。P4 会加一个知识探针：直接问模型是否知道这项实验和它的结果。如果知道的模型才翻转方向，这本身就是一条重要发现：模拟结果是在复述文献，而不是在模拟行为。

## 6. Eshwar Chandrasekharan 组的方法怎么用到这里

| 他们的做法 | 代表作（✓ 本次检索核实过；其余凭记忆，引用前核对） | 用在本项目 |
|---|---|---|
| 跨社区、大规模比较 | Chandrasekharan et al. CSCW 2018 "The Internet's Hidden Rules"（100 个 subreddit 的删帖）；Goyal, Lambert, Chandrasekharan CHI 2026 "The Language of Approval" ✓（100 个 subreddit、约 1,100 万条评论） | 从 1 个站点扩到 4 个；P5 扩到上百个社区，问模型能不能复现"哪些社区受影响更大"（社区层面效应的排序），而不只是平均效应 |
| 平台数据上的准实验 | Lambert, Saha, Chandrasekharan CHI 2025 ✓（分层倾向得分匹配 + DiD；newcomer 受正反馈的影响最大）；"You Can't Stay Here" CSCW 2017（ban 的效果） | 人类端的因果锚点不只靠随机实验。P5 照 Lambert 的设计估计正反馈对 newcomer 的效应，再和 agent 对"高分 / 感谢回复"的反应对照 |
| 正面治理 | Lambert et al. CSCW 2024（版主怎么看鼓励好行为）；Positive Queue 2025 ✓；modqueue 研究 Bajpai & Chandrasekharan 2025 ✓ | E6 和 E11：人类 newcomer 对正向线索（回复、得分）的回访几乎没反应，而多数模型反应很强，强弱和方向又因模型而异。所以**正面治理恰恰是模拟评估最不可靠的地方**，这一点可以直接和他们的方向对话 |
| 计算分析 + 版主访谈 + 做工具 | Crossmod CSCW 2019；modqueue 访谈 | P6：StickyLab 原型（多模型预测分布 + 人类锚点）+ 版主访谈。这是 HCI track 需要的 human-centered 部分 |
| r/popular 带来的 newcomer | Chan et al. ICWSM 2024 ✓（上过 r/popular 的帖子带来约 70 倍的 newcomer） | thesis 第 2 章（Reddit Newcomers）的直接前置工作 |

**他们的研究风格可以概括成三点：**
1. 问题来自版主的真实需要；
2. 用平台规模的数据加因果设计回答；
3. 最后落到一个给版主用的工具或设计建议上。

本项目目前只有第 2 点，第 1 和第 3 点要靠 P6 补上。

**11-07 用 Eshwar 视角（advisor skill，检索他约 60 篇论文）审一遍当前故事，他大概率会追问的四点：**

1. **"你拿哪个真实效应做校准？"** [inferred: goyal2026from p.2–4] 之前所有实验问的都是「会不会回来」，人类效应 ≈ 0，模型靠「什么都不敏感」也能过（E17）。
   sticky 真正的、他组论文反复引用的效应是 **合规 +8.4pp 和 newcomer 参与 +70%**（goyal2026language p.4 引 Matias 2019）。
   → 新增 **P9 entry**（「你会在这个帖子下评论吗？」），人类锚点在 CivilServant 原始数据上复现为 IRR 1.67 [1.53, 1.82]；
   设计、假设 H11–H13 已在跑之前冻结（freeze 6，`PREREG_addendum_P9.md`）。P7（合规）是另一个非零锚点。
2. **"对谁有效？"** [inferred: chandrasekharan2017you, lambert2025positive] 他总把 newcomer / 老用户 / 邻近社区分开。
   本项目只有 newcomer。可做的低成本补充：按 newcomer 的历史活跃度（n_hist 三分位）分层，看模型的效应异质性是否和人类一致——人类端的异质性可以在 r/science 2016 上直接估。
3. **"版主拿这个结果做什么？"** [inferred: chandrasekharan2020combatting p.143 —— 他主张评估要落到 moderator effort、user trust 上] 
   对版主的可操作结论是：**换一个模型，模拟给出的「sticky 有没有用」就会变号**，所以任何「policy wind tunnel」工具必须同时展示多模型分布和至少一个真实锚点。这正是 P6 StickyLab 的设计依据；P6 的访谈题应直接问「如果两个模型给出相反建议，你会怎么决定」。
4. **"你只测了坏的一面吗？"** [inferred: chandrasekharan2020combatting p.144, NSF CAREER] 本项目的干预本身（欢迎 + 规则）就是正面 + 惩罚的混合，P4 拆开后发现模型把「removal threat」读成主要驱动力（E16）。论文可以直接写：**模拟器系统性地高估了惩罚性措辞、低估（或乱估）了欢迎性措辞**——这和他的 CAREER 议程对话。

**标题按他的习惯（隐喻 + 冒号 + 精确结论）的候选：**
- "Same Sticky, Opposite Advice: Model Identity, Not Community, Determines How LLM Agents Respond to Moderation"
- "Ask a Different Model: Calibrating LLM Simulations of Online Governance Against a Field Experiment"

## 7. 论文结构（目标 CSCW 或 ICWSM；截稿日期待核实）

**题目候选：** *Same Sticky, Opposite Advice: Calibrating LLM-Simulated Newcomers Against Online Field Experiments*（Eshwar 式：意象 + 冒号 + 主张）；备选 *One Reading Fits All: …*

**RQ（按 unit 拆）：** RQ1 不同模型对同一干预给出的方向是否一致、是否跨社区稳定？RQ2 校准到真实非零效应（参与、合规）时是否对得上？RQ3 能否预测"对谁有效"？

1. **Intro：** 平台和研究者开始用 LLM 模拟预演治理设计（Goyal et al. 2026 指出 believability ≠ causality），但还没有和多个 field experiment 系统对照。
2. **Related work：** LLM social simulation 与预测实验结果；newcomer socialization 与 sticky 实验（Matias 2019）；正面治理。
3. **数据：** 四个 field experiment——r/science 2016（真实分组 RCT）；Futurology、ffxiv、science 2020（as-treated + 平衡检验）。
4. **Agent、估计量与预注册流程**（freeze 1–8；每个模型先过 readout-mass 检查：3 个模型因在线 fp8 失败被剔除或换权重）。
5. **Results：**
   - 5.1 RQ1：方向由模型决定（图 1 森林图；P8 文本 × 人群互换，模型解释 84% 方差）；预注册跨站预测 H8 14/16；反例 Phi-4（弱效应模型会翻号）。
   - 5.2 RQ2：校准失败但有结构（P9 entry H11–H13；P7 合规；scorecard：按杠杆的保真度）。
   - 5.3 RQ3：对谁（E21；H14/H15 Futurology 复制）。
   - 5.4 机制与稳健性：句子消融给出四种读法（§5 表；E23），污染探针不解释方向；prompt 多元宇宙（测量能翻号：Gemma-12B q2）。
6. **Discussion（写给版主、工具设计者、研究者）：**
   - 报告多个模型的预测分布，并标出跨模型分歧，不只给一个数；
   - 每个模拟工具都要锚定至少一个真实的非零效应（如 IRR 1.67）再谈预测；
   - 平均效应之外要检验异质性，因为新手保护正是版主最在意的"for whom"；
   - 正面治理：sticky 对参与和合规是正面的，模型里看不到这一点。
   - 别用模拟器挑措辞：措辞排序跨模型的平均 Spearman 只有 0.15–0.21，换一个模型就会换一个「最优」措辞（E23）。
7. **Limitations：** 结局只有 return/参与/合规；agent 人群是"留下了作者名"的幸存者；n_hist 上限 200；部分站点是 as-treated。

## 8. Thesis 结构（草案）

1. 引言：newcomer 的社会化，以及 LLM 模拟对社区设计的承诺和风险。
2. Reddit newcomer 的留存信号（Reddit Newcomers 项目；需要预注册复制 network fit）。
3. Agent 的行为保真度（benchmark FP0–FP3，跨 10 多个模型）。
4. 用 field experiment 检验模拟（本文）。
5. 版主研究和 StickyLab。
6. 结论。

## 9. 现在的弱点（直接列出）

1. **人类效应在所有已知站点都 ≈ 0。** 我们能证明 agent 错报了效应，但还不能检验人类效应不为 0 时，agent 能不能报对方向。
   - **11-06 用 CivilServant 原始数据核对了 r/science 2016 的另外两个 outcome（非 AMA 帖子）：**
     - **参与度**（每帖 newcomer 评论数）：均值 6.34 → 10.57。不加 FE 时 log RR 是 +0.51 [−0.02, +1.04]，噪声很大；加 block FE 后是 +0.06 [−0.05, +0.17]，几乎为 0。
     - **守规率**（newcomer 评论的可见比例；非 AMA、至少 1 条 newcomer 评论的 805 个帖子）：合并比例 0.447 → 0.508。按随机化设计加 block FE 的二项 GLM：logit(可见) **+0.44 [+0.07, +0.80]，p = .018**；不加 FE：+0.25 [−0.17, +0.66]；每帖比例 OLS + block FE：+5.3 pp，p = .07。是唯一的非零效应，但对设定敏感（11-06 重算，更正了之前的 0.429 → 0.504）。
   - **守规率没法在同一批人身上模拟。** 被删的评论在 Arctic Shift 里作者都变成了 `[deleted]`：29,862 条 `[removed]` 全部没有作者。所以我们能识别出来的 newcomer 全是守规的人。
   - **补法 1（P7，已排队，优先级提高）：** 让 agent 自己写评论（有 / 无 sticky，同一个 seed），再用两个本地 judge（Qwen3-32B、Gemma-3-27B）按 sticky 里列出的规则判定会不会被删。人类基准：logit(被删) −0.44。只能比较方向和大致大小，因为 agent 的人群是按守规筛选过的（`src/p7_*.py`）。
   - **补法 2：** P5 的正反馈准实验。r/science 2016 的人类 ATT 是 −0.14 [−0.40, +0.15]，也是 ≈ 0。
   - **结论：** "人类效应 ≈ 0、agent 给出稳定的非零预测"是现有数据能支持的最强说法。"agent 能不能报对真实的非零效应"需要另一类数据，例如删帖之后仍保留作者的平台，或者自己做一次 field experiment。
2. **Futurology 和新站点用的是 as-treated，不是 ITT**，靠平衡检验支撑。r/science 2016 是唯一有真实分组的站点。
3. **结果以"发了评论"为条件。** sticky 可能改变谁来评论。agent 和人类用的是同一批人，所以 gap 不受影响；但人类端的效应不是纯因果效应。
4. **单步问答，读第一个 token 的概率。** 不是多步模拟。早期的检查显示，基线比率主要来自 prompt 和读数方式。我们的主结论都用个体内的差值，对基线不敏感，但审稿人会问。
5. **Jev 是闭源的，gpt-4o-mini 也会被下线。** 主结论必须靠开源模型撑住，P1 就是为这个。
6. **可能有训练数据污染**（见第 5 节），要用 P4 的知识探针检查。
