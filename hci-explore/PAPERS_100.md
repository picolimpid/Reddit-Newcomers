# 100 篇论文：框架、模型模块与 JEV 切入点（v2）

**来源**：OpenAlex 检索 CHI / CSCW（PACM-HCI）/ UIST / IUI 2024–2026、标题或摘要含 LLM 相关词的论文，共 947 篇。

**怎么选的**：全部 947 个标题我都看过；入选的 100 篇我都读了完整摘要；三个 idea 依据的 15 篇读了 arXiv 全文的相关章节（标 [全文]）。v1 曾先用 JEV 初筛、只读截断摘要，v2 已改正：替换了 27 篇，标 ★ 的是被 JEV 初筛漏掉、这次补进来的。

**适配度**：高 = 判断步骤是核心，输出是离散选项或分数，并且量大、要实时或靠阈值行动；中 = 有判断步骤，但和生成纠缠，或依赖较强的视觉能力；低 = 作用边缘。本版分布：高 68，中 32，低 0。


## 标注 / 编码（23 篇）

| 论文 | 会议 | 框架与模型模块 | JEV 可替换或新增的环节 | 适配度 |
|---|---|---|---|---|
| [ArtScape: Detecting, Characterizing, and Interactively Exploring Urban Art at Scale Using Vision Language Models](https://doi.org/10.1145/3830398.3830557) | UIST 2026 | 五阶段 VLM 管线：街景全景图→检测城市艺术→分类与描述→给可达性/趣味性打分→去重（micro-F1 0.83） | 图像 noul 预筛（含壁画/雕塑/涂鸦？），VLM 只处理判正的帧；score 题打趣味性。注意 JEV 视觉能力有限 | 中 |
| [SenseMate: An Accessible and Beginner-Friendly Human-AI Platform for Qualitative Data Analysis](https://doi.org/10.1145/3640543.3645194) | IUI 2024 | 社区意见的质性编码；rationale extraction 模型（非 LLM，作者强调隐私和可控性）给主题推荐和理由。180 名新手实验：每条多花 49 秒，但编码者一致性 +29%、F 值 +10% | 整套 codebook 作为一组 noul 一次请求；概率提示新手哪些条目模型也拿不准 | 高 |
| [ScreenAudit: Detecting Screen Reader Accessibility Errors in Mobile Apps Using Large Language Models](https://doi.org/10.1145/3706598.3713797) | CHI 2025 | LLM 遍历 app 屏幕、抽取元数据与读屏转写，识别规则检查器漏掉的无障碍错误 | 错误类型清单做成多道 noul，一次请求判一屏；按概率给专家排序 | 中 |
| [DALL: Data Labeling via Data Programming and Active Learning Enhanced by Large Language Models](https://doi.org/10.1145/3772318.3791356) | CHI 2026 | 数据编程 + 主动学习 + LLM：LLM 帮用户纠正标签、建议 labeling function | JEV 每道 noul 就是一个 labeling function，概率即软标签；主动学习用 JEV 熵选样本 | 高 |
| [Into the Crossfire: Evaluating the Use of a Language Model to Crowdsource Gun Violence Reports](https://doi.org/10.1145/3757416) | CSCW 2025 | 微调 BERT 识别葡萄牙语推文中的枪击事件报告，接入人权组织的实时 web app | 零样本 noul 替代/补充 BERT，快速迁移到新语言和平台；阈值决定人工核查量 | 高 |
| [Empathy Practices in Social Media Discourse: A Multidimensional and Relational Perspective](https://doi.org/10.1145/3772318.3790967) | CHI 2026 | 基于心理学理论的多维共情标注框架（帖子如何求共情 × 回复如何表达），微调 LM 检测 | 多维 codebook 一次请求；post + reply 同在 state 中判断关系型匹配 | 高 |
| [Linguistically Differentiating Acts and Recalls of Racial Microaggressions on Social Media](https://doi.org/10.1145/3637366) | CSCW 2024 | 人工标注种族微侵犯的 act 与 recall，LM 分类并解释语言特征 | noul 区分 act/recall；概率与标注者分歧对照（E1 的同类问题） | 高 |
| [CollabCoder: A Lower-barrier, Rigorous Workflow for Inductive Collaborative Qualitative Analysis with Large Language Models](https://doi.org/10.1145/3613904.3642002) | CHI 2024 | CollabCoder：LLM 给开放编码建议、计算编码者分歧、建议 code group 的协作质性分析流程 | 候选 code 作为 noul，自动得到人—人—模型一致性矩阵，定位需要讨论的分歧条目 | 高 |
| [BIASsist: Empowering News Readers via Bias Identification, Explanation, and Neutralization](https://doi.org/10.1145/3706598.3713531) | CHI 2025 | BIASsist：6 类新闻偏见的识别→解释→中立化 | 识别步改为 6 道 noul 一次请求，便宜到可以逐句扫描；生成步留给 LLM | 高 |
| [Precise Duration Control for Video Editing via LLM Scoring and Constrained Optimization](https://doi.org/10.1145/3830398.3830510) | UIST 2026 | LLM 给视频片段打语义分，MILP 在毫秒级精确时长约束下选片段 | score 题的期望分直接作为 MILP 目标系数（概率比整数分更适合优化） | 高 |
| [Concept Induction: Analyzing Unstructured Text with High-Level Concepts Using LLooM](https://doi.org/10.1145/3613904.3642830) | CHI 2024 | LLooM：LLM 归纳高层概念和纳入标准；Score 步骤对 n 条文本 × c 个概念用多选 prompt 得到近似概率，阈值可调，形成 n×c 矩阵；每次运行平均约 $1.44 [全文] | Score 步骤：每个概念的纳入标准是一道 noul，一次请求覆盖全部概念，并给出真正的概率 | 高 |
| [Leveraging Prompt-Based Large Language Models: Predicting Pandemic Health Decisions and Outcomes Through Social Media Language](https://doi.org/10.1145/3613904.3642117) | CHI 2024 | RBIC：多步 prompt 从 subreddit 讨论中抽取 gist，再与疫苗接种等健康结果关联 | gist 定下后，对每条帖子判断是否含该 gist，做大规模计数 | 中 |
| [Mining Player Experience Trends From Game Reviews Using Large Language Models](https://doi.org/10.1145/3772318.3790760) | CHI 2026 | LLM 从 15 万条游戏评论预测 PXI 等问卷的 Likert 回答，分析体验趋势 | score 题直接输出 Likert 分布，一次请求答完整份问卷 | 高 |
| [Shortchanged: Uncovering and Analyzing Intimate Partner Financial Abuse in Consumer Complaints](https://doi.org/10.1145/3613904.3642033) | CHI 2024 | 270 万条金融投诉中用语言模型 + 专家复核识别亲密伴侣经济虐待 | 召回优先的 noul 初筛 + 校准阈值决定专家复核量 | 高 |
| [Media Bias Detector: Designing and Implementing a Tool for Real-Time Selection and Framing Bias Analysis in News Coverage](https://doi.org/10.1145/3706598.3713716) | CHI 2025 | Media Bias Detector：LLM 近实时标注新闻主题、语气、政治倾向、事实 | 多维 score/choice 一次请求，可以处理全量新闻流 | 高 |
| [Dynamic Surveys: Using LLMs to Blend Qualitative Depth, Quantitative Structure, and Collaborative Interaction](https://doi.org/10.1145/3757586) | CSCW 2025 | Dynamic Surveys：LLM 实时把开放回答聚类，再让答题者对聚类打分、排序、反思 | 新回答归入已有聚类（choice）的实时分配步骤 | 中 |
| ★ [Prompting in the Dark: Assessing Human Performance in Prompt Engineering for Data Labeling When Gold Labels Are Absent](https://doi.org/10.1145/3706598.3714319) | CHI 2025 | Prompting in the Dark：Google Sheets 插件里迭代 prompt 让 LLM 标注，**没有金标签**；20 人中只有 9 人迭代后准确率提升，DSPy 在金标签很少时也不行 [全文] | 用 JEV 熵挑出最值得人工标注的锚点，少量锚点再校准（直接对应 E1 的 20 条锚点结果） | 高 |
| ★ [End User Authoring of Personalized Content Classifiers: Comparing Example Labeling, Rule Writing, and LLM Prompting](https://doi.org/10.1145/3706598.3713691) | CHI 2025 | 个人内容分类器的三种创建方式：举例标注、写规则、LLM prompt（37 人）；LLM prompt 最准，但三者都难以迭代改进 | 规则写成 noul，输出概率和可调阈值，兼具写规则的可解释性和 LLM 的效果 | 高 |
| ★ [“Are Compliments Bad Now?”: Comparing LLMs and Human Interpretations of Gender Microaggressions in the Workplace](https://doi.org/10.1145/3772318.3790436) | CHI 2026 | 141 名参与者与 7 个 LLM 判断职场性别微侵犯的可能性：人与 LLM、不同性别和经历的人之间都有显著差异，主张检测系统接受解释多元性 | JEV 概率与不同人群的评分分布对照（E1 的同类问题） | 高 |
| ★ [Collecting Qualitative Data at Scale with Large Language Models: A Case Study](https://doi.org/10.1145/3710947) | CSCW 2025 | 用 LLM 聊天机器人大规模收集质性数据（n=399）：回答很少有具体动机和例子；**LLM 与人类在质量和丰富度评分上的一致性很低** | 评分步骤：检验锚点校准能否修复 LLM–人类低一致性 | 高 |
| ★ [LATA: A Pilot Study on LLM-Assisted Thematic Analysis of Online Social Network Data Generation Experiences](https://doi.org/10.1145/3711022) | CSCW 2025 | LATA：GPT-4/Gemini 按 Braun & Clarke 六阶段做主题分析，与人工结果的余弦相似度最高 0.76 | 主题生成是生成式任务；主题确定后可用 JEV 做演绎编码 | 中 |
| ★ [Steering Semantic Data Processing With DocWrangler](https://doi.org/10.1145/3746059.3747625) | UIST 2025 | DocWrangler：语义数据处理 IDE，map/filter/reduce 等算子由 LLM 执行，帮助用户在理解数据和写管线之间迭代 [全文] | filter/分类算子由 JEV 执行：便宜、带概率、每次修改都能全量重跑 | 高 |
| ★ [What Social Media Use Do People Regret? An Analysis of 34K Smartphone Screenshots with Multimodal LLM](https://doi.org/10.1145/3706598.3713724) | CHI 2025 | 每 5 秒截一次屏，用 MLLM 细粒度分析 3.4 万张截图，研究人们后悔的社交媒体使用 | 逐张截图的活动分类（图像 choice），量大；受视觉能力限制 | 中 |

## 状态检测 / 介入时机（28 篇）

| 论文 | 会议 | 框架与模型模块 | JEV 可替换或新增的环节 | 适配度 |
|---|---|---|---|---|
| [Sensing What Surveys Miss: Understanding and Personalizing Proactive LLM Support by User Modeling](https://doi.org/10.1145/3772318.3791191) | CHI 2026 | 皮电 + 鼠标轨迹→个性化分类器预测是否需要帮助→触发 LLM 澄清（准确率 +21%） | 生理信号不是 JEV 强项；可在展示前验证 LLM 的澄清是否切题 | 中 |
| [Be My Ears: Enabling Semantic Alerting for Earphone Wearers with Pheme](https://doi.org/10.1145/3830398.3830640) | UIST 2026 | Pheme：LLM 理解查询→音频分离→LALM 验证检测结果。准确率 82.2%，误报率降 11.9%，API 调用减少 48.1% | 验证步（声音事件是否符合用户意图）用 noul，进一步降延迟和成本 | 中 |
| [Inferring Affect and Intervention Opportunities for Cancer Survivors from Digital Diaries with Context-Aware LLMs](https://doi.org/10.1145/3772318.3791352) | CHI 2026 | CALLM：以相似同伴案例 + 个人轨迹为上下文，从极短日记推断情绪、调节意愿与可干预性（平衡准确率 60–74%）；作者发现 LLM 置信度与准确率相关，少量校准能改善个性化 | 三个目标作为 noul/score 一次请求，概率直接用于即时干预阈值；与 E1 的锚点校准一致 | 高 |
| [Human I/O: Towards a Unified Approach to Detecting Situational Impairments](https://doi.org/10.1145/3613904.3642065) | CHI 2024 | Human I/O：第一视角视频 + 多模态感知 + LLM 推理视觉/听觉/手/语音通道是否可用 | 4 个通道各一道 score 题，以场景描述或帧为 state；实时 | 中 |
| [Just-In-Time Objectives: A General Approach for Specialized AI Interactions](https://doi.org/10.1145/3772318.3790713) | CHI 2026 | Just-in-time objectives：被动观察行为→推断即时目标→据此生成并评估输出 | 评估步（输出是否服务当前目标）+ 在候选目标中做 choice | 高 |
| [CoNarrate: Computationally Modeling Information Prematureness and Scaffolding the Co-Construction of Patient Narratives in Asynchronous Telehealth](https://doi.org/10.1145/3830398.3830658) | UIST 2026 | CoNarrate：DIBH 框架检测患者自述中的 5 类信息不成熟→生成追问 | 5 类不成熟各一道 noul，随输入实时检测 | 高 |
| [ComPeer: A Generative Conversational Agent for Proactive Peer Support](https://doi.org/10.1145/3654777.3676430) | UIST 2024 | ComPeer：每轮对话后由 LLM 事件检测器抽取用户事件并推断合适时机，Schedule 模块安排主动消息 [全文] | 事件是否值得主动跟进、何时跟进的判断（对应 E2） | 高 |
| [An LLM-Based Motivation-Aware Framework For AI Coaching For Behaviour Change](https://doi.org/10.1145/3772318.3791123) | CHI 2026 | 动机感知教练：区分矛盾型和行动准备型来访者，切换 MI 情感支持或计划策略 | 改变阶段 choice 作为路由器 | 高 |
| [State Your Intention to Steer Your Attention: An AI Assistant for Intentional Digital Living](https://doi.org/10.1145/3772318.3791404) | CHI 2026 | 意图助手：LLM 看截图、应用标题、URL 判断当前活动是否偏离声明的意图并提醒 | 这就是一道 noul（活动与意图一致吗？），可每几秒跑一次，用户反馈校准阈值 | 高 |
| [Memoro: Using Large Language Models to Realize a Concise Interface for Real-Time Memory Augmentation](https://doi.org/10.1145/3613904.3642450) | CHI 2024 | Memoro：LLM 从对话推断记忆需求→语义检索→极简建议（含无查询模式） | 是否需要提示的门控 + 候选记忆相关性 choice | 高 |
| [Need Help? Designing Proactive AI Assistants for Programming](https://doi.org/10.1145/3706598.3714002) | CHI 2025 | Need Help?：**计时器触发**（停止输入 5 秒后、至少间隔 20 秒），LLM 生成建议；作者明确没有给建议排序的机制；模型返回太慢时直接丢弃建议 [全文] | 给候选建议打接受概率并排序/过滤；约 0.1 秒的延迟能避免建议因超时被丢弃（对应 E2） | 高 |
| [Guiding, Not Railroading: Design and Evaluation of a Multi-Agent System for Narrative Redirection in Role-playing Games](https://doi.org/10.1145/3742413.3789218) | IUI 2026 | SENNA：多 agent 游戏主持，用叙事图跟踪玩家进度并判断是否需要引导回主线 | 玩家行动是否偏离关键情节的 noul + 下一情节节点的 choice | 高 |
| [Epimopilot : An LLM-infused Epistemic Emotion Support System to Boost Self-regulated Asynchronous Online Video Learning](https://doi.org/10.1145/3772318.3791067) | CHI 2026 | Epimopilot：从弹幕识别认知情绪，检测到学习困难时触发 LLM 干预 | 弹幕情绪分类（量大、实时）+ 是否干预 | 高 |
| [AI of Oz: Enhancing Wizard of Oz Studies in HCI with AI Assistance for Human Moderation](https://doi.org/10.1145/3772318.3791324) | CHI 2026 | AI of Oz：LLM 监控 WoZ 交互、检测敏感时刻、建议回复 | 敏感时刻检测需要低延迟、高召回和可调阈值 | 高 |
| [InsightToast: Proactive Information Retrieval & Glanceable Visualization in the Side Channel of Data-Rich Meetings](https://doi.org/10.1145/3830398.3830522) | UIST 2026 | InsightToast：实时监听会议，识别信息需求，多 agent RAG 推送 toast | 这句话是否产生信息需求的门控，只有判正才启动昂贵的 RAG | 高 |
| [AiGet: Transforming Everyday Moments into Hidden Knowledge Discovery with AI Assistance on Smart Glasses](https://doi.org/10.1145/3706598.3713953) | CHI 2025 | AiGet：注视 + 环境 + 用户画像→LLM 在智能眼镜上主动推送知识 | 验证候选知识点此刻是否有趣、打扰小 | 中 |
| [Assistance or Disruption? Exploring and Evaluating the Design and Trade-offs of Proactive AI Programming Support](https://doi.org/10.1145/3706598.3713357) | CHI 2025 | Codellaborator：**规则触发**（空闲 30 秒、写完代码块、运行程序、写注释、选中代码）→ LLM 在动作列表里选一个（含不回应）[全文]。398 次主动介入：有效 53.3%、打扰 12.1%、被忽略 34.7% | 在 LLM 回应展示前加 JEV 验证（用户会接受吗？），针对近一半无效介入（对应 E2） | 高 |
| [ProMemAssist: Exploring Timely Proactive Assistance Through Working Memory Modeling in Multi-Modal Wearable Devices](https://doi.org/10.1145/3746059.3747770) | UIST 2025 | ProMemAssist：工作记忆建模 + 时机预测器平衡帮助价值与打扰代价 | 价值估计部分：这条帮助现在有没有用 | 中 |
| [Towards Better Health Conversations: The Benefits of Context-seeking](https://doi.org/10.1145/3772318.3793199) | CHI 2026 | Wayfinding AI：健康问答中先主动追问上下文再回答 | 当前信息是否足以安全作答的 noul，决定继续追问还是作答 | 高 |
| ★ [The Last JITAI? Exploring Large Language Models for Issuing Just-in-Time Adaptive Interventions: Fostering Physical Activity in a Prospective Cardiac Rehabilitation Setting](https://doi.org/10.1145/3706598.3713307) | CHI 2025 | The Last JITAI?：GPT-4 根据人设和情境决定是否触发即时干预并生成消息；450 个决定与 10 名普通人、10 名医护人员对比，GPT-4 各项都更好 [全文] | 触发决策（现在该不该干预）用 noul 概率，生成仍由 LLM 负责 | 高 |
| ★ [MindShift: Leveraging Large Language Models for Mental-States-Based Problematic Smartphone Use Intervention](https://doi.org/10.1145/3613904.3642790) | CHI 2024 | MindShift：结合应用使用、物理情境、心理状态（无聊/压力/惯性）和目标，由 LLM 生成说服内容；5 周现场实验，接受率 +4.7–22.5% | 心理状态 choice + 送达前预测用户是否会接受 | 高 |
| ★ [User Attention Is All chatLLMs Need: Proactive Intervention Against Attention Drift While Waiting for chatLLMs](https://doi.org/10.1145/3830398.3830580) | UIST 2026 | Q2Focus：用户等待 LLM 回复时立即推送任务相关的小问题，注意力漂移下降 50% | 必须在提交后立刻出现：适合用 JEV 快速决定是否推送、推送哪个问题 | 中 |
| ★ [Gensors: Authoring Personalized Visual Sensors with Multimodal Foundation Models and Reasoning](https://doi.org/10.1145/3708359.3712085) | IUI 2025 | Gensors：用户用自然语言定义个人视觉传感器（如"孩子在捣乱时提醒我"），系统拆成多条准则并行测试，MLLM 几秒内响应 | 每条准则是一道图像 noul，同一帧上多题只计费一次、约 0.1 秒；需要先验证 JEV 的视觉判断 | 中 |
| ★ [When Help Hurts: Verification Load and Fatigue with AI Coding Assistants](https://doi.org/10.1145/3772318.3791176) | CHI 2026 | When Help Hurts：固定同一个 LLM，比较内联、聊天、结构化三种交互（N=60），提出验证负担指数 | 只展示高接受概率的建议，以验证负担指数作为结果变量 | 中 |
| ★ [RageSense: Leveraging Acoustic Sensing and LLM-Based Intervention for Emotion Regulation in Mobile Gaming](https://doi.org/10.1145/3772318.3791076) | CHI 2026 | RageSense：用近超声声学感知估计效价和唤醒度，LLM 结合截图生成情绪调节干预（53 人现场研究） | 感知不适合；可做干预内容选择和接受度检查 | 中 |
| ★ [The Invisible Mentor: Inferring User Actions from Screen Recordings to Recommend Better Workflows](https://doi.org/10.1145/3772318.3790294) | CHI 2026 | InvisibleMentor：VLM 从屏幕录像重建操作，LLM 推荐更高效的工作流 | 推荐展示前判断其对该用户是否有用、是否新颖 | 中 |
| ★ [AI That Helps Us Help Each Other: A Proactive System for Scaffolding Mentor-Novice Collaboration in Entrepreneurship Coaching](https://doi.org/10.1145/3757549) | CSCW 2025 | 创业辅导系统：领域风险认知模型 + LLM 主动提出诊断性问题；导师可以检查并修改底层模型（CSCW'25） | 风险诊断项写成 noul，导师修改模型等于修改 JEV 问题 | 高 |
| ★ [Exploring the Design Space of Real-time LLM Knowledge Support Systems: A Case Study of Jargon Explanations](https://doi.org/10.1145/3706598.3714262) | CHI 2025 | StopGap：视频中实时解释术语的知识支持原型（N=24），发现偏好差异大，强调用户主导和个性化 | 判断某个术语对这位用户是否需要解释（个性化阈值），只对判正的生成解释 | 高 |

## 评估 / 打分（27 篇）

| 论文 | 会议 | 框架与模型模块 | JEV 可替换或新增的环节 | 适配度 |
|---|---|---|---|---|
| [Smarter Together: Enhancing Human-AI Collaborative Grading With Teacher-Cognition Multi-Agent LLM Framework](https://doi.org/10.1145/3742413.3789130) | IUI 2026 | TC-MAG：多 agent 模拟教师流程（建 rubric、盲双评、仲裁、交叉校验置信度）；2,000 份答卷上 κ=0.968；置信分数影响教师是否放手给系统 | 每个 rubric 项一道 score 题，概率即置信度；只有低置信才进入 LLM 仲裁或教师复核 | 高 |
| [ESR-Coach: Leveraging Large Language Models for Training People to Provide Emotionally Supportive Responses](https://doi.org/10.1145/3742413.3789094) | IUI 2026 | ESR-Coach：多 agent 生成情感支持练习场景、示范回答、评估学员回复 | 评估步按技能 rubric 多题打分 | 高 |
| [EvaluAId: Human-AI Collaborative Evaluation of Open-Ended Student Essays](https://doi.org/10.1145/3772318.3790814) | CHI 2026 | EvaluAId：人机协作作文评估（rubric—内容映射、自适应基准、反馈合成） | 段落 × rubric 维度的 noul 映射 | 高 |
| [PASTA: A Scalable Framework for Multi-Policy AI Compliance Evaluation](https://doi.org/10.1145/3772318.3790966) | CHI 2026 | PASTA：多政策 AI 合规评估，LLM 成对评估引擎 + 省钱策略；与专家 ρ ≥ .626；5 项政策不到 2 分钟、约 $3 | 同一份 model card × 大量政策条款的 noul（state 只计费一次），成本可再降一到两个数量级 | 高 |
| [SituationAdapt: Contextual UI Optimization in Mixed Reality with Situation Awareness via LLM Reasoning](https://doi.org/10.1145/3654777.3676470) | UIST 2024 | SituationAdapt：感知→VLM 判断 UI 放置是否合适→优化器 | 每个放置位置的适宜度 score 作为优化代价项 | 中 |
| [Human-LLM Collaborative Annotation Through Effective Verification of LLM Labels](https://doi.org/10.1145/3613904.3641960) | CHI 2024 | Human-LLM 协作标注（Wang et al.）：LLM 给标签和解释→训练一个验证器（用输入、标签、解释的特征）→人只复核低分标签；验证器优于基于 logit 的基线 | 验证器：比较 JEV 概率与训练验证器、logit 基线的效果和成本（对应 E1） | 高 |
| [Who's Keeping Score? Interactive Steering of LLM-Powered Scoring with Attune](https://doi.org/10.1145/3830398.3830567) | UIST 2026 | Attune：成对比较建立全局理解，再解析成一致的分数和规则 | 成对比较就是 choice 题；O(N²) 次比较在 JEV 上可以全量做 | 高 |
| [When Context Grows, So Does the Challenge: Human Oversight in LLM Evaluation of Financial Tables](https://doi.org/10.1145/3746058.3758457) | UIST 2025 | 金融表格 LLM-as-judge 在大表上退化，触发式监督把模糊情况交给专家 | 用校准概率决定何时触发（E3b 一致性维度的证据） | 高 |
| [EvalAgent: Interactive Comparative Evaluation of Computer-Using GUI Agents](https://doi.org/10.1145/3742414.3795096) | IUI 2026 | EvalAgent：按准则比较网页 agent 的行为和推理，LLM-as-judge | 准则 × 轨迹的批量判断 | 高 |
| [Generating Automatic Feedback on UI Mockups with Large Language Models](https://doi.org/10.1145/3613904.3642782) | CHI 2024 | Figma 插件：GPT-4 按启发式原则自动评估 UI mockup | 每条启发式一道 noul（是否违反），只对违反项生成建议 | 中 |
| [Building Benchmarks from the Ground Up: Community-Centered Evaluation of LLMs in Healthcare Chatbot Settings](https://doi.org/10.1145/3772318.3791172) | CHI 2026 | Samiksha：社区驱动的医疗聊天机器人评估管线 | 社区定义的标准写成 noul，大规模自动评估并与社区判断对齐 | 高 |
| [Efficient Management of LLM-Based Coaching Agents' Reasoning While Maintaining Interaction Quality and Speed](https://doi.org/10.1145/3706598.3713606) | CHI 2025 | 教练 agent 推理管理：用差距信号决定投入多少推理，成本降约 10 倍且质量几乎不变 | 差距信号本身可以换成 JEV 的概率判断（System-1/2 路由），对比现有机制 | 高 |
| [EvaluLLM: LLM assisted evaluation of generative outputs](https://doi.org/10.1145/3640544.3645216) | IUI 2024 | EvaluLLM：LLM 辅助评估生成结果的应用 | 评估后端 | 高 |
| [Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences](https://doi.org/10.1145/3654777.3676450) | UIST 2024 | EvalGen：LLM 生成准则和断言，人给少量好/坏评分来挑选对齐的实现，提出 criteria drift。选哪些输出给人评用交替策略；作者明确说其分数只表示输出差的可能性，**不能做不确定性采样** [全文] | 提供 EvalGen 缺的不确定性信号：按 JEV 熵挑选给人评的输出（对应 E3b） | 高 |
| [EvalLM: Interactive Evaluation of Large Language Model Prompts on User-Defined Criteria](https://doi.org/10.1145/3613904.3642216) | CHI 2024 | EvalLM：用户自定义准则，GPT-4 对输出对逐准则比较并给解释；与手工评估相比，参与者检查的输出多一倍、修改次数少 59% [全文] | 逐准则比较的执行端 | 高 |
| [Think Together and Work Better: Combining Humans' and LLMs' Think-Aloud Outcomes for Effective Text Evaluation](https://doi.org/10.1145/3706598.3713181) | CHI 2025 | InteractEval：人和 LLM 出声思考生成清单属性，做清单式文本评估（SummEval） | 清单执行端（E3：清单拆分没有超过直接打分） | 高 |
| [Evalet: Evaluating Large Language Models through Functional Fragmentation](https://doi.org/10.1145/3772318.3790285) | CHI 2026 | Evalet：把输出拆成片段并解释每个片段的功能，做片段级评估 | 片段 × 准则的 noul（片段多、量大） | 高 |
| [HILL: A Hallucination Identifier for Large Language Models](https://doi.org/10.1145/3613904.3642428) | CHI 2024 | HILL：幻觉识别界面 | 每个陈述是否有依据的 noul（E3b：一致性维度不确定性很准） | 高 |
| ★ [Data-Prompt Co-Evolution: Growing Test Sets to Refine LLM Behavior](https://doi.org/10.1145/3772318.3791222) | CHI 2026 | Data-Prompt Co-Evolution：测试集与 prompt 一起演化，帮开发者发现边界情况、写明期望行为，并在不断增长的测试集上评估修改后的 prompt [全文] | 每次修改都在全部测试用例上重跑 JEV 检查；不确定条目即候选边界情况 | 高 |
| ★ [Designing Staged Evaluation Workflows for LLMs: Integrating Domain Experts, Lay Users, and Model-Generated Evaluation Criteria](https://doi.org/10.1145/3772318.3790897) | CHI 2026 | 分阶段评估工作流：比较专家、普通用户、LLM 三类来源的准则，发现各阶段间的漂移和后期收敛 | 三类来源的准则作为同一 state 上的多题同时执行 | 中 |
| ★ [MultEval: Collaboratively Creating Criteria for LLM-as-a-Judge Systems](https://doi.org/10.1145/3742414.3795100) | IUI 2026 | MultEval：多人协作制定 LLM-as-judge 准则，尽早暴露分歧并诊断冲突来源 | 每个人的准则版本各跑一遍，按概率差异定位分歧出现在哪些输出上 | 高 |
| ★ [Prism-UX: Turning Evaluative Divergence into Traceable Design Evidence for LLM-Based Process-Level UX Evaluation](https://doi.org/10.1145/3830398.3830622) | UIST 2026 | Prism-UX：把屏幕录像转成可引用的证据，从多个可配置视角评估并保留分歧；逻辑冲突率降 84% | 不同视角的判断作为同一 state 上的多题（只计费一次） | 中 |
| ★ [Limitations of the LLM-as-a-Judge Approach for Evaluating LLM Outputs in Expert Knowledge Tasks](https://doi.org/10.1145/3708359.3712091) | IUI 2025 | 专家知识任务中的 LLM-as-judge：营养学和心理健康专家与 LLM 裁判的一致率只有 68% 和 64% | 检验 JEV + 少量专家锚点能否提高一致性；也是一个警示 | 中 |
| ★ [ChainForge: A Visual Toolkit for Prompt Engineering and LLM Hypothesis Testing](https://doi.org/10.1145/3613904.3642016) | CHI 2024 | ChainForge：开源可视化 prompt 工程与假设检验工具，跨模型、跨 prompt 比较大量回答 | 评估节点：快速、带概率的批量判断 | 高 |
| ★ [RELIC: Investigating Large Language Model Responses using Self-Consistency](https://doi.org/10.1145/3613904.3641904) | CHI 2024 | RELIC：用同一 LLM 多次采样的自洽性估计每个陈述的可信度，帮助用户核查长文本 | 用 JEV 的陈述—证据概率替代 N 次采样，成本对比 | 高 |
| ★ [DxHF: Providing High-Quality Human Feedback for LLM Alignment with Interactive Decomposition](https://doi.org/10.1145/3746059.3747600) | UIST 2025 | DxHF：把长回答拆成陈述来收集人类偏好，并可视化陈述与对话的相关性；准确率 +5%，每条多花 18 秒 | 陈述相关性评分（可视化编码所需的那一步） | 中 |
| ★ [iRULER: Intelligible Rubric-Based User-Defined LLM Evaluation for Revision](https://doi.org/10.1145/3772318.3790539) | CHI 2026 | iRULER：基于用户自定义 rubric 的写作评估，解释"为什么是这个分、为什么不是更高分"并给出改写示例 [全文] | rubric 分数分布：相邻两档的概率就是"为什么不是 4 分"的量化依据 | 中 |

## 审核 / 安全（5 篇）

| 论文 | 会议 | 框架与模型模块 | JEV 可替换或新增的环节 | 适配度 |
|---|---|---|---|---|
| [APOLLO: A GPT-based tool to detect phishing emails and generate explanations that warn users](https://doi.org/10.1145/3733049) | CSCW 2025 | APOLLO：GPT-4o 判别钓鱼邮件并生成警示解释 | 分类步（实时、量大），只对判正的邮件生成解释 | 高 |
| [Twits, Toxic Tweets, and Tribal Tendencies: Trends in Politically Polarized Posts on Twitter](https://doi.org/10.1145/3757684) | CSCW 2025 | 训练 DeBERTa 毒性检测器，分析党派与情感极化和毒性的关系 | 作为测量工具的替代或对照，多维毒性（对应 E1） | 高 |
| [CLOAK: A Privacy-Preserving Assistant for Online Social Media Users](https://doi.org/10.1145/3830398.3830723) | UIST 2026 | CLOAK：跨帖子聚合用户画像，评估再识别风险 | 这组属性是否足以缩小到少数人的 noul + 每个帖子的 PII 类别 | 中 |
| [Think Fast, Think Slow, Think Critical: Designing an Automated Propaganda Detection Tool](https://doi.org/10.1145/3613904.3642805) | CHI 2024 | ClarifAI：LLM 检测新闻中的宣传手法并解释 | 宣传手法多题检测 | 高 |
| [Exploring Multidimensional Checkworthiness: Designing AI-assisted Claim Prioritization for Human Fact-checkers](https://doi.org/10.1145/3757473) | CSCW 2025 | 多维可核查性：辅助事实核查员给 claim 排优先级 | 每个维度一道 score，按核查员偏好加权排序 | 高 |

## 排序 / 选择（7 篇）

| 论文 | 会议 | 框架与模型模块 | JEV 可替换或新增的环节 | 适配度 |
|---|---|---|---|---|
| [RankCut: A Ranking-Based LLM Approach to Extractive Summarization for Transcript-Based Video Editing](https://doi.org/10.1145/3742413.3789115) | IUI 2026 | RankCut：三阶段 LLM 管线（总体摘要→片段排序→编辑在时间线上确认） | 片段重要性 score / choice | 高 |
| [Embedding Democratic Values into Social Media AIs via Societal Objective Functions](https://doi.org/10.1145/3641002) | CSCW 2024 | 社会目标函数：把反民主态度量表写成 LLM 打分，作为 feed 重排目标。人工标注 α=.895，模型与人工 ρ=.75；降权使党派敌意下降（d=.25） | 8 个变量一次请求，让在线重排变得可行；有现成人工标注可直接对比 | 高 |
| [OmniActions: Predicting Digital Actions in Response to Real-World Multimodal Sensory Inputs with LLMs](https://doi.org/10.1145/3613904.3642068) | CHI 2024 | OmniActions：LLM 根据多模态感知输入预测用户想执行的数字动作 | 离散动作空间上的 choice | 高 |
| [Leveraging Large Language Models for Collective Decision-Making](https://doi.org/10.1145/3757418) | CSCW 2025 | LLM 协调群体决策：从对话中抽取每个成员的偏好并推荐选项 | 选项 × 成员偏好的满意度 score 矩阵 | 高 |
| [Promptimizer: User-Led Prompt Optimization for Personal Content Classification](https://doi.org/10.1145/3772318.3790923) | CHI 2026 | Promptimizer：用户主导的个人内容分类器提示优化 | 分类器执行端：每轮优化都要全量评估，JEV 让每轮几乎免费 | 高 |
| [CustomSight: Enhancing LLM-Powered Visual Assistance for Blind Individuals using Goal-Directed Dynamic Filters](https://doi.org/10.1145/3746058.3758401) | UIST 2025 | CustomSight：动态过滤器自动捕获与目标相关的帧，再交给 LLM | 过滤器本身（帧是否含目标信息的 noul）；视觉 | 中 |
| [Social Media Feed Elicitation](https://doi.org/10.1145/3772318.3791569) | CHI 2026 | Feed elicitation：访谈引导用户写出定制 feed 偏好，LLM 按偏好给 Bluesky 帖子打分排序 | feed 打分器：帖子 × 偏好规则 | 高 |

## 模拟用户 / 人设（10 篇）

| 论文 | 会议 | 框架与模型模块 | JEV 可替换或新增的环节 | 适配度 |
|---|---|---|---|---|
| [CoBRA: Programming Cognitive Bias in Social Agents Using Classic Social Science Experiments](https://doi.org/10.1145/3772318.3790804) | CHI 2026 | CoBRA：用经典社会科学实验测量并闭环控制 agent 的认知偏差 | 测量步：agent 回答是否显示偏差 | 中 |
| [Wisdom of the Crowd, Without the Crowd: A Socratic LLM for Asynchronous Deliberation on Perspectivist Data](https://doi.org/10.1145/3757707) | CSCW 2025 | 苏格拉底式 LLM 作为审议伙伴，替代其他众包工人（perspectivist 数据） | 先判断哪些条目值得审议（模型不确定或有争议，对应 E1） | 高 |
| [Can LLM-Simulated Practice and Feedback Upskill Human Counselors? A Randomized Study with 90+ Novice Counselors](https://doi.org/10.1145/3772318.3791821) | CHI 2026 | LLM 模拟练习 + 反馈，94 名新手咨询师随机试验 | 反馈步的逐句技能判定 | 高 |
| [Proxona: Supporting Creators' Sensemaking and Ideation with LLM-Powered Audience Personas](https://doi.org/10.1145/3706598.3714034) | CHI 2025 | Proxona：从评论提炼受众维度→聚类成人设→模拟反馈 | 评论→维度值的大规模映射 | 中 |
| [Interview-Informed Generative Agents for Product Discovery: A Validation Study](https://doi.org/10.1145/3772318.3791918) | CHI 2026 | 访谈驱动的生成式 agent：群体层面校准、个体层面不准 | choice 概率作为一次性分布预测，不必多次采样 | 中 |
| [Rehearsal: Simulating Conflict to Teach Conflict Resolution](https://doi.org/10.1145/3613904.3642159) | CHI 2024 | Rehearsal：模拟冲突对话，按冲突策略给反馈 | 冲突策略识别的 noul | 中 |
| [Scaffolding Empathy: Training Counselors with Simulated Patients and Utterance-level Performance Visualizations](https://doi.org/10.1145/3706598.3714014) | CHI 2025 | 模拟病人 + 逐句表现可视化，训练 MI 咨询技能 | 逐句 MI 行为编码：量大、实时 | 高 |
| [TeachTune: Reviewing Pedagogical Agents Against Diverse Student Profiles with Simulated Students](https://doi.org/10.1145/3706598.3714054) | CHI 2025 | TeachTune：模拟学生与教学 agent 对话，教师审查适应性 | 每段对话判断教学 agent 是否适应了该学生画像 | 中 |
| ★ [Emulating Aggregate Human Choice Behavior and Biases with GPT Conversational Agents](https://doi.org/10.1145/3772318.3791835) | CHI 2026 | 用 GPT 对话 agent 模拟群体选择行为与偏差：1,100 人实验，GPT-4/5 能准确复现偏差 | choice 概率一次给出群体分布预测 | 中 |
| ★ [Predicting Biased Human Decision-Making with Large Language Models in Conversational Settings](https://doi.org/10.1145/3742413.3789127) | IUI 2026 | LLM 预测对话情境下的个人有偏决策（预注册，N=1,648），GPT-4 系列最贴近人类 | 同上；可与 behavioral-fidelity 项目衔接 | 中 |
