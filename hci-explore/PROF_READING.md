# 两位老师的 20 篇代表作：读书笔记（2026-11-07）

用途：以 Koustuv Saha 和 Eshwar Chandrasekharan 两位老师的视角评价本项目（组会推演见 `GROUP_MEETING.md`）。
全文都是我自己读的（PDF 转文本后读摘要、方法、结果、讨论和局限）；没有用任何模型筛选或总结论文。
PDF 和文本在 `data/ext/prof_papers/`、`data/ext/prof_papers_txt/`（下载的文件只当数据读）。

两位老师合作很多（LLM-Mod、SLM-Mod、MoMoE、VASTU、LLUMI、Positive Reinforcement 都是两人合著）。
分配原则：合著的论文按通讯/资深作者（作者表最后一位）归到那位老师名下，两边各 10 篇，不重复。

每篇记四件事：做了什么、关键数字、方法上的立场、对我们项目的意义。

## Koustuv Saha（OnCARE lab：社会计算、心理健康与福祉、以人为中心的 AI、因果推断）

### K1. LLM-Mod: Can Large Language Models Assist Content Moderation?（Kolla, Salunkhe, Chandrasekharan, Saha；CHI EA 2024）
- 做了什么：GPT-3.5 拿到社区规则和例子后判断帖子是否违规，9 个文本社区、744 个帖子（600 个“热门帖”当作不违规，144 个违规帖来自手写、人工挑选和 ChatGPT 生成后改写）。研究者扮演版主追问。
- 关键数字：不违规帖的真阴性率 92.3%，违规帖只抓到 43.1%（论文的混淆矩阵口径）。模型偏保守：容易放过违规帖。
- 观察到的问题：追问“你有多确定”时模型来回改判（vacillating）；用含糊的措辞回避判断（non-committal）。作者的结论是 LLM 更适合解释决定，不适合直接做决定；人要留在回路里。
- 对我们：这两个毛病正是 JEV 的设计要解决的：JEV 直接给出概率，不靠追问，也就没有“改口”的空间。LLM 偏保守是阈值没对上社区的问题，每个社区一个截距就能调。数据小、违规样本有人工生成的，是作者自己承认的局限。

### K2. SLM-Mod: Small Language Models Surpass LLMs at Content Moderation（Zhan, Goyal, Chen, Chandrasekharan, Saha；NAACL 2025）
- 做了什么：15 个热门社区，每个社区从 Chandrasekharan et al. (2018) 的删帖数据里取 5K 被删 + 5K 未删的评论（共 150K）。每个社区用 LoRA 微调一个 8–12B 的小模型（Llama-3.1-8B、Gemma-2-9B、Mistral-NeMo），对比零样本和少样本的 GPT-4o、GPT-4o-mini、Command R+。提示里给社区名、上一条评论和社区规则，要求回答 True/False。
- 关键数字：微调小模型比零样本大模型准确率平均高 11.5%，召回高 25.7%；大模型精确率更高（14/15 个社区最高），召回低。少样本只带来一点点提升。不平衡测试（违规率 1%、5%、10%）下 Mistral-NeMo 的 AUC 约 0.79，GPT-4o 约 0.70。跨社区（拿 A 社区的模型去管 B 社区）不如本社区，但最好的跨社区小模型仍胜过大模型。
- 方法立场：社区特定的模型；关注精确率和召回的取舍；关注新社区（冷启动）和模型版本漂移（GPT-4o 两个版本结果不同）。
- 对我们：他们的大模型输出的是 True/False，阈值固定，所以“精确率高、召回低”这个差距里有多少只是阈值没校准，论文没法回答。固定判官 + 每社区截距正好能测这一点，而且每个社区只要几条版主决定，不要几千条微调数据。新社区的冷启动正是 k 很小时的学习曲线。

### K3. LLUMI: Improving LLM Writing Assistance for Mental Health Support with Online Community Feedback（Kim, Ajit, Gong, Shimgekar, Yoo, Chandrasekharan, Saha；EMNLP 2026 Oral，最佳论文提名）
- 做了什么：可以在机构内部部署的开源模型（Mistral-7B），两个部件：生成模型（起草支持性回复）和改进模型（修改人写的回复）。用 Reddit 心理健康社区的点赞和点踩构造“好 / 差”回复对，做 SFT 和 DPO；再用 Prolific 人工评价五个维度（可读性、共情、连接感、可操作性、安全）。
- 结论：小的开源模型加社区反馈，在语言分析和人工评价上和闭源 GPT 相当；强调心理健康场景下隐私和数据治理，不该把敏感对话发到云端模型。
- 对我们：两点。① 心理健康场景下 Koustuv 一定会问隐私：JEV 是云端 API。我们的方法里“每人一个数”可以只存在本地，判官也可以换成本地开源模型（我们已经有本地 Qwen3-32B 的对照）。② 社区反馈（点赞）是一种群体标签，和我们“校准到个人”是两个层次，可以互补。

### K4. AI Psychosis: Does Conversational AI Amplify Delusion-Related Language?（Shimgekar, Gunda, Kim, Rodriguez, Sundaram, Saha；EMNLP 2026）
- 做了什么：用 Reddit 用户的长期发帖历史构造模拟用户（SimUsers），和 GPT、LLaMA、Qwen 进行多轮长对话；构造 DelusionScore 衡量每一轮的妄想相关语言强度，并和人工标注做了验证。
- 发现：先前有妄想相关表述的用户（处理组）在对话中 DelusionScore 逐轮上升，对照组持平或下降；“现实怀疑”和“强迫推理”主题上升最快；让 AI 根据当前 DelusionScore 调整回复，能明显压住上升趋势。
- 对我们：一个“判官分数反过来调节生成”的例子，和 Idea 2 的门控同构。Koustuv 关心的是长期、多轮下的动态和对弱势用户的影响，我们的每人截距是静态的，没有处理“这个人在对话中变了”。

### K5. TUX: Measuring Human–AI Tacit Understanding（Li, Min, Das Swain, Saha；Findings of EMNLP 2026）
- 做了什么：仿 Wavelength 桌游，人和 AI 各自把概念放到主观光谱上（没有对错、没有反馈），用两者放置的相似度定义 Tacit Understanding Index。241 名参与者，200 个按人设画像条件化的 LLM 代理（四个模型家族）。
- 发现：画像上最接近的人–代理配对 TUX 显著更高，说明默契和个人特质有关；预测变量越丰富（个人特质、决策风格、自信程度）TUX 越可解释；但按画像条件化能捕捉的个人判断有限。
- 对我们：和我们的结果直接呼应：人口学和画像对新人的判断帮助很小（E1e 更正后 0.498），而每人从几条真实判断估出的截距有效。可以说成：要对上一个人的标准，问他几条，比给模型一份画像有用。

### K6. Mental Health Impacts of AI Companions: Triangulating Social Media Quasi-Experiments, User Perspectives, and Relational Theory（Yuan, Zhang, Aledavood, Zhang, Saha；CHI 2026）
- 做了什么：Reddit 纵向数据上的分层倾向得分匹配 + 双重差分，看使用 AI 伴侣（如 Replika）后语言的变化；再做 18 个半结构访谈，用 Knapp 的关系发展模型解释。
- 发现：效果有好有坏：悲伤表达和人际关注增加，同时关于孤独、抑郁、自杀意念的语言也增加；访谈里看到“开始、升级、依恋”的轨迹，既有情感确认和社交演练，也有过度依赖和退缩。
- 方法立场：三角验证（准实验 + 用户视角 + 理论），不靠单一方法下结论。
- 对我们：Koustuv 会要求我们的离线指标配上用户视角：人到底愿不愿意被“校准”，觉得系统懂不懂他。这就是 Study 2 的位置。

### K7. Observer Effect in Social Media Use（Saha, Gupta, Mark, Kiciman, De Choudhury；CHI 2024 最佳论文荣誉提名）
- 做了什么：300 多名参与者自愿共享平均 82 个月的 Facebook 历史，看加入研究（知道被观察）后 5 个月的变化；时间序列预测“本来会怎样”，再看偏离。
- 发现：认知能力高、神经质低的人加入后立刻少发帖；开放性高的人多发帖；自我相关内容减少，话题更分散。
- 对我们：Idea 2 的卖点是“不需要遥测”。Koustuv 的视角会把这一点从“工程上方便”提升成“少监视”：用户知道自己的每次停顿、每个按键都被记录，行为会变，数据本身也会偏。E2g 的“只看文本 + JEV 打平全部遥测”因此也是一个隐私和伦理上的结果。

### K8. Causal Factors of Effective Psychosocial Outcomes in Online Mental Health Communities（Saha, Sharma；ICWSM 2020）
- 做了什么：TalkLife 上约 3.9 万人的 30 多万条帖子；反过来问“什么样的回复导致了好的结果”：按情感、行为、认知三方面的心理社会变化定义结果，取改善最多（Case）和最少（Control）的人，在相似个体的匹配簇里比较他们收到的回复。
- 发现：有效的支持在于语言上的多样性、适应性和风格；回复的数量和速度这类简单指标在因果上不相关。
- 对我们：提醒我们区分“预测有用”和“因果有用”。比如 E2g 里遥测特征预测接受很准，但不说明它们是接受的原因；JEV 的增量也是预测上的。论文措辞要注意。

### K9. Linguistic Comparison of AI- and Human-Written Responses to Online Mental Health Queries（Saha, Jain, Rodriguez, De Choudhury；npj Artificial Intelligence 2026）
- 做了什么：Reddit 55 个心理健康社区的 24,114 个帖子、138,758 条社区回复；让 GPT-4-Turbo、Llama-3、Mistral-7B 回答同样的帖子，比较心理语言学和词汇语义特征，再做定性分析。
- 发现：AI 回复更长、更易读、更有分析结构，但缺少语言多样性和个人经历；立场中立，不会来回追问澄清。
- 对我们：LLM 判官可能偏好“像 LLM 写的”内容，这是判官的系统偏差；在需要“人味”的场景（同伴支持）里，固定判官的偏差可能不只是一个截距。

### K10. Sources of Inequity and Fairness Risks in Wellbeing Sensing（Zhang, Das Swain, Saha, Dey, Mankoff；AIES 2026）
- 做了什么：访谈 5 个国家的 14 位被动感知研究者和从业者，看公平风险在系统生命周期里从哪来、怎么传播。
- 发现：五种“情境性”的不平等来源（例如对被监测的接受程度、行为的规律性），它们系统性地影响公平，而且超出了按身份属性（性别、种族）做的事后比较；总结了 15 种风险和对应的缓解办法，以及制度上的障碍。
- 对我们：公平审计不能只按人口学分组。我们可以问：固定判官最对不上的是哪些人？很可能是标准最极端（最严或最松）的人，而不是某个人口学群体。每人截距恰恰是给这些人用的，这一点可以直接用数据检验。

## Eshwar Chandrasekharan（社区治理、内容审核、Reddit、人机协作的审核系统）

### E6. MoMoE: Mixture of Moderation Experts Framework for AI-Assisted Online Governance（Goyal, Zhan, Chen, Saha, Chandrasekharan；EMNLP 2025）
- 做了什么：四个算子 Allocate → Predict → Aggregate → Explain。两种专家：7 个“社区专家”（每个社区一个微调模型），5 个“违规类型专家”。新评论按相似度或分类概率分配给专家，加权合并，再用 LLM 事后生成解释。
- 关键数字：在 30 个没见过的社区上，Micro-F1 分别为 0.72（社区专家）和 0.67（违规类型专家），持平或超过微调基线；社区专家峰值高，违规类型专家更稳。
- 方法立场：反对“每个社区一个模型”（新社区没有历史数据），也反对黑箱；要跨社区、可解释。
- 对我们：同一个问题（新社区没有数据）的另一种解法。他们用专家组合，我们用固定判官 + 一个社区截距，少几个数量级的训练。可解释性方面，JEV 的每个问题都是一句可读的自然语言判断，天然可以拿来解释。

### E7. VASTU: Language Models Struggle to Recognize Online Community Values（Goyal, Zhan, Lambert, Saha, Chandrasekharan；EMNLP 2026 Oral）
- 做了什么：15 个社区的 75,000 条评论，标签 = 是否进入该社区点赞数前 5%（社区认可的内容，不是违规）。比较提示词大模型、LoRA 小模型、监督编码器和特征模型，分全局训练、本地训练、加上下文三种设置。
- 关键数字：监督编码器 AUROC 0.74，微调小模型 0.64–0.71，最好的提示词结果只有 0.62；朴素提示漏掉 80% 以上被社区认可的内容。本地模型一致胜过全局模型。父评论上下文帮助最大。
- 一句很关键的原话：只加社区描述的效果不稳定，“community descriptions shift decision thresholds rather than providing reliable alignment signal”；用 Schwartz 价值框架提示能改善阈值，但 AUROC 不变。
- 对我们：他们自己的数据说明，提示里加社区信息主要是在挪阈值，挪阈值正是每社区一个截距干的事，而且几条标签就能估。另一方面，排序本身（AUROC）的差距要靠社区特定的训练信号，截距补不了。这是对我们主张边界的提醒：截距修的是“标准高低”，修不了“看重什么”。

### E8. Venire: A Machine Learning-Guided Panel Review System for Community Content Moderation（Koshy, Choi, Chiang, Sundaram, Chandrasekharan, Karahalios；arXiv 2024 / CSCW）
- 做了什么：Reddit 版主队列上的插件，模型预测每位版主会怎么判，预测团队会分歧的案例就推荐交给多人小组投票。三个研究：版主访谈（6 人）、两个数据集上的技术评估、版主出声思考的使用研究。
- 模型：BERT，把评分者 ID 作为特殊 token 放在文本前（学每个人的嵌入），输出“这个人会怎么标”；分歧分数 = 随机两人预测不一致的概率。
- 关键数字：Kumar et al. (2021) 毒性数据（107,620 条评论、17,280 名评分者、每条 5 人）上，个人标签 AUROC 0.88；分歧预测不看是谁 0.69、看是谁 0.74。Prolific 众包按 r/ChangeMyView 规则标注的小数据上效果差一些：要把 60% 的案例交给小组才接近全员复审的一致性。版主普遍认为一致性重要，但要和工作量权衡。
- 原话里的限制：每人模型需要每条有多人标注的训练数据，“real moderation log data is largely singly-labeled”。
- 对我们：E1j（每人截距预测 3 人小组分歧，AUROC +0.09）就是 Venire 的问题。我们的做法不训练 BERT：内容难度来自固定判官，每人只有一个截距，可以从这个人自己的单人标注记录里估出来，正好对上 Venire 说的“版主日志几乎都是单人标注”。值得补的实验：照 Venire 的仿真（交给小组的预算 vs 找出的分歧、和多数意见的一致率）在 DICES 上跑一遍，另外检验只用单人标注的日志能不能学到同样的东西。

### E3. Crossmod: A Cross-Community Learning-based System to Assist Reddit Moderators（Chandrasekharan, Gandhi, Mustelier, Gilbert；CSCW 2019）
- 做了什么：访谈 11 位版主后设计的开源审核系统。后端是 108 个预训练分类器（100 个社区各一个“这个社区的版主会不会删”的分类器，加 8 个宏观规范违规分类器），对目标社区的新评论给出“如果发在 r/science，那里的版主会删吗”这样的反事实估计。
- 关键设计：agreement_score = 认为该删的源社区所占比例；版主用条件语句自己配置，比如 `if agreement_score > 0.95`，或者只看某几个相近社区（小社区借大社区的标准）。混合主动：系统推荐，人决定。
- 关键数字：在两个千万级订阅的社区里回放真实讨论，4 位版主审核；准确率 86%，召回 87.5% 以上；版主说 Crossmod 标出的评论里 95.3% 他们会删，而其中 98.3% 当时还在线上（现有审核漏掉了）。
- 对我们：Crossmod 的核心其实就是“固定的跨社区判官 + 每个社区自己定阈值”，只是阈值靠版主手写规则来定。我们的社区截距是把这个阈值从几条版主决定里自动估出来。这让 E4a 的对照很自然：TF-IDF 跨社区分类器就是 Crossmod 式基线。

### E2. The Internet's Hidden Rules: An Empirical Study of Reddit Norm Violations at Micro, Meso, and Macro Scales（Chandrasekharan, Samory, Jhaver, Charvat, Bruckman, Lampe, Eisenstein, Gilbert；CSCW 2018）
- 做了什么：100 个热门社区 10 个月里被版主删除的 280 万条评论；为每个社区训练“会不会被删”的分类器，再看这些分类器之间的一致程度，归纳出三层规范：宏观（全站多数社区都删：人身攻击、仇恨言论、骂版主等）、中观（一部分社区共有：例如不许个人经历、不许推广链接；只说谢谢的评论在一些社区也会被删）、微观（个别社区独有）。
- 方法立场：规范是涌现的、局部的；用大规模删帖记录 + 定性编码把“看不见的规则”找出来；用跨社区分类器的一致来定义规范的层级。
- 对我们：给“截距能修什么”画了边界。宏观规范上固定判官本来就对，截距只需要很小；中观、微观规范上社区和判官的差距更大，但如果差距在“看重什么内容”而不只是“标准高低”，一个截距补不了（和 VASTU 的结论一致）。E4a 里 r/soccerstreams 的推广规则就是中观规范的例子。SLM-Mod 和 VASTU 用的都是这份删帖数据。

### E5. Measuring User-Moderator Alignment on r/ChangeMyView（Koshy, Bajpai, Chandrasekharan, Sundaram, Karahalios；CSCW 2023）
- 做了什么：在 r/ChangeMyView 调查 798 名用户，分三层衡量用户和版主是否一致：知道规则吗（policy awareness）、知道规则怎么执行吗（practice awareness）、支持规则和执行方式吗（policy / practice support）。
- 关键数字：对规则本身的支持度高（5 条规则里 4 条超过 70% 支持），对执行方式的支持度低；层级贝叶斯模型估计，用户意见和版主决定的相关只有 0.14–0.45（因规则而异）；让用户预测版主会怎么做，相关也只高一点点，说明用户并不清楚规则实际怎么执行。
- 方法立场：多层次地量化“谁和谁不一致”；层级贝叶斯模型；主张给社区工具让他们自己做这种分析。
- 对我们：两点。① 我们的“每人截距”就是一个层级贝叶斯模型，和这篇的统计框架同源，Eshwar 组会觉得自然。② 它提出一个我们一直没正面回答的问题：校准到谁？版主、普通用户、受影响的群体，彼此的相关只有 0.14–0.45。我们的方法对目标是谁中立，但论文里必须说清楚每个实验校准的是哪一方。

### E1. You Can't Stay Here: The Efficacy of Reddit's 2015 Ban Examined Through Hate Speech（Chandrasekharan, Pavalanathan, Srinivasan, Glynn, Eisenstein, Gilbert；CSCW 2017）
- 做了什么：Reddit 2015 年封禁 r/fatpeoplehate 和 r/CoonTown，用 1 亿多条帖子和评论、自动生成的仇恨言论词典、匹配对照组和中断时间序列这类因果方法，看封禁对用户和“被迁入”社区的影响。
- 关键数字：比预期更多的账号离开了网站；留下来的用户仇恨言论至少减少 80%；接收“移民”的社区仇恨言论没有显著变化。
- 方法立场：用大规模行为数据做因果推断；结论落在平台政策上（封禁有用）。
- 对我们：这是 Eshwar 评价一切审核工具的标准：最终要看对社区行为的影响，而不是分类器的 AUROC。我们的实验全是离线预测，离“对社区有什么影响”还差一步。

### E4. A Just and Comprehensive Strategy for Using NLP to Address Online Abuse（Jurgens, Hemphill, Chandrasekharan；ACL 2019 立场论文）
- 主张三点：① 扩大问题范围，不只做仇恨言论这类窄定义，还要覆盖更隐蔽（微冒犯、居高临下）和更严重（威胁、人肉）的滥用；② 做主动式技术，在伤害发生之前抑制；③ 把工作放进“正义”的框架里：考虑受害者需要的是确认和解决，也考虑惩罚的公平和改过的机会。
- 对我们：“正义”框架下，个性化（每人截距）有两面：它尊重不同人的标准，但也可能把“对冒犯不敏感的人”的标准当作合理默认。论文里要讨论校准对谁有利、谁承担误判。

### E9. Does Positive Reinforcement Work?: A Quasi-Experimental Study of the Effects of Positive Feedback on Reddit（Lambert, Saha, Chandrasekharan；CHI 2025）
- 做了什么：4 个月 1,100 万条帖子，因果推断（匹配对照用户）看收到正面反馈（例如版主或社区的认可）对用户后续行为的影响，并看对谁效果最大。
- 关键数字：收到正面反馈的用户每天多发 2% 的帖子，帖子得分高 57%，每天被删的少 2%。
- 方法立场：审核不只是惩罚，还要鼓励好的行为；用准实验设计回答“有没有用”。
- 对我们：和 VASTU 一起说明这个组正在从“抓坏内容”转向“认出好内容”。固定判官 + 社区截距同样可以用在“这条会不会被社区认可”上。

### E10. "Think about it like you're…": How Reddit Moderators Use the Modqueue（Bajpai, Chandrasekharan；CHI 2026）
- 做了什么：调查 110 位版主（共管 400 多个社区），看版主队列（modqueue）怎么用、怎么排优先级、缺什么。
- 发现：用法差别很大（每日清单、看模式、找苗头）；经常要离开界面去找上下文、和队友协调；问题包括多人同时处理同一条（collision）、举报信号不完整或有噪声、界面版本割裂、依赖第三方工具。
- 对我们：我们的输出要落在 modqueue 的工作流里：按概率排序、给出“这条团队可能有分歧，请小组看”（Venire）、说明理由。举报信号有噪声，正是固定判官 + 校准可以补的地方（对每个举报来源、每位版主做校准）。
