# 读过的论文清单

更新 2026-11-05。所有阅读和归类都是我自己做的，没有用 JEV 或其他模型。按读到的深度分组：

| 组 | 读到什么程度 | 篇数 |
|---|---|---|
| A | 全文（arXiv 版，读了系统设计、研究方法和结果等相关章节） | 14 |
| B | 核心论文，但只拿到摘要或标题 | 3 |
| C | 试点用到的数据集和方法论文（用了数据或数字，没有通读） | 7 |
| D | 入选 100 篇中的其余论文：完整摘要 | 86 |
| E | 我标出来、读了完整摘要但没有入选 | 26 |
| F | 只看了标题：全部 947 篇检索结果（`data/candidates.jsonl`，不列出） | 947 |

“用在哪”一列标出它支撑哪个 idea；★ 表示 v1 的 JEV 初筛漏掉、v2 由我补进来的论文。

## A. 读了全文（14 篇）

| # | 简称 | 论文 | 会议 | 年份 | 链接 | 用在哪 | 要点 |
|---|---|---|---|---|---|---|---|
| 1 | **LLooM** | Concept Induction: Analyzing Unstructured Text with High-Level Concepts Using LLooM | CHI | 2024 | [DOI](https://doi.org/10.1145/3613904.3642830) · [arXiv](https://arxiv.org/abs/2404.12259) | Idea 1 | 文本→概念归纳；Score 步骤（每个概念×每条文本打分）占 79.9% 成本 |
| 2 | **CollabCoder** | CollabCoder: A Lower-barrier, Rigorous Workflow for Inductive Collaborative Qualitative Analysis with Large Language Models | CHI | 2024 | [DOI](https://doi.org/10.1145/3613904.3642002) · [arXiv](https://arxiv.org/abs/2304.07366) | Idea 1 | 两人质性编码：独立编码→按相似度排出分歧讨论→编码表 |
| 3 | **Prompting in the Dark** | Prompting in the Dark: Assessing Human Performance in Prompt Engineering for Data Labeling When Gold Labels Are Absent | CHI | 2025 | [DOI](https://doi.org/10.1145/3706598.3714319) · [arXiv](https://arxiv.org/abs/2502.11267) | Idea 1 | 没有金标签时迭代 prompt 不可靠：20 人中只有 9 人提高 |
| 4 | **DocWrangler** | Steering Semantic Data Processing With DocWrangler | UIST | 2025 | [DOI](https://doi.org/10.1145/3746059.3747625) · [arXiv](https://arxiv.org/abs/2504.14764) | Idea 1 | 语义数据处理 IDE，map/filter 等算子由 LLM 执行 |
| 5 | **Codellaborator** | Assistance or Disruption? Exploring and Evaluating the Design and Trade-offs of Proactive AI Programming Support | CHI | 2025 | [DOI](https://doi.org/10.1145/3706598.3713357) · [arXiv](https://arxiv.org/abs/2502.18658) | Idea 2 | 6 条规则触发的主动编程助手；398 次介入：53.3% 有效、12.1% 打扰、34.7% 被忽略 |
| 6 | **Need Help?** | Need Help? Designing Proactive AI Assistants for Programming | CHI | 2025 | [DOI](https://doi.org/10.1145/3706598.3714002) · [arXiv](https://arxiv.org/abs/2410.04596) | Idea 2 | 计时器触发的主动聊天助手；65 人，任务完成 +12–18%，建议过多时偏好减半 |
| 7 | **ProactiveBench** | Proactive Agent: Shifting LLM Agents from Reactive Responses to Active Assistance | ICLR | 2025 | [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/75c37811e830bf029584b1c6fac17726-Abstract-Conference.html) · [arXiv](https://arxiv.org/abs/2410.12361) | Idea 2（E2 数据） | 主动提议的接受/拒绝标注；1,640 条微调的奖励模型 0.917，GPT-4o 零样本 0.508 |
| 8 | **ComPeer** | ComPeer: A Generative Conversational Agent for Proactive Peer Support | UIST | 2024 | [DOI](https://doi.org/10.1145/3654777.3676430) · [arXiv](https://arxiv.org/abs/2407.18064) | Idea 2 | 每轮对话后 LLM 检测事件、排程主动消息 |
| 9 | **The Last JITAI?** | The Last JITAI? Exploring Large Language Models for Issuing Just-in-Time Adaptive Interventions: Fostering Physical Activity in a Prospective Cardiac Rehabilitation Setting | CHI | 2025 | [DOI](https://doi.org/10.1145/3706598.3713307) · [arXiv](https://arxiv.org/abs/2402.08658) | Idea 2 | GPT-4 决定是否触发健康干预并写消息，与 20 名人类比较 |
| 10 | **Memoro** | Memoro: Using Large Language Models to Realize a Concise Interface for Real-Time Memory Augmentation | CHI | 2024 | [DOI](https://doi.org/10.1145/3613904.3642450) · [arXiv](https://arxiv.org/abs/2403.02135) | Idea 2 | 实时记忆增强；无查询模式下主动推断记忆需求 |
| 11 | **EvalGen** | Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences | UIST | 2024 | [DOI](https://doi.org/10.1145/3654777.3676450) · [arXiv](https://arxiv.org/abs/2404.12272) | Idea 3 | 人给少量输出打分来对齐 LLM 评估断言；criteria drift |
| 12 | **EvalLM** | EvalLM: Interactive Evaluation of Large Language Model Prompts on User-Defined Criteria | CHI | 2024 | [DOI](https://doi.org/10.1145/3613904.3642216) · [arXiv](https://arxiv.org/abs/2309.13633) | Idea 3 | 用户自定义准则，GPT-4 逐准则比较输出 |
| 13 | **Data-Prompt Co-Evolution** | Data-Prompt Co-Evolution: Growing Test Sets to Refine LLM Behavior | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3791222) · [arXiv](https://arxiv.org/abs/2510.12728) | Idea 3 | 测试集与 prompt 共同演化 |
| 14 | **iRULER** | iRULER: Intelligible Rubric-Based User-Defined LLM Evaluation for Revision | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3790539) · [arXiv](https://arxiv.org/abs/2602.12779) | Idea 3 | 基于用户 rubric 的可解释写作评估 |

## B. 核心论文，但没拿到全文（3 篇）

| 简称 | 论文 | 会议 | 年份 | 链接 | 用在哪 | 读到什么程度 |
|---|---|---|---|---|---|---|
| **Wang et al.** | Human-LLM Collaborative Annotation Through Effective Verification of LLM Labels | CHI | 2024 | [DOI](https://doi.org/10.1145/3613904.3641960) | Idea 1（最接近的已有工作） | 完整摘要；ACM 拒绝下载全文 |
| **Live in the Loop** | Live in the Loop: Rapid Run-time Feedback for Prompts | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3791731) | Idea 3 | OpenAlex 只有两句摘要；全文未拿到 |
| **Talk to the Hand** | Talk to the Hand: an LLM-powered Chatbot with Visual Pointer as Proactive Companion for On-Screen Tasks | CHI | 2025 | [DOI](https://doi.org/10.1145/3706598.3715579) | Idea 2 | **只看了标题**（OpenAlex 没有摘要，ACM 拒绝下载），待补读 |

## C. 试点用到的数据集和方法论文（4 篇）

| 简称 | 论文 | 会议/期刊 | 年份 | 链接 | 用在哪 | 读到什么程度 |
|---|---|---|---|---|---|---|
| **MHS** | Measuring a hate speech spectrum with faceted Rasch item response theory and perspective-aware, explainable-by-design deep learning (Kennedy et al.) | arXiv | 2020 | [arXiv](https://arxiv.org/abs/2009.10277) | E1 数据 | 只用了数据集（Measuring Hate Speech 语料） |
| **SummEval** | SummEval: Re-evaluating Summarization Evaluation (Fabbri et al.) | TACL | 2021 | [DOI](https://doi.org/10.1162/tacl_a_00373) | E3 数据 | 只用了数据集 |
| **G-Eval** | G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment (Liu et al.) | EMNLP | 2023 | [DOI](https://doi.org/10.18653/v1/2023.emnlp-main.153) | E3 基线 | 用了它的评分 prompt 格式和报告的 Spearman，没有通读 |
| **CDHF** | When to Show a Suggestion? Integrating Human Feedback in AI-Assisted Programming (Mozannar et al.) | AAAI | 2024 | [DOI](https://doi.org/10.1609/aaai.v38i9.28878) · [arXiv](https://arxiv.org/abs/2306.04930) | Idea 2（最接近的已有工作；E2f 在 RealHumanEval 上复刻） | 读了正文第 1–8 节和附录 C（v7）：XGBoost 默认参数，特征含 Copilot 置信度、建议长度、关键词、前 5 次动作、codeBERTa 向量；跨用户 AUC 0.780，按会话划分并加程序员 ID 0.795，不看建议的第一阶段 0.631；藏 25.3% 的建议时其中 94.7% 本会被拒；把接受率当奖励会偏向短建议。附录中的证明和扩展相关工作没读 |
| **Jury Learning** | Jury Learning: Integrating Dissenting Voices into Machine Learning Models (Gordon et al.) | CHI | 2022 | [arXiv](https://arxiv.org/abs/2202.02950) | Idea 1 / Idea 2（按人建模标注者，v5 的 E1c） | 只读了 arXiv 摘要：为数据集里每个标注者建模，从中抽样组成评审团再分类；正文待读 |
| **DICES** | DICES Dataset: Diversity in Conversational AI Evaluation for Safety (Aroyo et al.) | NeurIPS Datasets | 2023 | [GitHub](https://github.com/google-research-datasets/dices-dataset) | E1e 数据（v6） | 只看了仓库说明和 DICES-350 的字段说明：350 段对话 × 123 名评分者（每人评全部，质量过滤后 104 人）、990 段 × 60–70 人；CC BY 4.0 |
| **RealHumanEval** | The RealHumanEval: Evaluating Large Language Models' Abilities to Support Programmers (Mozannar et al.) | arXiv | 2024 | [arXiv](https://arxiv.org/abs/2404.02806) · [数据](https://huggingface.co/datasets/hsseinmz/realhumaneval) | E2e 数据（v6，Idea 2 第一份带用户 ID 的接受日志） | 读了摘要、界面与日志一节（停顿 2 秒弹出补全，Tab 接受）和讨论中对自动补全的改进建议（参与者要求减少弹出频率；主动请求的建议接受率高 3–10 倍）；其余章节没读 |

## D. 入选 100 篇中的其余论文（读了完整摘要）

每篇的框架、模型模块和 JEV 切入点见 [PAPERS_100.md](PAPERS_100.md)。A、B 组已列出的不再重复。

### 标注 / 编码（19 篇）

| 论文 | 会议 | 年份 | 链接 | 用在哪 |
|---|---|---|---|---|
| DALL: Data Labeling via Data Programming and Active Learning Enhanced by Large Language Models | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3791356) | Idea 1 |
| Empathy Practices in Social Media Discourse: A Multidimensional and Relational Perspective | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3790967) | – |
| Mining Player Experience Trends From Game Reviews Using Large Language Models | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3790760) | – |
| “Are Compliments Bad Now?”: Comparing LLMs and Human Interpretations of Gender Microaggressions in the Workplace ★ | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3790436) | Idea 1 |
| ArtScape: Detecting, Characterizing, and Interactively Exploring Urban Art at Scale Using Vision Language Models | UIST | 2026 | [DOI](https://doi.org/10.1145/3830398.3830557) | – |
| Precise Duration Control for Video Editing via LLM Scoring and Constrained Optimization | UIST | 2026 | [DOI](https://doi.org/10.1145/3830398.3830510) | – |
| BIASsist: Empowering News Readers via Bias Identification, Explanation, and Neutralization | CHI | 2025 | [DOI](https://doi.org/10.1145/3706598.3713531) | – |
| End User Authoring of Personalized Content Classifiers: Comparing Example Labeling, Rule Writing, and LLM Prompting ★ | CHI | 2025 | [DOI](https://doi.org/10.1145/3706598.3713691) | Idea 1 |
| Media Bias Detector: Designing and Implementing a Tool for Real-Time Selection and Framing Bias Analysis in News Coverage | CHI | 2025 | [DOI](https://doi.org/10.1145/3706598.3713716) | – |
| ScreenAudit: Detecting Screen Reader Accessibility Errors in Mobile Apps Using Large Language Models | CHI | 2025 | [DOI](https://doi.org/10.1145/3706598.3713797) | – |
| What Social Media Use Do People Regret? An Analysis of 34K Smartphone Screenshots with Multimodal LLM ★ | CHI | 2025 | [DOI](https://doi.org/10.1145/3706598.3713724) | – |
| Collecting Qualitative Data at Scale with Large Language Models: A Case Study ★ | CSCW¹（PACM HCI 9(2)） | 2025 | [DOI](https://doi.org/10.1145/3710947) | – |
| Dynamic Surveys: Using LLMs to Blend Qualitative Depth, Quantitative Structure, and Collaborative Interaction | CSCW¹（PACM HCI 9(7)） | 2025 | [DOI](https://doi.org/10.1145/3757586) | – |
| Into the Crossfire: Evaluating the Use of a Language Model to Crowdsource Gun Violence Reports | CSCW¹（PACM HCI 9(7)） | 2025 | [DOI](https://doi.org/10.1145/3757416) | – |
| LATA: A Pilot Study on LLM-Assisted Thematic Analysis of Online Social Network Data Generation Experiences ★ | CSCW¹（PACM HCI 9(2)） | 2025 | [DOI](https://doi.org/10.1145/3711022) | – |
| Leveraging Prompt-Based Large Language Models: Predicting Pandemic Health Decisions and Outcomes Through Social Media Language | CHI | 2024 | [DOI](https://doi.org/10.1145/3613904.3642117) | – |
| Shortchanged: Uncovering and Analyzing Intimate Partner Financial Abuse in Consumer Complaints | CHI | 2024 | [DOI](https://doi.org/10.1145/3613904.3642033) | – |
| SenseMate: An Accessible and Beginner-Friendly Human-AI Platform for Qualitative Data Analysis | IUI | 2024 | [DOI](https://doi.org/10.1145/3640543.3645194) | Idea 1 |
| Linguistically Differentiating Acts and Recalls of Racial Microaggressions on Social Media | CSCW（PACM HCI 8, CSCW1） | 2024 | [DOI](https://doi.org/10.1145/3637366) | – |

### 状态检测（主动助手、干预时机）（23 篇）

| 论文 | 会议 | 年份 | 链接 | 用在哪 |
|---|---|---|---|---|
| AI of Oz: Enhancing Wizard of Oz Studies in HCI with AI Assistance for Human Moderation | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3791324) | – |
| An LLM-Based Motivation-Aware Framework For AI Coaching For Behaviour Change | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3791123) | – |
| Epimopilot : An LLM-infused Epistemic Emotion Support System to Boost Self-regulated Asynchronous Online Video Learning | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3791067) | – |
| Inferring Affect and Intervention Opportunities for Cancer Survivors from Digital Diaries with Context-Aware LLMs | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3791352) | Idea 2 |
| Just-In-Time Objectives: A General Approach for Specialized AI Interactions | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3790713) | Idea 2 |
| RageSense: Leveraging Acoustic Sensing and LLM-Based Intervention for Emotion Regulation in Mobile Gaming ★ | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3791076) | – |
| Sensing What Surveys Miss: Understanding and Personalizing Proactive LLM Support by User Modeling | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3791191) | Idea 2 |
| State Your Intention to Steer Your Attention: An AI Assistant for Intentional Digital Living | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3791404) | Idea 2 |
| The Invisible Mentor: Inferring User Actions from Screen Recordings to Recommend Better Workflows ★ | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3790294) | Idea 2 |
| Towards Better Health Conversations: The Benefits of Context-seeking | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3793199) | – |
| When Help Hurts: Verification Load and Fatigue with AI Coding Assistants ★ | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3791176) | Idea 2 |
| Guiding, Not Railroading: Design and Evaluation of a Multi-Agent System for Narrative Redirection in Role-playing Games | IUI | 2026 | [DOI](https://doi.org/10.1145/3742413.3789218) | – |
| Be My Ears: Enabling Semantic Alerting for Earphone Wearers with Pheme | UIST | 2026 | [DOI](https://doi.org/10.1145/3830398.3830640) | – |
| CoNarrate: Computationally Modeling Information Prematureness and Scaffolding the Co-Construction of Patient Narratives in Asynchronous Telehealth | UIST | 2026 | [DOI](https://doi.org/10.1145/3830398.3830658) | – |
| InsightToast: Proactive Information Retrieval & Glanceable Visualization in the Side Channel of Data-Rich Meetings | UIST | 2026 | [DOI](https://doi.org/10.1145/3830398.3830522) | Idea 2 |
| User Attention Is All chatLLMs Need: Proactive Intervention Against Attention Drift While Waiting for chatLLMs ★ | UIST | 2026 | [DOI](https://doi.org/10.1145/3830398.3830580) | Idea 2 |
| AiGet: Transforming Everyday Moments into Hidden Knowledge Discovery with AI Assistance on Smart Glasses | CHI | 2025 | [DOI](https://doi.org/10.1145/3706598.3713953) | Idea 2 |
| Exploring the Design Space of Real-time LLM Knowledge Support Systems: A Case Study of Jargon Explanations ★ | CHI | 2025 | [DOI](https://doi.org/10.1145/3706598.3714262) | Idea 2 |
| Gensors: Authoring Personalized Visual Sensors with Multimodal Foundation Models and Reasoning ★ | IUI | 2025 | [DOI](https://doi.org/10.1145/3708359.3712085) | – |
| AI That Helps Us Help Each Other: A Proactive System for Scaffolding Mentor-Novice Collaboration in Entrepreneurship Coaching ★ | CSCW¹（PACM HCI 9(7)） | 2025 | [DOI](https://doi.org/10.1145/3757549) | – |
| ProMemAssist: Exploring Timely Proactive Assistance Through Working Memory Modeling in Multi-Modal Wearable Devices | UIST | 2025 | [DOI](https://doi.org/10.1145/3746059.3747770) | Idea 2 |
| Human I/O: Towards a Unified Approach to Detecting Situational Impairments | CHI | 2024 | [DOI](https://doi.org/10.1145/3613904.3642065) | – |
| MindShift: Leveraging Large Language Models for Mental-States-Based Problematic Smartphone Use Intervention ★ | CHI | 2024 | [DOI](https://doi.org/10.1145/3613904.3642790) | Idea 2 |

### 评估（22 篇）

| 论文 | 会议 | 年份 | 链接 | 用在哪 |
|---|---|---|---|---|
| Building Benchmarks from the Ground Up: Community-Centered Evaluation of LLMs in Healthcare Chatbot Settings | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3791172) | – |
| Designing Staged Evaluation Workflows for LLMs: Integrating Domain Experts, Lay Users, and Model-Generated Evaluation Criteria ★ | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3790897) | Idea 3 |
| Evalet: Evaluating Large Language Models through Functional Fragmentation | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3790285) | – |
| EvaluAId: Human-AI Collaborative Evaluation of Open-Ended Student Essays | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3790814) | – |
| PASTA: A Scalable Framework for Multi-Policy AI Compliance Evaluation | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3790966) | – |
| ESR-Coach: Leveraging Large Language Models for Training People to Provide Emotionally Supportive Responses | IUI | 2026 | [DOI](https://doi.org/10.1145/3742413.3789094) | – |
| EvalAgent: Interactive Comparative Evaluation of Computer-Using GUI Agents | IUI | 2026 | [DOI](https://doi.org/10.1145/3742414.3795096) | – |
| MultEval: Collaboratively Creating Criteria for LLM-as-a-Judge Systems ★ | IUI | 2026 | [DOI](https://doi.org/10.1145/3742414.3795100) | Idea 3 |
| Smarter Together: Enhancing Human-AI Collaborative Grading With Teacher-Cognition Multi-Agent LLM Framework | IUI | 2026 | [DOI](https://doi.org/10.1145/3742413.3789130) | – |
| Prism-UX: Turning Evaluative Divergence into Traceable Design Evidence for LLM-Based Process-Level UX Evaluation ★ | UIST | 2026 | [DOI](https://doi.org/10.1145/3830398.3830622) | Idea 3 |
| Who's Keeping Score? Interactive Steering of LLM-Powered Scoring with Attune | UIST | 2026 | [DOI](https://doi.org/10.1145/3830398.3830567) | – |
| Efficient Management of LLM-Based Coaching Agents' Reasoning While Maintaining Interaction Quality and Speed | CHI | 2025 | [DOI](https://doi.org/10.1145/3706598.3713606) | – |
| Think Together and Work Better: Combining Humans' and LLMs' Think-Aloud Outcomes for Effective Text Evaluation | CHI | 2025 | [DOI](https://doi.org/10.1145/3706598.3713181) | – |
| Limitations of the LLM-as-a-Judge Approach for Evaluating LLM Outputs in Expert Knowledge Tasks ★ | IUI | 2025 | [DOI](https://doi.org/10.1145/3708359.3712091) | – |
| DxHF: Providing High-Quality Human Feedback for LLM Alignment with Interactive Decomposition ★ | UIST | 2025 | [DOI](https://doi.org/10.1145/3746059.3747600) | – |
| When Context Grows, So Does the Challenge: Human Oversight in LLM Evaluation of Financial Tables | UIST | 2025 | [DOI](https://doi.org/10.1145/3746058.3758457) | – |
| ChainForge: A Visual Toolkit for Prompt Engineering and LLM Hypothesis Testing ★ | CHI | 2024 | [DOI](https://doi.org/10.1145/3613904.3642016) | – |
| Generating Automatic Feedback on UI Mockups with Large Language Models | CHI | 2024 | [DOI](https://doi.org/10.1145/3613904.3642782) | – |
| HILL: A Hallucination Identifier for Large Language Models | CHI | 2024 | [DOI](https://doi.org/10.1145/3613904.3642428) | – |
| RELIC: Investigating Large Language Model Responses using Self-Consistency ★ | CHI | 2024 | [DOI](https://doi.org/10.1145/3613904.3641904) | Idea 3 |
| EvaluLLM: LLM assisted evaluation of generative outputs | IUI | 2024 | [DOI](https://doi.org/10.1145/3640544.3645216) | – |
| SituationAdapt: Contextual UI Optimization in Mixed Reality with Situation Awareness via LLM Reasoning | UIST | 2024 | [DOI](https://doi.org/10.1145/3654777.3676470) | – |

### 审核（5 篇）

| 论文 | 会议 | 年份 | 链接 | 用在哪 |
|---|---|---|---|---|
| CLOAK: A Privacy-Preserving Assistant for Online Social Media Users | UIST | 2026 | [DOI](https://doi.org/10.1145/3830398.3830723) | – |
| APOLLO: A GPT-based tool to detect phishing emails and generate explanations that warn users | PACM HCI 9(4)² | 2025 | [DOI](https://doi.org/10.1145/3733049) | – |
| Exploring Multidimensional Checkworthiness: Designing AI-assisted Claim Prioritization for Human Fact-checkers | CSCW¹（PACM HCI 9(7)） | 2025 | [DOI](https://doi.org/10.1145/3757473) | – |
| Twits, Toxic Tweets, and Tribal Tendencies: Trends in Politically Polarized Posts on Twitter | CSCW¹（PACM HCI 9(7)） | 2025 | [DOI](https://doi.org/10.1145/3757684) | – |
| Think Fast, Think Slow, Think Critical: Designing an Automated Propaganda Detection Tool | CHI | 2024 | [DOI](https://doi.org/10.1145/3613904.3642805) | – |

### 排序 / 选择（7 篇）

| 论文 | 会议 | 年份 | 链接 | 用在哪 |
|---|---|---|---|---|
| Promptimizer: User-Led Prompt Optimization for Personal Content Classification | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3790923) | Idea 1 |
| Social Media Feed Elicitation | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3791569) | – |
| RankCut: A Ranking-Based LLM Approach to Extractive Summarization for Transcript-Based Video Editing | IUI | 2026 | [DOI](https://doi.org/10.1145/3742413.3789115) | – |
| Leveraging Large Language Models for Collective Decision-Making | CSCW¹（PACM HCI 9(7)） | 2025 | [DOI](https://doi.org/10.1145/3757418) | – |
| CustomSight: Enhancing LLM-Powered Visual Assistance for Blind Individuals using Goal-Directed Dynamic Filters | UIST | 2025 | [DOI](https://doi.org/10.1145/3746058.3758401) | – |
| OmniActions: Predicting Digital Actions in Response to Real-World Multimodal Sensory Inputs with LLMs | CHI | 2024 | [DOI](https://doi.org/10.1145/3613904.3642068) | – |
| Embedding Democratic Values into Social Media AIs via Societal Objective Functions | CSCW（PACM HCI 8, CSCW1） | 2024 | [DOI](https://doi.org/10.1145/3641002) | – |

### 模拟（10 篇）

| 论文 | 会议 | 年份 | 链接 | 用在哪 |
|---|---|---|---|---|
| Can LLM-Simulated Practice and Feedback Upskill Human Counselors? A Randomized Study with 90+ Novice Counselors | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3791821) | – |
| CoBRA: Programming Cognitive Bias in Social Agents Using Classic Social Science Experiments | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3790804) | – |
| Emulating Aggregate Human Choice Behavior and Biases with GPT Conversational Agents ★ | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3791835) | – |
| Interview-Informed Generative Agents for Product Discovery: A Validation Study | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3791918) | – |
| Predicting Biased Human Decision-Making with Large Language Models in Conversational Settings ★ | IUI | 2026 | [DOI](https://doi.org/10.1145/3742413.3789127) | – |
| Proxona: Supporting Creators' Sensemaking and Ideation with LLM-Powered Audience Personas | CHI | 2025 | [DOI](https://doi.org/10.1145/3706598.3714034) | – |
| Scaffolding Empathy: Training Counselors with Simulated Patients and Utterance-level Performance Visualizations | CHI | 2025 | [DOI](https://doi.org/10.1145/3706598.3714014) | – |
| TeachTune: Reviewing Pedagogical Agents Against Diverse Student Profiles with Simulated Students | CHI | 2025 | [DOI](https://doi.org/10.1145/3706598.3714054) | – |
| Wisdom of the Crowd, Without the Crowd: A Socratic LLM for Asynchronous Deliberation on Perspectivist Data | CSCW¹（PACM HCI 9(7)） | 2025 | [DOI](https://doi.org/10.1145/3757707) | Idea 1 |
| Rehearsal: Simulating Conflict to Teach Conflict Resolution | CHI | 2024 | [DOI](https://doi.org/10.1145/3613904.3642159) | – |

## E. 读了完整摘要但没入选（26 篇）

v2 重读全部标题时我标出了 53 篇可能被漏掉的论文，读完整摘要后补进了 27 篇，下面是没补进的。

| 论文 | 会议 | 年份 | 链接 |
|---|---|---|---|
| AI-Mediated Feedback Improves Student Revisions: A Randomized Trial with FeedbackWriter in a Large Undergraduate Course | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3791121) |
| Behavioral Indicators of Overreliance During Interaction with Conversational Language Models | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3790332) |
| Forage: Understanding LLM-facilitated sensemaking of conversation data | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3791640) |
| From Expectation to Evaluation: Expectation Cues Systematically Bias LLM and Human Judgment | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3790492) |
| Live in the Loop: Rapid Run-time Feedback for Prompts | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3791731) |
| Open-ended Structured Question Assessment with Human-LLM Collaboration | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3791034) |
| PolicyPad: Collaborative Prototyping of LLM Policies | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3791689) |
| The Impact of Response Latency and Task Type on Human-LLM Interaction and Perception | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3790716) |
| TurnStyle: A Framework for Analyzing Human Conversational Behaviors to Predict Success in LLM-Assisted Tasks | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3790459) |
| Unveiling the Capabilities of Large Language Models in Simulating Student Behavioral Dynamics and Supporting Peer Feedback to Augment Task Performance | CHI | 2026 | [DOI](https://doi.org/10.1145/3772318.3790346) |
| Improving Human Verification of LLM Reasoning through Interactive Explanation Interfaces | IUI | 2026 | [DOI](https://doi.org/10.1145/3742413.3789134) |
| Veriscope: Provenance-Aware Verification of LLM Reasoning Artifacts in Conversational Data Analysis | UIST | 2026 | [DOI](https://doi.org/10.1145/3830398.3830599) |
| How Do Hackathons Foster Creativity? Towards Automated Evaluation of Creativity at Scale | CHI | 2025 | [DOI](https://doi.org/10.1145/3706598.3713447) |
| PLAID: Supporting Computing Instructors to Identify Domain-Specific Programming Plans at Scale | CHI | 2025 | [DOI](https://doi.org/10.1145/3706598.3713832) |
| Plurals: A System for Guiding LLMs via Simulated Social Ensembles | CHI | 2025 | [DOI](https://doi.org/10.1145/3706598.3713675) |
| Simulacrum of Stories: Examining Large Language Models as Qualitative Research Participants | CHI | 2025 | [DOI](https://doi.org/10.1145/3706598.3713220) |
| Talk to the Hand: an LLM-powered Chatbot with Visual Pointer as Proactive Companion for On-Screen Tasks | CHI | 2025 | [DOI](https://doi.org/10.1145/3706598.3715579) |
| Evaluation Workflows for Large Language Models (LLMs) that Integrate Domain Expertise for Complex Knowledge Tasks | IUI | 2025 | [DOI](https://doi.org/10.1145/3708557.3716146) |
| Expanding Automated Accessibility Evaluations: Leveraging Large Language Models for Heading-Related Barriers | IUI | 2025 | [DOI](https://doi.org/10.1145/3708557.3716329) |
| ShareFlows: Seamless Knowledge Capture and Proactive Push for Efficient Teacher Workflows in Higher Education | IUI | 2025 | [DOI](https://doi.org/10.1145/3708359.3712138) |
| DanModCap: Designing a Danmaku Moderation Tool for Video-Sharing Platforms that Leverages Impact Captions with Large Language Models | CSCW¹（PACM HCI 9(2)） | 2025 | [DOI](https://doi.org/10.1145/3710954) |
| MeetMap: Real-Time Collaborative Dialogue Mapping with LLMs in Online Meetings | CSCW¹（PACM HCI 9(2)） | 2025 | [DOI](https://doi.org/10.1145/3711030) |
| ThemeViz: Understanding the Effect of Human-AI Collaboration in Theme Development with an LLM-enhanced Interactive Visual System | CSCW¹（PACM HCI 9(7)） | 2025 | [DOI](https://doi.org/10.1145/3757675) |
| Towards Insider Summarization for Mediation Instead of Moderation: Examining Wikipedian Views on Key Elements of Discussion Summaries | CSCW¹（PACM HCI 9(7)） | 2025 | [DOI](https://doi.org/10.1145/3757681) |
| AI-Driven Mediation Strategies for Audience Depolarisation in Online Debates | CHI | 2024 | [DOI](https://doi.org/10.1145/3613904.3642322) |
| ConstitutionMaker: Interactively Critiquing Large Language Models by Converting Feedback into Principles | IUI | 2024 | [DOI](https://doi.org/10.1145/3640543.3645144) |

## 待读

| 论文 | 会议 | 年份 | 链接 | 为什么要读 |
|---|---|---|---|---|
| Principles of Mixed-Initiative User Interfaces (Horvitz) | CHI | 1999 | [链接](https://doi.org/10.1145/302979.303030) | Idea 2：用期望效用决定是否行动的经典框架，可作为阈值 τ = C/(B+C) 的依据 |
| Talk to the Hand（正文） | CHI | 2025 | [链接](https://doi.org/10.1145/3706598.3715579) | Idea 2：主动屏幕伙伴，需确认它怎么决定何时开口 |
| Jury Learning（正文） | CHI | 2022 | [链接](https://arxiv.org/abs/2202.02950) | Idea 1：每个标注者的模型用什么输入、怎么评估，和 E1c 的偏移模型怎么对照 |

---

¹ PACM HCI 9(2)（2025 年 5 月）和 9(7)（2025 年 10 月）按出版时间推断是 CSCW 的两期，没有逐篇核实；8(CSCW1) 由期号确认。
² 这一期对应哪个会议未核实。

生成脚本：`src/make_papers_read.py`（读 `data/final100.json`、`data/flagged_by_me.json`；卷期来自 OpenAlex，缓存在 `data/pacm_issues.json`）。
