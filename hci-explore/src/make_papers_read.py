"""Build PAPERS_READ.md: every paper read for this project, with venue, year, link, depth of reading and which idea
it informs.  Sources: data/final100.json (the 100 selected, full abstracts read), data/flagged_by_me.json +
data/unseen_by_me.jsonl (candidates I flagged and read in full but did not select), and hand-written tables for the
full-text papers and the dataset/method papers.  PACM HCI volume/issue comes from OpenAlex (cached in
data/pacm_issues.json).  No model was used to read or tag anything."""
import json
import os
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = lambda *p: os.path.join(ROOT, *p)
final = json.load(open(D("data", "final100.json")))
unseen = [json.loads(l) for l in open(D("data", "unseen_by_me.jsonl"))]
flagged = [unseen[i] for i in json.load(open(D("data", "flagged_by_me.json"))).values()]
doi_of = {c["id"]: c.get("doi") for c in map(json.loads, open(D("data", "candidates.jsonl")))}
for x in flagged:
    x["doi"] = doi_of.get(x["id"])
titles100 = {x["title"] for x in final}

# ---- PACM HCI issues (OpenAlex), cached
cache_p = D("data", "pacm_issues.json")
issues = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
s = requests.Session(); s.headers["User-Agent"] = "hci-explore/0.1"
for x in final + flagged:
    if x["venue"].startswith("PACM") and x["doi"] and x["doi"] not in issues:
        r = s.get("https://api.openalex.org/works/doi:" + x["doi"].replace("https://doi.org/", ""),
                  params={"select": "biblio"}, timeout=30)
        b = r.json().get("biblio", {}) if r.status_code == 200 else {}
        issues[x["doi"]] = [b.get("volume"), b.get("issue")]
        time.sleep(0.2)
json.dump(issues, open(cache_p, "w"), indent=0)


def venue(x):
    v = x["venue"]
    if not v.startswith("PACM"):
        return v
    vol, iss = issues.get(x["doi"], [None, None])
    if iss and str(iss).startswith("CSCW"):
        return f"CSCW（PACM HCI {vol}, {iss}）"
    if (vol, iss) in (("9", "7"), ("9", "2")):
        return f"CSCW¹（PACM HCI {vol}({iss})）"
    return f"PACM HCI {vol}({iss})²" if vol else "PACM HCI"


IDEA = {  # title prefix -> idea it informs
    "Idea 1": ["Concept Induction", "CollabCoder", "Prompting in the Dark", "Steering Semantic Data Processing",
               "Human-LLM Collaborative Annotation", "SenseMate", "DALL", "Promptimizer",
               "End User Authoring of Personalized Content Classifiers", "“Are Compliments Bad Now?”",
               "Wisdom of the Crowd, Without the Crowd"],
    "Idea 2": ["Assistance or Disruption?", "Need Help?", "ComPeer", "The Last JITAI?", "Memoro", "ProMemAssist",
               "Sensing What Surveys Miss", "Exploring the Design Space of Real-time LLM Knowledge Support",
               "When Help Hurts", "Inferring Affect and Intervention Opportunities", "MindShift", "AiGet",
               "InsightToast", "User Attention Is All chatLLMs Need", "The Invisible Mentor",
               "State Your Intention to Steer Your Attention", "Just-In-Time Objectives"],
    "Idea 3": ["Who Validates the Validators?", "EvalLM", "Data-Prompt Co-Evolution", "iRULER", "MultEval",
               "Prism-UX", "Designing Staged Evaluation Workflows", "RELIC"],
}
FULLTEXT = ["Concept Induction", "CollabCoder", "Prompting in the Dark", "Steering Semantic Data Processing",
            "Assistance or Disruption?", "Need Help?", "ComPeer", "The Last JITAI?", "Memoro",
            "Who Validates the Validators?", "EvalLM", "Data-Prompt Co-Evolution", "iRULER"]


def idea_of(t):
    return ", ".join(k for k, v in IDEA.items() if any(t.startswith(p) for p in v))


def link(doi):
    return f"[DOI]({doi})" if doi else "–"


# ---- A: full text (hand-written; venue/year/DOI copied from final100 or verified via Crossref / arXiv)
A = [  # short, title, venue, year, links, idea, gist
    ("LLooM", "Concept Induction: Analyzing Unstructured Text with High-Level Concepts Using LLooM", "CHI", 2024,
     "[DOI](https://doi.org/10.1145/3613904.3642830) · [arXiv](https://arxiv.org/abs/2404.12259)", "Idea 1",
     "文本→概念归纳；Score 步骤（每个概念×每条文本打分）占 79.9% 成本"),
    ("CollabCoder", "CollabCoder: A Lower-barrier, Rigorous Workflow for Inductive Collaborative Qualitative Analysis with Large Language Models",
     "CHI", 2024, "[DOI](https://doi.org/10.1145/3613904.3642002) · [arXiv](https://arxiv.org/abs/2304.07366)", "Idea 1",
     "两人质性编码：独立编码→按相似度排出分歧讨论→编码表"),
    ("Prompting in the Dark", "Prompting in the Dark: Assessing Human Performance in Prompt Engineering for Data Labeling When Gold Labels Are Absent",
     "CHI", 2025, "[DOI](https://doi.org/10.1145/3706598.3714319) · [arXiv](https://arxiv.org/abs/2502.11267)", "Idea 1",
     "没有金标签时迭代 prompt 不可靠：20 人中只有 9 人提高"),
    ("DocWrangler", "Steering Semantic Data Processing With DocWrangler", "UIST", 2025,
     "[DOI](https://doi.org/10.1145/3746059.3747625) · [arXiv](https://arxiv.org/abs/2504.14764)", "Idea 1",
     "语义数据处理 IDE，map/filter 等算子由 LLM 执行"),
    ("Codellaborator", "Assistance or Disruption? Exploring and Evaluating the Design and Trade-offs of Proactive AI Programming Support",
     "CHI", 2025, "[DOI](https://doi.org/10.1145/3706598.3713357) · [arXiv](https://arxiv.org/abs/2502.18658)", "Idea 2",
     "6 条规则触发的主动编程助手；398 次介入：53.3% 有效、12.1% 打扰、34.7% 被忽略"),
    ("Need Help?", "Need Help? Designing Proactive AI Assistants for Programming", "CHI", 2025,
     "[DOI](https://doi.org/10.1145/3706598.3714002) · [arXiv](https://arxiv.org/abs/2410.04596)", "Idea 2",
     "计时器触发的主动聊天助手；65 人，任务完成 +12–18%，建议过多时偏好减半"),
    ("ProactiveBench", "Proactive Agent: Shifting LLM Agents from Reactive Responses to Active Assistance", "ICLR", 2025,
     "[ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/75c37811e830bf029584b1c6fac17726-Abstract-Conference.html) · [arXiv](https://arxiv.org/abs/2410.12361)",
     "Idea 2（E2 数据）", "主动提议的接受/拒绝标注；1,640 条微调的奖励模型 0.917，GPT-4o 零样本 0.508"),
    ("ComPeer", "ComPeer: A Generative Conversational Agent for Proactive Peer Support", "UIST", 2024,
     "[DOI](https://doi.org/10.1145/3654777.3676430) · [arXiv](https://arxiv.org/abs/2407.18064)", "Idea 2",
     "每轮对话后 LLM 检测事件、排程主动消息"),
    ("The Last JITAI?", "The Last JITAI? Exploring Large Language Models for Issuing Just-in-Time Adaptive Interventions: Fostering Physical Activity in a Prospective Cardiac Rehabilitation Setting",
     "CHI", 2025, "[DOI](https://doi.org/10.1145/3706598.3713307) · [arXiv](https://arxiv.org/abs/2402.08658)", "Idea 2",
     "GPT-4 决定是否触发健康干预并写消息，与 20 名人类比较"),
    ("Memoro", "Memoro: Using Large Language Models to Realize a Concise Interface for Real-Time Memory Augmentation",
     "CHI", 2024, "[DOI](https://doi.org/10.1145/3613904.3642450) · [arXiv](https://arxiv.org/abs/2403.02135)", "Idea 2",
     "实时记忆增强；无查询模式下主动推断记忆需求"),
    ("EvalGen", "Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences",
     "UIST", 2024, "[DOI](https://doi.org/10.1145/3654777.3676450) · [arXiv](https://arxiv.org/abs/2404.12272)", "Idea 3",
     "人给少量输出打分来对齐 LLM 评估断言；criteria drift"),
    ("EvalLM", "EvalLM: Interactive Evaluation of Large Language Model Prompts on User-Defined Criteria", "CHI", 2024,
     "[DOI](https://doi.org/10.1145/3613904.3642216) · [arXiv](https://arxiv.org/abs/2309.13633)", "Idea 3",
     "用户自定义准则，GPT-4 逐准则比较输出"),
    ("Data-Prompt Co-Evolution", "Data-Prompt Co-Evolution: Growing Test Sets to Refine LLM Behavior", "CHI", 2026,
     "[DOI](https://doi.org/10.1145/3772318.3791222) · [arXiv](https://arxiv.org/abs/2510.12728)", "Idea 3",
     "测试集与 prompt 共同演化"),
    ("iRULER", "iRULER: Intelligible Rubric-Based User-Defined LLM Evaluation for Revision", "CHI", 2026,
     "[DOI](https://doi.org/10.1145/3772318.3790539) · [arXiv](https://arxiv.org/abs/2602.12779)", "Idea 3",
     "基于用户 rubric 的可解释写作评估"),
]
B = [  # core papers whose full text I could not get
    ("Wang et al.", "Human-LLM Collaborative Annotation Through Effective Verification of LLM Labels", "CHI", 2024,
     "https://doi.org/10.1145/3613904.3641960", "Idea 1（最接近的已有工作）", "完整摘要；ACM 拒绝下载全文"),
    ("Live in the Loop", "Live in the Loop: Rapid Run-time Feedback for Prompts", "CHI", 2026,
     "https://doi.org/10.1145/3772318.3791731", "Idea 3", "OpenAlex 只有两句摘要；全文未拿到"),
    ("Talk to the Hand", "Talk to the Hand: an LLM-powered Chatbot with Visual Pointer as Proactive Companion for On-Screen Tasks",
     "CHI", 2025, "https://doi.org/10.1145/3706598.3715579", "Idea 2", "**只看了标题**（OpenAlex 没有摘要，ACM 拒绝下载），待补读"),
]
C = [  # datasets and method papers used in the pilots
    ("MHS", "Measuring a hate speech spectrum with faceted Rasch item response theory and perspective-aware, explainable-by-design deep learning (Kennedy et al.)",
     "arXiv", 2020, "[arXiv](https://arxiv.org/abs/2009.10277)", "E1 数据", "只用了数据集（Measuring Hate Speech 语料）"),
    ("SummEval", "SummEval: Re-evaluating Summarization Evaluation (Fabbri et al.)", "TACL", 2021,
     "[DOI](https://doi.org/10.1162/tacl_a_00373)", "E3 数据", "只用了数据集"),
    ("G-Eval", "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment (Liu et al.)", "EMNLP", 2023,
     "[DOI](https://doi.org/10.18653/v1/2023.emnlp-main.153)", "E3 基线", "用了它的评分 prompt 格式和报告的 Spearman，没有通读"),
    ("CDHF", "When to Show a Suggestion? Integrating Human Feedback in AI-Assisted Programming (Mozannar et al.)", "AAAI", 2024,
     "[DOI](https://doi.org/10.1609/aaai.v38i9.28878) · [arXiv](https://arxiv.org/abs/2306.04930)", "Idea 2（最接近的已有工作；E2f 在 RealHumanEval 上复刻）",
     "读了正文第 1–8 节和附录 C（v7）：XGBoost 默认参数，特征含 Copilot 置信度、建议长度、关键词、前 5 次动作、codeBERTa 向量；跨用户 AUC 0.780，按会话划分并加程序员 ID 0.795，不看建议的第一阶段 0.631；藏 25.3% 的建议时其中 94.7% 本会被拒；把接受率当奖励会偏向短建议。附录中的证明和扩展相关工作没读"),
    ("Jury Learning", "Jury Learning: Integrating Dissenting Voices into Machine Learning Models (Gordon et al.)", "CHI", 2022,
     "[arXiv](https://arxiv.org/abs/2202.02950)", "Idea 1 / Idea 2（按人建模标注者，v5 的 E1c）",
     "只读了 arXiv 摘要：为数据集里每个标注者建模，从中抽样组成评审团再分类；正文待读"),
    ("DICES", "DICES Dataset: Diversity in Conversational AI Evaluation for Safety (Aroyo et al.)", "NeurIPS Datasets", 2023,
     "[GitHub](https://github.com/google-research-datasets/dices-dataset)", "E1e 数据（v6）",
     "只看了仓库说明和 DICES-350 的字段说明：350 段对话 × 123 名评分者（每人评全部，质量过滤后 104 人）、990 段 × 60–70 人；CC BY 4.0"),
    ("RealHumanEval", "The RealHumanEval: Evaluating Large Language Models' Abilities to Support Programmers (Mozannar et al.)", "arXiv", 2024,
     "[arXiv](https://arxiv.org/abs/2404.02806) · [数据](https://huggingface.co/datasets/hsseinmz/realhumaneval)", "E2e 数据（v6，Idea 2 第一份带用户 ID 的接受日志）",
     "读了摘要、界面与日志一节（停顿 2 秒弹出补全，Tab 接受）和讨论中对自动补全的改进建议（参与者要求减少弹出频率；主动请求的建议接受率高 3–10 倍）；其余章节没读"),
]
TODO = [
    ("Principles of Mixed-Initiative User Interfaces (Horvitz)", "CHI", 1999, "https://doi.org/10.1145/302979.303030",
     "Idea 2：用期望效用决定是否行动的经典框架，可作为阈值 τ = C/(B+C) 的依据"),
    ("Talk to the Hand（正文）", "CHI", 2025, "https://doi.org/10.1145/3706598.3715579", "Idea 2：主动屏幕伙伴，需确认它怎么决定何时开口"),
    ("Jury Learning（正文）", "CHI", 2022, "https://arxiv.org/abs/2202.02950", "Idea 1：每个标注者的模型用什么输入、怎么评估，和 E1c 的偏移模型怎么对照"),
]
ROLE = {"annotate": "标注 / 编码", "detect_state": "状态检测（主动助手、干预时机）", "evaluate": "评估",
        "moderate": "审核", "rank_select": "排序 / 选择", "simulate": "模拟"}

out = ["# 读过的论文清单", "",
       "更新 2026-11-05。所有阅读和归类都是我自己做的，没有用 JEV 或其他模型。按读到的深度分组：",
       "",
       "| 组 | 读到什么程度 | 篇数 |", "|---|---|---|",
       f"| A | 全文（arXiv 版，读了系统设计、研究方法和结果等相关章节） | {len(A)} |",
       f"| B | 核心论文，但只拿到摘要或标题 | {len(B)} |",
       f"| C | 试点用到的数据集和方法论文（用了数据或数字，没有通读） | {len(C)} |",
       f"| D | 入选 100 篇中的其余论文：完整摘要 | {sum(1 for x in final if not any(x['title'].startswith(p) for p in FULLTEXT) and x['doi'] not in {b[4] for b in B})} |",
       f"| E | 我标出来、读了完整摘要但没有入选 | {sum(1 for x in flagged if x['title'] not in titles100)} |",
       "| F | 只看了标题：全部 947 篇检索结果（`data/candidates.jsonl`，不列出） | 947 |",
       "", "“用在哪”一列标出它支撑哪个 idea；★ 表示 v1 的 JEV 初筛漏掉、v2 由我补进来的论文。", ""]

out += ["## A. 读了全文（14 篇）", "", "| # | 简称 | 论文 | 会议 | 年份 | 链接 | 用在哪 | 要点 |", "|---|---|---|---|---|---|---|---|"]
out += [f"| {i} | **{a[0]}** | {a[1]} | {a[2]} | {a[3]} | {a[4]} | {a[5]} | {a[6]} |" for i, a in enumerate(A, 1)]
out += ["", "## B. 核心论文，但没拿到全文（3 篇）", "", "| 简称 | 论文 | 会议 | 年份 | 链接 | 用在哪 | 读到什么程度 |", "|---|---|---|---|---|---|---|"]
out += [f"| **{b[0]}** | {b[1]} | {b[2]} | {b[3]} | [DOI]({b[4]}) | {b[5]} | {b[6]} |" for b in B]
out += ["", "## C. 试点用到的数据集和方法论文（4 篇）", "", "| 简称 | 论文 | 会议/期刊 | 年份 | 链接 | 用在哪 | 读到什么程度 |", "|---|---|---|---|---|---|---|"]
out += [f"| **{c[0]}** | {c[1]} | {c[2]} | {c[3]} | {c[4]} | {c[5]} | {c[6]} |" for c in C]

out += ["", "## D. 入选 100 篇中的其余论文（读了完整摘要）", "",
        "每篇的框架、模型模块和 JEV 切入点见 [PAPERS_100.md](PAPERS_100.md)。A、B 组已列出的不再重复。", ""]
bdois = {b[4] for b in B}
for role, name in ROLE.items():
    rows = [x for x in final if x["role"] == role and not any(x["title"].startswith(p) for p in FULLTEXT) and x["doi"] not in bdois]
    rows.sort(key=lambda x: (-x["year"], x["venue"], x["title"]))
    out += [f"### {name}（{len(rows)} 篇）", "", "| 论文 | 会议 | 年份 | 链接 | 用在哪 |", "|---|---|---|---|---|"]
    out += [f"| {x['title']}{' ★' if x['added'] else ''} | {venue(x)} | {x['year']} | {link(x['doi'])} | {idea_of(x['title']) or '–'} |" for x in rows]
    out.append("")

rest = sorted((x for x in flagged if x["title"] not in titles100), key=lambda x: (-x["year"], x["venue"], x["title"]))
out += [f"## E. 读了完整摘要但没入选（{len(rest)} 篇）", "",
        "v2 重读全部标题时我标出了 53 篇可能被漏掉的论文，读完整摘要后补进了 27 篇，下面是没补进的。", "",
        "| 论文 | 会议 | 年份 | 链接 |", "|---|---|---|---|"]
out += [f"| {x['title']} | {venue(x)} | {x['year']} | {link(x.get('doi'))} |" for x in rest]
out += ["", "## 待读", "", "| 论文 | 会议 | 年份 | 链接 | 为什么要读 |", "|---|---|---|---|---|"]
out += [f"| {t[0]} | {t[1]} | {t[2]} | [链接]({t[3]}) | {t[4]} |" for t in TODO]
out += ["", "---", "",
        "¹ PACM HCI 9(2)（2025 年 5 月）和 9(7)（2025 年 10 月）按出版时间推断是 CSCW 的两期，没有逐篇核实；8(CSCW1) 由期号确认。",
        "² 这一期对应哪个会议未核实。", "",
        "生成脚本：`src/make_papers_read.py`（读 `data/final100.json`、`data/flagged_by_me.json`；卷期来自 OpenAlex，缓存在 `data/pacm_issues.json`）。"]
open(D("PAPERS_READ.md"), "w").write("\n".join(out) + "\n")
print("wrote PAPERS_READ.md;", "A", len(A), "B", len(B), "C", len(C), "E", len(rest))
