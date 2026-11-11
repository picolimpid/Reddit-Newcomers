"""PAPERS_100.md, version 2 (2026-11-03): selection and notes are my own reading.

Changes vs v1 (src/paper_notes.py):
- v1 only looked at the 363 papers JEV's screen passed, and read abstracts truncated to 650-850 chars.
- v2: I read all 947 titles myself, full abstracts of every listed paper, and the arXiv full text of the papers the
  three ideas rely on (marked [全文]). 27 weak entries removed, 27 papers JEV's screen had dropped added, several
  notes corrected. The JEV-assigned tag column is removed.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from paper_notes import N as N1  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOVED = [48, 10, 33, 58, 28, 53, 77, 1, 63, 104, 121, 151, 103, 133, 76, 8, 36, 62, 108,
           39, 187, 196, 221, 57, 87, 218, 170]
FIX = {  # corrected / extended notes for retained papers (k in data/pool.jsonl)
    12: ("社区意见的质性编码；rationale extraction 模型（非 LLM，作者强调隐私和可控性）给主题推荐和理由。180 名新手实验：每条多花 49 秒，但编码者一致性 +29%、F 值 +10%",
         "整套 codebook 作为一组 noul 一次请求；概率提示新手哪些条目模型也拿不准", "高"),
    11: ("Pheme：LLM 理解查询→音频分离→LALM 验证检测结果。准确率 82.2%，误报率降 11.9%，API 调用减少 48.1%",
         "验证步（声音事件是否符合用户意图）用 noul，进一步降延迟和成本", "中"),
    15: ("CALLM：以相似同伴案例 + 个人轨迹为上下文，从极短日记推断情绪、调节意愿与可干预性（平衡准确率 60–74%）；作者发现 LLM 置信度与准确率相关，少量校准能改善个性化",
         "三个目标作为 noul/score 一次请求，概率直接用于即时干预阈值；与 E1 的锚点校准一致", "高"),
    0: ("TC-MAG：多 agent 模拟教师流程（建 rubric、盲双评、仲裁、交叉校验置信度）；2,000 份答卷上 κ=0.968；置信分数影响教师是否放手给系统",
        "每个 rubric 项一道 score 题，概率即置信度；只有低置信才进入 LLM 仲裁或教师复核", "高"),
    43: ("PASTA：多政策 AI 合规评估，LLM 成对评估引擎 + 省钱策略；与专家 ρ ≥ .626；5 项政策不到 2 分钟、约 $3",
         "同一份 model card × 大量政策条款的 noul（state 只计费一次），成本可再降一到两个数量级", "高"),
    5: ("社会目标函数：把反民主态度量表写成 LLM 打分，作为 feed 重排目标。人工标注 α=.895，模型与人工 ρ=.75；降权使党派敌意下降（d=.25）",
        "8 个变量一次请求，让在线重排变得可行；有现成人工标注可直接对比", "高"),
    130: ("教练 agent 推理管理：用差距信号决定投入多少推理，成本降约 10 倍且质量几乎不变",
          "差距信号本身可以换成 JEV 的概率判断（System-1/2 路由），对比现有机制", "高"),
    206: ("Dynamic Surveys：LLM 实时把开放回答聚类，再让答题者对聚类打分、排序、反思",
          "新回答归入已有聚类（choice）的实时分配步骤", "中"),
    143: ("Codellaborator：**规则触发**（空闲 30 秒、写完代码块、运行程序、写注释、选中代码）→ LLM 在动作列表里选一个（含不回应）[全文]。398 次主动介入：有效 53.3%、打扰 12.1%、被忽略 34.7%",
          "在 LLM 回应展示前加 JEV 验证（用户会接受吗？），针对近一半无效介入（对应 E2）", "高"),
    75: ("Need Help?：**计时器触发**（停止输入 5 秒后、至少间隔 20 秒），LLM 生成建议；作者明确没有给建议排序的机制；模型返回太慢时直接丢弃建议 [全文]",
         "给候选建议打接受概率并排序/过滤；约 0.1 秒的延迟能避免建议因超时被丢弃（对应 E2）", "高"),
    50: ("ComPeer：每轮对话后由 LLM 事件检测器抽取用户事件并推断合适时机，Schedule 模块安排主动消息 [全文]",
         "事件是否值得主动跟进、何时跟进的判断（对应 E2）", "高"),
    168: ("LLooM：LLM 归纳高层概念和纳入标准；Score 步骤对 n 条文本 × c 个概念用多选 prompt 得到近似概率，阈值可调，形成 n×c 矩阵；每次运行平均约 $1.44 [全文]",
          "Score 步骤：每个概念的纳入标准是一道 noul，一次请求覆盖全部概念，并给出真正的概率", "高"),
    159: ("EvalGen：LLM 生成准则和断言，人给少量好/坏评分来挑选对齐的实现，提出 criteria drift。选哪些输出给人评用交替策略；作者明确说其分数只表示输出差的可能性，**不能做不确定性采样** [全文]",
          "提供 EvalGen 缺的不确定性信号：按 JEV 熵挑选给人评的输出（对应 E3b）", "高"),
    55: ("Human-LLM 协作标注（Wang et al.）：LLM 给标签和解释→训练一个验证器（用输入、标签、解释的特征）→人只复核低分标签；验证器优于基于 logit 的基线",
         "验证器：比较 JEV 概率与训练验证器、logit 基线的效果和成本（对应 E1）", "高"),
    160: ("EvalLM：用户自定义准则，GPT-4 对输出对逐准则比较并给解释；与手工评估相比，参与者检查的输出多一倍、修改次数少 59% [全文]",
          "逐准则比较的执行端", "高"),
}
NEW = {  # idx in data/unseen_by_me.jsonl -> (role, framework & module, JEV slot, fit)
    255: ("annotate", "Prompting in the Dark：Google Sheets 插件里迭代 prompt 让 LLM 标注，**没有金标签**；20 人中只有 9 人迭代后准确率提升，DSPy 在金标签很少时也不行 [全文]",
          "用 JEV 熵挑出最值得人工标注的锚点，少量锚点再校准（直接对应 E1 的 20 条锚点结果）", "高"),
    114: ("annotate", "个人内容分类器的三种创建方式：举例标注、写规则、LLM prompt（37 人）；LLM prompt 最准，但三者都难以迭代改进",
          "规则写成 noul，输出概率和可调阈值，兼具写规则的可解释性和 LLM 的效果", "高"),
    82: ("annotate", "141 名参与者与 7 个 LLM 判断职场性别微侵犯的可能性：人与 LLM、不同性别和经历的人之间都有显著差异，主张检测系统接受解释多元性",
         "JEV 概率与不同人群的评分分布对照（E1 的同类问题）", "高"),
    467: ("annotate", "用 LLM 聊天机器人大规模收集质性数据（n=399）：回答很少有具体动机和例子；**LLM 与人类在质量和丰富度评分上的一致性很低**",
          "评分步骤：检验锚点校准能否修复 LLM–人类低一致性", "高"),
    453: ("annotate", "LATA：GPT-4/Gemini 按 Braun & Clarke 六阶段做主题分析，与人工结果的余弦相似度最高 0.76",
          "主题生成是生成式任务；主题确定后可用 JEV 做演绎编码", "中"),
    574: ("annotate", "DocWrangler：语义数据处理 IDE，map/filter/reduce 等算子由 LLM 执行，帮助用户在理解数据和写管线之间迭代 [全文]",
          "filter/分类算子由 JEV 执行：便宜、带概率、每次修改都能全量重跑", "高"),
    145: ("annotate", "每 5 秒截一次屏，用 MLLM 细粒度分析 3.4 万张截图，研究人们后悔的社交媒体使用",
          "逐张截图的活动分类（图像 choice），量大；受视觉能力限制", "中"),
    225: ("detect_state", "The Last JITAI?：GPT-4 根据人设和情境决定是否触发即时干预并生成消息；450 个决定与 10 名普通人、10 名医护人员对比，GPT-4 各项都更好 [全文]",
          "触发决策（现在该不该干预）用 noul 概率，生成仍由 LLM 负责", "高"),
    173: ("detect_state", "MindShift：结合应用使用、物理情境、心理状态（无聊/压力/惯性）和目标，由 LLM 生成说服内容；5 周现场实验，接受率 +4.7–22.5%",
          "心理状态 choice + 送达前预测用户是否会接受", "高"),
    598: ("detect_state", "Q2Focus：用户等待 LLM 回复时立即推送任务相关的小问题，注意力漂移下降 50%",
          "必须在提交后立刻出现：适合用 JEV 快速决定是否推送、推送哪个问题", "中"),
    702: ("detect_state", "Gensors：用户用自然语言定义个人视觉传感器（如\"孩子在捣乱时提醒我\"），系统拆成多条准则并行测试，MLLM 几秒内响应",
          "每条准则是一道图像 noul，同一帧上多题只计费一次、约 0.1 秒；需要先验证 JEV 的视觉判断", "中"),
    366: ("detect_state", "When Help Hurts：固定同一个 LLM，比较内联、聊天、结构化三种交互（N=60），提出验证负担指数",
          "只展示高接受概率的建议，以验证负担指数作为结果变量", "中"),
    107: ("detect_state", "RageSense：用近超声声学感知估计效价和唤醒度，LLM 结合截图生成情绪调节干预（53 人现场研究）",
          "感知不适合；可做干预内容选择和接受度检查", "中"),
    384: ("detect_state", "InvisibleMentor：VLM 从屏幕录像重建操作，LLM 推荐更高效的工作流",
          "推荐展示前判断其对该用户是否有用、是否新颖", "中"),
    497: ("detect_state", "创业辅导系统：领域风险认知模型 + LLM 主动提出诊断性问题；导师可以检查并修改底层模型（CSCW'25）",
          "风险诊断项写成 noul，导师修改模型等于修改 JEV 问题", "高"),
    147: ("detect_state", "StopGap：视频中实时解释术语的知识支持原型（N=24），发现偏好差异大，强调用户主导和个性化",
          "判断某个术语对这位用户是否需要解释（个性化阈值），只对判正的生成解释", "高"),
    72: ("evaluate", "Data-Prompt Co-Evolution：测试集与 prompt 一起演化，帮开发者发现边界情况、写明期望行为，并在不断增长的测试集上评估修改后的 prompt [全文]",
         "每次修改都在全部测试用例上重跑 JEV 检查；不确定条目即候选边界情况", "高"),
    141: ("evaluate", "分阶段评估工作流：比较专家、普通用户、LLM 三类来源的准则，发现各阶段间的漂移和后期收敛",
          "三类来源的准则作为同一 state 上的多题同时执行", "中"),
    609: ("evaluate", "MultEval：多人协作制定 LLM-as-judge 准则，尽早暴露分歧并诊断冲突来源",
          "每个人的准则版本各跑一遍，按概率差异定位分歧出现在哪些输出上", "高"),
    562: ("evaluate", "Prism-UX：把屏幕录像转成可引用的证据，从多个可配置视角评估并保留分歧；逻辑冲突率降 84%",
          "不同视角的判断作为同一 state 上的多题（只计费一次）", "中"),
    602: ("evaluate", "专家知识任务中的 LLM-as-judge：营养学和心理健康专家与 LLM 裁判的一致率只有 68% 和 64%",
          "检验 JEV + 少量专家锚点能否提高一致性；也是一个警示", "中"),
    5: ("evaluate", "ChainForge：开源可视化 prompt 工程与假设检验工具，跨模型、跨 prompt 比较大量回答",
        "评估节点：快速、带概率的批量判断", "高"),
    115: ("evaluate", "RELIC：用同一 LLM 多次采样的自洽性估计每个陈述的可信度，帮助用户核查长文本",
          "用 JEV 的陈述—证据概率替代 N 次采样，成本对比", "高"),
    546: ("evaluate", "DxHF：把长回答拆成陈述来收集人类偏好，并可视化陈述与对话的相关性；准确率 +5%，每条多花 18 秒",
          "陈述相关性评分（可视化编码所需的那一步）", "中"),
    112: ("evaluate", "iRULER：基于用户自定义 rubric 的写作评估，解释\"为什么是这个分、为什么不是更高分\"并给出改写示例 [全文]",
          "rubric 分数分布：相邻两档的概率就是\"为什么不是 4 分\"的量化依据", "中"),
    247: ("simulate", "用 GPT 对话 agent 模拟群体选择行为与偏差：1,100 人实验，GPT-4/5 能准确复现偏差",
          "choice 概率一次给出群体分布预测", "中"),
    646: ("simulate", "LLM 预测对话情境下的个人有偏决策（预注册，N=1,648），GPT-4 系列最贴近人类",
          "同上；可与 behavioral-fidelity 项目衔接", "中"),
}
ROLE_ZH = {"annotate": "标注 / 编码", "detect_state": "状态检测 / 介入时机", "evaluate": "评估 / 打分",
           "moderate": "审核 / 安全", "rank_select": "排序 / 选择", "simulate": "模拟用户 / 人设"}

if __name__ == "__main__":
    sel = [json.loads(l) for l in open(os.path.join(ROOT, "data", "selected100.jsonl"))]
    un = {r["idx"]: r for r in map(json.loads, open(os.path.join(ROOT, "data", "unseen_by_me.jsonl")))}
    rows = []
    for r in sel:
        if r["k"] in REMOVED:
            continue
        fw, slot, fit = FIX.get(r["k"], N1[r["k"]])
        rows.append({**{k: r[k] for k in ("role", "title", "venue", "year", "doi")}, "fw": fw, "slot": slot, "fit": fit, "added": False})
    for i, (role, fw, slot, fit) in NEW.items():
        u = un[i]
        doi = None
        for l in open(os.path.join(ROOT, "data", "candidates.jsonl")):
            c = json.loads(l)
            if c["id"] == u["id"]:
                doi = c["doi"]; break
        rows.append({"role": role, "title": u["title"], "venue": u["venue"], "year": u["year"], "doi": doi,
                     "fw": fw, "slot": slot, "fit": fit, "added": True})
    assert len(rows) == 100, len(rows)
    json.dump(rows, open(os.path.join(ROOT, "data", "final100.json"), "w"), ensure_ascii=False, indent=1)
    fits = {f: sum(r["fit"] == f for r in rows) for f in ("高", "中", "低")}
    out = ["# 100 篇论文：框架、模型模块与 JEV 切入点（v2）\n",
           "**来源**：OpenAlex 检索 CHI / CSCW（PACM-HCI）/ UIST / IUI 2024–2026、标题或摘要含 LLM 相关词的论文，共 947 篇。\n",
           "**怎么选的**：全部 947 个标题我都看过；入选的 100 篇我都读了完整摘要；三个 idea 依据的 15 篇读了 arXiv 全文的相关章节（标 [全文]）。"
           "v1 曾先用 JEV 初筛、只读截断摘要，v2 已改正：替换了 27 篇，标 ★ 的是被 JEV 初筛漏掉、这次补进来的。\n",
           "**适配度**：高 = 判断步骤是核心，输出是离散选项或分数，并且量大、要实时或靠阈值行动；中 = 有判断步骤，但和生成纠缠，或依赖较强的视觉能力；低 = 作用边缘。"
           f"本版分布：高 {fits['高']}，中 {fits['中']}，低 {fits['低']}。\n"]
    for role, zh in ROLE_ZH.items():
        rs = [r for r in rows if r["role"] == role]
        out.append(f"\n## {zh}（{len(rs)} 篇）\n")
        out.append("| 论文 | 会议 | 框架与模型模块 | JEV 可替换或新增的环节 | 适配度 |")
        out.append("|---|---|---|---|---|")
        for r in rs:
            t = r["title"].replace("|", "/")
            link = f"[{t}]({r['doi']})" if r.get("doi") else t
            star = "★ " if r["added"] else ""
            out.append(f"| {star}{link} | {r['venue'].replace('PACM-HCI(CSCW)', 'CSCW')} {r['year']} | {r['fw']} | {r['slot']} | {r['fit']} |")
    open(os.path.join(ROOT, "PAPERS_100.md"), "w").write("\n".join(out) + "\n")
    print(fits, {z: sum(r["role"] == k for r in rows) for k, z in ROLE_ZH.items()})
