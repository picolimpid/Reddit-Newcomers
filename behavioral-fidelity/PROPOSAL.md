## Project Proposal: Behavioral Fidelity of LLM Social Agents

**核心问题：Can LLM agents behave like real humans, rather than merely generate socially plausible responses?**

### Motivation

现有的 LLM 社交模拟器（如 OASIS）能够模拟用户发帖、评论和互动，但这些 Agent 的行为是否真正符合现实中的人类社会规律，仍然缺乏充分验证。

我们的 Reddit Newcomers 研究发现了一些有意思的 **behavioral fingerprints**：

- **Pre-entry trajectory matters:** 用户进入社区前最近 30 天的 community fit 能够预测长期留存，而更早的历史没有表现出同样的显著关联。
- **Social feedback has limited predictive value:** 是否收到回复、欢迎或排斥，与长期留存没有显著的观察关联。
- **Selection over socialization:** 留存更明显地与进入前的用户特征相关，而不是首次互动中的社交反馈。这支持 selection 的解释，但还不能证明因果关系。

这些结果为评估 LLM Agent 的 **Behavioral Fidelity** 提供了真实世界的参照。

### Research Questions

**RQ1 — Behavioral Fidelity:** LLM Agent 能否自然复现这些人类行为规律，而不只是生成看起来合理的社交反应？

**RQ2 — Social Cue Sensitivity:** LLM Agent 是否会对简单的社交信号（如 welcoming replies）过度敏感，而忽略更隐性的历史行为轨迹？

**RQ3 — Post-training:** 能否通过真实用户行为轨迹进行 SFT 或 RL，提高 Agent 的行为真实性以及跨社区泛化能力？

### Proposed Method

1. **Build a behavioral benchmark:** 从真实 Reddit 用户轨迹中提取评估任务，使用已有的 behavioral fingerprints 作为群体层面的检验指标。
2. **Evaluate base LLM agents:** 在 Reddit-like 模拟环境中，测试未经后训练的 Agent 是否能够复现真实用户的回访、再次参与等行为模式。
3. **Post-train agents:** 使用行为轨迹进行 SFT / Behavior Cloning，必要时尝试 RL，并与未经训练的模型比较。
4. **Evaluate generalization:** 检查 Agent 能否在未见过的社区中自然复现这些规律，而不是简单地拟合训练中给定的统计指标。

### Expected Contribution

提出一个基于真实社交平台行为规律的 LLM Agent 评估框架，并研究后训练能否缩小 **social plausibility 与 behavioral fidelity** 之间的差距。

**核心思路：Instead of predicting noisy individual outcomes, use real-world behavioral fingerprints to evaluate and improve the behavioral realism of LLM social agents.**
