# OSF 提交说明：r/Futurology 预注册

这一步需要你本人提交，因为要用你的 OSF 账号。提交之前，r/Futurology 的回访结果（outcomes 阶段）一律不跑。

## 步骤

1. 登录 https://osf.io，新建一个 project，例如 "LLM-simulated newcomers vs field experiments"。
2. 在这个 project 里选 Registrations → New registration，模板选 **"OSF Preregistration"**。如果想简单一些，可以选 "Open-Ended Registration"，把 `PREREG_futurology.md` 整篇贴进去。
3. 上传 3 个文件：
   - `PREREG_futurology.md`
   - `results/futurology/FREEZE.txt`（各次冻结的 SHA-256）
   - `src/futurology_analyze.py`（冻结的分析代码）
4. "Existing data" 一项，选 **"Registration prior to analysis of the data"**，并照实说明：
   - 帖子和评论的公开存档数据已经存在，我们也已经抓取了 agent 的输入，即发帖前的历史和线程；
   - 评论者进入实验之后的活动（结果变量）还没有抓取，也没有查看过；
   - 研究团队公开报告过：sticky 对 r/Futurology newcomer 的参与和守规率没有可检测的效应。他们没有报告回访，而回访正是我们的结果变量。
5. 选择公开或设 embargo，然后提交。拿到 registration DOI 或 URL 后发给我。
6. 我收到链接后会运行：

   ```
   python3 src/futurology_collect.py outcomes
   python3 src/futurology_analyze.py futurology-qwen3-8b futurology-gpt-4o-mini futurology-jev futurology-qwen3-32b
   ```

   然后把结果写进 `results/futurology/summary.md`，偏离计划的地方会逐条列出。

## 注意

- 如果在提交之前想改 `PREREG_futurology.md`，改完要重新运行 `src/freeze.py`，在 FREEZE.txt 里留一条新的记录，并写清原因。
- 如果你决定不预注册，直接揭晓也可以。但论文里就只能写成 "retrodiction"，不能写成 "pre-registered"。

## 补充（2026-11-06）：提交时必须写明的一点

截至提交时，所有 agent 分数都已经冻结，并且已经算过一次（freeze 3 和 freeze 4）。所以：

- 只用 agent 数据就能判定的假设（H2、H3、H5、H6、H7）**在提交前已经看过结果**，要在 "Existing data" 里如实说明。
- 结果：H2 成立；H3 不成立（gpt 的 rules −0.32）；H5 的个体内部分成立；H6 的 5 条全部成立；H7 成立（8B −0.60，32B +0.61）。
- 这几条的前瞻性来自另一件事：假设写在 freeze 2，比 freeze 3、4 的分数早；FREEZE.txt 里的时间戳可以证明这一点。
- 真正还没看过数据的是 **H1（主假设）和 H4**，两者都依赖人类 outcome。

建议在 OSF 表格的 "Existing data" 栏里写成：

> Agent scores (all models) were generated and frozen before registration; hypotheses H2, H3, H5–H7 concern agent outputs only and were evaluated before registration (results stated in the attached FREEZE.txt notes). Post-entry human activity (the outcome for H1 and H4) has not been collected or observed.

另外，P1 面板新加的模型（gemma、llama、ministral 等）在揭晓之前跑完，也会在 freeze 5 里冻结，作为探索性的前瞻预测。它们不属于预注册的假设。


## Addendum H10 (2026-11-07 02:22 UTC, freeze 5)

File `PREREG_addendum_H10.md` together with PREREG_futurology.md and PREREG_sites.md. It adds one secondary hypothesis
(panel median of open-weight models equivalent to the human effect within ±0.25; expected NOT supported) and was
hashed before any sealed-site outcome was fetched. The newer panel models' Futurology scores were hashed in the same freeze.
