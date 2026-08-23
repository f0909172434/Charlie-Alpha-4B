# FORGE：一晚内可验证的三语选择性蒸馏

FORGE 是 **Focused One-pass Relative-Gap Gradient Equivalence** 的缩写，也是 Charlie
alpha v0.2 面向 24 GB Apple Silicon 笔记本的研究配方。它不宣称创造新的基础模型层，
而是把多个可独立验证的想法组合为一套受算力约束的训练系统。唯一成功标准是：在冻结的
全新测试中，正确率必须高于同一个 Qwen3.5-4B 底模。

## 设计

1. **适配混合架构。** Apache-2.0 的 Qwen3.5-4B 混合了 Gated DeltaNet 线性注意力与
   完整注意力。LoRA 同时覆盖两条路径，而不是只训练完整注意力层。
2. **逐 token 相对差距。** 每条已验证、去污染的答案分别通过 4B 学生与同 tokenizer 的
   9B 教师做一次 teacher-forced 前向。优先选择学生损失高于教师的样本；训练只保留正差距
   最大的 token，并始终保留答案末尾至少 32 tokens。
3. **能力平衡 replay。** 52 个 optimizer 更新组固定为数学 26、Python 13、C++ 13；
   多样性约束避免单一模板只因损失较高而占满数据。
4. **三语耦合更新。** 每组先放同一题的英文、简中、繁中版本，再放五条同能力英文 replay，
   八个 microsteps 合并为一次更新。样本权重让英文／简中／繁中的名义 loss 质量精确为
   70%／15%／15%。
5. **无损翻译。** 程序码、LaTeX、URL 和数字先替换成不可更改的 placeholder，只让 9B
   教师生成一次简中翻译，再用保护区段的 OpenCC 产生繁中。所有 placeholder 与结构在接纳
   前都会逐项验证。
6. **等成本消融。** 比较最后 16 层 rank-8 LoRA、rank-8 LoRA+、全部 32 层 rank-4
   LoRA+；三者可训练参数必须完全相等。九题锁定三语 canary 先比较正确率，再以验证 loss 与
   耗时打破平手。

设计依据包括 [Rho-1](https://arxiv.org/abs/2404.07965)、
[LESS](https://arxiv.org/abs/2402.04333)、[BIDS](https://arxiv.org/abs/2501.12147)、
[LoRA+](https://arxiv.org/abs/2402.12354)、[xCoT](https://arxiv.org/abs/2401.07037)、
[STaR](https://arxiv.org/abs/2203.14465)、[LIMO](https://arxiv.org/abs/2502.03387) 与
[s1](https://arxiv.org/abs/2501.19393)。这些论文支持各个组件，并不能证明组合后必然有效；
FORGE 是否有效只能由锁定评测决定。

## 评测防火墙

- 已提交的题锁包含 34 题开发集与 62 题最终集；两者彼此不重叠，也不与 v0.1 评测重叠，
  每道规范化题目都有 SHA-256 指纹。
- 开发结果可以选择配方。只有 `forge freeze` 对配置、来源、题锁、训练／验证数据与 adapter
  全部取哈希后，最终评测才可运行。
- 冻结后任一文件改变都会阻止最终评测。
- 发布门槛是总分至少增加 2 个百分点，且任一语言或领域下降不超过 2 点。能力门槛失败只能
  标记为 Experimental；安全、授权、加载或泄漏检查失败则禁止发布权重。

使用 `make forge` 可运行完整且可续跑的一晚流程；各阶段入口见根目录 Makefile。完整命令
列表也收录于繁体中文版 [`FORGE.md`](FORGE.md)。
