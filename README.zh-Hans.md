# Charlie alpha

Charlie alpha（模型标识：`Charlie-Alpha-4B`）是支持繁体中文、简体中文、英文的数学与
编程实验模型。它基于 Apache-2.0 许可的
[`Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B)，使用 MLX 4-bit QLoRA；这是衍生
微调模型，并非从零预训练。

> 发布状态：**Experimental v0.2.0**。两组完全不重叠、各 62 题的冻结测试都比同条件底模
> 多答对 1 题，但 +1.62 和 +1.61 个百分点均未达到预先设定的 +2 点门槛。这是可复现的正向
> 观察，不足以证明全面或具有统计显著性的优势。

[繁體中文](README.md) · [English](README.en.md) · [模型卡](MODEL_CARD.md) ·
[FORGE 方法](docs/FORGE.zh-Hans.md)

## 推理架构

正式运行时不会同时加载两个模型，也不会生成两次再挑答案。它只加载一份 4B 底模和一个
8.52 MB adapter，在生成前切换最后 4 层的 8 个 LoRA 模块：

- 中文或编程题：启用 2,129,920 个 LoRA 参数。
- 英文非编程题：将 LoRA 增量置零，直接走底模路径。
- 每个请求只生成一次；推理时没有教师、评审模型或第二份 4B 权重。
- 数值测试确认，旁路与独立底模的 logits 最大误差为 `0.0`，恢复 adapter 后的最大误差也为
  `0.0`。

固定路由是在第一组 final 结果之后提出，随后才建立完全不重叠的确认集。确认集上 Charlie
alpha 为 43/62，底模为 42/62；编程领域从 11/16 提升至 12/16，其他语言和领域都没有少答
对。完整证据见 [`reports/v3/evaluation.json`](reports/v3/evaluation.json)。

## 方法与训练结果

FORGE（Focused One-pass Relative-Gap Gradient Equivalence）把计算集中在高价值更新上：

- 9B 教师不重写大量英文答案，而是用 teacher-forced 4B/9B 逐 token loss 找出学生明显落后
  的位置；英文答案 token 保留率为 52.7%。
- 52 个语义组固定为数学 26、Python 13、C++ 13。每次梯度累积包含同一题的英文、简中、
  繁中版本与 3 条英文 replay；语言梯度质量精确为 70%／15%／15%。
- 四组等参数短跑比较标准 LoRA、LoRA+ 与选择性 loss。胜出配方只训练最后 4 层 rank-32；
  最佳 validation loss 从 0.8640 降至 0.6867。
- 只在封存 dev 上进行 LoRA-B 增量线搜索，选出 0.22，无需重新训练或提前查看 final。

第一组冻结 final 上，直接 adapter 为 44/62，底模为 43/62。它改善了编程和中文，却损害了
英文 MATH-500，因此没有将 adapter 永久应用于所有题目。第二组全新确认集上的稀疏路由仍
多答对 1 题，且没有分组正确题数下降。两组测试规模仍小，不能外推为所有任务都更强。

## 使用

需要 Apple Silicon Mac、Python 3.12 和 `uv`：

```bash
make setup
make forge-router-verify
make forge-chat
```

`make forge-chat` 使用本机训练结果。若只使用公开 adapter、无需重新训练：

```bash
uv run charlie-alpha chat --config configs/pipeline.v2.yaml \
  --adapter-path f0909172434/Charlie-Alpha-4B-MLX-4bit
```

聊天中可用 `/route auto`、`/route base` 或 `/route adapter` 覆盖自动路由。本地非流式
OpenAI-compatible API 用 `make forge-serve` 启动；请求可加入 `"charlie_route":"base"` 或
`"adapter"`。

完整可续跑流程与冻结评测见 [`docs/FORGE.zh-Hans.md`](docs/FORGE.zh-Hans.md)。来源 revision
和许可见 [`configs/sources.lock.json`](configs/sources.lock.json) 与
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。训练文本、缓存、凭据与机器路径不会提交。

当前证据仅包含两个 62 题套件。自动路由规则可解释但仍可能误判跨领域英文题，重要场景可
手动覆盖。生成的证明、计算与代码都可能出错。动态路由无法忠实融合为单一 GGUF，因此
v0.2.0 不发布未通过行为等价门槛的 GGUF。

项目代码与模型衍生物采用 [Apache-2.0](LICENSE)；上游数据仍遵循各自许可。
