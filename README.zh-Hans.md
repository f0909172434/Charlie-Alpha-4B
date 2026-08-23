# Charlie alpha

Charlie alpha（模型标识：`Charlie-Alpha-4B`）是一个面向数学与编程的繁体中文、
简体中文、英文实验模型。它基于 Apache-2.0 许可的
[`Qwen3-4B-Thinking-2507`](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)，
在 Apple Silicon 上使用 MLX 4-bit QLoRA 微调，并非从零预训练。

> 当前状态：`Experimental v0.1.0`。60 题同条件小型评测从底模 38.33% 提升至
> 45.00%，但简中从 80.00% 降至 40.00%，未通过语言退步门槛，不宣称全面提升。

[繁體中文](README.md) · [English](README.en.md) · [模型卡](MODEL_CARD.md)

## 一晚训练方案

- 600 条已验证的英文短轨迹：数学 300、Python 150、C++ 150。
- 最多各 50 条繁中与简中教师精炼样本；若教师生成太慢，各语言最低 10 条后继续。
- 1,024 tokens；用两个 40-iteration 短试跑比较最后 16 层的 rank-8 Q/V 与
  rank-16 Q/K/V/O LoRA，再只使用胜出的设置。
- 正式训练最多 6 小时、2 epochs；完整验证集连续两次无改善即停止。显存不足时降至
  768 tokens，再降至最后 8 层。
- 优先交付 MLX adapter；完整 GGUF 与大型评测不占用核心训练时间。

## 实际一晚结果

- rank-8 Q/V、最后 16 层胜出；最佳完整验证 loss 从 1.106 降至 0.586。
- 固定 Metal 降级耗尽后在累计第 490 步资源早停，发布累计第 440 步的最佳检查点。
- MATH-500 38.46% → 53.85%、GSM8K 66.67% → 75.00%、MBPP+ 0% → 20.00%。
- 英文 32.00% → 44.00%、繁中维持 60.00%、简中 80.00% → 40.00%。
- adapter、融合 MLX、沙箱、隐私与干净环境加载均通过；GGUF 等价评测延期。

完整结果见 [`reports/evaluation.json`](reports/evaluation.json) 和
[`MODEL_CARD.md`](MODEL_CARD.md)。

```bash
make setup
make data
make distill
make mix
make pilot
make train
make eval
make export
```

重型步骤可续跑，数据与模型来源锁定至 commit SHA。详见
[`configs/sources.lock.json`](configs/sources.lock.json) 和
[`DATA_SOURCES.md`](DATA_SOURCES.md)。结果与限制会完整保存在 `reports/`。

本项目代码与模型衍生物计划采用 [Apache-2.0](LICENSE)；数据仍遵循上游许可。
