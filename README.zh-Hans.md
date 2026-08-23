# Charlie alpha

Charlie alpha（模型标识：`Charlie-Alpha-4B`）是一个面向数学与编程的繁体中文、
简体中文、英文实验模型。它基于 Apache-2.0 许可的
[`Qwen3-4B-Thinking-2507`](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)，
在 Apple Silicon 上使用 MLX 4-bit QLoRA 微调，并非从零预训练。

> 当前状态：训练流程正在构建，尚未发布权重，也不宣称能力已有提升。若质量门槛未通过，
> `v0.1.0` 将明确标记为 Experimental。

[繁體中文](README.md) · [English](README.en.md) · [模型卡](MODEL_CARD.md)

## 一晚训练方案

- 600 条已验证的英文短轨迹：数学 300、Python 150、C++ 150。
- 最多各 50 条繁中与简中教师精炼样本；若教师生成太慢，各语言最低 10 条后继续。
- 1,024 tokens、rank 8、最后 8 层 Q/V LoRA，只运行一组高可信参数。
- 正式训练最多 6 小时、1 epoch；显存不足时降至 768 tokens，再降至最后 4 层。
- 优先交付 MLX adapter；完整 GGUF 与大型评测不占用核心训练时间。

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

