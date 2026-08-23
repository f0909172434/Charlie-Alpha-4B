# Charlie alpha

Charlie alpha (`Charlie-Alpha-4B`) is an experimental Traditional Chinese, Simplified Chinese,
and English model focused on mathematics and programming. It is a derivative fine-tune of the
Apache-2.0-licensed
[`Qwen3-4B-Thinking-2507`](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507),
trained with 4-bit QLoRA on Apple Silicon using MLX. It is not trained from scratch.

> Status: the pipeline is being built. No weights have been released and no improvement is
> claimed yet. If the quality gate is missed, `v0.1.0` will be labeled Experimental.

[繁體中文](README.md) · [简体中文](README.zh-Hans.md) · [Model card](MODEL_CARD.md)

## Overnight profile

- 600 verified short English trajectories: 300 math, 150 Python, and 150 C++.
- Up to 50 teacher-refined examples in each Chinese script, with a floor of 10 when generation
  would exceed its one-hour budget.
- 1,024 tokens, rank 8, Q/V LoRA on the last 8 layers, and one high-confidence candidate.
- At most six hours and one epoch for the main run; fixed OOM fallbacks are 768 tokens and then
  four trainable layers.
- The MLX adapter is the priority artifact. GGUF conversion and large evaluations never displace
  the core training run.

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

Heavy steps are resumable. Every model and dataset source is pinned to a commit SHA; see
[`configs/sources.lock.json`](configs/sources.lock.json) and
[`DATA_SOURCES.md`](DATA_SOURCES.md). Scores, failures, and limitations are recorded under
`reports/`.

Project code and derivative model artifacts are intended for release under
[Apache-2.0](LICENSE). Upstream datasets retain their own terms.

