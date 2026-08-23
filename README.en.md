# Charlie alpha

Charlie alpha (`Charlie-Alpha-4B`) is an experimental Traditional Chinese, Simplified Chinese,
and English model focused on mathematics and programming. It is a derivative fine-tune of the
Apache-2.0-licensed
[`Qwen3-4B-Thinking-2507`](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507),
trained with 4-bit QLoRA on Apple Silicon using MLX. It is not trained from scratch.

> Status: `Experimental v0.1.0`. The identically prompted 60-task suite rose from 38.33% to
> 45.00%, but Simplified Chinese fell from 80.00% to 40.00%. The subgroup gate failed, so no
> broad improvement is claimed.

> The `v0.2` research branch is testing **FORGE** (Focused One-pass Relative-Gap Gradient
> Equivalence): a Qwen3.5-4B hybrid base, 4B/9B token-level learning-gap selection, and coupled
> English/Simplified/Traditional updates for each task. It will claim a stronger model only if a
> frozen 62-task unseen suite improves by at least two points against the same base with no
> language or domain falling by more than two points. See
> [`docs/FORGE.en.md`](docs/FORGE.en.md).

[繁體中文](README.md) · [简体中文](README.zh-Hans.md) · [Model card](MODEL_CARD.md)

## Overnight profile

- 600 verified short English trajectories: 300 math, 150 Python, and 150 C++.
- Up to 50 teacher-refined examples in each Chinese script, with a floor of 10 when generation
  would exceed its one-hour budget.
- 1,024 tokens; two 40-iteration pilots compare rank-8 Q/V and rank-16 Q/K/V/O LoRA on
  the last 16 layers, after which only the winner continues.
- At most six hours and two epochs for the main run, with full-validation early stopping. Fixed
  OOM fallbacks are 768 tokens and then eight trainable layers.
- The MLX adapter is the priority artifact. GGUF conversion and large evaluations never displace
  the core training run.

## Actual overnight result

- Rank-8 Q/V on the last 16 layers won; best full-validation loss fell from 1.106 to 0.586.
- Fixed Metal fallbacks were exhausted at cumulative iteration 490; the best checkpoint from
  cumulative iteration 440 is released.
- MATH-500 rose from 38.46% to 53.85%, GSM8K from 66.67% to 75.00%, and MBPP+ from 0% to 20.00%.
- English rose from 32.00% to 44.00%, Traditional Chinese stayed at 60.00%, and Simplified Chinese
  fell from 80.00% to 40.00%.
- Adapter, fused MLX, sandbox, privacy, and fresh-environment load gates passed. GGUF parity is
  deferred.

See [`reports/evaluation.json`](reports/evaluation.json) and [`MODEL_CARD.md`](MODEL_CARD.md) for
the complete result and limitations.

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
