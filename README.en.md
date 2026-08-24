# Charlie alpha

Charlie alpha (`Charlie-Alpha-4B`) is a statistical procedure-selection model and local data
analysis interface for English, Traditional Chinese, and Simplified Chinese. It is an Apple MLX
4-bit QLoRA derivative of the Apache-2.0-licensed
[`Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B).

> Release status: **Experimental v0.3.0**. DGP-Regret reduced regret on the sealed DGP suite, but it
> did not improve P-Bench or StatQA and regressed on underspecified cases. Treat this release as a
> procedure-selection research artifact, not an autonomous statistician.

[繁體中文](README.md) · [简体中文](README.zh-Hans.md) · [Model card](MODEL_CARD.md) ·
[DGP-Regret report](docs/DGP_REGRET.md)

## What it does

The runtime holds one 4-bit base and a 2,129,920-parameter adapter:

- Data attachments and statistical questions use the `stats` route. Other requests bypass the
  adapter through the identical `base` path.
- The adapter selects among 28 training procedures. The data agent exposes seven more fixed
  evaluation procedures. Generated text cannot be executed as code.
- A plan records the estimand, sampling unit, study design, dependence, missingness, method, and
  diagnostics.
- CSV, TSV, JSON, and Parquet files stay local. Python and R run inside a macOS sandbox with no
  network or access to unrelated user files.

The API retains `adapter` as an alias for the `stats` route. Numerical tests measured a `0.0`
maximum logit error between the base bypass and an independently loaded base, and `0.0` after the
adapter route was restored.

## Sealed evaluation

All three variants used the same prompts, tools, temperature 0, and sealed task IDs. Lower
normalized regret is better.

| Model | Final DGP regret | Method accuracy | Invalid selection rate |
| --- | ---: | ---: | ---: |
| Qwen3.5-4B base | 0.6727 | 20.83% | 63.33% |
| Hard-label ablation | 0.7016 | 17.50% | 65.83% |
| DGP-Regret | **0.4437** | **45.00%** | **38.33%** |

DGP-Regret reduced regret by 34.04% relative to the base. The paired-bootstrap mean absolute
improvement was 0.2290 with a 95% CI of `[0.1199, 0.3361]`. The invalid selection rate fell by
39.47% relative. On pilot dev, the full method reduced regret by 8.54% against hard-label and passed
the predeclared 5% ablation gate.

| Evaluation | Base | DGP-Regret | Result |
| --- | ---: | ---: | --- |
| Trilingual method accuracy, English | 16.67% | 43.33% | +26.67 points |
| Trilingual method accuracy, Traditional Chinese | 16.67% | 26.67% | +10.00 points |
| Trilingual method accuracy, Simplified Chinese | 30.00% | 40.00% | +10.00 points |
| P-Bench Raw / Strict | 0% / 0% | 0% / 0% | No improvement |
| StatQA exact | 1.00% | 1.00% | No improvement |
| Underspecified cases | 43.33% | 0% | Regression |
| Math/code/STEM/general retention | 100% | 100% | No change |

[`reports/stats/evaluation.json`](reports/stats/evaluation.json) contains the aggregate metrics,
confidence interval, and gate decisions. The release missed the complete capability gate and is
therefore Experimental.

## Install and run

You need an Apple Silicon Mac, Python 3.12, `uv`, and the locked Pixi Python/R environment:

```bash
make setup
```

Analyze a local file:

```bash
uv run charlie-alpha stats analyze \
  --data survey.csv \
  --question "Compare treatment and control means under independent random assignment." \
  --language en \
  --adapter-path f0909172434/Charlie-Alpha-4B-MLX-4bit
```

Start a multi-turn session:

```bash
uv run charlie-alpha stats chat \
  --data survey.csv \
  --adapter-path f0909172434/Charlie-Alpha-4B-MLX-4bit
```

The local OpenAI-compatible API binds to `127.0.0.1` by default:

```bash
uv run charlie-alpha stats serve \
  --adapter-path f0909172434/Charlie-Alpha-4B-MLX-4bit

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"Charlie-Alpha-4B",
    "messages":[{"role":"user","content":"Estimate the treatment effect on outcome"}],
    "charlie_files":["/absolute/path/survey.csv"],
    "charlie_route":"stats",
    "charlie_tools":true
  }'
```

The response includes the route, tool count, isolation state, and analysis plan. It does not expose
hidden reasoning.

## Known limits

The v0.3 DGP engine is an inspectable semiparametric operating-characteristic emulator. It uses
common random numbers and 128, 256, or 512 samples. It does not refit all 28 methods to raw tables
inside each replication. Final DGP scores measure agreement with this simulator, not general
statistical optimality.

On P-Bench, the data agent almost always returned `needs_clarification` and produced no scoreable p
values. The adapter also failed to request missing information on the held-out clarification suite.
Users should state the estimand, sampling unit, paired or clustered structure, assignment mechanism,
and missingness assumptions, then inspect the selected method.

Medical, policy, and financial analyses require review by a qualified statistician. GGUF is
withheld because the pinned upstream path does not yet provide a verified fix for the relevant
Qwen3.5 hybrid tensors. The MLX adapter and fused MLX copy passed clean-environment loading.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the resumable pipeline. Pinned sources
and licenses are recorded in [`configs/sources.lock.json`](configs/sources.lock.json) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The repository excludes training text, caches,
credentials, and machine paths. The v0.2 FORGE code and artifacts remain at Git tag `v0.2.0`.

Project code and derivative model artifacts use [Apache-2.0](LICENSE). Upstream data retains its
original terms.
