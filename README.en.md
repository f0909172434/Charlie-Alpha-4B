# Charlie alpha

Charlie alpha (`Charlie-Alpha-4B`) is an experimental English, Traditional Chinese, and
Simplified Chinese math-and-code model. It is a 4-bit MLX QLoRA derivative of the Apache-2.0-licensed
[`Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B), not a model pretrained from scratch.

> Release status: **Experimental v0.2.0**. Charlie alpha answered one additional task correctly on
> each of two disjoint frozen 62-task suites. The gains, +1.62 and +1.61 percentage points, missed
> the predeclared +2-point release threshold. They are reproducible positive observations, not
> evidence of broad or statistically established superiority.

[繁體中文](README.md) · [简体中文](README.zh-Hans.md) · [Model card](MODEL_CARD.md) ·
[FORGE method](docs/FORGE.en.md)

## Inference architecture

The canonical runtime does not load two models or generate two candidate answers. It holds one 4B
base plus an 8.52 MB adapter and switches eight LoRA modules in the last four layers before
generation:

- Chinese or coding prompt: enable 2,129,920 LoRA parameters.
- English non-coding prompt: zero the LoRA contribution and use the base path.
- One generation per request, with no teacher, judge, or second 4B copy at inference time.
- Numerical tests show an exact `0.0` maximum logit error between bypass and an independently
  loaded base, and `0.0` between the restored and original adapter paths.

The route was formulated after the first final suite and then tested once on a newly locked,
fully disjoint confirmation suite. Routed Charlie alpha scored 43/62 versus 42/62 for the base;
code rose from 11/16 to 12/16, while no other language or domain lost a correct answer. See
[`reports/v3/evaluation.json`](reports/v3/evaluation.json).

## Method and training result

FORGE (Focused One-pass Relative-Gap Gradient Equivalence) concentrates the compute budget on
high-value updates:

- The 9B teacher does not regenerate a large English corpus. Teacher-forced 4B/9B token losses
  identify where the student lags; 52.7% of English answer tokens are retained.
- Fifty-two semantic groups are fixed at 26 math, 13 Python, and 13 C++. Each optimizer update
  couples the same task in English, Simplified Chinese, and Traditional Chinese with three English
  replays. Gradient mass is exactly 70%/15%/15% by language.
- Four equal-parameter pilots compare standard LoRA, LoRA+, and selective loss. The winning
  rank-32 recipe trains only the final four layers and reduced best validation loss from 0.8640
  to 0.6867.
- A sealed-development LoRA-B delta line search selected 0.22 without retraining or opening final.

The direct adapter scored 44/62 against 43/62 on the first frozen final suite. It improved code and
Chinese but hurt English MATH-500, which is why the adapter is not applied globally. The disjoint
router confirmation again gained one correct answer with no subgroup losing one. Both suites are
small, so these results must not be generalized to all math and programming tasks.

## Run it

An Apple Silicon Mac, Python 3.12, and `uv` are required:

```bash
make setup
make forge-router-verify
make forge-chat
```

`make forge-chat` uses the locally trained artifact. To use the public adapter without retraining:

```bash
uv run charlie-alpha chat --config configs/pipeline.v2.yaml \
  --adapter-path f0909172434/Charlie-Alpha-4B-MLX-4bit
```

Use `/route auto`, `/route base`, or `/route adapter` in chat. The local, non-streaming
OpenAI-compatible endpoint starts with `make forge-serve`; requests may include
`"charlie_route":"base"` or `"adapter"` to override automatic routing.

The full resumable pipeline and frozen evaluation procedure are documented in
[`docs/FORGE.en.md`](docs/FORGE.en.md). Pinned revisions and licenses are in
[`configs/sources.lock.json`](configs/sources.lock.json) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Training text, caches, credentials, and machine
paths are never committed.

The current evidence consists of only two 62-task suites. Routing is an explainable, high-precision
rule but can still misclassify cross-domain English prompts; manual override is available. Proofs,
calculations, and generated code can be wrong. Dynamic routing cannot be faithfully represented by
one fused GGUF, so v0.2.0 does not publish a GGUF that has not passed behavioral parity.

Project code and derivative model artifacts use [Apache-2.0](LICENSE); upstream datasets retain
their own terms.
