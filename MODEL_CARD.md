---
language:
  - en
  - zh
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
library_name: mlx
pipeline_tag: text-generation
tags:
  - mlx
  - qlora
  - mathematics
  - code
  - experimental
  - sparse-routing
---

# Charlie alpha model card

## Model description

Charlie alpha is an experimental 4B-class derivative for English, Traditional Chinese, and
Simplified Chinese mathematics and programming. It starts from Apache-2.0-licensed Qwen3.5-4B and
uses a quantized MLX LoRA adapter; it is not pretrained from scratch.

The canonical v0.2 runtime is a single-model, deterministic sparse route. One 4-bit base and one
8.52 MB adapter remain loaded. Chinese or coding prompts enable eight LoRA modules in the final four
layers; English non-coding prompts bypass their contribution. A request is generated once. No
teacher, judge, second candidate, or second base-model copy is used at inference time. Users can
override the route when the prompt classifier is inappropriate.

The adapter contains 2,129,920 parameters. The published numerical check reports zero maximum
logit error between its bypass path and an independently loaded base, and zero error after restoring
the adapter path. See `reports/v2/dynamic-router.json`.

## Training method

FORGE combines several compute-saving techniques under a frozen evaluation protocol:

- Correct, locally decontaminated math and Python/C++ trajectories are scored once with pinned 4B
  student and 9B teacher models under teacher forcing. Positive student-teacher token-loss gaps
  select the useful portion of each English answer; 52.7% of English target tokens are retained.
- Fifty-two semantic groups are split into 26 math, 13 Python, and 13 C++. Each optimizer update
  couples one source problem in English, Simplified Chinese, and Traditional Chinese with three
  same-capability English replays. Loss weights make the language gradient mass exactly
  70%/15%/15% and the category mass 50%/25%/25%.
- Formulae, numbers, URLs, and code are replaced with validated placeholders during local 9B
  translation and restored byte-for-byte. Training has 312 records; validation has 18 records.
- Four equal-parameter pilots compare standard LoRA, LoRA+, and selective/full target loss. The
  selected rank-32 recipe trains eight projections in the final four hybrid Qwen3.5 layers.
- Full training stopped after two validation checks without improvement. It took 2,896 seconds,
  peaked at 16.05 GB, and reduced best validation loss from 0.8640 to 0.6867 at iteration 431.
- A development-only LoRA-B delta-scale search selected 0.22. The final suite remained sealed
  until the recipe, data, prompt, adapter, and task hashes were frozen.

Model, dataset, teacher, and tool revisions are pinned in `configs/sources.lock.json`. Public data
manifests contain metadata and hashes, not redistributed training text.

## Evaluation

All comparisons use the same pinned Qwen3.5-4B MLX base, prompts, greedy decoding, task scoring, and
generation limits. Dev can guide selection; final and router-confirm are one-time, disjoint frozen
suites. Neither 62-task result reached the predeclared +2 percentage-point normal-release gate.

### Direct adapter final

| Group | Base | Adapter | Delta (points) |
| --- | ---: | ---: | ---: |
| Overall (62) | 43/62, 69.35% | 44/62, 70.97% | +1.62 |
| Code (16) | 11/16 | 12/16 | +6.25 |
| Math (40) | 28/40 | 28/40 | 0.00 |
| English (42) | 27/42 | 26/42 | -2.39 |
| Simplified Chinese (10) | 10/10 | 10/10 | 0.00 |
| Traditional Chinese (10) | 6/10 | 8/10 | +20.00 |

The direct adapter gained one total answer but traded two English MATH-500 answers for one MBPP+
and two Chinese MGSM answers. Because it did not preserve every subgroup, it is not the canonical
always-on runtime.

### Disjoint sparse-router confirmation

The fixed rule—adapter for code or either Chinese script, base otherwise—was written after the
first final result. A new 62-task lock was then created before any routed generation and excludes all
v0.1 and v0.2 dev/final tasks.

| Group | Base | Routed | Delta (points) |
| --- | ---: | ---: | ---: |
| Overall (62) | 42/62, 67.74% | 43/62, 69.35% | +1.61 |
| HumanEval+ (8) | 7/8 | 8/8 | +12.50 |
| Code (16) | 11/16 | 12/16 | +6.25 |
| Math (40) | 27/40 | 27/40 | 0.00 |
| English (42) | 24/42 | 25/42 | +2.38 |
| Simplified Chinese (10) | 9/10 | 9/10 | 0.00 |
| Traditional Chinese (10) | 9/10 | 9/10 | 0.00 |

No measured language or domain lost a correct answer in this confirmation, but one additional
answer in 62 tasks is not statistically persuasive. This result is evidence for further study, not
proof that Charlie alpha is broadly stronger.

## Release status and limitations

**Experimental v0.2.0.** Both frozen suites observed a positive one-answer difference, but both
missed the predeclared +2-point threshold. The test sets are small, and benchmark accuracy is a
limited proxy for real use. Auto-routing can misclassify cross-domain English prompts. Generated
proofs, calculations, explanations, and programs can be wrong; run code in an isolated environment
and independently verify important answers.

Dynamic sparse routing cannot be faithfully collapsed into one fused GGUF. GGUF is therefore not
published without a separate behavioral-parity pass. The fused always-on MLX export is a specialist
artifact and does not reproduce the canonical route.

## MLX usage

Clone the source repository, install its pinned environment, and pass either a local adapter
directory or the public Hugging Face adapter repository. The latter is downloaded at its immutable
Hub revision and applied to the pinned 4-bit base:

```bash
make setup
uv run charlie-alpha chat --config configs/pipeline.v2.yaml \
  --adapter-path f0909172434/Charlie-Alpha-4B-MLX-4bit
```

The same `--adapter-path` option works with `charlie-alpha serve`. `/route auto` is the canonical
policy; `/route base` and `/route adapter` are explicit overrides.

## License and provenance

Project code and derivative artifacts use Apache-2.0. Upstream datasets retain their own licenses;
see `THIRD_PARTY_NOTICES.md` and `DATA_SOURCES.md`. The release excludes training corpora, caches,
credentials, and machine-specific paths.
