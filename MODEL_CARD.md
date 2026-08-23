---
language:
  - en
  - zh
license: apache-2.0
base_model: Qwen/Qwen3-4B-Thinking-2507
library_name: mlx
pipeline_tag: text-generation
tags:
  - mlx
  - qlora
  - mathematics
  - code
  - experimental
---

# Charlie alpha model card

## Model description

Charlie alpha is a 4B-parameter-class derivative fine-tune for introductory through early
university mathematics, Python, C++, data structures, and algorithms. Its response languages are
English, Traditional Chinese, and Simplified Chinese. The base is
`Qwen/Qwen3-4B-Thinking-2507`; training uses a quantized MLX LoRA adapter rather than full
pretraining.

## Training data

- Verified correct trajectories from `open-r1/OpenR1-Math-220k`.
- Decontaminated Python and C++ trajectories from `open-r1/codeforces-cots`.
- A small locally generated Chinese translation/refinement layer using a pinned Qwen3.5 teacher.

Records are split by original problem identity before translation. Exact duplicates, incomplete
answers, over-length rows, and local 8-gram benchmark overlaps are rejected. Source revisions and
record hashes are published without redistributing the raw corpora.

The released overnight mix contains 829 records and 123,737 assistant tokens. The train split has
781 records and 115,644 assistant tokens, balanced to 50.01% math / 49.99% code, 50.00% Python /
50.00% C++ within code, and 84.00% English / 8.01% Traditional Chinese / 7.99% Simplified Chinese.
The one-hour teacher pass accepted 35 Traditional Chinese and 17 Simplified Chinese records before
bounded train-only repetition and script conversion; rejected translations are counted in the
published manifest.

## Overnight training profile

The initial run is deliberately compute-bounded: 1,024-token sequences, two 40-iteration pilots on
the last 16 layers, batch size 1, gradient accumulation 4, prompt masking, gradient checkpointing,
3% warm-up, cosine decay, full-validation early stopping, and at most two epochs or six hours. The
configured seed is 42. The released configuration records which pilot won.

Rank-8 Q/V won the trilingual pilot canary. The main run reached 490 cumulative iterations before
the fixed Metal OOM fallbacks were exhausted. The best full-validation loss fell from 1.106 to
0.586 at cumulative iteration 440; that 16-layer checkpoint was restored for evaluation and
release. The run is therefore resource-stopped, not a completed two-epoch run.

## Evaluation and release status

**Experimental v0.1.0.** On the fixed 60-task overnight suite, Charlie alpha scored 27/60 (45.00%)
against 23/60 (38.33%) for the identically prompted base model, a gain of 6.67 percentage points.
The suite uses temperature 0 and the same direct-answer prompt for both variants.

| Group | Base | Charlie alpha | Delta |
| --- | ---: | ---: | ---: |
| MATH-500 (13) | 38.46% | 53.85% | +15.39 |
| GSM8K (12) | 66.67% | 75.00% | +8.33 |
| HumanEval+ (10) | 20.00% | 20.00% | 0.00 |
| MBPP+ (10) | 0.00% | 20.00% | +20.00 |
| Trilingual canary (9) | 33.33% | 11.11% | -22.22 |
| Retention canary (6) | 83.33% | 100.00% | +16.67 |

English rose from 32.00% to 44.00% and Traditional Chinese stayed at 60.00%, but Simplified Chinese
fell from 80.00% to 40.00%. This exceeds the three-point subgroup-regression limit, so the model is
not a stable candidate and no broad multilingual improvement is claimed. The adapter, fused MLX
model, source/data gates, privacy checks, sandbox tests, and fresh-environment load tests passed.
GGUF is deferred because its behavioral parity gate was not run.

## Limitations

The one-night data volume and 60-task evaluation are small; the Chinese language groups contain
only five evaluation tasks each. Exact calculations, proofs, citations, and generated programs can
still be wrong. The measured Simplified Chinese regression is material. Users should independently
verify answers and run code in an isolated environment.
