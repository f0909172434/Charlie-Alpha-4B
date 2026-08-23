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

## Overnight training profile

The initial run is deliberately compute-bounded: 1,024-token sequences, rank-8 Q/V LoRA on the
last eight layers, batch size 1, gradient accumulation 4, prompt masking, gradient checkpointing,
3% warm-up, cosine decay, and at most one epoch or six hours. The configured seed is 42.

## Evaluation and release status

No score or improvement claim is valid until `reports/generated/evaluation.json` and the release
gate are present. The stable gate requires a two-point aggregate gain over the identically prompted
base model, no language or domain regression above three points, clean loading, licensing,
decontamination, and sandbox tests. Otherwise the release is Experimental; a safety, loading,
license, or data-leak failure blocks weight publication.

The compact suite also contains original trilingual STEM and general-ability canaries. These are
reported separately to surface catastrophic forgetting, not to claim broad benchmark coverage.

## Limitations

The one-night data volume is small. Exact calculations, proofs, citations, and generated programs
can still be wrong. Multilingual fluency from the base model does not guarantee equal specialist
performance across scripts. Users should independently verify answers and run code in an isolated
environment.
