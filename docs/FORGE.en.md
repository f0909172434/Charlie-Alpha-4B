# FORGE: verifiable trilingual distillation in one night

FORGE stands for **Focused One-pass Relative-Gap Gradient Equivalence**. It is Charlie alpha
v0.2's research recipe for a 24 GB Apple Silicon laptop. It does not claim a new foundation-model
layer. It combines independently testable ideas into a compute-bounded training system whose only
success criterion is higher accuracy than the same Qwen3.5-4B base on a frozen unseen suite.

## Design

1. **Hybrid-aware base.** The Apache-2.0
   [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) base mixes Gated DeltaNet linear attention
   with full attention. LoRA targets both paths instead of adapting only the full-attention layers.
2. **Token-level relative gap.** Each verified, decontaminated answer receives one teacher-forced
   pass through the 4B student and aligned 9B teacher. Selection favors examples with positive
   student-minus-teacher loss. Training retains the highest positive-gap tokens plus at least the
   final 32 answer tokens.
3. **Capability-balanced replay.** Fifty-two optimizer-update groups are fixed at 26 mathematics,
   13 Python, and 13 C++ groups. Diversity-aware replay prevents one narrow template from winning
   merely because its loss is large.
4. **Coupled language updates.** Every group contains one task in English, Simplified Chinese, and
   Traditional Chinese, followed by three high-value English replay examples from the same ability.
   All six microsteps form one optimizer update. Per-example weights make the nominal loss mass
   exactly 70/15/15 across English/Simplified/Traditional.
5. **Lossless translation.** Code, LaTeX, URLs, and numbers become immutable placeholders before a
   single 9B translation to Simplified Chinese. Protected OpenCC conversion then produces the
   Traditional version. Every placeholder and preserved structure is verified before acceptance.
6. **Equal-cost ablations.** On-device trials showed that 16 trainable layers OOM and eight layers
   cause severe swapping. Candidates therefore use the final four layers at rank 32, covering
   three DeltaNet layers and one full-attention layer. Standard LoRA, conservative LoRA+, fast
   LoRA+, and fast LoRA+ without token selection have identical trainable parameter counts. A
   nine-task locked trilingual canary chooses by accuracy, then validation loss and elapsed time.
7. **Compile-shape compression.** Sequences are capped at 704 tokens and padded into only 384,
   544, or 704-token buckets, allowing MLX to reuse backward graphs. Reducing replay from five to
   three also cuts microsteps per capability update by 25% without changing the 70/15/15 gradient
   mass.
8. **Capability-calibrated line search.** The minimum-validation-loss adapter damaged English
   arithmetic on dev. Scaling only LoRA B continuously scales the low-rank delta without retraining.
   A locked-dev search over 0.125, 0.16, 0.18, 0.22, and 0.25 selected 0.22 by accuracy, maximum
   subgroup regression, then smaller delta; final remained sealed throughout. This explicitly
   demonstrates that validation loss is not a sufficient checkpoint selector for reasoning.
9. **Single-model dynamic sparse LoRA.** The first final suite showed gains on code and Chinese but
   a loss on English math. The fixed runtime therefore enables the adapter for code or Chinese and
   bypasses it otherwise. It loads one 4B model plus an 8.52 MB adapter and switches eight LoRA
   scales, with no second generation. Bypass exactly matches an independent base at the logit level.

The ingredients are motivated by [Rho-1](https://arxiv.org/abs/2404.07965),
[LESS](https://arxiv.org/abs/2402.04333), [BIDS](https://arxiv.org/abs/2501.12147),
[LoRA+](https://arxiv.org/abs/2402.12354), [xCoT](https://arxiv.org/abs/2401.07037),
[STaR](https://arxiv.org/abs/2203.14465), [LIMO](https://arxiv.org/abs/2502.03387), and
[s1](https://arxiv.org/abs/2501.19393). These papers support components, not the combined recipe;
only the locked evaluation can establish whether FORGE works here.

## Evaluation firewall

- The committed lock contains 34 development and 62 final tasks. They do not overlap each other
  or the v0.1 evaluation tasks, and every canonical task has a SHA-256 fingerprint.
- Development results may select a recipe. Final evaluation is disabled until `forge freeze`
  hashes the pipeline, sources, task lock, training data, validation data, and adapter.
- Any post-freeze change blocks final evaluation.
- Release requires at least +2 overall percentage points with no language or domain losing more
  than 2 points. A failed capability gate permits only an Experimental label; a safety, licensing,
  loading, or leakage failure prohibits weight release.

## Observed result

The 312-record training run took 2,896 seconds and peaked at 16.05 GB. Best validation loss fell
from 0.8640 to 0.6867. On the first frozen 62-task final suite, the direct adapter scored 44/62
against 43/62 (+1.62 points), while trading English MATH performance for code and Chinese gains.

The sparse route was fixed after that result. Only then was a fully disjoint 62-task confirmation
lock created. Routed Charlie alpha scored 43/62 against 42/62 (+1.61 points); code rose from 11/16
to 12/16, and no other language or domain lost a correct answer. Both suites observed one additional
correct answer, but both missed the +2-point gate. The release is therefore **Experimental v0.2.0**
and does not claim statistically established or broad superiority.

Run the entire resumable workflow with `make forge`, or inspect the individual `make forge-*`
targets in the root Makefile. The Traditional Chinese document contains the complete command list.
Use `make forge-router-verify`, `make forge-chat`, and `make forge-serve` for the canonical dynamic
runtime.
