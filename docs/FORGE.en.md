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
   Traditional Chinese, followed by five high-value English replay examples from the same ability.
   All eight microsteps form one optimizer update. Per-example weights make the nominal loss mass
   exactly 70/15/15 across English/Simplified/Traditional.
5. **Lossless translation.** Code, LaTeX, URLs, and numbers become immutable placeholders before a
   single 9B translation to Simplified Chinese. Protected OpenCC conversion then produces the
   Traditional version. Every placeholder and preserved structure is verified before acceptance.
6. **Equal-cost ablations.** Last-16-layer rank-8 LoRA, rank-8 LoRA+, LoRA+ without token
   selection, and all-32-layer rank-4 LoRA+ must have identical trainable parameter counts. A
   nine-task locked trilingual canary picks the highest-accuracy pilot; validation loss and elapsed
   time break ties.

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

Run the entire resumable workflow with `make forge`, or inspect the individual `make forge-*`
targets in the root Makefile. The Traditional Chinese document contains the complete command list.
