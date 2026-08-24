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
  - statistics
  - dgp-regret
  - tool-agent
  - multilingual
  - experimental
---

# Charlie alpha model card

## Model details

Charlie alpha v0.3 is a statistical procedure-selection derivative of Qwen3.5-4B. It supports
English, Traditional Chinese, and Simplified Chinese. The canonical artifact uses one 4-bit MLX
base and a LoRA adapter with 2,129,920 parameters across eight modules in the final four layers.

The runtime has two routes. `stats` enables the adapter for statistical questions or attached data;
`base` bypasses LoRA for other requests. The compatibility name `adapter` maps to `stats`. Each
request generates one answer. No teacher or judge model runs during inference.

Release classification: **Experimental v0.3.0**. The model improved its sealed DGP procedure
selection metrics, but the external P-Bench and StatQA gates failed. It also regressed on the
held-out clarification suite.

## Intended use

The model supports research on statistical procedure selection under explicit assumptions. The
local agent can inspect CSV, TSV, JSON, or Parquet files and call fixed Python/R procedures. It
returns a structured plan with these fields:

`status`, `estimand`, `sampling_unit`, `study_design`, `outcome_type`, `dependence`, `missingness`,
`method_id`, `uncertainty`, `diagnostics`, `tool`, and `variables`.

The current release is unsuitable for autonomous analysis. A user should state the estimand,
sampling unit, assignment mechanism, dependence structure, and missingness assumptions, then have a
statistician check the plan and result. Medical, financial, and policy decisions require qualified
review.

## Training method

DGP-Regret casts procedure selection as a bounded decision problem. The generator covers 12 DGP
families and 28 procedures. Each item presents at most six candidates, including a robust
alternative and `needs_clarification` where appropriate. The simulator estimates Type I error,
coverage, bias, RMSE, power, calibration, and cost with common random numbers. It starts at 128
replications and escalates to 256 or 512 when the method ranking remains uncertain.

The simulator converts normalized regret to a listwise target:

\[
q_j=\frac{\exp(-r_j/0.15)}{\sum_k\exp(-r_k/0.15)}.
\]

The primary split contains 240 training semantic groups, 30 validation groups, 60 dev DGPs, and
120 sealed final DGPs. Splitting occurs before language rendering. Each training group has two
English boundary views plus one Traditional Chinese and one Simplified Chinese view. Loss weights
of 1.4, 1.4, 0.6, and 0.6 yield a 70%/15%/15% language gradient ratio.

The assistant objective assigns 45% to method soft-label loss, 35% to plan/tool tokens, and 20% to
the report. User content and tool output receive zero loss. A local Qwen3.5-9B teacher edited only
explanations at temperature 0. It could not choose methods or alter simulator fields. Of 120 target
explanations, 76 passed validation and 44 used the deterministic template.

Three equal-compute pilots used 160 microsteps, rank 32, the final four layers, batch size 1, four
steps of gradient accumulation, and a 640-token limit:

| Pilot | Dev normalized regret | Method accuracy | Invalid selection | Final validation loss |
| --- | ---: | ---: | ---: | ---: |
| Hard-label | 0.6848 | 18.33% | 61.67% | 1.6983 |
| Regret, random DGP | 0.6644 | 18.33% | 60.00% | 1.7328 |
| DGP-Regret curriculum | **0.6263** | **30.00%** | 61.67% | 1.7125 |

The DGP-Regret pilot reduced dev regret by 8.54% relative to hard-label. Formal training planned at
most 1,920 microsteps and stopped after 1,120 because two validation checks did not improve. The
best checkpoint occurred at microstep 800 with validation loss 1.0529. Dev calibration compared
adapter delta scales 0.5, 0.75, and 1.0 and selected 1.0 using validity first, then normalized regret
and retention.

The v0.3 simulator is a semiparametric operating-characteristic emulator. It does not generate a
raw table and refit all 28 methods inside every replication. Its response functions are checked in,
seeded, and testable, but they encode modeling judgment. Sealed DGP performance measures agreement
with that declared simulator.

## Evaluation

All variants use the pinned Qwen3.5-4B MLX base, the same prompts and tools, greedy decoding, and the
same sealed IDs. The evaluation lock contains 120 final DGPs, 90 P-Bench tasks, 200 StatQA tasks, 30
trilingual semantic tasks rendered in three languages, 30 clarification cases, and 24 retention
items. An 8-gram audit found no overlap with training prompts.

### DGP procedure selection

| Model | Normalized regret | Method accuracy | Invalid selection rate |
| --- | ---: | ---: | ---: |
| Base | 0.6727 | 20.83% | 63.33% |
| Hard-label | 0.7016 | 17.50% | 65.83% |
| DGP-Regret | **0.4437** | **45.00%** | **38.33%** |

Relative regret improved by 34.04%. The paired-bootstrap mean absolute improvement was 0.2290 with a
95% CI of `[0.1199, 0.3361]`. The invalid selection rate fell by 39.47% relative to the base.

### External, language, and retention results

| Metric | Base | Hard-label | DGP-Regret |
| --- | ---: | ---: | ---: |
| P-Bench Raw | 0% | 0% | 0% |
| P-Bench Strict | 0% | 0% | 0% |
| StatQA exact | 1.00% | 1.00% | 1.00% |
| StatQA method-set accuracy | 5.00% | 6.00% | 5.50% |
| StatQA column-set accuracy | 19.50% | 23.00% | 22.00% |
| Trilingual English method accuracy | 16.67% | 10.00% | 43.33% |
| Trilingual Traditional Chinese method accuracy | 16.67% | 20.00% | 26.67% |
| Trilingual Simplified Chinese method accuracy | 30.00% | 30.00% | 40.00% |
| Clarification accuracy | 43.33% | 40.00% | 0% |
| Retention accuracy | 100% | 100% | 100% |

The adapter passed the final-regret, invalid-selection, trilingual, retention, category, and
DGP-Regret ablation gates. P-Bench and StatQA failed their improvement gates. On P-Bench, all base
and hard-label cases returned `needs_clarification`; DGP-Regret did so on 88 of 90 cases, and no
variant produced a scoreable p value. The complete capability gate therefore failed.

The public aggregate is `reports/stats/evaluation.json`. Generated task text, per-item predictions,
and model answers remain outside the release tree.

## Runtime and isolation

The planner can call a checked-in Python or R implementation at most four times per analysis. Each
call has a 20-second timeout, 2 GiB memory limit, 32 MiB write limit, and 64 KiB output limit. The
macOS sandbox blocks networking, reads outside the runtime and temporary input directory, writes
outside the temporary directory, and unapproved child processes. Release tests exercised each
escape class in Python and R.

The adapter, fused MLX export, and dynamic router loaded in a clean Python 3.12 environment. Router
tests measured eight LoRA modules, a nonzero adapter effect, zero restoration error, and zero error
between the bypass and an independently loaded base.

## Limitations

- The model's gains are confined to the declared DGP selection task and trilingual variants of it.
- P-Bench and StatQA provide no evidence of improved end-to-end statistical analysis.
- The adapter failed all held-out clarification cases, so it may choose a method when it should ask
  for design information.
- The finite emulator does not establish optimality on unseen statistical problems.
- Generated plans, code outputs, and prose can contain errors even when the selected method is
  valid.

GGUF is withheld. The pinned llama.cpp path lacks a verified upstream resolution for the relevant
Qwen3.5 hybrid-tensor and 4B block-count issues. The project will not publish an unverified GGUF.

## License and provenance

Project code and derivative model artifacts use Apache-2.0. Upstream datasets retain their own
licenses. `configs/sources.lock.json`, `DATA_SOURCES.md`, and `THIRD_PARTY_NOTICES.md` record pinned
revisions and terms. The release excludes corpora, caches, credentials, evaluation question text,
and machine paths. The v0.2 FORGE release remains available at Git tag `v0.2.0`.
