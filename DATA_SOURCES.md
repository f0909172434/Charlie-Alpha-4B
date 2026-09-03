# Data sources

Charlie alpha does not vendor its training corpora. The preparation command downloads pinned
revisions, records source IDs and hashes, and writes only reproducibility manifests to Git.

| Purpose | Source | Configuration | License |
|---|---|---|---|
| Mathematics SFT | `open-r1/OpenR1-Math-220k` | `default` | Apache-2.0 |
| C++ reasoning SFT | `open-r1/codeforces-cots` | `solutions_short_and_long_decontaminated` | ODC-By-4.0 / dataset metadata CC-BY-4.0 |
| Python reasoning SFT | `open-r1/codeforces-cots` | `solutions_py_decontaminated` | ODC-By-4.0 / dataset metadata CC-BY-4.0 |
| Math evaluation | `HuggingFaceH4/MATH-500` | `test` | MIT |
| Arithmetic evaluation | `openai/gsm8k` | `main/test` | MIT |
| Chinese arithmetic evaluation | `CohereLabs/global-mgsm` | `zh/test` | CC-BY-4.0 |
| Code evaluation | EvalPlus HumanEval+ and MBPP+ | packaged releases | Apache-2.0 / original dataset terms |
| Research reference only | `GAIR/LIMO` | `train` | Apache-2.0 |

Synthetic Traditional and Simplified Chinese examples are translations of accepted source
records. They are generated locally with the pinned Apache-2.0 Qwen3.5 teacher and retain the
source record ID and source license in their manifest.

For the compute-bounded overnight profile, mathematics rows must contain at least one trajectory
that passes the upstream `math_verify` and completeness flags. The pipeline then chooses the
shorter of those verified generations and the source's canonical solution, keeping the student
sequence at or below 1,024 tokens without truncation.

For code, the repeated long-form `<think>` section is excluded and the structured problem fields
are rendered into a compact prompt. The final Python or C++ implementation must compile/run and
pass the available public or example tests inside the no-network macOS sandbox before it can enter
the training set.

See `configs/sources.lock.json` for immutable revisions.

The v0.2 FORGE profile reuses only verified, decontaminated source records and applies a second
Qwen3.5 tokenizer limit of 832 tokens. It removes rows whose prose is not detected as English,
scores every remaining answer under the pinned 4B student and 9B teacher, and selects a balanced
subset by positive teacher-student excess loss plus diversity. LIMO is cited as a data-efficiency
reference but is not training input because its trajectories exceed the one-night sequence budget.

## v0.3 statistics profile

v0.3 does not train on StatQA or P-Bench. Its supervised records are generated from 450 seeded DGP
blueprints split into train, validation, development, and sealed-final groups before any language
rendering. The public repository keeps the generator, 28-procedure catalog, configuration, split
IDs, and hashes; generated JSONL records and simulation caches stay local.

| Purpose | Source | Locked use | License |
|---|---|---|---|
| Base model | `mlx-community/Qwen3.5-4B-MLX-4bit` | MLX 4-bit base at the commit in `sources.lock.json` | Apache-2.0 |
| Wording editor | `mlx-community/Qwen3.5-9B-MLX-4bit` | At most 120 explanations; no correctness decisions | Apache-2.0 |
| Open-ended evaluation | `May2222/P-Bench` | 90 frozen tasks, 45 Easy and 45 Hard, all 17 categories | CC-BY-4.0 |
| Applicability evaluation | `HKUSTDial/StatQA` | 200 frozen indices; questions and answers are not redistributed | GPL-3.0-only |

The 9B editor runs at temperature zero with thinking disabled. A candidate is accepted only when
its method identifier and every number match the deterministic template; otherwise one retry is
allowed and then the template is retained. Statistical labels, operating-characteristic fields,
tool programs, and final numerical results never come from the teacher.

The evaluation lock contains only task IDs, indices, categories, revisions, and hashes. An 8-gram
audit is performed against the generated training prompts before the lock is written. P-Bench and
StatQA remain evaluation-only and are loaded from their pinned upstream revisions at evaluation
time.
