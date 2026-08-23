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
