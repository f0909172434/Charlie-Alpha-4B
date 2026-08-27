# H4 champion replacement study handoff

Date: 2026-08-27 (Asia/Taipei)

Status: **complete and externally rejected**. This is the terminal state for the H4 research
cycle. Do not reinterpret the internal confirmations as promotion evidence and do not continue to
an H5 without a separate research contract.

## Decision

- Official champion remains `v0.3.0-parent`.
- Champion adapter SHA-256 remains
  `e644b7087c00321f16add940997f8809204458ff6bc51c97a795fed32d3e0b16`.
- H4 was `0.90 sufficiency guard + reduced family router + frozen experts`.
- H4 passed its fresh synthetic confirmation but failed its frozen historical-external gate.
- The failed gate was StatQA exact improvement: required `+5` points, observed `+0` points.
- No untouched external final was selected, generated, opened, or scored.
- No champion pointer, release artifact, tag, Hugging Face repository, or public model was changed.
- The H4 candidate must not be tuned against this historical outcome within this study.

## Confirmatory sequence

| Stage | Surface | Main result | Decision |
| --- | --- | --- | --- |
| Full route replication | Fresh 900 blueprints, 2,700 language views | Regret `-12.10%`, but experimental/causal regret `+0.15035` exceeded the `0.15` ceiling | Rejected |
| Leave-one-expert diagnosis | Retired replication | Causal fallback restored all gates and retained `11.92%` gain | Diagnostic only |
| Reduced route | New 900 blueprints | Regret `-10.62%`; paired CI `[0.03221, 0.05775]`; all gates passed | Survived |
| Argmax sufficiency guard | New paired 900 blueprints | Sensitivity `100%`, specificity `94.19%`; English `82.56%`; route gain erased | Rejected |
| Margin diagnosis | Retired failed guard surface | Threshold `0.90` separated complete/incomplete views | Diagnostic only |
| Thresholded H4 guard | New paired 900 blueprints | Sensitivity and specificity `100%` in every language/family; regret `-10.72%` | Survived |
| Historical external | Opened P-Bench 90 + StatQA 200 | P-Bench raw `0% -> 61.11%`; StatQA exact `1% -> 1%` | H4 rejected |

Every fresh synthetic surface was prepared from a frozen contract before scoring and was disjoint
from prior blueprint surfaces. The historical suites were already opened by v0.3 and were used only
for falsification, never as a new final claim.

## Historical evaluator incident

Evaluator v1 could not score `other/dfeep__paper_191__task_3`: its 351-column input produced an
86,215-byte inspection JSON, above the 65,536-byte sandbox transport ceiling. The process exited by
`SIGKILL` with `output_exceeded=true` before producing a score.

Evaluator v2 raised only the historical evaluator's minimum output ceiling to 131,072 bytes. It
recomputed all 290 tasks under one new fingerprint and changed no model, prompt, adapter, source,
or gate. The formerly blocked task then completed normally. The final public report fingerprint is
`5cbcfbbaf33fcdf112ab10b939e81cd78f99c297c923924a91fa33269fe1ca35`.

## Evidence map

- Full-route contract/result:
  `reports/evolve/family-router-replication-contract.json`,
  `reports/evolve/family-router-replication.json`.
- Rejection diagnosis: `reports/evolve/family-router-replication-failure.json`.
- Reduced-route contract/result: `reports/evolve/family-router-reduced-contract.json`,
  `reports/evolve/family-router-reduced.json`.
- Argmax guard contract/result and margin diagnosis:
  `reports/evolve/sufficiency-guard-contract.json`,
  `reports/evolve/sufficiency-guard.json`,
  `reports/evolve/sufficiency-guard-margin.json`.
- H4 thresholded guard contract/result:
  `reports/evolve/sufficiency-guard-thresholded-contract.json`,
  `reports/evolve/sufficiency-guard-thresholded.json`.
- Historical external contract/result:
  `reports/evolve/router-historical-external-contract.json`,
  `reports/evolve/router-historical-external.json`.
- Human-readable research record: `docs/DGP_EVOLVE.md`.
- Private row-level artifacts and resumable progress remain ignored under
  `artifacts/evolve/common-descent/family-llm-router-v2/`.

## Repository boundary

The work lives on branch `codex/family-router-replication`, based on main commit
`96e20732f0ef83cf3548777f3000807f448efe69`. The unrelated pre-existing modification to
`tests/test_stats.py` was preserved and must not be staged as part of the H4 commit. Generated raw
surfaces, adapter weights, row-level predictions, and progress files remain ignored and must not be
published.

Before any future research cycle, verify the live Git state, champion pointer, adapter hash, public
report fingerprints, and ignored-artifact integrity. A new cycle needs a new finite goal and new
external-evidence contract; this completed H4 outcome cannot become development data retroactively.
