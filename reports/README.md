# Reports

## Experimental v0.1.0 result

The fixed 60-task overnight comparison scored 23/60 (38.33%) for the base model and 27/60
(45.00%) for Charlie alpha. The overall delta is +6.67 percentage points, with gains on MATH-500,
GSM8K, MBPP+, English, and the retention canary. Simplified Chinese regressed by 40 points and the
trilingual canary regressed by 22.22 points, so the stable-candidate gate did not pass. Weight
publication is permitted under an Experimental label because every loading, source, data, privacy,
and sandbox hard gate passed. GGUF remains unpublished pending behavioral parity.

This directory holds the reviewable outputs used by the release gate. Generated model responses,
large logs, and machine-local paths stay under the ignored `reports/generated/` directory. Compact
aggregate reports are copied here before a release.

Expected files after an overnight run:

- `evaluation.json`: identical-parameter base/adapter comparison and subgroup deltas.
- `export.json`: adapter and fused-MLX checksums and load tests.
- `clean-load.json`: fresh-environment adapter and fused-model load validation.
- `release-gate.json`: stable, experimental, or blocked classification.
- `gguf-export.json`: conversion revision, matrix-equivalence result, and quantized checksum when
  the optional GGUF path is completed.

Missing files mean the corresponding gate has not run; they are never interpreted as a pass.
