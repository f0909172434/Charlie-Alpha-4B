# Reports

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

