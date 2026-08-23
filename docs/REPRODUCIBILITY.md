# Reproducibility and overnight execution

## Immutable inputs

`configs/sources.lock.json` pins every model and dataset to a full commit SHA. EvalPlus artifacts
are pinned by version and SHA-256, while `uv.lock` fixes the Python environment. The v0.2 and router
evaluation locks store canonical task hashes and are committed without generated answers.

Prepared training text, model caches, and weights are intentionally ignored. Public manifests keep
source IDs, revisions, category and language weights, record hashes, and output hashes. The Forge
seed is 20260824; the disjoint router confirmation seed is 20260825.

## Resumable workflow

```bash
make setup
make forge-lock
make forge-prepare
make forge-score
make forge-select
make forge-distill
make forge-build
make forge-pilot
make forge-train
make forge-calibrate
make forge-dev
make forge-freeze
make forge-final
make forge-router-lock
make forge-router-freeze
make forge-router-eval
make forge-router-verify
make forge-export
make forge-clean-load
make forge-release-check
```

Heavy stages use fingerprints, append-only per-task generations, or checkpoints. A matching result
is reused; changed inputs invalidate only the affected stage. Final evaluation refuses to run until
the recipe is frozen, and any changed frozen hash blocks it. Router confirmation has an independent
freeze and excludes every v0.1/v0.2 task.

## Actual compute profile

The selected dataset contains 52 semantic groups, 312 train records, and 18 validation records.
Training uses batch size 1, six-step gradient accumulation, 704-token maximum length, and only
384/544/704 padding buckets. Four rank-32 pilots train the same 2,129,920 parameters in the final
four layers. The winning full run took 2,896 seconds, peaked at 16.05 GB, and early-stopped with its
best checkpoint at iteration 431.

The 9B model is used only for one-pass teacher-forced scoring and protected Chinese translation.
It is absent from inference. The canonical runtime loads one 4B base and one 8.52 MB adapter, then
changes eight LoRA scales before a single generation.

## Security boundary

Generated Python and C++ are evaluated through the macOS sandbox with no network access, writes
limited to a temporary directory, CPU and memory limits, and a wall-clock timeout. Source/data
gates verify schema, original-problem split isolation, locked revisions, code checks, language and
category gradient ratios, and benchmark separation. Release scans reject training text, caches,
credentials, and machine-specific home paths from tracked or published artifacts.
