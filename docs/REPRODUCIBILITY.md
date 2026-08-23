# Reproducibility and overnight execution

## Immutable inputs

Model and dataset repositories are resolved to full commit SHAs in
`configs/sources.lock.json`. EvalPlus release contents are pinned by version and SHA-256. The Python
environment is locked in `uv.lock`; the user-local `uv` installer itself is fixed at 0.12.5 in CI.

Prepared corpora and weights are intentionally not committed. Tracked manifests retain source IDs,
revisions, licenses, split assignments, token counts, and content hashes. Rebuilding with the same
configuration produces the same deterministic problem-level split (seed 42).

## One command

```bash
make setup
make overnight
```

`make overnight` runs each stage in a separate `caffeinate`-protected process and imposes an
11-hour overall cap. Its status is updated atomically in
`reports/generated/overnight-status.json`. Completed data, translations, training checkpoints,
generations, and exports are reused when their input fingerprint is unchanged.

## Time allocation

| Stage | Hard budget |
|---|---:|
| English preparation and verification | 1 hour |
| Chinese teacher refinement | 1 hour |
| QLoRA pilot | 30 minutes |
| Main QLoRA | 6 hours |
| Compact base/adapter evaluation and export | 2 hours |
| Buffer | 30 minutes |

The already prepared English data makes the normal resumed run substantially shorter than the
upper bound. GGUF is optional in the overnight profile because downloading BF16 weights, merging,
building llama.cpp, quantizing, and parity evaluation can displace the core training run.

## Fixed fallbacks

The first pilot uses 1,024 tokens and Q/V LoRA on the last eight layers. A Metal out-of-memory
failure changes only one dimension at a time: first 768 tokens, then four LoRA layers. Other errors
stop the run and preserve logs. Checkpoints are written every 50 iterations; the main run resumes
from the selected pilot adapter.

## Security boundary

Generated Python and C++ execute through the macOS sandbox with no network access, writes limited
to a temporary directory, five CPU-seconds per process, a 1.5 GiB resident-memory monitor, and a
wall-clock timeout. C++ is compiled with the installed Command Line Tools compiler inside the same
file/network sandbox. `make test` includes live probes for denied network and external writes.

