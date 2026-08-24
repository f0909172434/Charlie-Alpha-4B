# Reproducibility

## v0.4 development profile

`configs/pipeline.evolve.yaml` continues training from the frozen v0.3 adapter without changing its
release artifacts. Generated records and candidates live under `data/evolve/` and
`artifacts/evolve/`. The archive retains rejected candidates and changes the champion pointer only
after every promotion gate passes.

```bash
make evolve-prepare
make evolve
make evolve-status
```

Task generation uses the v0.3 dev surface as a discovery set. Each cycle receives a deterministic
144-DGP promotion shard with a new seed. The program fingerprints that shard before training and
does not use it for proposal scoring, training, or replay. Parent and candidate are scored as a
pair after training. The sealed v0.3 final surface remains unopened. See
[`DGP_EVOLVE.md`](DGP_EVOLVE.md) for the objective, archive format, and promotion gates.

## v0.3 statistics profile

`configs/pipeline.stats.yaml` is the complete v0.3 recipe. Model, teacher, dataset, and conversion
inputs are pinned to full revisions in `configs/sources.lock.json`; `uv.lock` fixes the MLX
environment and `pixi.lock` fixes the Python/R analysis runtime. The seed is 42.

The DGP blueprint is split before any language rendering. Generated training records, simulator
caches, evaluation answers, and weights are intentionally ignored. The repository retains the
generator, method catalog, revisions, sealed task IDs and hashes, configuration, aggregate results,
and release checks.

Run or resume the stages separately:

```bash
make setup
make stats-simulate
make stats-distill
make stats-data
make stats-lock
make stats-baseline
make stats-pilot
make stats-train
make stats-eval
make stats-export
make stats-release-check
```

`make overnight` runs the same sequence under a single budget. Simulation results, teacher edits,
training stages, and individual long-running evaluation items use input fingerprints. A matching
completed result is reused. Training writes adapter checkpoints every 100 microsteps; external
evaluation writes per-item progress below the ignored generated-report directory.

Three pilots receive the same 160 microsteps and trainable parameter count. A Metal memory failure
changes every pilot to the same fallback profile: first 512 tokens, then rank 16. The winner is
chosen by development normalized regret, statistical method accuracy, and validation loss, in that
order. Delta scale is chosen on development data. `stats freeze` then binds the selected adapter,
scale, data, and sealed evaluation lock before final evaluation.

The published aggregate report contains no evaluation prompts or model answers. Reproduction of a
sealed score requires the pinned public evaluation sources and the committed ID/hash lock.

## Isolation boundary

Python and R tools run in a locked Pixi environment through the macOS application sandbox. Each
analysis call has a wall-clock, CPU, memory, write-volume, and output limit. Network access,
reading outside the temporary input/runtime roots, writing outside the temporary directory, and
unapproved child processes are denied. Release checks exercise these restrictions for both Python
and R.

The planner cannot submit arbitrary generated code. It selects from a checked-in procedure catalog,
maps explicit column roles, and invokes a fixed Python or R implementation. Input files remain on
the local machine.

## v0.2 reproduction

The v0.2 FORGE workflow and artifacts remain available at tag `v0.2.0`. Its commands retain the
`forge-` prefix, for example `make forge-pilot`, `make forge-train`, `make forge-router-verify`, and
`make forge-export`. See `docs/FORGE.md` for that frozen recipe.
