# Charlie-Alpha-4B context migration handoff

Recovery date: 2026-08-27 (Asia/Taipei)

> This recovery snapshot is historical. For the current terminal research state, continue with
> [`2026-08-27-h4-champion-study.md`](2026-08-27-h4-champion-study.md).

Recovery status: **successful**. The repository, committed reports, ignored local artifacts,
manifests, checkpoints, and evaluation locks were reconciled. No training, data generation,
promotion, final evaluation, publication, deletion, reset, cleanup, or force-rebuild was run.

This file is the first state source for the next Codex conversation. Treat chat history only as a
lead. For a live continuation, recheck Git and artifact hashes before mutating anything.

## Executive state

- Official champion: `v0.3.0-parent`.
- Champion adapter SHA-256:
  `e644b7087c00321f16add940997f8809204458ff6bc51c97a795fed32d3e0b16`.
- Trainable base: `mlx-community/Qwen3.5-4B-MLX-4bit` at revision
  `32f3e8ecf65426fc3306969496342d504bfa13f3`.
- Public release: Experimental v0.3.0, Hugging Face
  `f0909172434/Charlie-Alpha-4B-MLX-4bit`, revision
  `dd6afeb30a3126b62535a5e2f5920f4fe5d33142`, tag `v0.3.0`.
- Latest completed research stage: v0.6 targeted repair selection. It completed and failed its
  registered selection gate; no v0.6 downstream surface opened.
- Most advanced evaluation lineage: the earlier family-router branch passed promotion, opened and
  scored the locked v0.3 synthetic final surface, then failed the parent paired-bootstrap gate.
  External benchmarks never opened and the route never became the default.
- No experiment is resumable or waiting for compute. All registered v0.5 and v0.6 experts are
  complete. Do not rerun them.
- Do not start v0.7 automatically.

## Repository state

Project path:

`~/Documents/Codex/2026-08-23/Charlie-Alpha-4B`

| Field | Recovered value |
| --- | --- |
| Branch | `main` |
| Local HEAD | `96e20732f0ef83cf3548777f3000807f448efe69` |
| Local `origin/main` | `96e20732f0ef83cf3548777f3000807f448efe69` |
| Live remote `refs/heads/main` | `96e20732f0ef83cf3548777f3000807f448efe69` |
| Ahead / behind | `0 / 0` |
| HEAD message | `research: evaluate gated DGP expert repair` |
| HEAD parent | `e06e83c4cee518837c5dd62992c9dc05bd0319f0` |
| v0.3.0 tag | `3c0d80e075bd8ce0095ab3b7be6fa88af0c81f39` |

Pre-existing dirty state at recovery start:

- `tests/test_stats.py` was modified but unstaged: 14 insertions and 2 deletions.
- The change replaces a dependency on ignored `data/stats/surface/dev.jsonl` with deterministic
  blueprint reconstruction for one test, and skips two Pixi-runtime tests when the locked Pixi
  interpreter is absent.
- This is a test-environment robustness change, not a research result. It was preserved and not
  edited, staged, reverted, or committed during recovery.
- There were no staged files and no non-ignored untracked files before this handoff was created.
- This handoff is a new untracked file until the user chooses to stage or commit it.

Ignored local state was preserved. The DGP-Evolve scope contains 4,188,453,161 bytes under
`artifacts/evolve/` and 30,638,679 bytes under `data/evolve/`. The wider ignored workspace is larger
and includes the v0.3 release artifacts and generated evaluation outputs.

## Evidence authority and integrity audit

The recovery used this order: local Git, hashed local artifacts, local status/progress/manifests,
committed machine-readable reports, committed documentation, live remote branch, then the migration
baseline.

Read-only integrity results:

- 427 JSON files in `reports/evolve/`, `artifacts/evolve/`, `data/evolve/`, and `configs/` parsed
  successfully; none were malformed.
- 24 manifest-to-JSONL hash/count pairs were re-read; all matched.
- 369 distinct files referenced by explicit path/SHA pairs were re-hashed; all existed and all
  matched.
- All 117 recovered evolve status files have `complete=true`.
- All 56 expert progress files are complete.
- v0.5: 36/36 expert status fingerprints match the committed training report.
- v0.6: 8/8 expert status fingerprints match the committed training report.
- All five registered v0.5/v0.6 target-data hashes match the local JSONL files.
- Champion adapter exists, is 8,521,749 bytes, and matches the official SHA above.
- The locked v0.3 final surface exists locally with 120 rows and SHA-256
  `435e635921f8c5999118e783e1ad3dd9b91802dd60d9b1b79fcfc223ffe8d8bc`.
- The evaluation lock file matches SHA-256
  `04205b3e4d292bd823b7fdadb0e2d06eee249cc527b2e97c4d1110f61ed8dfc8`.
- The family-router final lock's prior base and parent report hashes also match their local files.
- No checkpoint corruption, missing registered artifact, or fingerprint mismatch was found.
- The zero-byte `.iterate.lock` exists as a file-lock target, but no training/evolve process was
  running during recovery. It is not evidence of an active partial run.

The audit re-hashed every file explicitly covered by recovered status/manifest SHA fields. It did
not invent hashes for files whose schema records only a canonical experiment fingerprint.

## Official model and release state

### v0.3 champion

The authoritative champion pointer is
`artifacts/evolve/archive/index.json -> champion -> v0.3.0-parent`, mirrored by
`artifacts/evolve/selected.json` and `reports/evolve/development.json`.

Local adapter:

`artifacts/stats/adapters/calibrated-scale-1p0/adapters.safetensors`

The v0.3 release is intentionally classified as Experimental. The release gate allows weight
publication, but the complete ability gate failed: sealed DGP regret improved, while P-Bench and
StatQA did not improve and clarification behavior regressed. This is not a production statistical
analyst claim.

### Development routes

- No Cycle 1-5 adapter was promoted.
- No projection, common-descent, calibration, or block-projection candidate was promoted.
- The family-router route is a valuable failed experimental route, not the champion.
- v0.5 and v0.6 both selected `null` and opened no downstream surface.
- No new weights were released or uploaded after v0.3 as part of these development lineages.

## Research timeline

### v0.3 DGP-Regret release

Hypothesis: simulator soft targets improve statistical procedure selection over the 4B base and a
hard-label ablation.

Control/candidate: Qwen3.5-4B base and hard-label adapter versus DGP-Regret adapter.

Evaluation: locked v0.3 final DGP plus P-Bench, StatQA, trilingual, clarification, retention,
runtime, privacy, and load gates.

Result: final normalized regret `0.6727 -> 0.4437`, method accuracy `20.83% -> 45.00%`, invalid
selection `63.33% -> 38.33%`; paired improvement CI was positive. P-Bench and StatQA did not
improve, and clarification accuracy fell `43.33% -> 0%`.

Decision: Experimental v0.3.0 was released; the adapter remains champion.

### DGP-Evolve Cycles 1-5

| Stage | Hypothesis and control | Compute / surface | Result and gate | Downstream state |
| --- | --- | --- | --- | --- |
| Cycle 1 | Adaptive failure training from v0.3 | 160 planned microsteps; validation before promotion | Best candidate numerically identical to parent; rejected | Promotion shard generated but never scored; final sealed |
| Cycle 2 | Same basic self-iteration recipe | 160 microsteps; independent 60-DGP promotion | Validation regret improved 24.44%; promotion regret improved 19.46%, but paired CI lower `-0.0536` missed `-0.01` floor | Promotion opened/scored and retired; no final |
| Cycle 3 | Revised adaptive cycle | 160 microsteps; validation selection | No checkpoint passed validation selection | 144-case promotion generated but unopened and retired with lineage |
| Cycle 4 | Adaptive curriculum versus matched random control | 160 planned microsteps per arm | Both arms selected parent; adaptive improvement over control `0%` | Promotion unopened; final sealed |
| Cycle 5 | Decision-only constrained adaptive curriculum versus matched random control | 80 completed early-stopped microsteps per arm | Random last checkpoint improved regret but failed invalidity; adaptive worsened; both retained parent | Promotion unopened; final sealed |

Mechanistic finding: lower token-level validation loss did not reliably imply lower statistical
decision regret. Do not repeat Cycles 1-5.

### 4B / 9B base bake-off

Hypothesis: Qwen3.5-9B might provide enough capability gain to justify its local cost.

Control/candidate: locked Qwen3.5-4B versus Qwen3.5-9B, with identical reusable-dev scoring and a
32-microstep QLoRA smoke test.

Result: trilingual regret was `0.6409` for 4B and `0.6737` for 9B. The 9B route used about 10.63 GB
peak memory versus 5.92 GB and took 1.048x the training time, so feasibility passed but capability
and language gates failed.

Decision: continue with Qwen3.5-4B. No promotion or final surface opened.

### Static policy projection

Hypothesis: a conservative parent-constrained counterfactual target would improve the selector at
equal training compute.

Control/candidate: oracle-control soft labels versus policy-projection targets, same parent, records,
validation surface, optimizer, and seeds `42`, `314`, `2718`.

Result: initial projection improved mean regret 3.47% and won 1/3 seeds. Balanced prefixes improved
4.13%, reduced invalidity `30.00% -> 28.89%`, but again won 1/3 seeds. Both missed the 5% and seed-win
gates.

Decision: rejected. Promotion and final remained sealed.

### Gradient-conflict diagnostic

Hypothesis: cross-family gradient interference explains unstable aggregate gains.

Control/candidate: paired oracle-control and policy-projection gradients at the frozen parent on 23
groups across 12 families.

Result: 49.17% of cross-family pairs had negative cosine for both objectives. Projection mean-gradient
coverage was 19/23 groups and its mean norm was 9.01 versus 17.90 for control. The most negative
projected family cosine was about `-0.715`, between binary/count GLM and experimental/causal.

Decision: diagnostic supported interference as a bottleneck; it opened no promotion/final surface.

### Common descent / MGDA and uniform-family control

Hypothesis: a minimum-norm common-cone direction would reduce destructive interference better than
uniform family averaging.

Control/candidate: uniform-family versus MGDA/common-cone, both four full sweeps over the same 160
records with matched backward compute and fixed update norm.

Result: MGDA retained the parent (`0.3607` valid regret). Uniform-family selected update 4 at
`0.3161`, with accuracy `46.67% -> 50.00%` and invalidity `30.00% -> 26.67%`. Deterministic replay
reproduced all metrics and adapter SHA exactly.

Confirmation: the fixed uniform candidate improved reusable-dev trilingual regret only 3.20%; English
regret worsened 4.93%, English accuracy fell `45.00% -> 38.33%`, and English invalidity rose
`35.00% -> 40.00%`.

Decision: common-cone failed; uniform-family confirmation failed. Promotion and final remained sealed.

### Delta calibration

Hypothesis: a smaller effective-weight step along the parent-to-uniform path would retain aggregate
gain while removing granular regressions.

Control/candidate: parent versus fixed effective-weight scales `0.25`, `0.50`, `0.75`, `1.00` on
reusable valid and dev.

Result: every exact scale improved aggregate mean regret on both surfaces, but none passed every
language/domain/family gate. Scale 0.75 had the strongest worst-surface aggregate gain (valid 14.62%,
dev 8.92%) but still failed Traditional Chinese, prediction/analysis, predictive-calibration, and
time-series constraints.

Decision: `selected_scale=null`; promotion and final sealed. The earlier factor-space interpolation
was invalid because separate LoRA A/B interpolation creates quadratic cross terms; it is retained as
superseded evidence only.

### Block projection

Hypothesis: sparse layer support could preserve aggregate gain while avoiding granular regressions.

Control/candidate: parent versus 30 preregistered nonempty layer-subset/amplitude profiles on valid.

Result: the best all-four-layer 0.50 profile improved aggregate regret 7.68%, accuracy, and invalidity,
but failed language accuracy, domain accuracy, and family regret gates.

Decision: `selected_profile=null`. The new confirmation shard was not generated; Cycle 6 promotion
and final remained unopened.

### Oracle block-profile family route

Hypothesis: family-specific block masks could preserve local gains hidden by a single global mask.

Control/candidate: parent versus an oracle family mapping selected on valid.

Result: the fixed route improved its 120-DGP single-use confirmation by 4.29%, below the registered
5% minimum. Granular gates passed, but the primary gain gate failed.

Decision: rejected; no router implementation or promotion from this route.

### Compute-matched family experts

Hypothesis: separately trained family LoRA experts can retain specialization that a single adapter
loses through cross-family interference.

Control/candidate: a shared uniform-family control versus 12 family experts trained from the same 160
source records. Combined expert and shared-control backward record exposures were both 640.

Selection: six non-parent experts were selected: categorical update 4, clustered/repeated update 4,
experimental/causal update 2, group comparison update 4, predictive calibration update 4, and time
series/leakage update 3.

Confirmation: the oracle route improved trilingual regret 11.47% on a new immutable 240-DGP shard;
all granular gates passed.

Decision: family specialization passed and authorized router development, but did not itself promote
weights or open final.

### Router attempts and successful v2 development route

Rejected intermediate routers:

- Character n-gram Naive Bayes reached 100% synthetic template accuracy but only 38.89% family
  accuracy on manually written canaries, with severe Chinese transfer failure. It was rejected
  before adapter-regret validation.
- Frozen-parent LLM router v1 reached 58.33% family accuracy and at least 10% wrong-expert rate; no
  confidence threshold passed. Ambiguous menu boundaries were revised.

LLM router v2:

- Frozen parent predicts one A-M family letter; threshold 0.6 falls back to parent.
- Validation selected the threshold without opening confirmation.
- On the new 360-DGP confirmation shard, routed experts improved regret 9.49%; router family accuracy
  was 88.89%, wrong-expert rate 1.11%, paired CI `[0.0172, 0.0612]`, and all gates passed.
- On the new 720-DGP promotion shard, regret improved 10.89%, accuracy `46.81% -> 50.14%`, invalidity
  `32.22% -> 28.75%`, router family accuracy stayed 88.89%, wrong-expert rate stayed 1.11%, and paired
  CI `[0.0299, 0.0611]` passed.
- Promotion passed and authorized final evaluation.

Final result:

- The route reused the locked 120-row v0.3 synthetic final DGP surface.
- Versus v0.3 parent, trilingual regret improved 9.21%, accuracy `44.17% -> 51.67%`, and invalidity
  `30.83% -> 26.67%`.
- Versus base, regret improved 42.77% and invalid selections fell 53.62% relatively.
- The parent paired-bootstrap CI was `[-0.0059, 0.0769]`, so the registered lower-bound gate failed.
- `passed=false`, `proceed_to_external_evaluation=false`, and
  `external_benchmarks_opened=false`.

Decision: useful but failed experimental route. It is not the default route or official champion.

### v0.5 robust family experts

Hypothesis: exact direct expected-regret gradients, optionally CVaR-weighted, would beat Boltzmann
projection across family experts.

Control/candidates: Boltzmann-mean control, direct-mean, direct-CVaR, plus a mixed route. All three
trained arms covered 12 families, six updates per expert, 6,912 weighted record exposures, and 3,456
backward calls per arm.

Data/evaluation: 288 training groups and three mutually disjoint 360-group selection folds. The
confirmation, promotion, and final blueprints were registered but never materialized or scored.

Result:

- Boltzmann selected clustered/repeated update 5 and time-series/leakage update 6. It improved pooled
  selection regret 8.39%, with fold gains 8.23%, 7.66%, and 9.25%; paired CI was
  `[0.0292, 0.0445]`.
- Direct-mean selected no non-parent expert; only 1/72 direct-mean checkpoints survived all granular
  fold gates before cross-fit selection.
- Direct-CVaR selected no non-parent expert; 0/72 checkpoints survived.
- Both direct routes were 9.16% worse than the matched Boltzmann route; their bootstrap intervals
  were entirely below zero.
- The mixed route was identical to Boltzmann and therefore improved 0% over its control.

Decision: `eligible_candidates=[]`, `selected=null`, `passed=false`. Selection completed; confirmation,
promotion, final simulations, and final scores remained unopened.

### v0.6 triggered anchor repair

Hypothesis: update only high-regret/invalid regions while preserving low-regret anchors.

Anchor/control/candidate: the v0.5 Boltzmann route as anchor; matched Boltzmann replay as control;
triggered repair as candidate. Target families were binary/count GLM, group comparison,
experimental/causal, and linear/robust.

Data: 288 candidate-pool semantic groups; 96 selected groups (72 repair, 24 anchor); 384 rows per arm;
70/15/15 language gradient ratio; matched rows and loss weights. Every realized repair trigger
saturated at lambda 1 and every anchor used lambda 0, so the experiment tested binary repair plus
25% anchor replay, not a genuinely graded trust region.

Compute: both arms completed all four families and six updates per family. Each arm used 2,304
backward record exposures and 1,152 backward calls. All eight expert fingerprints and all 48 local
update checkpoint files are present and hash-valid.

Selection:

- Both arms selected only experimental/causal update 6 beyond the anchor route.
- Control versus anchor mean fold improvement: 3.75%; candidate versus anchor: 3.82%. Both missed the
  registered 5% mean threshold although their paired intervals versus anchor were positive.
- Candidate versus control pooled improvement: 0.075%; mean absolute improvement `0.000282`, CI
  `[-0.002559, 0.003096]`. One fold also failed granular invalidity/family-regret constraints.

Decision: `selected=null`, `passed=false`. Confirmation, promotion, final simulations, and final
scores remain unopened. Do not rerun any v0.6 expert.

## Evaluation surface ledger

`G` means generated/materialized, `O` opened to the candidate, `S` scored, and `I` immutable or locked.
“Retired” means the lineage is closed and the surface must not be opened again merely because chat
context was lost.

| Lineage | Selection / validation | Confirmation | Promotion | Final | External | Status |
| --- | --- | --- | --- | --- | --- | --- |
| v0.3 release | dev scored | release gates scored | n/a | G/O/S/I, 120 rows | P-Bench and StatQA scored for v0.3 | Released Experimental; final later reused by router |
| Cycle 1 | scored | n/a | G/I, not O/S | not opened | not opened | Rejected; promotion retired unopened |
| Cycle 2 | scored | n/a | G/O/S/I | not opened | not opened | Rejected; promotion retired after scoring |
| Cycles 3-5 | scored | n/a | G/I, not O/S | not opened | not opened | Rejected; promotion shards retired unopened |
| Base bake-off | reusable dev O/S | n/a | not opened | not opened | not opened | 4B retained |
| Policy projection | reusable valid/dev O/S | n/a | not opened | not opened | not opened | Rejected |
| Gradient diagnostic | paired training groups O/S | n/a | not opened | not opened | not opened | Diagnostic only |
| Common descent | reusable valid O/S | reusable dev O/S | not opened | not opened | not opened | Rejected |
| Delta calibration | reusable valid/dev O/S | same reusable surfaces | not opened | not opened | not opened | Rejected |
| Block projection | valid O/S | not generated | not opened | not opened | not opened | Rejected |
| Oracle block family route | valid O/S | G/O/S/I, 120 | not opened | not opened | not opened | Rejected; confirmation retired |
| Family experts | valid+dev O/S | G/O/S/I, 240 | not opened | not opened | not opened | Passed to router development |
| LLM router v2 | G/O/S/I, 240 plus dev canary | G/O/S/I, 360 plus canary | G/O/S/I, 720 plus canary | G/O/S/I via reused v0.3 final | not opened | Final failed; all these surfaces retired |
| v0.5 | G/O/S/I: three folds of 360 | not generated/opened/scored | not generated/opened/scored | blueprints registered only; no simulations/scores | not opened | Selection failed |
| v0.6 | G/O/S/I: three folds of 360 | not generated/opened/scored | not generated/opened/scored | blueprints registered only; no simulations/scores | not opened | Selection failed |

## Provenance discrepancies and interpretation

### `development.json` versus family-router final

`reports/evolve/development.json` says `invariants.sealed_final_surface_opened=false`, while
`reports/evolve/family-router-final.json` says `sealed_final_surface_opened=true`.

These fields describe different scopes:

- `development.json` is the aggregate for the original Cycle 1-5 / projection / common-descent /
  calibration / block-projection line. It was not updated into a global ledger when the separate
  family-expert/router branch later opened final.
- `family-router-final.json` is the authoritative report for that later branch. Its lock names the
  exact v0.3 evaluation lock, 120-row final-surface SHA, parent/base prior-report SHAs, promotion
  fingerprint, router fingerprint, expert selection fingerprint, and prompt SHA. All local hashes
  were verified during recovery.

Therefore there is no evidence that Cycles 1-5 themselves opened final. There is definite evidence
that the family-router branch did.

### Final-surface reuse and leakage boundary

The family-router final did not generate a new pristine final surface. It reused the already scored
v0.3 locked synthetic final surface. The recovered manifests state that router/expert training,
threshold selection, validation, confirmation, and promotion used other surfaces, and no raw final
rows were referenced by their training manifests. No direct train-row or answer leakage was found.

However, because v0.3 aggregate results from this same surface were already known before later method
development, it is not an untouched confirmatory surface for future candidates. Treat it as a locked
reused regression surface that is now retired, not as sealed future evidence. Any future confirmatory
claim needs a fresh preregistered surface and must not silently reuse this one.

### Immutable manifest flags versus runtime reports

The family-router promotion manifest records `used_for_promotion=true` but
`promotion_surface_opened=false`; the promotion report records `promotion_shard_opened=true`.
The manifest is a creation-time immutable provenance record and was intentionally not rewritten when
scoring began. The runtime report is authoritative for whether it was actually opened. A later
metadata-only maintenance change resolved the ambiguity for newly generated immutable surfaces by
adding an explicit `open_state_semantics` field while preserving the legacy open-state booleans for
backward compatibility. Historical manifests remain immutable and must still be interpreted using
their runtime reports.

### v0.5/v0.6 public training snapshots

At recovery time the local ignored `training-status.json` files had `selection_opened=true`, while
the committed `robust-family-experts-training.json` and `targeted-repair-training.json` still showed
`selection_opened=false`. Their fingerprints otherwise matched. A later metadata-only maintenance
change made those public training snapshots mutable lifecycle summaries, synchronized them after
selection, and corrected the committed snapshots to `selection_opened=true`. The dedicated committed
selection reports remain the authoritative outcome evidence: both selections opened, scored, and
failed; all later surfaces remained closed.

### Uncommitted test change

`tests/test_stats.py` is the only pre-existing non-ignored worktree modification. It may be worth
keeping, but it must be reviewed and tested separately before commit. It does not alter any recovered
research conclusion.

## Local artifact inventory

| Path | Files | Bytes | Safetensors | Role / disposition |
| --- | ---: | ---: | ---: | --- |
| `artifacts/evolve/archive` | 51 | 238,780,788 | 28 | Cycles 1-5 candidates, comparisons, champion ledger; immutable historical evidence |
| `artifacts/evolve/base-bakeoff` | 7 | 37,252,197 | 4 | 4B/9B smoke adapters and status; complete |
| `artifacts/evolve/policy-projection` | 76 | 417,744,400 | 49 | Three-seed projection/control checkpoints; complete, rejected |
| `artifacts/evolve/common-descent` | 268 | 1,160,012,202 | 99 | Common descent, calibration, block profiles, family experts, routers, promotion/final reports; complete |
| `artifacts/evolve/robust-family-experts` | 340 | 1,908,592,483 | 216 | v0.5 36 experts x 6 updates plus status/selection; complete, expensive to reproduce |
| `artifacts/evolve/targeted-repair` | 90 | 426,070,769 | 48 | v0.6 8 experts x 6 updates plus status/cache/selection; complete, expensive to reproduce |
| `data/evolve/cycles` | 45 | 7,753,287 | 0 | Cycle training/valid/promotion data and manifests; preserve |
| `data/evolve/policy-projection` | 5 | 1,222,554 | 0 | Matched projection datasets; preserve |
| `data/evolve/robust-family-experts` | 11 | 15,007,694 | 0 | v0.5 train targets and three selection folds; hash-valid, preserve |
| `data/evolve/targeted-repair` | 10 | 6,655,144 | 0 | v0.6 train targets and three selection folds; hash-valid, preserve |

Other important local-only artifacts:

- v0.3 champion adapter, verified SHA:
  `artifacts/stats/adapters/calibrated-scale-1p0/adapters.safetensors`.
- Family router model artifacts:
  `artifacts/evolve/common-descent/family-router/router.npz` SHA
  `d4481c64d3ffa71a4d93eae29a4e1e98b0af3cf67dee9e849846b477630ceeb5` and
  `model.json` SHA `7db98478f14ad9e29f9c6656710b484902207a1b3b0d4034bc2a8d3cd5de7eae`.
  This n-gram router is rejected evidence, not the successful LLM router.
- LLM router v2 immutable surfaces: validation 240, confirmation 360, promotion 720; all manifest
  hashes and row counts match.
- The successful LLM router is a frozen-parent prompt/threshold procedure, not a separately trained
  weight file. Its state is carried by the prompt SHA, selection/confirmation/promotion fingerprints,
  threshold 0.6, and selected family-expert checkpoint hashes.
- v0.5 and v0.6 contracts, generated data, training states, selection states, parent caches, and all
  registered checkpoints are local and should not be deleted even though the public summaries are
  committed.

## Negative-results registry

Do not repeat these unchanged recipes:

- DGP-Evolve Cycles 1-5 adaptive curricula, including decision-only constrained sampling.
- Qwen3.5-9B as the trainable base based only on feasibility; it failed capability/language gates.
- Initial or balanced static policy projection with the same objective and seeds.
- The same MGDA/common-cone finite-update recipe.
- Global effective-weight delta interpolation over the same update.
- LoRA factor-space interpolation; it is mathematically the wrong path.
- The same exhaustive four-layer block projection.
- Oracle block-profile family route without a new mechanism; it missed its registered confirmation
  gain.
- Character n-gram family routing; it failed manual Chinese transfer.
- LLM router v1 menu; its family boundaries and wrong-expert rate failed.
- v0.5 direct-mean and direct-CVaR targets; granular selection rejected them and they underperformed
  Boltzmann.
- v0.5 robust-mixed route; it was identical to control.
- v0.6 realized binary triggered repair plus 25% anchors; it only tied matched replay.
- Additional epochs, steps, learning-rate sweeps, or seeds without a mechanism-level change, unless a
  preregistered power analysis specifically shows the uncertainty is the bottleneck.

## Surviving useful findings

- Qwen3.5-4B remains the correct trainable base for this hardware/evidence set.
- Token-level language-model loss and statistical decision regret are not aligned enough to use loss
  as a promotion surrogate.
- Cross-family interference is substantial: nearly half of paired family gradients conflict.
- Uniform-family updates can find useful aggregate directions, but their transfer and granular safety
  are unstable.
- A single global scale or layer mask did not remove the granular regressions.
- Family specialization produced repeatable useful signal: the family-expert oracle confirmation
  passed, and the LLM-routed route passed a much larger promotion surface.
- Router reliability was not the failing final gate for v2: family accuracy was 88.89% and wrong
  expert rate 1.11%.
- The family-router final point estimate was favorable but underpowered/variable relative to the
  parent on 120 reused cases; only the paired-bootstrap gate failed.
- Direct expected-regret and CVaR objectives did not improve selector learning under the registered
  setup.
- The realized v0.6 trigger collapsed to binary repair and almost exactly matched Boltzmann replay,
  so the test did not establish a benefit from targeted repair.

## Unresolved questions and recommended next question

Open questions:

- Does the fixed family-router candidate's favorable point estimate replicate on a genuinely fresh,
  sufficiently powered surface, or was the promotion/final gap caused by generator variance?
- Are family regressions best handled by uncertainty-aware fallback and conservative expert
  activation, rather than more selector-only fine-tuning?
- Has one-token soft-target selector tuning reached a representation limit for unresolved families?
- Would explicit expert isolation or a representation-level objective reduce cross-family
  interference more reliably than target reshaping?
- Can router uncertainty be calibrated to expected regret/harm rather than only family-label
  confidence?

Recommended next research question, before inventing another training recipe:

> With the existing family experts, prompt v2, and threshold 0.6 frozen, does the routed candidate
> reproduce its parent-regret gain and granular safety on a new preregistered synthetic surface whose
> size is chosen by power analysis, without reusing the retired v0.3 final surface?

Why this is highest-information: promotion on 720 cases passed with a positive paired interval, while
the 120-case reused final retained a favorable 9.21% point estimate but its paired interval crossed
zero. A fresh powered replication can distinguish evaluation variance from a real generalization
failure without retraining or changing the candidate after seeing the new surface. If it fails, the
next mechanism should be uncertainty-aware safe fallback or representation/expert isolation; if it
passes, external benchmark design becomes the next gate.

This is only a proposed preregistered question. Do not generate the surface, score it, open external
benchmarks, or call it v0.7 without explicit user authorization.

## Safe continuation checklist

Before any future mutation:

1. Read this handoff, then verify live `HEAD`, `origin/main`, and `git status`.
2. Preserve the user's `tests/test_stats.py` modification.
3. Revalidate the exact candidate/checkpoint fingerprints that the proposed experiment would use.
4. Treat every old promotion/final surface listed as retired; never reopen it for a new decision.
5. Write a new immutable contract and power analysis before generating any new confirmation/final
   surface.
6. Keep champion, release, Hugging Face, external benchmarks, and publication unchanged unless the
   user explicitly authorizes those actions after gates pass.
7. Do not use `--force`, reset, clean, or delete ignored artifacts during normal continuation.
