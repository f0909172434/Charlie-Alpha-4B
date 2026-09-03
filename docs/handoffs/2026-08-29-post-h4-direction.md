# Post-H4 direction: matched historical replay before H5

Date: 2026-08-29 (Asia/Taipei)

H4 is closed as `external-rejected`; `v0.3.0-parent` remains the official champion. This note does
not reopen H4 and does not authorize H5 training, a new external final, a release, or publication.

## Why the next action changed

The older migration handoff proposed another powered synthetic family-router replication. H4 later
superseded that question: the reduced routed policy and thresholded guard already passed large,
disjoint synthetic confirmations, then failed historical StatQA exact improvement (`+0` versus the
required `+5` points). More synthetic confirmation of the same mechanism is therefore low
information.

A post-H4 paired audit exposed a more basic attribution issue. Across all 200 historical StatQA
rows, the frozen H4 candidate and the legacy v0.3 report had identical exact, method-set, and
column-set correctness. On P-Bench, however, 29 of the 53 H4 tasks that still routed to the
unchanged parent adapter changed from incorrect in the legacy v0.3 report to correct under the H4
evaluation run. The reported historical P-Bench gain therefore cannot be attributed cleanly to the
router or experts without a matched control replay.

## Finite next action

Recompute the frozen `v0.3` parent on all 90 locked P-Bench tasks and all 200 locked StatQA tasks
using the same historical runtime/evaluator path as H4 evaluator v2. Compare three quantities
separately:

1. legacy v0.3 report -> matched v0.3 replay (evaluator/runtime drift),
2. matched v0.3 replay -> frozen H4 candidate (mechanism effect),
3. legacy v0.3 report -> frozen H4 candidate (historical number, retained only for provenance).

The replay is diagnostic-only. It may select the *class* of the next research question, but it may
not modify H4, promote a model, or open a sealed external final.

## Branching rule after replay

- If the matched replay removes most of the P-Bench gain while StatQA remains inert, prioritize a
  new representation/transfer hypothesis rather than more routing or synthetic scaling.
- If a substantial matched P-Bench gain survives but StatQA remains inert, treat the next problem as
  task-format transfer: design a separately preregistered mechanism that can alter method/column
  extraction behavior without using the H4 historical rows as fresh evidence.
- If matched StatQA behavior differs materially, first explain that discrepancy before any new
  training.

In every branch, a future capability claim requires a new external-evidence contract and genuinely
independent evidence.

## Execution refinement

The first matched-replay contract (`historical-matched-replay-v1`) completed all 200 StatQA rows
and began a full 90-row P-Bench parent replay. That P-Bench plan was compute-redundant: the frozen
H4 route already contains exact same-evaluator, same-adapter parent outputs for the 53 tasks routed
to `parent`. Recomputing those 53 rows cannot add attribution information.

The v1 P-Bench run was stopped before completion and preserved. A replacement v2 contract reuses
the exact 53 H4 parent-route rows, reuses any already-computed v1 parent counterfactuals among the
37 expert-routed tasks, and computes only the remaining expert-routed parent counterfactuals. The
37-task set is determined solely by the frozen H4 routing map, not by correctness or observed
effect, so this refinement changes compute but not the estimand or evidence coverage.

## Matched replay result and H5 mechanism

The completed v2 matched replay changes the interpretation of H4 materially but not its terminal
decision. Replaying the same v0.3 parent under the H4 evaluator moves P-Bench raw accuracy from
0/90 to 55/90 and strict accuracy from 0/90 to 33/90. The frozen H4 routed candidate is exactly
55/90 raw and 33/90 strict as well: matched parent versus H4 candidate changes zero P-Bench
correctness outcomes. StatQA is stable across both evaluator replay and H4: exact 2/200,
method-set 11/200, and column-set 44/200, again with zero matched correctness changes.

Therefore H5 must not be another router, threshold, or selector-only expert study. The concrete
mechanistic mismatch is that the post-v0.3 evolution and expert updates use `selector_only=True`:
training is truncated immediately after the one-token method label, so plan, variable-role, tool,
and report tokens receive no gradient. Both historical external formats require behavior beyond
that one token.

H5 is preregistered as a matched-compute cross-format representation-repair pilot. Both arms start
from the unchanged v0.3 parent and receive the exact same fresh DGP groups, full token sequences,
shuffle seed, optimizer, learning rates, 48 microsteps, and fixed final checkpoint. The control
uses method-only loss on the full sequences. The candidate restores the original v0.3 objective:
0.45 method, 0.35 plan/tool, and 0.20 report. This isolates gradient support rather than sequence
length or training compute.

Selection uses a fresh DGP shard transformed into a different output contract: the candidate menu
is removed and the model must return only `{"methods": [...], "columns": [...]}`. Gold method IDs
and column roles are compiled deterministically from each DGP's registered analysis plan. The
candidate must gain at least five exact-accuracy points over both the parent and the matched
selector-only control while preserving selector regret, accuracy, and invalidity. Only then may a
separate registered confirmation shard be simulated. P-Bench and StatQA remain unavailable for H5
tuning or selection.

### H5 v1 implementation rollover before training

The first H5 contract was invalidated before either training arm began. Its initial renderer used
the in-memory insertion order of `Scenario.parameters`; on re-entry, the same simulations were read
from canonical JSONL whose nested keys had been sorted, changing the textual order of DGP audit
values and therefore the derived train/selection file hashes. The pilot integrity guard stopped
before model loading: both arm status files were absent, completed training microsteps were zero,
and the confirmation simulation file did not exist.

V1 is preserved as `superseded-before-training`, not interpreted as a research result. H5 v2 uses
new train/selection/confirmation seeds and defines all downstream rendering from the canonical
persisted simulation JSONL. This makes first-run and resumed prompts byte-identical while leaving
the causal question, matched A/B compute, loss weights, and gates unchanged.

### H5 v2 evaluation amendment after training, before selection

Both v2 arms completed the registered 48 microsteps and 12 optimizer updates. The first selection
attempt then failed before producing any model output: the selector scorer requires `(record,
simulation)` pairs, while the pilot passed only the frozen record list. The failure occurred before
the selection progress directory was populated; confirmation was still unopened.

The frozen v2 training implementation and both adapter files were left unchanged. A narrow
evaluation amendment pairs each frozen selection record with its frozen simulation by exact
`blueprint_id` before selector scoring. The 24 menu-free cases, their gold labels, adapter hashes,
selection gates, confirmation rule, and all training settings are unchanged. The amendment is
hashed separately and must be present in every v2 pilot/confirmation report.

### H5 result and H6 direction

H5 failed its selection gate and its registered 60-case confirmation therefore remains unopened.
On the 24-case menu-free selection shard, parent, selector-only control, and multi-token candidate
all scored 0 exact cases, 1/24 exact method IDs, and 22/24 exact column sets. The multi-token arm
also failed selector retention: normalized regret rose from about 0.5025 to 0.6624, accuracy fell
from 37.5% to 20.8%, and invalid selection rose from 41.7% to 58.3%.

The outputs show a narrower failure than generic extraction. Columns are already nearly perfect,
while method strings are usually generic human-readable names (`linear regression`, `t-test`,
`generalized linear model`) rather than the DGP-specific canonical IDs used by the project. H6 is
therefore a canonical semantic bottleneck study, not a larger H5. Two matched arms start again from
the unchanged v0.3 parent on 48 new DGPs. Both learn the same JSON methods+columns schema; the
control target uses the catalog display name and the candidate target uses the canonical method ID.
Selection uses a new 24-case DGP shard and still requires canonical IDs, with explicit selector
retention gates. A separately registered 60-case H6 confirmation remains unsimulated until all
selection gates pass.

### H6 result: canonical target supervision is insufficient

H6 was preregistered only after fixing two pre-run implementation issues: both arms now use an
identical system/user prompt so the only model-visible training difference is the assistant method
target string, and a duplicated YAML `confirmation_gates` block was corrected before the immutable
contract was created. The final H6 contract fingerprint is
`2c00f51cbde033844af110f65511aff4cd2d92a2854fc18fd6deb65a259fe7ac`; the prepared-data
fingerprint is `3ab382c21d3845b835753f19bfe0f83b908c77e1466beaeb8d4e4ef4883fe204`.

Both H6 arms completed the fixed 48 microsteps / 12 optimizer updates. On the fresh 24-case
selection shard, the canonical-ID arm still scored 0/24 exact and only 1/24 normalized method-set
correctness, versus 0/24 for the display-name control. The registered method gain therefore missed
its 10-point gate. The candidate did improve the pre-existing selector on the same fresh shard:
normalized regret moved from about 0.5017 to 0.4600, accuracy from 33.3% to 37.5%, and invalid
selection from 45.8% to 41.7%. Those secondary improvements do not rescue the failed canonical
extraction hypothesis. H6 is rejected with result fingerprint
`1c5851ee2add1a6b8eac39d970bce9fc40018335ada7fcfcfdec864eeabbf366`, and its registered
60-case confirmation remains unopened.

Post-H6 inspection also rejects a simple alias-resolver explanation. Roughly 20/24 generated method
strings per arm do not exactly match either a repository method ID or its existing catalog display
name; outputs such as `t-test`, `generalized linear model`, and `linear mixed effects model` collapse
the fine-grained distinctions needed by the DGP oracle. The useful signal is instead that canonical
supervision improved the bounded selector while free generation stayed poor, suggesting that the
model can rank some fine-grained distinctions when the local taxonomy is externally represented.

### H7 result: fixed global catalog grounding unlocks a confirmed interface mechanism

H7 tests that interpretation without changing any weights. Both arms use the exact same frozen
v0.3 parent and the same fresh menu-free questions. The control uses the H6 canonical extraction
prompt unchanged. The candidate adds one fixed repository-wide catalog containing all 28 existing
`method_id — display name` pairs. This catalog is identical for every case: it is not a
case-specific candidate menu and contains no simulator result, family label, or oracle-selected
subset.

The H7 contract fingerprint is
`5bcec5fab7ff97601d6994039b1f58746949ff74979d731510b9717b4f131b33`. Its 84 registered
selection+confirmation blueprints have zero ID or semantic overlap with all 132 H6 registered
blueprints, including H6's still-sealed confirmation blueprint set. The fresh-selection data
fingerprint is `ab20622a3936725ee41fd22a3ed19aa1b0ca1cafba35cc01caedac7f08565656`.

On the 24-case H7 selection shard, the unchanged menu-free parent scored 0/24 exact and 1/24 method
correct, while the same parent with the fixed catalog scored 7/24 exact and 8/24 method correct.
That is +29.17 points on both metrics with no column regression (23/24 for both). All registered
selection gates passed, authorizing H7's independent 60-case confirmation.

The confirmation reproduced the effect. Menu-free control again scored 0/60 exact and 2/60 method
correct. Fixed-catalog grounding reached 13/60 exact (21.67%) and 17/60 method correct (28.33%),
gains of +21.67 and +25.00 points respectively, while column accuracy remained exactly 55/60 in
both arms. All confirmation gates passed. H7 therefore confirms a synthetic prompt-interface
mechanism with result fingerprint
`f9e3b27a91b41f11a44cf21fde90f928cf198c36e751b2686dbe41a87809295c`.

This is not evidence that the weights themselves improved, and it does not reopen H4's historical
external benchmarks. It is stronger and more actionable than H6: the frozen 4B model contains
substantial latent fine-grained statistical discrimination that is inaccessible under the
menu-free interface but becomes reproducibly available when the repository taxonomy is supplied.

The preferred next weight-level hypothesis is therefore catalog-grounding distillation, not more
lexical ID rehearsal. A future H8 should start from the unchanged v0.3 parent, use fresh disjoint
DGPs, and compare equal-compute canonical-target training with versus without deterministic catalog
grounding/dropout. Selection must return to the menu-free format; success means the grounded arm
retains a substantial portion of H7's method/exact advantage after the catalog is removed, while
preserving the selector. Historical P-Bench and StatQA must remain unavailable for H8 tuning. If a
weight-level mechanism is confirmed, external evidence must be separately preregistered before any
capability or champion claim.

### H8 result: catalog grounding did not distill into menu-free weights

H8 implemented the registered weight-level test from the unchanged v0.3 parent. Both arms learned
the same 96 fresh canonical JSON targets for 96 microsteps / 24 optimizer updates. The control never
saw the catalog. The candidate saw the fixed H7 catalog on exactly 48/96 training rows, balanced as
4/8 rows in each of the 12 families; the other 48 candidate rows were byte-identical to the control
apart from the registered metadata flag. Held-out evaluation removed the catalog from all models.

On the 24-case menu-free selection shard, parent, control, and catalog-dropout candidate all scored
only 1/24 method correct and 0/24 exact. The candidate retained 0% of H7's registered exact and
method effects, although it recovered some column accuracy relative to the control. H8 therefore
failed every method/exact transfer gate and its 60-case confirmation remained sealed. The H8 result
fingerprint is `c5b2604147f3f0fb741b2ad7335f51dee6e1e034915eb30eea47a1b187621639`.

This falsifies the simplest weight-distillation story: intermittent catalog context plus ordinary
canonical-target SFT is not enough to make the menu-free model internalize the repository taxonomy.

### H9 result: constrained likelihood ranking did not beat flat catalog decoding

H9 returned to the unchanged v0.3 parent and tested whether the fine-grained signal existed in
sequence likelihood even when free generation was unreliable. Every fresh case received the same
fixed 28-method catalog. The control decoded one canonical method ID at temperature zero; the
candidate scored all 28 canonical IDs by mean assistant-token negative log likelihood and selected
the minimum-loss sequence.

On the 24-case selection shard, both free decoding and constrained likelihood ranking reached 9/24
method correct (37.5%). Ranking had mean gold rank 1.96 and MRR 0.632, but produced zero paired
accuracy gain over free decoding, so the registered gate failed and H9 confirmation stayed sealed.
The H9 result fingerprint is
`7f6517fecd3f0582f846388723156a6fee35355f3afbdd0914a09f8a951da0a0`.

### H10 result: output factorization did not confirm

H10 tested whether asking for method and columns in the same JSON object suppresses method quality.
For each case, the control used H7's joint fixed-catalog JSON response. The candidate reused the
exact same generated columns but replaced only the method with an independent H9-style method-only
fixed-catalog generation.

The 24-case selection looked promising: joint method accuracy was 33.3% and factorized method
accuracy 45.8%; exact rose from 29.2% to 45.8%, passing all registered gates. The independent
60-case confirmation did not reproduce the method gain. Joint method accuracy was 33.3% and
factorized method accuracy 35.0%, only +1.67 points versus the required +8. Exact improved from
25.0% to 35.0%, but the full confirmation gate therefore failed. H10 is rejected with result
fingerprint `c9128f29e597e40bc5b6356264bcf99ecee42de3f09da0621335475a31530f4b`.

### H11 result: richer semantic catalog did not beat the flat catalog robustly

H11 kept the unchanged v0.3 parent and compared H7's fixed `method_id — display name` catalog with
a fixed richer catalog built only from pre-existing repository metadata: each procedure's
assumptions, strengths, and uncertainty description. No family label, simulator score, case-specific
menu, or newly authored rule was added.

Selection again looked positive: semantic catalog versus flat catalog gained +12.5 method points,
+12.5 exact points, and +8.33 column points on 24 fresh cases. Confirmation reduced the gains to
only +1.67 method points and +3.33 exact points on 60 new cases. Semantic catalog absolute method
accuracy was 48.3% and exact 41.7%, but the paired improvement gates failed. H11 is rejected with
result fingerprint `f2164e7bb2ed732ec325e1e66ac4ba0ceb03027badb85349fb531b17373ed02b`.

### H12 result: H7 flat-catalog grounding replicates across three fresh seeds

Because the absolute flat-catalog accuracy varied substantially across fresh shards, H12 stopped
prompt micro-tuning and preregistered a direct stability replication of H7. Three disjoint 60-case
folds (180 total cases) were frozen together before any H12 simulation. Every fold compared the
unchanged v0.3 parent under the menu-free canonical JSON prompt against the same prompt plus H7's
fixed 28-method `method_id — display name` catalog. No fold was used for tuning.

All three folds independently reproduced a large effect:

- fold 1: exact +30.00 points, method +33.33 points, columns +0.00;
- fold 2: exact +36.67 points, method +41.67 points, columns -1.67;
- fold 3: exact +36.67 points, method +41.67 points, columns -1.67.

Pooled over all 180 cases, menu-free accuracy was 0.0% exact, 2.22% method, and 88.33% columns.
Flat-catalog grounding reached 34.44% exact, 41.11% method, and 87.22% columns: paired gains of
+34.44 exact and +38.89 method points with only -1.11 column points. Every registered aggregate
gate passed and all three folds exceeded the per-fold +10 exact / +10 method replication threshold.
The H12 result fingerprint is
`b105ee64dcad97e3c55dd9d8fddf205409cd38df1ec8c990551e69088a2ab9a3`.

The scientific interpretation is now substantially stronger than after H7 alone. The frozen 4B
weights contain reproducible fine-grained statistical discrimination that is largely inaccessible
under the menu-free canonical interface but becomes available when the repository method taxonomy
is supplied in context. Ordinary SFT did not internalize this capability (H8), likelihood ranking
did not improve it (H9), and two later interface refinements did not survive confirmation (H10/H11).
The stable mechanism is the simple fixed flat catalog itself.

H12 still establishes only a synthetic interface mechanism, not an external capability claim and
not a champion replacement. The next permissible step is a separately preregistered evaluation on
a genuinely new external source that was not used in H4/H5/H6/H7-H12 development. Historical
P-Bench and StatQA must remain unavailable for tuning or fresh-evidence claims.

### E1 closure: the preregistered source is unavailable, not a negative model result

E1 froze one independent external fixed-catalog evaluation before opening its intended source: the
27 expert-validated statistical-test vignettes associated with Mondal et al. The official PMC HTML,
NCBI BioC full text, and visually reviewed five-page PDF expose aggregate study results and one
example vignette, but not the complete 27-case set or its answer key. A version-4 source audit
therefore closed the source as `published-source-omits-case-set` before any model evaluation.

The terminal E1 report has result fingerprint
`6d2c9f155e8d66581e05e1483ad676aac5ffbc79e996dec7c854da2ffb0e7cb0`; its source-audit
fingerprint is `cb94076ba59bb3b6455fd0556a505f77625f1b2e4e9c3905353bdbf91270490d`.
`model_evaluation_started=false`, the external gate is explicitly `decidable=false`, and the
independent external interface claim is neither supported nor rejected. The official champion
remains `v0.3.0-parent`, no release is authorized, and E1 must not be repaired by substituting a
different source under the frozen E1 contract.

### E2 direction: source qualification before another one-shot evaluation

The next research action is a new E2 source qualification and preregistration, not another synthetic
mechanism search and not a rerun of E1. A source may be locked for E2 only if all of the following
can be established before seeing any E2 model output:

1. the public source has a stable URL or repository revision and terms that permit the intended
   non-commercial research use;
2. the same frozen source package contains complete case text plus an authoritative answer or gold
   method for every included case;
3. case extraction is mechanical from source structure or a rule frozen before model output, rather
   than hand-picking examples by perceived difficulty;
4. the existing exact alias table yields at least 12 eligible cases and at least 40% source coverage;
5. any unmapped method remains out-of-catalog and visible in coverage accounting rather than being
   added to the taxonomy after inspection;
6. all eligible cases are retained in source order, and neither prompts, aliases, gates, cases, nor
   exclusions may change after the first model answer is observed;
7. historical P-Bench and StatQA remain unavailable, and a pass can support only the external
   fixed-catalog interface claim, never automatic champion replacement or release.

A source reconnaissance on 2026-08-29 deliberately stopped before model evaluation. The public
`paytonyau/biostats-book` revision `4b97fd547f70988d6d80c55e5f4fee7b7239eb5f` exposes hidden
solutions but has fewer than 12 clearly source-authored method-selection cases, so it is not
sufficient as a standalone E2 source. The public OpenIntro IMS revision
`b88f367ac03a7b2998812d0dd49d57a8493c3b84` is openly licensed and contains exercises with public
solutions, but it does not present a pre-existing, mechanically identifiable method-selection case
set; manually mining convenient cases would weaken the external claim. Neither candidate is
authorized for model evaluation in its current form.

The finite next gate is therefore: identify one source that satisfies the seven E2 qualification
conditions, freeze its exact revision/asset hashes plus the unchanged 28-method alias table and
evaluation gates, materialize the full case set once, and stop again if the 12-case / 40%-coverage
gate fails. Only a passing materialization gate may authorize the one-shot menu-free versus fixed-
catalog model comparison.

### E2-v1 closure: no single source passed both frozen qualification thresholds

The bounded single-source search was completed before any E2 model output. Four structured
candidates were checked with the unchanged E1 exact-alias table and without adding aliases after
inspection:

- the Sheffield CC BY-SA decision graph produced 37 complete root-to-test paths, of which 13 mapped
  to the existing 28-method catalog (35.14% coverage): enough eligible cases, but below the frozen
  40% coverage floor;
- Haq and Nazir's CC BY-NC-SA 2016 article contributed every non-NA cell from its two published
  statistical-test tables: 10/22 mapped (45.45% coverage), so coverage passed but the 12-case floor
  did not;
- the complete Conyso CC BY 4.0 selector CSV mapped 3/8 rows (37.5%); and
- the complete MIT-licensed CrucibleBench parametric/non-parametric README tables mapped 4/7 rows
  (57.14%).

No candidate passed both thresholds, so E2-v1 is closed as source-qualification-negative rather
than weakening either threshold or cherry-picking source rows. No E2 model answer existed at this
point.

### E2-v2 result: a frozen direct-tabular multi-source pool passes external interface gates

E2-v2 changed only the evidence design, not the model mechanism. It preregistered a direct-tabular
pool of three independently authored, openly licensed sources whose source rows map conditions to a
single published test: Haq 2016 (all 22 non-NA Table 1/2 cells), Conyso 2026 (all 8 CSV rows), and
CrucibleBench at revision `cfc785c66eccf3a2b1df25f6fd8179f034e39142` (all 7 rows in the two
frozen README test tables). Sheffield was not included because deriving cases from its graph would
require an additional path-construction layer rather than direct tabular rows.

The contract explicitly records that source content was opened for qualification before the lock;
E2-v2 is therefore source-qualified, not source-blinded. Crucially, no model output had been opened,
and the three source identities, exact assets/revision hashes, complete extraction rules, unchanged
28-method aliases, prompts, gates, case order, and adaptation policy were all frozen before the first
model answer. The contract fingerprint is
`115fb480320d969d69e0c074f73647ea96057083393b34627a914e1e64608629`.

Materialization reproduced exactly 37 source rows with 17 exact-alias-eligible cases (45.95%
coverage), passing every source/coverage gate. Its data fingerprint is
`ecfc0a947d3232aab19791784c0f14ec8a1f71863bf372374665c63e217599c4`, and the immutable data
receipt fingerprint is `efa9ea848c7b2e06c6f1d445a586319443b17aa08bec5d10fb88d6f409f4fc57`.

The authorized one-shot evaluation used the unchanged `v0.3.0-parent` weights in both arms. On the
17 eligible external cases, menu-free control accuracy was 10/17 (58.82%) and fixed-catalog
accuracy was 12/17 (70.59%), a preregistered +11.76-point gain. Paired outcomes were 10 both-correct,
2 catalog-only, 0 control-only, and 5 both-wrong, for net +2. Across all 37 source rows, valid output
rate rose from 62.16% to 97.30% (+35.14 points). Source-level net improvements were 0 for Conyso,
+1 for CrucibleBench, and +1 for Haq, so all three sources were nonnegative and the frozen
cross-source robustness gates passed.

Every registered E2-v2 gate passed. The completed result fingerprint is
`e60e5a612368b68ba281fed11c21c972ed2d066b5ab1c614fb402f12d352e70f`.

This is genuine independent external evidence for the fixed-catalog *interface mechanism*, but it
is deliberately a narrow claim. The eligible denominator is only 17 and the two discordant paired
improvements give an exact two-sided McNemar p-value of 0.5; the preregistered effect/robustness gate
passed, but this is not strong evidence for broad free-form statistical competence. It does not
change weights, replace `v0.3.0-parent`, authorize release, or reopen historical P-Bench/StatQA.
The result should be preserved as positive external decision-source evidence without automatic
champion promotion.

### H13 result: menu-free final representations are already strongly linearly decodable

H13 changed no model weights and used the unchanged `v0.3.0-parent`. Before opening any H13
model representation, it froze 240 training, 72 selection, and 120 confirmation DGPs, all disjoint
from the prior registered surfaces. The probe location was fixed at the final RMS-normalized hidden
state on the last `add_generation_prompt` token with thinking disabled, immediately before the LM
head. Both menu-free and flat-catalog arms used the same frozen parent. A 28-slot ridge linear probe
was trained only on the 24 canonical methods observed in the training shard; the four unobserved
methods remained explicit coverage limitations and were masked from argmax rather than treated as
negative evidence about model capacity.

Selection showed that the menu-free representation was already highly decodable: 57/72 method
predictions were correct (79.17%) and top-3 accuracy was 98.61%, versus an 11.11% majority baseline.
The flat-catalog representation reached 59/72 (81.94%), only +2.78 points above menu-free. The
preregistered selector-head route therefore advanced while the catalog-dependent representation
route did not.

The untouched 120-case confirmation reproduced the result. The menu-free linear probe reached
90/120 method accuracy (75.00%) and 119/120 top-3 accuracy (99.17%), while the flat-catalog probe
reached 96/120 (80.00%). The +5.00-point catalog lift was small relative to the very large absolute
menu-free decodability. Every registered selector-head confirmation gate passed. The H13 result
fingerprint is `340ffe7c4f3630adde436e6841196aa1a03da3537057b1096f2e4130fc35faff`.

The interpretation is narrower and stronger than H12 alone: the 4B parent is not merely helped by
seeing taxonomy text. On these fresh DGPs, the information required to distinguish the observed
canonical methods is already present in the menu-free final hidden state; most of the catastrophic
2-3% free-generation method accuracy is therefore downstream of that representation, at the
decision/output interface.

### H14 result: a frozen menu-free selector head converts the latent signal into end-to-end JSON

H14 froze the H13 menu-free ridge head fit only on H13 training+selection representations (312
cases, lambda 0.01) before opening any H14 model output. It then compared ordinary menu-free joint
JSON generation with a factorized candidate on fresh DGPs. The candidate reused the control's exact
generated columns and replaced only `methods` with the frozen hidden-state head argmax. No catalog
was present, no v0.3 weight changed, and the H13 confirmation cases were not used to fit the head.

On the fresh 72-case H14 selection shard, ordinary menu-free generation scored 2/72 method correct
(2.78%), 0/72 exact, and 65/72 columns correct (90.28%). The frozen selector head scored 58/72 method
correct (80.56%) and 53/72 exact (73.61%) while column accuracy remained exactly 90.28%. The paired
gains were +77.78 method points and +73.61 exact points; all registered selection gates passed.

The independent 120-case confirmation reproduced the architecture effect. Menu-free control scored
3/120 method correct (2.50%), 0/120 exact, and 109/120 columns correct (90.83%). The selector-head
architecture scored 93/120 method correct (77.50%) and 84/120 exact (70.00%), with the same 90.83%
column accuracy. The confirmed paired gains were therefore +75.00 method points and +70.00 exact
points with zero column change. Every H14 confirmation gate passed. The H14 result fingerprint is
`cbec02c2f27e1938b1a218866ffcc0da637565acddb8f589c9090ba89eed93ee`.

H14 establishes a synthetic architecture result, not a new official champion or external
capability claim. `v0.3.0-parent` remains the official champion and release remains unauthorized.
The next research action is to freeze a runtime version of the selector-head architecture and seek
genuinely new external evidence for it. The four catalog methods absent from H13 head training also
need an explicit coverage strategy before any broad 28-method deployment claim.

### E3 result: the frozen H14 selector does not transfer to source-authored natural scenarios

The H14 runtime was frozen and hash-verified before E3. E3 then used the 20 expert-prepared full
natural-language statistical scenarios in the 2025 Cureus source, mechanically paired in source
order with the source's 20 published gold tests. The existing catalog could represent 10/20 source
tests, while the already-frozen H14 head covered 9/20; the primary paired capability denominator was
therefore preregistered as those nine cases, without adding a class after seeing model output.

The first evaluator attempt stopped before the first model answer because of a prompt/interface
field mismatch. Progress was verified as zero, and an evaluator-only amendment with its own hash
was frozen without changing the source, mapping, eligibility, H14 head, parent, or capability gates.
The amended one-shot E3 then completed. On the nine eligible cases, ordinary menu-free `v0.3`
generation scored 5/9 (55.56%), while the frozen H14 selector scored 1/9 (11.11%). The paired table
was 1 both-correct, 0 selector-only, 4 control-only, and 4 both-wrong, for net -4 and a -44.44-point
selector effect. The exact two-sided McNemar p-value was 0.125. Across all 20 source cases, valid
output rose from 40% to 100%, but every preregistered capability gate failed.

The terminal E3 result fingerprint is
`7525bbbe08c2ca003c0bb5b73bd7e38795b1b2ceb56953e96032bf50cbd6671a`.
This is a terminal negative result for the frozen H14 external-transfer claim. It does not imply
that the 4B parent lacks the underlying statistical capability: on these same external cases the
unmodified free generator was substantially better than the added head. E3 therefore localizes the
failure to transfer of the synthetic decision interface rather than just model size. No E3 case was
used for post-hoc head fitting, no champion changed, and release remains unauthorized.

### H15 result: matched-semantic prose style is not the source of the E3 collapse

H15 was preregistered only after E3 closed. It reused none of the 20 E3 scenarios and changed no
model weights. It froze 144 training, 48 selection, and 72 confirmation DGP semantic points from
new splits and seeds. Every semantic point had three deterministic renderings before representation
scoring: the existing repository audit prose, researcher-like natural English, and a concise
applied-statistics vignette. All three renderings declared the same statistical facts and shared
the same simulator-selected gold method. The contract fingerprint is
`6feb195fd1f715c2ab1a9af903bcccddc6095c34ae7f30ec648fbb0751f8d7ef`.

The data gate passed before any H15 representation was opened: 143/144 training semantic points
and 48/48 selection points were covered by the immutable 24-class H14 head. On selection, the
frozen H14 head scored 75.00% on audit prose, 66.67% on researcher prose, and 70.83% on the concise
vignette. The worst style drop was only 8.33 points, below the preregistered 15-point collapse
threshold, so H15 selected the `frozen-head-style-stable` route. A separately fit style-diverse
ridge probe reached 72.92% on all three selection styles, but its small gain was not needed to
explain the external failure.

The untouched 72-point confirmation reproduced the style-stable result with 100% H14-head
coverage. The frozen H14 head scored 80.56% on audit prose, 72.22% on researcher prose, and 75.00%
on the concise vignette; the maximum style gap was again exactly 8.33 points. The preregistered
style-stability confirmation gate passed. The fresh style-diverse probe reached 79.17%, 76.39%,
and 79.17% respectively, again showing that method information remains linearly accessible across
these renderings. H15's terminal result fingerprint is
`c0107dd174d755a89959975e496fe7ea3d52893813f30676b62750054aa18dfc`.

H15 therefore rejects the simple explanation that E3 failed merely because researcher prose uses a
different surface style from the synthetic DGP audit language. The next bounded research question
is narrower: identify which external task/semantic features move source-authored scenarios outside
the simulator-trained decision representation. The immediate diagnostic may reuse E3 only as
historical, already-opened evidence and must not fit or tune on E3; any new capability claim must be
tested on a genuinely new external source after the mechanism is frozen.

### Historical E3 readback after H15: style-diverse linear decoding still does not transfer

After H15 closed, the already-opened nine E3 head-eligible scenarios were read back once as
historical diagnostic evidence. No probe was fit or tuned on E3. The frozen H14 head was compared
with two probes refit exclusively from the fresh H15 training+selection representations using the
H15-selected lambdas: an audit-only probe and the style-diverse probe.

The frozen H14 head remained 1/9 (11.11%, top-3 also 11.11%). The H15 audit-only probe covered
8/9 E3 methods and scored 0/8 on that covered denominator (top-3 3/8). The H15 style-diverse probe
also covered 8/9 and scored only 1/8 (12.5%), although top-3 rose to 4/8. Thus the fresh synthetic
style-diverse boundary did not recover the external cases. The only E3 case it got exactly right was
the Fisher-exact scenario; several otherwise straightforward external cases were still mapped into
synthetic-specialized classes such as `posterior_predictive`, `calibrated_logistic`, or
`blocked_time_series_cv`.

The historical diagnostic fingerprint is
`d8a70d5f7f4118e8bb3d1356a9847be262552d100eb1d4cd8aec7379fce607d5`.
It is explicitly not fresh external evidence. Its value is localization: E3 remains far outside the
synthetic selector geometry even after the representation boundary is trained across multiple prose
styles. The next mechanism should therefore be selective rather than universal: use the H14-style
selector only when its decision is sufficiently supported, and fall back to the original 4B
generation path otherwise. Any confidence/sufficiency threshold must be calibrated without E3 and
then tested on a new external source before making another transfer claim.
