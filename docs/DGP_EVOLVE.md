# DGP-Evolve

DGP-Evolve is the development self-iteration line for Charlie alpha after v0.3. It trains small
statistics adapters on failures that a deterministic simulator can verify. Each candidate remains
separate from the released adapter until it passes its pre-registered paired promotion tests.

The implementation is experimental. One or more successful cycles do not establish a general
self-improving model, and this repository does not claim that the method is novel before matched
ablations and external evaluation support that claim.

## Scope

The loop can change LoRA matrices and generate synthetic DGP records. It cannot edit its trainer,
simulator, evaluation gates, runtime sandbox, or source locks. It never trains on the sealed v0.3
final DGP surface. A rejected candidate stays in the archive and the selected adapter remains
unchanged.

The v0.4 runtime also adds a deterministic statistics compiler. It parses declared column roles,
constructs a menu of audited methods, and compiles the selected method into a complete tool plan.
The language model chooses one menu label. It cannot invent a column name or executable analysis.
If a tool call fails, the agent can try the next audited candidate, up to the existing four-call
limit.

## One cycle

The current adapter is the parent policy. A cycle performs these operations:

1. Score the parent on the discovery DGP surface.
2. Mutate parameters near high-regret cases while keeping every value inside its family bounds.
3. Run the common-random-number simulator and discard invalid or duplicate proposals.
4. Rank proposals by validity, novelty, frontier proximity, and measured family learning progress,
   subject to family and oracle-method exposure ceilings.
5. Train the existing LoRA matrices for up to 160 microsteps with 20% replay from v0.3. During
   evolution, only the simulator's method-distribution loss produces gradients.
6. Compare parent and candidate on a cycle-specific promotion shard plus the retention suite.
7. Promote only when every gate passes. Otherwise retain the parent.

For proposal \(x\) in family \(f\), the selection score is

\[
S(x)=V(x)N(x)\exp\left[-\frac{(r(x)-r^*)^2}{2\sigma^2}\right]L_f.
\]

Here, \(V\) verifies that at least one valid method exists and that simulator soft targets sum to
one. \(N\) is normalized distance from prior DGPs, \(r\) is parent regret, and \(r^*=0.65\) targets
cases near the current failure frontier. \(L_f\) is a family-specific learnability signal computed
only on the reusable validation surface. Relative regret change passes through a bounded tanh
transform, so the signal stays between 0.25 and 0.75 and returns to 0.5 when no checkpoint advances.
Promotion outcomes never choose the next cycle's training tasks.

The greedy constrained selector first keeps one stepping stone from every represented family, then
fills the remaining slots by score. No DGP family and no simulator-selected oracle method may
occupy more than four of the 32 new groups. The ceilings prevent one high-reward procedure from
dominating a small update and are applied identically to adaptive and random-control arms.

Each training DGP produces two English boundary views, one Traditional Chinese view, and one
Simplified Chinese view. Their loss weights preserve the 70/15/15 gradient ratio. The selector
keeps 32 proposals and adds eight replay groups. This makes 160 records, so 160 microsteps cover one
complete group epoch. The trainer updates 2,129,920 adapter parameters on the current Qwen3.5-4B
MLX model; base-model parameters remain frozen.

The iterator places one replay group after every four new groups. The v0.3 adapter already contains
the plan and report behavior; short evolution cycles optimize only the regret soft-label selector
so that token imitation cannot improve while the promotion objective worsens. Selector-only batches
end immediately after the one-token method label; later plan, tool, tool-result, and report tokens
cannot affect that causal logit and are not sent through the model. Weight decay is zero
because the parent LoRA already contains useful nonzero matrices; decaying those matrices toward
zero would alter the parent even when the task gradient is weak. The LoRA+ rates are
`5e-7 / 2e-6`.

The trainer saves 40-, 80-, 120-, and 160-step weights in addition to the lowest-loss checkpoint. It
scores each distinct checkpoint once on the frozen validation DGPs, before reading promotion data.
A checkpoint must reduce validation regret by at least 1%, keep invalid selections from increasing,
and stay within the accuracy regression limit. At most one checkpoint advances to the promotion
shard. If none qualifies, the cycle records a rejection without opening that shard.

## Promotion isolation

Discovery and promotion use different DGPs. Before training starts, the program creates a fresh
144-case Latin-hypercube shard from a cycle-specific seed and records its SHA-256. Task selection and
training never read that shard. Parent and candidate receive the same cases after training, then the
program retires the shard from future promotion decisions. The next cycle receives a different
seed.

Promotion requires all of the following:

- relative normalized-regret improvement of at least 1%;
- paired-bootstrap lower 95% bound of at least -0.01 regret;
- method accuracy no more than 3 points below the parent;
- invalid-method rate no more than 2 points above the parent;
- retention accuracy no more than 1 point below the parent;
- each language's accuracy no more than 3 points below the parent and regret no more than 0.05
  higher;
- domain accuracy no more than 10 points lower and family regret no more than 0.15 higher.

These are development gates. Public capability claims still require the sealed final suite and the
release checks defined for v0.3. DGP-Evolve does not open that final suite during iteration.

## Matched control

Development cycles train a uniformly sampled random-DGP control and the adaptive curriculum from
the same parent. Both arms use 32 new groups, eight replay groups, 160 microsteps, the same frozen
validation records, optimizer, seed, and 2,129,920 trainable parameters. The better validation
checkpoint is selected by regret, invalidity, accuracy, and then loss; an exact tie retains the
random control. Adaptive selection must lower regret by at least 3% relative to the control before
the result can be attributed to the curriculum.

The match covers training compute, not the full pipeline. Both arms simulate and parent-score 72
new proposals, while adaptive selection additionally scores the 60-case reusable dev surface to
locate failure regions. Reports record this overhead explicitly. Neither arm sees promotion data
during task or checkpoint selection.

## Archive and rollback

`artifacts/evolve/archive/index.json` stores the champion, parent-child relationships, promotion
decisions, and per-family learning progress. Every candidate has its own directory under
`artifacts/evolve/archive/cycle-NNNN/`. Promotion changes one pointer in the archive; it does not
overwrite the previous adapter. The initial node always points to the frozen v0.3 adapter.

The generated DGPs, checkpoints, and local paths are ignored by Git. Public commits contain the
trainer, configuration, tests, and reports, but no local data cache or model weights.

## Commands

```bash
make evolve-prepare       # generate and fingerprint one cycle without training
make evolve               # train, evaluate, and promote or reject one candidate
make evolve-status        # show the current champion and archive history
make evolve-bakeoff       # compare locked Qwen3.5 4B/9B quality and local QLoRA cost
make evolve-project-prepare # build and audit parent-constrained targets without training
make evolve-project       # run the matched three-seed conservative policy pilot
make evolve-project-balanced # repeat with family/method-balanced checkpoint prefixes
make evolve-diagnose      # measure paired per-family LoRA gradient conflict at the parent
make evolve-cone          # compare deterministic uniform-family and common-descent updates
make evolve-cone-confirm  # confirm the fixed valid winner on reusable trilingual dev data
make evolve-calibrate     # test fixed parent-to-winner delta scales on valid and dev
make evolve-cone-promote  # open a fresh promotion shard only if every cone pilot gate passed
make evolve-robust-experts-prepare # freeze the v0.5 blueprint and gate contract
make evolve-robust-experts-data    # build only training and three selection folds
make evolve-robust-experts-train   # run the three matched family-expert arms
make evolve-robust-experts-select  # select or stop without opening downstream shards
make evolve-targeted-repair-prepare # freeze the v0.6 discovery and gate contract
make evolve-targeted-repair-data    # build fresh training and three selection folds
make evolve-targeted-repair-train   # train the two compute-matched targeted arms
make evolve-targeted-repair-select  # select or stop without opening downstream shards
make evolve-llm-router-replication-prepare # freeze the powered family-router replication
make evolve-llm-router-replicate    # score the frozen route on its fresh replication surface
make evolve-llm-router-replication-diagnose # post-rejection mechanism diagnosis only
make evolve-llm-router-reduced-prepare # freeze the causal-expert exclusion before fresh scoring
make evolve-llm-router-reduced-confirm # confirm the reduced route on a new surface
make evolve-sufficiency-guard-prepare # freeze the argmax sufficiency guard
make evolve-sufficiency-guard-confirm # confirm or reject the argmax guard
make evolve-sufficiency-guard-diagnose # retired-surface probability-margin diagnosis
make evolve-sufficiency-guard-thresholded-prepare # freeze the 0.90 guard
make evolve-sufficiency-guard-thresholded-confirm # score it on a fourth fresh surface
make evolve-router-historical-external-prepare # freeze the opened-suite falsification gate
make evolve-router-historical-external # run routed adapters on historical P-Bench and StatQA
```

The equivalent CLI supports at most two cycles per invocation:

```bash
uv run charlie-alpha stats iterate --cycles 1 \
  --config configs/pipeline.evolve.yaml
```

Every completed stage uses content fingerprints. Re-running a command reuses valid work. `--force`
can deliberately regenerate mutable discovery or training work, but it cannot replace a promotion
shard once that shard has been prepared. Changed promotion settings take effect in a new cycle.

`evolve-bakeoff` uses only the reusable 60-case dev surface. It evaluates both locked models in
English, Traditional Chinese, and Simplified Chinese, then runs one identical 32-microstep,
eight-update QLoRA smoke test. Both candidates use a 4 GB MLX free-cache ceiling so variable-length
batches do not retain enough cached buffers to distort the 24 GB feasibility test. The 9B route is
recommended only if it lowers mean trilingual regret by at least 10%, stays within three accuracy
points in every language, peaks below 18 GB, and takes no more than 2.5 times the 4B training time.
It does not open a sealed final or promotion surface.

On the locked reusable dev surface, the 4B base reached `0.6409` trilingual normalized regret and
the 9B base reached `0.6737`; lower is better. The 9B arm trained in 1.05 times the 4B wall time but
used `10.63 GB` peak memory instead of `5.92 GB`. It therefore passed the local memory and
throughput gates but failed the capability and language gates. The v0.4 route keeps Qwen3.5-4B as
the trainable base and reserves Qwen3.5-9B for text refinement only. The complete locked comparison
is in [`reports/evolve/base-bakeoff.json`](../reports/evolve/base-bakeoff.json).

## DGP Policy Projection

The next development hypothesis changes the learning objective rather than selecting another set
of failures. For each candidate menu, the frozen parent policy supplies probabilities
\(p_0(a\mid x)\). Simulator soft labels are inverted to recover normalized candidate regrets
\(r(a,x)\). A conservative counterfactual target is then formed as

\[
\tilde q(a\mid x) \propto
\left[(1-\epsilon)p_0(a\mid x)+\epsilon/|A|\right]
\exp[-r(a,x)/\tau],
\qquad
q=(1-\alpha)p_0+\alpha\tilde q.
\]

The current pilot uses \(\epsilon=0.05\), \(\tau=0.15\), and \(\alpha=0.5\). A target is replaced
by the unchanged parent distribution if its simulator-expected regret would increase. Training
minimizes the menu-level cross-entropy to this target at one causal decision token. It does not
imitate a hidden chain of thought or let the 9B text teacher decide correctness.

The matched control uses the same Cycle 5 uniformly sampled records, parent adapter, validation
surface, optimizer, microsteps, and seeds (`42`, `314`, and `2718`), but retains the original oracle
soft labels. Projection proceeds to a fresh promotion shard only if it wins at least two seeds,
lowers mean reusable-dev regret by at least 5%, and does not raise mean invalid selections. These
were registered as development gates rather than capability claims.

The first matched pilot improved mean reusable-dev regret by `3.47%` and won one of three seeds.
Balancing every checkpoint prefix across DGP families and oracle methods raised the mean improvement
to `4.13%` and reduced invalid selections from `30.00%` to `28.89%`, but it again won only one of
three seeds. Both pre-registered gates failed, so no promotion shard or sealed final case was
opened. These results reject the current static target-projection recipe. The next diagnostic
measures the per-family LoRA gradient geometry at the unchanged parent; it does not authorize more
runs of the same recipe. Machine-readable results are in
[`policy-projection.json`](../reports/evolve/policy-projection.json) and
[`policy-projection-balanced.json`](../reports/evolve/policy-projection-balanced.json).

## Gradient conflict and common descent

The paired diagnostic differentiates every selected training group at the unchanged v0.3 parent.
It disables dropout and compares the oracle soft-label and parent-constrained objectives on the
same 23 groups, covering all 12 families. For policy projection, `49.17%` of cross-family gradient
pairs have negative cosine. The ordinary mean gradient is a descent direction for 19 of 23 groups,
leaving four groups with a negative first-order alignment. The oracle control has the same
cross-family negative fraction and covers 16 of 23 groups. Projection therefore does not create the
conflict, but its mean family-gradient norm is about half the control's (`9.01` versus `17.90`), so
order and dropout noise are large relative to the intended update.

The most negative projected family cosine is `-0.715` between binary/count GLMs and experimental or
causal tasks. A minimum-norm convex combination of the 12 normalized family gradients has positive
alignment with every family on this diagnostic (`minimum = 0.03283`). This is a first-order result
at one parameter point, not evidence that the corresponding finite update improves held-out
regret. The complete matrix is in
[`gradient-conflict.json`](../reports/evolve/gradient-conflict.json).

The `DGP Common-Descent Cone` pilot removes SGD order and dropout as variables. Both arms
make four complete sweeps over the same 160 records. The control takes an equal-weight mean of the
12 normalized family gradients. The candidate solves for the minimum-norm point in their convex
hull, then applies a fixed adapter-L2 step of `0.012`. It advances only if reusable-dev regret is at
least 5% lower than the uniform-family control and 1% lower than the parent, invalidity does not
increase, accuracy remains within three points, and every candidate update keeps positive
first-order family alignment. Promotion and final surfaces remain sealed until all those gates pass.

The minimum-norm convex-hull update is not a new optimizer. It is the established
[Multiple-Gradient Descent Algorithm](https://www.numdam.org/item/CRMATH_2012__350_5-6_313_0.pdf)
(MGDA) applied at the level of DGP families. Nearby alternatives include
[PCGrad](https://arxiv.org/abs/2001.06782),
[CAGrad](https://papers.neurips.cc/paper_files/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html),
and [Nash-MTL](https://proceedings.mlr.press/v162/navon22a.html). The project-specific hypothesis is
that simulator-defined family objectives, regret-projected targets, selector-only LoRA gradients,
and sealed statistical promotion gates form a useful compute-efficient combination. No novelty
claim is attached to the gradient solver itself.

The matched result rejected MGDA for this selector. The common-cone arm retained the parent at
`0.3607` reusable-valid regret; its finite checkpoints did not pass the validation selection gate.
The simpler uniform-family arm reached `0.3161`, `50.00%` accuracy, and `26.67%` invalid selections,
compared with the parent's `0.3607`, `46.67%`, and `30.00%`. A forced deterministic replay after a
fingerprint-scope bug reproduced all four checkpoint metrics and the final adapter SHA exactly.

That valid result did not transfer cleanly. With the candidate fixed before scoring the reusable
60-DGP dev surface, mean trilingual regret improved only `3.20%`; English regret worsened `4.93%`,
English accuracy fell from `45.00%` to `38.33%`, and invalid selections rose from `35.00%` to
`40.00%`. Retention stayed at `100%`. The confirmation gates failed, so the cycle-specific
promotion shard remains unopened. These results are in
[`common-descent.json`](../reports/evolve/common-descent.json) and
[`common-descent-confirmation.json`](../reports/evolve/common-descent-confirmation.json).

The next low-cost check interpolated the effective adapter weight from the parent at fixed
scales `0.25`, `0.50`, `0.75`, and `1.00`. A scale is eligible only if both reusable valid and dev
surfaces pass language, domain, family, invalidity, accuracy, and retention noninferiority gates,
and its worst-surface trilingual regret improves by at least 1%. The scale list and selection rule
were frozen before those intermediate adapters were scored.

The effective update is represented exactly as a rank-64 concatenation of the rank-32 parent and
candidate factors. This matters because separately interpolating LoRA's A and B factors introduces
quadratic cross terms and is not a linear path in effective weight space. The original factor-space
pilot exposed this implementation flaw; it was replaced and all scales were rescored before making
a selection decision. Its superseded aggregate result is retained in
[`delta-calibration-factor-space-superseded.json`](../reports/evolve/delta-calibration-factor-space-superseded.json)
rather than silently discarded.

Every exact scale lowered mean trilingual regret on both surfaces, but none passed all granular
gates. Scale `0.75` produced the strongest worst-surface aggregate result: valid improved `14.62%`
and dev improved `8.92%`; aggregate valid accuracy rose from `56.67%` to `63.33%` and invalidity
fell from `23.33%` to `16.67%`. On valid, however, Traditional Chinese accuracy fell from `46.67%`
to `43.33%`, prediction-and-analysis accuracy fell from `100%` to `66.67%`, and both predictive
calibration and time-series family regret rose by `0.3333`. Dev also missed the family gate because
time-series regret rose by `0.1667`, just beyond the frozen `0.15` allowance. No global scale was
selected. The machine-readable result is in
[`delta-calibration.json`](../reports/evolve/delta-calibration.json).

The remaining development test is a pre-registered exhaustive block-support projection over the
four adapted layers. It evaluates every nonempty layer subset at effective-weight amplitudes `0.25`
and `0.50` (30 profiles) on valid only. The single selected profile is then scored on a new,
immutable 60-DGP confirmation shard generated from seed `53000042`; no rejected profile sees that
shard. This is a sparse-support model-merging
test, not a novel optimizer: layer-wise coefficients and interference-aware task-vector masking have
prior art in [AdaMerging](https://openreview.net/forum?id=nZP6NgD3QY) and
[TIES-Merging](https://arxiv.org/abs/2306.01708). The project-specific question is whether DGP, language,
domain, family, invalidity, and retention verifiers can identify a compact update that preserves the
aggregate signal without the observed statistical regressions. This confirmation layer was added
because an interrupted prototype had already exposed reusable dev to three profiles, making dev
unsuitable for a clean selection confirmation. A confirmed mask still has to pass the fresh
single-use Cycle 6 promotion shard; final data remain sealed.

No support passed valid. The strongest profile kept all four layers at amplitude `0.50`, lowering
trilingual regret from `0.3875` to `0.3577` (`7.68%`), raising aggregate accuracy from `56.67%` to
`66.67%`, and lowering invalid selections from `23.33%` to `16.67%`. It nevertheless reduced
Traditional Chinese accuracy from `46.67%` to `43.33%`, reduced prediction-and-analysis accuracy
from `100%` to `83.33%`, and raised predictive-calibration family regret from `0.0000` to `0.3333`.
Every sparse alternative either lost the aggregate gain or retained granular regressions. No mask
was selected, so the confirmation shard was never generated or scored. The promotion entry point
returned a gated skip and Cycle 6 remains unopened. Full comparisons are in
[`block-projection.json`](../reports/evolve/block-projection.json).

## v0.5 robust family-expert result

The v0.5 experiment compared three equal-backward-compute objectives across all 12 DGP families:
the existing Boltzmann projection, the exact local gradient of simulator expected regret, and the
same direct-regret target with CVaR-style tail weighting. Each arm saw 6,912 weighted record
exposures and 3,456 backward calls. Every family had six fixed-size updates and was selected using
the same three fresh, mutually disjoint folds. Training, selection, and all file hashes completed;
confirmation, promotion, and final surfaces remained sealed.

The direct objectives failed the registered selection gate. Only one of 72 direct-mean checkpoints
and none of 72 direct-CVaR checkpoints survived the per-fold language, accuracy, invalidity,
domain, and family noninferiority checks. Neither direct route selected a non-parent expert. Both
were `9.16%` worse than the matched Boltzmann route in pooled regret; their paired mean regret
difference was `-0.0366`, with 95% bootstrap intervals wholly below zero. CVaR weighting therefore
amplified rather than repaired the failing direction.

The Boltzmann control selected only the `clustered_repeated` update-5 and
`time_series_leakage` update-6 experts. That development route lowered pooled selection-fold regret
by `8.39%` relative to the unchanged parent, improved all three folds by `7.66%` to `9.25%`, and had
a paired mean absolute improvement of `0.0366` with 95% interval `[0.0292, 0.0445]`. It is useful
evidence for a later anchor, but it does not validate the new direct-regret or CVaR hypothesis: a
mixed route was identical to the control and therefore improved `0%` over it. No v0.5 candidate
qualified, so no downstream shard was opened and no released route changed. The complete result is
in [`robust-family-experts-selection.json`](../reports/evolve/robust-family-experts-selection.json).

## v0.6 targeted anchor-repair result

The v0.6 test asked whether repair should be restricted to failures of the existing v0.5 route.
The old selection folds were used only to rank four unresolved families. Training and selection
then used new blueprints and seeds. Each family contributed 24 semantic training groups: 18
high-regret repair groups and six low-regret anchors. Every group had two English, one Traditional
Chinese, and one Simplified Chinese view, with loss weights preserving the 70/15/15 gradient
ratio.

The matched control used simulator Boltzmann targets on all groups. The candidate preserved the
parent distribution on anchors and moved repair cases toward the simulator oracle. In the realized
sample, every repair trigger saturated at `lambda = 1` and every anchor used `lambda = 0`.
Consequently, this experiment tested binary failure repair plus 25% anchor replay, not a graded
trust-region method. Both arms still used exactly 2,304 weighted record exposures and 1,152
backward calls. All 56 registered checkpoint hashes were independently re-read without a mismatch.

Only the experimental-or-causal family produced an eligible non-parent checkpoint. Both arms chose
update 6; every checkpoint in binary/count GLM, group comparison, and linear/robust regression
failed at least one per-fold granular gate. Against the v0.5 anchor route, the control lowered
pooled regret by `3.76%` and the candidate by `3.83%`. Both paired bootstrap intervals were above
zero, but both missed the frozen 5% mean-improvement requirement.

The candidate improved pooled regret over the matched control by only `0.075%`. Its paired mean
absolute improvement was `0.000282`, with 95% interval `[-0.002559, 0.003096]`; one fold also failed
the invalidity and family-regret gates. The selected route is therefore null. Confirmation,
promotion, and final shards were not generated or scored, and the released adapter did not change.
This result rejects the registered claim that the realized triggered-repair recipe improves on
ordinary Boltzmann replay at equal training compute. It does not rule out a genuinely graded
trigger, a different anchor fraction, or a representation-level continual-learning method; those
would require new contracts and data.

The machine-readable contract, data, training, and selection reports are
[`targeted-repair-contract.json`](../reports/evolve/targeted-repair-contract.json),
[`targeted-repair-data.json`](../reports/evolve/targeted-repair-data.json),
[`targeted-repair-training.json`](../reports/evolve/targeted-repair-training.json), and
[`targeted-repair-selection.json`](../reports/evolve/targeted-repair-selection.json).

## Champion Replacement Study — H4

This finite study asked whether a frozen policy composed of a high-confidence sufficiency guard,
a reduced family router, and the existing frozen experts should replace the v0.3 champion. Its
terminal rule was binary: failure of any historical-external or sealed-final gate retains v0.3 and
ends the study without tuning the failed candidate. A new sealed final could be designed only after
the historical falsification gate passed.

The original routed-expert candidate was first independently replicated on 900 new blueprints
(2,700 language views). It lowered trilingual normalized regret by `12.10%`; the paired absolute
improvement was `0.05094`, with 95% interval `[0.03692, 0.06530]`. Router family accuracy was
`88.96%`, expert coverage `50.00%`, and wrong-expert routing `1.04%`. The candidate nevertheless
failed its preregistered family-safety gate: experimental/causal regret increased by `0.15035`
against a ceiling of `0.15`. It was rejected without relaxing that ceiling.

A post-rejection leave-one-expert-out diagnosis localized the failure to the
`experimental_causal` expert. Falling only that family back to the parent restored every granular
gate while retaining an `11.92%` aggregate gain, but this was diagnostic evidence on a retired
surface. The exclusion was therefore frozen prospectively and scored on another disjoint
900-blueprint confirmation. The reduced route passed every gate, improving trilingual regret by
`10.62%`; its paired absolute improvement was `0.04467`, with 95% interval
`[0.03221, 0.05775]`.

The first binary sufficiency guard intercepted whenever the parent preferred `insufficient` over
`sufficient`. On a third paired complete/incomplete surface it found every incomplete prompt, but
complete specificity was only `94.19%`; English specificity was `82.56%`, and the
missing-selection and time-series families were each `66.67%`. Its false positives erased the
router gain and the paired interval crossed zero, so that guard was rejected. A single permitted
post-rejection margin diagnosis found that requiring `P(insufficient) >= 0.90` separated the
retired complete and incomplete views. The threshold and prompt were then frozen before a fourth
disjoint 900-blueprint paired confirmation. That thresholded guard achieved `100%` complete
specificity and `100%` incomplete sensitivity in every language and family cell while the full
policy retained a `10.72%` trilingual regret improvement. The paired absolute improvement was
`0.04431`, with 95% interval `[0.03180, 0.05699]`.

H4 therefore survived synthetic confirmation but failed the preregistered historical-external
falsification gate. Across the previously opened 90-task P-Bench suite, routed adapters improved
raw accuracy from `0%` to `61.11%` and strict accuracy from `0%` to `36.67%`. Across the 200-task
StatQA suite, exact accuracy remained `1.0%`, method-set accuracy remained `5.5%`, and column-set
accuracy remained `22.0%`. The required StatQA exact improvement was five points; the observed
improvement was zero. Every other historical gate passed, but the conjunction failed.

Historical evaluator v1 initially stopped before scoring one 351-column P-Bench task because its
complete 86,215-byte inspection JSON exceeded the 65,536-byte sandbox transport ceiling. Evaluator
v2 raised only this bounded transport ceiling to 131,072 bytes and recomputed all 290 tasks under a
new fingerprint. It did not alter a model, prompt, adapter, data source, or decision gate.

The study is closed as `external-rejected`. No untouched external final was selected, generated,
opened, or scored; the H4 policy was not modified after the external result; and no H5 was started.
The official champion remains `v0.3.0-parent`. Machine-readable evidence is in
[`family-router-replication.json`](../reports/evolve/family-router-replication.json),
[`family-router-replication-failure.json`](../reports/evolve/family-router-replication-failure.json),
[`family-router-reduced.json`](../reports/evolve/family-router-reduced.json),
[`sufficiency-guard.json`](../reports/evolve/sufficiency-guard.json),
[`sufficiency-guard-margin.json`](../reports/evolve/sufficiency-guard-margin.json),
[`sufficiency-guard-thresholded.json`](../reports/evolve/sufficiency-guard-thresholded.json), and
[`router-historical-external.json`](../reports/evolve/router-historical-external.json).

### Provenance open-state semantics

Immutable surface manifests describe the state **when the surface was created**. New manifests retain
the legacy `promotion_surface_opened` and `final_surface_opened` fields for compatibility, but also
include `open_state_semantics` stating that these are creation-time snapshots; once a surface is
scored, the dedicated runtime report is authoritative for whether it was actually opened. Historical
immutable manifests are never rewritten just to make those flags look current.

Training status reports are different: they are mutable lifecycle snapshots. When v0.5 or v0.6
selection is opened, both the ignored local `training-status.json` and the tracked public training
snapshot are refreshed to `selection_opened=true`. Selection reports remain authoritative for the
selection outcome, and no later surface is implied to be open unless its own runtime evidence says
so.

## Development cycles

The completed cycles remain development evidence and do not change the v0.3 release.

| Cycle | Checkpoint selection | Promotion regret | Accuracy | Invalid selection | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | Parent retained | not opened | not opened | not opened | Rejected; best checkpoint was identical to parent |
| 2 | 160-step checkpoint | 0.4446 to 0.3581 | 41.67% to 51.67% | 35.00% to 25.00% | Rejected; paired CI lower bound failed |
| 3 | Parent retained | not opened | not opened | not opened | Rejected; no checkpoint passed validation selection |
| 4 | Both A/B arms retained parent | not opened | not opened | not opened | Rejected; adaptive curriculum did not beat random control |
| 5 | Both A/B arms retained parent | not opened | not opened | not opened | Rejected; balanced decision-only adaptive training did not beat random control |

Cycle 2 reduced validation regret from 0.3780 to 0.2856 before promotion. On its independent
60-DGP promotion shard, relative regret fell 19.46% and retention stayed at 100%. The paired mean
improvement was 0.0865 with a 95% bootstrap interval of `[-0.0536, 0.2315]`. The lower bound missed
the frozen -0.01 noninferiority floor, so the archive kept v0.3 as champion. Later cycles use 144
new promotion DGPs and the additional language and group gates listed above; cycle 2 is not retested.
Cycle 3 also showed that the original unsmoothed learning-progress multiplier over-focused the next
curriculum. Cycle 4 tested the bounded validation-only signal in a matched training-compute A/B.
Both arms selected the unchanged parent on validation regret (`0.3780`); the adaptive arm therefore
had 0% relative improvement over random control and the sealed 144-case promotion shard remained
unopened. Inspection found that eight of 32 adaptive groups shared the same oracle method. The
current constrained selector and decision-only loss were direct responses to that negative result.
Cycle 5 tested them with identical 80-microstep early-stopped training runs. The random arm reduced
reusable-dev regret from `0.3780` to `0.3712`, but raised invalid selections from 33.33% to 36.67%,
so the hard gate retained the parent. The adaptive arm reached `0.4501` regret and 43.33% invalid
selections. It did not beat the random control, and the sealed promotion shard again remained
unopened. This result rejects the current failure-frontier curriculum recipe; additional compute
alone is not evidence that it will become beneficial.
The machine-readable aggregate is in [`reports/evolve/development.json`](../reports/evolve/development.json).

## Research context

DGP-Evolve combines ideas that appear separately in self-generated task training, archive-based
agent improvement, and verified statistical simulation. Relevant comparisons include
[Absolute Zero](https://arxiv.org/abs/2505.03335),
[SEAL](https://arxiv.org/abs/2506.10943),
[Darwin Godel Machine](https://arxiv.org/abs/2505.22954), and
[Ornith 1.5](https://ornith.ai/ornith_1_5.html). The optimizer comparison also includes MGDA,
PCGrad, CAGrad, and Nash-MTL as linked above. Nearby ingredients also have direct prior art:
[SIFT](https://openreview.net/forum?id=VPa8OUPGzg) studies active fine-tuning,
[GORP](https://arxiv.org/abs/2507.02503) projects continual gradients into low-rank subspaces,
[LiRA](https://openreview.net/forum?id=rZBWRkcqXZ) uses cross-lingual anchoring, and
[amortized Bayesian decision making](https://arxiv.org/abs/2312.02674) learns decisions for
simulation-based models. DGP-Evolve differs in implementation by using operating-characteristic
regret as the verifier, a constrained statistics compiler at inference, and cycle-specific paired
promotion shards. The repository does not claim these combinations are first in the literature.
The completed matched ablations have not established an advantage for adaptive curricula,
direct-regret targets, CVaR weighting, or triggered anchor repair over their registered controls.
Additional cycles require a changed hypothesis and remain tests rather than evidence of a generally
self-improving model.

## Limitations

The simulator is a declared semiparametric approximation, so a lower DGP regret can reflect closer
agreement with its finite method catalog rather than better statistical judgment in open-ended
work. Even 144 promotion cases give a noisy estimate for individual families. Repeated development
decisions can still overfit the generator distribution even though shards do not repeat. External
benchmarks, real-data studies, and review by a statistician remain necessary before a public v0.4
capability claim.
