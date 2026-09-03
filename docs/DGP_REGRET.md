# DGP-Regret technical report

## Scope

DGP-Regret is the v0.3 training and evaluation profile for Charlie alpha. It treats statistical
analysis as a decision problem: given a declared estimand, sampling unit, study design, outcome
type, dependence structure, and missingness assumptions, choose a procedure whose repeated-sample
behavior is acceptable. The method does not assume that one procedure is always the only correct
answer.

This report describes an experimental implementation. It does not claim that DGP-Regret is the
first use of synthetic statistical tasks, simulation, regret, soft labels, active sampling, or tool
agents. A claim that this particular combination improves the model is allowed only if the frozen
ablation gate passes.

## Related work and distinction

- [StatQA](https://arxiv.org/abs/2406.07815) separates method/column applicability from numerical
  execution. DGP-Regret uses StatQA only as an evaluation set and adds a training objective based on
  relative operating characteristics across candidate procedures.
- [BoxLM](https://arxiv.org/abs/2402.17879) automates Box's loop: a language model proposes
  probabilistic programs, fits them, and uses model criticism to propose revisions. DGP-Regret does
  not search an open-ended model space; it learns a bounded policy over audited procedures and can
  defer when the design is underspecified.
- [Fisher-R1](https://arxiv.org/html/2608.07437) uses executable synthetic hypothesis-testing tasks,
  supervised trajectories, and outcome-grounded reinforcement learning. Its task generator already
  includes null, borderline, and medium effect regimes, so synthetic DGPs and boundary calibration
  are prior art. Fisher-R1 deliberately omits an explicit method-correctness reward because several
  procedures can be defensible. DGP-Regret addresses that same multiplicity with a probability
  distribution over a finite candidate menu, derived from relative procedure regret.
- [Ornith 1.5](https://ornith.ai/ornith_1_5.html) jointly improves task generation, executable
  scaffolds, and solution rollouts. Its task reward gates on validity and combines model-frontier
  difficulty with novelty. That design is relevant to future DGP selection, but its published 9B
  results cover reasoning, coding, and agentic benchmarks rather than statistical procedure
  selection. The frozen v0.3 comparison therefore does not use Ornith as a statistical judge,
  teacher, or replacement base model. A later ablation may add current-policy success and DGP-space
  novelty after simulator validity checks; the simulator must remain the source of statistical
  labels.
- Decision-focused learning and learning-to-rank already optimize downstream regret rather than an
  intermediate prediction error. DGP-Regret applies that principle to statistical procedure
  selection; this application still requires empirical evidence and should not be described as a
  new general learning theory.

## DGP surface

The committed catalog defines 12 families and 28 core procedures. Seven additional fixed
procedures are available to the audited runtime for external evaluation, but do not enter the DGP
training comparison.

| Domain | Families |
|---|---|
| Statistical inference and research design (60%) | group/paired comparison, categorical data, linear and robust regression, binary/count GLMs, clustered/repeated measurements, survival, missingness/selection, experiments/causal inference |
| Probability and Bayesian analysis (20%) | probability distributions, Bayesian estimation and model checking |
| Prediction and data analysis (20%) | predictive calibration, time series and leakage |

For every split, parameter points begin with deterministic Latin-hypercube coverage. The primary
surface has 240 training groups, 30 validation groups, 60 development DGPs, and 120 sealed-final
DGPs. Training is divided evenly among central LHS points and two searches. Round one evaluates
family-specific boundary candidates and selects validity failures first. Round two combines two
boundary stresses and selects the smallest candidate-method ranking gap, then the largest regret
of the family's central method. This implements the predeclared ranking-uncertainty branch; it does
not claim to have measured base-model regret during surface construction. Each selected blueprint
stores the candidate count, boundary keys, top gap, ambiguity flag, invalid-method count, and
central-method regret before language rendering.

The random-DGP ablation uses 240 additional unspecialized LHS training points with no boundary
search. It shares the primary validation and development sets, sequence budget, optimizer, and
microstep count. These ablation-only points are not part of the 450-group primary split.

### Current simulator boundary

The v0.3 engine is a fast semiparametric operating-characteristic emulator, not a brute-force refit
of all 28 procedures to raw synthetic tables on every replication. Each family has explicit
structural response functions for central validity and assumption violations. The engine then uses
common random numbers to sample rejection, coverage, detection, estimation error, and calibration
indicators at 128 replications, escalating to 256 or 512 when the top-method ranking is uncertain.

This design makes the complete surface reproducible and inexpensive enough for local ablation, but
its structural response functions encode modeling judgment. Therefore:

1. final DGP regret measures agreement with this declared simulator, not universal statistical
   optimality;
2. external P-Bench and StatQA results are required before any capability claim;
3. raw-data Monte Carlo validation of every family remains a priority for a later revision.

The repository exposes the response functions, parameters, central sentinels, random seeds,
adaptive stopping history, and cache fingerprints so this limitation is inspectable rather than
hidden.

## Regret and soft targets

For candidate procedure (j), the emulator records Type I error

\[
\widehat{\alpha}_j = R^{-1}\sum_{r=1}^{R} I(\text{reject}_{jr}\mid H_0),
\]

coverage, bias, RMSE, power, calibration error, and a declared computational cost. Validity is
lexicographically dominant. In the implementation, an invalid procedure receives a base penalty of
2 before secondary accuracy, power, and cost terms are added; a valid procedure is compared using

\[
0.42\min(|\widehat{\mathrm{bias}}_j|,1)
+0.34\frac{\min(\widehat{\mathrm{RMSE}}_j,1.5)}{1.5}
+0.24\frac{\min(\widehat{\mathrm{calibration}}_j,0.5)}{0.5}
+0.18(1-\widehat{\mathrm{power}}_j)+0.05c_j.
\]

Within each candidate menu, raw regret is min-max normalized to \(r_j\in[0,1]\). The method target
is listwise rather than one-hot:

\[
q_j=\frac{\exp(-r_j/0.15)}{\sum_k\exp(-r_k/0.15)}.
\]

Consequently, several procedures can retain meaningful target probability. A separate
`needs_clarification` action is the hard target when the estimand, sampling unit, design,
dependence, missingness, or assignment mechanism is not declared.

## Training objective

Every training group contains two complete English boundary views, one Traditional Chinese view,
and one Simplified Chinese view. Their loss weights are 1.4, 1.4, 0.6, and 0.6, yielding an exact
70%/15%/15% language gradient ratio. The second English view challenges a cheap standard procedure
instead of converting a quarter of the training mass into clarification examples.

The assistant loss is

\[
\mathcal{L}=0.45\,\mathcal{L}_{\text{method-soft-CE}}
+0.35\,\mathcal{L}_{\text{plan+tool}}
+0.20\,\mathcal{L}_{\text{report}}.
\]

User messages and tool observations are context-only and receive zero loss. The model is trained to
emit a validated plan with `status`, `estimand`, `sampling_unit`, `study_design`, `outcome_type`,
`dependence`, `missingness`, `method_id`, `uncertainty`, `diagnostics`, `tool`, and explicit variable
roles.

The 9B model edits at most 120 high-regret explanations at temperature zero with thinking disabled.
It never chooses a method or changes a computed field. A rewrite is accepted only if the method ID
and every number are unchanged, it is a direct explanation rather than editing commentary, and it
fits the sequence budget. One stricter retry is allowed; otherwise the deterministic template is
used.

## Equal-compute ablation

The three pilot variants are:

1. hard-label SFT using the single minimum-regret method;
2. regret soft labels on the separate random-LHS DGP surface;
3. regret soft labels with active failure-region and assumption-boundary curriculum.

All variants use the same 160 microsteps, last four layers, rank 32, target modules, batch size 1,
four-step gradient accumulation, 640-token limit, seed 42, optimizer, and LoRA+ rates. A Metal OOM
changes all variants together to 512 tokens and then rank 16. Selection is development normalized
regret, then method accuracy, then validation loss. Formal training runs the winning variant for at
most two epochs and three hours. Adapter delta scales 0.5, 0.75, and 1.0 are compared only on dev.

## Local analysis agent

At inference, one 4-bit base and one adapter stay loaded. A data attachment or statistical prompt
uses the stats route; other prompts bypass the adapter and use the identical base logits. The
planner may invoke only checked-in Python/R procedures, at most four times. It cannot emit an
arbitrary program for execution.

The locked Pixi runtime supports Python and R. Each call is run through macOS `sandbox-exec` with no
network, no reads from other user directories, no writes outside a new temporary directory, no
unapproved executable, a 20-second timeout, 2 GiB memory cap, 32 MiB write cap, and 64 KiB output
cap. User data remains local.

## Frozen evaluation and claims

The committed lock contains 120 final DGP IDs, 90 P-Bench tasks (45 Easy, 45 Hard, all 17
categories), 200 StatQA indices, 30 trilingual semantic tasks, 30 clarification cases, and a
math/code/STEM/general retention set. It contains no StatQA questions or answers. An 8-gram audit
must find no overlap with training prompts.

P-Bench Raw uses the reference reject/fail-to-reject decision. Strict additionally requires the
reported p-value to be within 0.5 on the two-sided normal z scale, matching the benchmark's stated
criterion. StatQA is exact only when both the complete method set and complete column set match.

Normal v0.3 publication requires all predeclared capability gates. If the safety, license, source,
split, load, or privacy gates pass but a capability gate fails, the release is labeled
Experimental and reports the negative result. If the DGP-Regret pilot does not beat hard-label by
5% dev regret, the release must not attribute any improvement to DGP-Regret.

## Results

### Pilot and formal training

The equal-compute pilot selected the active-boundary DGP-Regret curriculum:

| Variant | Dev normalized regret | Accuracy | Invalid selection | Validation loss |
|---|---:|---:|---:|---:|
| Hard-label | 0.6848 | 18.33% | 61.67% | 1.6983 |
| Regret, random LHS | 0.6644 | 18.33% | 60.00% | 1.7328 |
| DGP-Regret | **0.6263** | **30.00%** | 61.67% | 1.7125 |

DGP-Regret reduced dev regret by 8.54% relative to hard-label, above the frozen 5% ablation gate.
Formal training stopped at 1,120 of the planned 1,920 microsteps after two validation checks without
improvement. The best checkpoint was microstep 800 with validation loss 1.0529. Scale calibration
selected 1.0; its dev regret was 0.4840, method accuracy was 36.67%, and invalid selection was
43.33%. The 24-item retention score was 100% for all three scale candidates.

### Sealed evaluation

| Model | Final DGP regret | Method accuracy | Invalid selection |
|---|---:|---:|---:|
| Base | 0.6727 | 20.83% | 63.33% |
| Hard-label | 0.7016 | 17.50% | 65.83% |
| DGP-Regret | **0.4437** | **45.00%** | **38.33%** |

The final relative regret improvement over the base was 34.04%. Across the 120 paired DGPs, the
mean absolute improvement was 0.2290; a 10,000-draw paired bootstrap produced a 95% interval of
`[0.1199, 0.3361]`. The relative reduction in invalid selections was 39.47%.

Method accuracy improved by 15.28 points for inference and design, 33.33 points for prediction and
analysis, and 41.67 points for probability and Bayesian tasks. The 30 shared trilingual scenarios
improved by 26.67 points in English and 10 points in each Chinese script. No retention language or
domain subgroup changed.

The external results did not support an end-to-end capability claim:

| Metric | Base | Hard-label | DGP-Regret |
|---|---:|---:|---:|
| P-Bench Raw | 0% | 0% | 0% |
| P-Bench Strict | 0% | 0% | 0% |
| StatQA exact | 1.00% | 1.00% | 1.00% |
| StatQA method set | 5.00% | 6.00% | 5.50% |
| StatQA column set | 19.50% | 23.00% | 22.00% |
| Clarification | 43.33% | 40.00% | 0% |

Every base and hard-label P-Bench case returned `needs_clarification`. DGP-Regret returned that
status on 88 of 90 cases; the remaining two also produced no scoreable p value. The planner's
schema validation prevented unsafe execution, but the agent failed the benchmark as an analysis
system. On the separate clarification suite, DGP-Regret never selected `needs_clarification` at the
method decision position.

The final-regret, invalid-selection, trilingual, retention, ability-category, and ablation gates
passed. P-Bench and StatQA failed. The release is **Experimental v0.3.0** and may claim measured
DGP-Regret procedure-selection gains only. It may not claim broader statistical analysis gains.
The aggregate metrics are in `reports/stats/evaluation.json`; per-item prompts and responses remain
outside the release tree.

## Intended use and limitations

Charlie alpha is a local research-analysis assistant, not an autonomous statistician. It can choose
the wrong estimand, map variables incorrectly, miss a design dependency, or produce a fluent but
invalid interpretation. The DGP catalog is finite; the response-surface emulator is partly
hand-specified; external evaluations are sampled; and a 4B model has limited planning capacity.
Medical, financial, and policy analyses require review by a qualified statistician and relevant
domain experts.
