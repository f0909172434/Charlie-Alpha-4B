# DGP-Evolve

DGP-Evolve is the development self-iteration loop for Charlie alpha v0.4. It trains a small
statistics adapter on failures that a deterministic simulator can verify. Each candidate remains
separate from the current adapter until it passes a paired promotion test.

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
4. Rank proposals by validity, novelty, frontier proximity, and measured family learning progress.
5. Train the existing LoRA matrices for 160 microsteps with 20% replay from v0.3.
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

Each training DGP produces two English boundary views, one Traditional Chinese view, and one
Simplified Chinese view. Their loss weights preserve the 70/15/15 gradient ratio. The selector
keeps 32 proposals and adds eight replay groups. This makes 160 records, so 160 microsteps cover one
complete group epoch. The trainer updates 2,129,920 adapter parameters on the current Qwen3.5-4B
MLX model; base-model parameters remain frozen.

The iterator places one replay group after every four new groups. Continued training assigns 80%
of the loss to the simulator's method distribution, 15% to the compiled plan, and 5% to the report.
Weight decay is zero because the parent LoRA already contains useful nonzero matrices; decaying
those matrices toward zero would alter the parent even when the task gradient is weak.

The trainer saves the 80-step and final weights in addition to the lowest-loss checkpoint. It
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
```

The equivalent CLI supports at most two cycles per invocation:

```bash
uv run charlie-alpha stats iterate --cycles 1 \
  --config configs/pipeline.evolve.yaml
```

Every completed stage uses content fingerprints. Re-running a command reuses valid work. `--force`
can deliberately regenerate mutable discovery or training work, but it cannot replace a promotion
shard once that shard has been prepared. Changed promotion settings take effect in a new cycle.

## Development cycles

The first three cycles remain development evidence and do not change the v0.3 release.

| Cycle | Checkpoint selection | Promotion regret | Accuracy | Invalid selection | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | Parent retained | not opened | not opened | not opened | Rejected; best checkpoint was identical to parent |
| 2 | 160-step checkpoint | 0.4446 to 0.3581 | 41.67% to 51.67% | 35.00% to 25.00% | Rejected; paired CI lower bound failed |
| 3 | Parent retained | not opened | not opened | not opened | Rejected; no checkpoint passed validation selection |

Cycle 2 reduced validation regret from 0.3780 to 0.2856 before promotion. On its independent
60-DGP promotion shard, relative regret fell 19.46% and retention stayed at 100%. The paired mean
improvement was 0.0865 with a 95% bootstrap interval of `[-0.0536, 0.2315]`. The lower bound missed
the frozen -0.01 noninferiority floor, so the archive kept v0.3 as champion. Later cycles use 144
new promotion DGPs and the additional language and group gates listed above; cycle 2 is not retested.
Cycle 3 also showed that the original unsmoothed learning-progress multiplier over-focused the next
curriculum. The current implementation replaces it with the validation-only bounded signal above;
that revision has unit coverage but has not yet produced a new trained candidate.
The machine-readable aggregate is in [`reports/evolve/development.json`](../reports/evolve/development.json).

## Research context

DGP-Evolve combines ideas that appear separately in self-generated task training, archive-based
agent improvement, and verified statistical simulation. Relevant comparisons include
[Absolute Zero](https://arxiv.org/abs/2505.03335),
[SEAL](https://arxiv.org/abs/2506.10943),
[Darwin Godel Machine](https://arxiv.org/abs/2505.22954), and
[Ornith 1.5](https://ornith.ai/ornith_1_5.html). DGP-Evolve differs in implementation by using
operating-characteristic regret as the verifier, a constrained statistics compiler at inference,
and cycle-specific paired promotion shards. An equal-compute ablation is still required to tell
whether those choices improve on ordinary continued SFT or random DGP replay.

## Limitations

The simulator is a declared semiparametric approximation, so a lower DGP regret can reflect closer
agreement with its finite method catalog rather than better statistical judgment in open-ended
work. Even 144 promotion cases give a noisy estimate for individual families. Repeated development
decisions can still overfit the generator distribution even though shards do not repeat. External
benchmarks, real-data studies, and review by a statistician remain necessary before a public v0.4
capability claim.
