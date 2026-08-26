# Reports

## Post-v0.3 development

- `evolve/development.json`: gated DGP-Evolve cycles; no candidate was promoted.
- `evolve/robust-family-experts-*.json`: the v0.5 matched Boltzmann, direct-regret, and CVaR
  ablation. The new direct objectives failed selection.
- `evolve/targeted-repair-*.json`: the v0.6 matched targeted-repair ablation. It improved only
  0.075% over its control and did not open a downstream shard.

The selected model therefore remains the Experimental v0.3 adapter. Public development reports
contain aggregate metrics and hashes; generated DGPs, predictions, local paths, and weights remain
ignored.

## Experimental v0.2.0

- `v2/development.json`: sealed-development calibration result, 22/34 versus 21/34.
- `v2/evaluation.json`: first frozen 62-task final, direct adapter 44/62 versus base 43/62.
- `v2/dynamic-router.json`: logit-level proof that runtime bypass exactly equals the base and restore
  exactly equals the calibrated adapter.
- `v3/evaluation.json`: disjoint frozen router confirmation, routed model 43/62 versus base 42/62,
  with no language or domain losing a correct answer.

Both unseen 62-task suites observed one additional correct answer, but +1.62 and +1.61 percentage
points miss the predeclared +2-point normal-release threshold. The release is Experimental and does
not claim broad or statistically established superiority.

Generated answers, logs, weights, and machine-local paths stay under ignored `generated/` or
`artifacts/` directories. Compact aggregate reports are tracked. Missing reports are never treated
as a pass; license, source, data isolation, privacy, sandbox, adapter, fused-model, clean-environment,
and dynamic-router checks are hard gates for weight publication. GGUF remains unpublished because
one fused GGUF cannot reproduce the canonical dynamic route and behavioral parity was not run.

The original `evaluation.json`, `export.json`, `clean-load.json`, and `release-gate.json` document
the earlier Experimental v0.1.0 result and are retained for provenance.
