# Release procedure

Charlie alpha v0.2.0 is an Experimental release. Its direct-adapter final and disjoint routed
confirmation each improved by one correct answer, but neither reached the predeclared +2
percentage-point normal-release threshold.

1. Run or resume `make forge`; do not change the frozen recipe or task locks after generation.
2. Run `make forge-router-verify`. Bypass must exactly match an independently loaded base, restore
   must exactly match the adapter, and all eight intended LoRA modules must be present.
3. Run `make forge-export` and `make forge-clean-load`.
4. Run `make forge-release-check`. License, source revision, split isolation, sandbox, privacy,
   adapter, fused MLX, clean-environment, and dynamic-router gates are hard blockers.
5. Commit and push the exact release tree on `main`, then run `make forge-publish-github`. The
   GitHub release includes the adapter archive, frozen configurations, both compact evaluations,
   public data manifest, release gate, and SHA-256 file.
6. Hugging Face authentication is interactive and user-owned. Run `hf auth login` personally,
   then publish only after reviewing the model card. The project does not read, copy, or print a
   token.

The canonical artifact is the MLX adapter plus dynamic routing code. A fused MLX export is tested
for recoverability but behaves as an always-on specialist. One fused GGUF cannot reproduce dynamic
routing; do not publish GGUF unless a separate behavioral-parity gate has passed.
