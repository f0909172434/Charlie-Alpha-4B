# Release procedure

## v0.3.0

1. Complete `make stats-eval`. Final evaluation is allowed only after the adapter, delta scale,
   data fingerprint, and sealed evaluation lock have been frozen.
2. Run `make stats-export`. The dynamic MLX router must reproduce an independently loaded base when
   bypassed and the adapter when restored. The adapter and fused MLX copy must load in a clean
   environment.
3. Run `make stats-release-check`. Source licenses, DGP schema and split isolation, 8-gram
   decontamination, Python/R isolation, privacy, clean loading, and preservation of `v0.2.0` are
   hard gates.
4. If every hard gate passes but a capability threshold fails, publish as `Experimental v0.3.0`
   with the negative result. A failed hard gate blocks weight publication.
5. Commit and push the exact release tree on `main`. Run `make stats-publish-hf`, then
   `make stats-publish-github`. Both publishers verify the uploaded artifacts. Authentication is
   interactive and user-owned; the project does not read, copy, or print access tokens.

The canonical artifact is the MLX adapter plus the deterministic `base`/`stats` router. The GitHub
release contains the adapter archive, configuration, aggregate evaluation, release gate, sealed ID
lock, and SHA-256 list. Generated training records and evaluation answers are not release assets.

GGUF is a separate conditional artifact. It is built only from the pinned official BF16 base after
adapter-matrix equivalence succeeds, and only when the pinned llama.cpp revision has a verified
Qwen3.5 hybrid-model fix. Clean loading, readable generation, and 30-item parity within two points
are also required. Otherwise MLX is released without GGUF.

## Preserved v0.2 release

`v0.2.0`, its GitHub release, and its existing Hugging Face revision are not rewritten. The v0.2
FORGE release procedure remains available from that tag and in `docs/FORGE.md`.
