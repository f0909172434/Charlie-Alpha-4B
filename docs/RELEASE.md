# Release procedure

1. Run `make overnight`, then rerun any incomplete resumable stage.
2. Run `make export` and `make clean-load`.
3. Run `make release-check` and inspect every hard gate.
4. If the aggregate improvement is at least two points and no language/domain subgroup loses more
   than three points, use the normal `v0.1.0` label. Otherwise publish as Experimental with the full
   result table. A load, license, split-leak, sandbox, or clean-environment failure blocks weights.
5. Publish the small MLX adapter, configuration, compact reports, and SHA-256 file as GitHub release
   assets. Publish the fused 4-bit MLX model on Hugging Face only after the user runs `hf auth login`.
6. Run the optional `make gguf` only with enough remaining time. Publish GGUF only after its
   numerical mapping and behavioral parity gates pass.

Authentication is always interactive and user-owned. The project never reads, prints, copies, or
commits tokens.

