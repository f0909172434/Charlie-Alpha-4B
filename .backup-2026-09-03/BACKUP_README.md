# Charlie-Alpha-4B archival backup

Created for archival deletion on 2026-09-03 Asia/Taipei.

- Files: 28343
- Bytes: 13830483110
- Git HEAD: `af97379446ea81a400aea55180b57be35b54f6b8`
- Original Git history: `original-history.bundle`
- Integrity manifest: `SHA256SUMS.json`
- GitHub: `f0909172434/Charlie-Alpha-4B`, branch `archive/backup-2026-09-03`
- Hugging Face: `f0909172434/Charlie-Alpha-4B-MLX-4bit`, prefix `archive/2026-09-03/`

The complete research workspace is stored under the Hugging Face archive prefix. The 9 original symbolic-link targets are recorded in the integrity manifest and are not materialized on Hugging Face. GitHub stores a source/report/data snapshot plus the original Git bundle on a separate archive branch. Local `.venv`, `.pixi`, caches, and `.env` are intentionally excluded; recreate dependencies from `uv.lock`, `pixi.lock`, and `pyproject.toml`.

## Hugging Face storage layout

The Hub accepted 19975 workspace files individually before reaching its 20,000-file repository limit. The remaining 8367 regular files and 9 symbolic links are preserved in `REMAINDER-8367-files-plus-9-symlinks.tar`. Extract that tar at the root of a downloaded `archive/2026-09-03/` tree. Tar SHA-256: `4e2c9e20e98b3266dbef27e8f1da1487db3ce1fd04d90d4f093550706f33637e`.
