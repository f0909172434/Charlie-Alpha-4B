from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    path: Path
    root: Path
    values: dict[str, Any]
    sources: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        value = self.values.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"Missing configuration section: {name}")
        return value

    def resolve(self, configured_path: str) -> Path:
        return (self.root / configured_path).resolve()

    def path_for(self, key: str) -> Path:
        value = self.section("paths").get(key)
        if not isinstance(value, str):
            raise ValueError(f"Missing path setting: paths.{key}")
        return self.resolve(value)


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).expanduser().resolve()
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    root = config_path.parent.parent
    lock_rel = values.get("paths", {}).get("source_lock")
    if not isinstance(lock_rel, str):
        raise ValueError("paths.source_lock is required")
    lock_path = (root / lock_rel).resolve()
    sources = json.loads(lock_path.read_text(encoding="utf-8"))
    if sources.get("schema_version") != 1:
        raise ValueError("Unsupported source lock schema")
    return ProjectConfig(path=config_path, root=root, values=values, sources=sources)
