from __future__ import annotations

from pathlib import Path

import yaml


def load_repo_config(repo_root: Path) -> dict:
    cfg_path = repo_root / "config" / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_runtime_root(repo_root: Path, cfg: dict | None = None) -> Path:
    use_cfg = cfg if cfg is not None else load_repo_config(repo_root)
    runtime_rel = (use_cfg.get("runtime", {}) or {}).get("root")
    if runtime_rel:
        return (repo_root / runtime_rel).resolve()
    return repo_root
