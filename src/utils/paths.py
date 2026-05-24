"""Path and configuration helpers for experiment files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def resolve_path(path_value: str | Path | None, base_dir: str | Path | None = None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    base = Path(base_dir).expanduser() if base_dir is not None else PROJECT_ROOT
    return (base / path).resolve()


def require_path(path_value: str | Path, base_dir: str | Path | None = None) -> Path:
    path = resolve_path(path_value, base_dir=base_dir)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Missing required path: {path_value}")
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON configuration file."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return load_yaml(path)
