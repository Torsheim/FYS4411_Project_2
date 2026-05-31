"""YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config file {path} must contain a YAML mapping.")
    return data


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(config), f, sort_keys=False)


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Return recursive ``base`` updated by ``updates``."""
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def set_by_dotted_path(config: dict[str, Any], dotted_path: str, value: Any) -> dict[str, Any]:
    """Return a copy where ``a.b.c`` has been set to value."""
    result = dict(config)
    current = result
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        current[part] = dict(current.get(part, {}))
        current = current[part]
    current[parts[-1]] = value
    return result
