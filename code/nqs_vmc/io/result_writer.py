"""Write experiment outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_history_csv(history: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    rows = list(history)
    path = ensure_parent(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_local_energies_csv(energies: np.ndarray, path: str | Path) -> None:
    path = ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "local_energy"])
        for i, energy in enumerate(np.asarray(energies, dtype=float).reshape(-1)):
            writer.writerow([i, float(energy)])


def write_json(data: Mapping[str, Any], path: str | Path) -> None:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
