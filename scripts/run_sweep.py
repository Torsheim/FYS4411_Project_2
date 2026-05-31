#!/usr/bin/env python3
"""Run a parameter sweep from a YAML config file."""

from __future__ import annotations

import argparse
import csv
import copy
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))

from nqs_vmc.io.config_loader import load_config, save_config, set_by_dotted_path
from run_experiment import run


def safe_value_string(value: Any) -> str:
    """Make a value safe to use in generated filenames."""
    return str(value).replace(".", "p").replace("-", "m").replace("/", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    sweep = config.get("sweep", {})

    parameter = sweep.get("parameter")
    values = sweep.get("values", [])
    output_path = Path(sweep.get("output_path", "results/selected/sweep.csv"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not parameter or not values:
        raise ValueError("Sweep config must contain sweep.parameter and sweep.values.")

    sweep_name = sweep.get("name", output_path.stem)
    parameter_column = parameter.replace(".", "_")

    rows: list[dict[str, Any]] = []
    temp_dir = Path("results/logs/sweep_configs")
    temp_dir.mkdir(parents=True, exist_ok=True)

    for index, value in enumerate(values):
        run_config = copy.deepcopy(config)
        run_config.pop("sweep", None)
        run_config = set_by_dotted_path(run_config, parameter, value)

        stem = f"{sweep_name}_{parameter_column}_{index}_{safe_value_string(value)}"

        run_config.setdefault("output", {})
        run_config["output"]["history_path"] = f"results/selected/{stem}_history.csv"
        run_config["output"]["samples_path"] = f"results/processed/{stem}_local_energies.csv"
        run_config["output"]["checkpoint_path"] = f"results/checkpoints/{stem}_model.npz"

        temp_config = temp_dir / f"{stem}.yaml"
        save_config(run_config, temp_config)

        print(f"Running sweep value {parameter}={value}")
        summary = run(temp_config, verbose=False)

        summary[parameter_column] = value
        summary["sweep_parameter"] = parameter
        summary["sweep_value"] = value
        rows.append(summary)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote sweep results to {output_path}")


if __name__ == "__main__":
    main()
