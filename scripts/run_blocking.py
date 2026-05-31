#!/usr/bin/env python3
"""Run blocking analysis on saved local-energy samples."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from nqs_vmc.io.config_loader import load_config
from nqs_vmc.statistics.blocking import blocking_analysis, blocking_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = Path(config.get("blocking", {}).get("input_path", config.get("output", {}).get("samples_path", "results/processed/local_energies.csv")))
    output_path = Path(config.get("blocking", {}).get("output_path", "results/selected/blocking.csv"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if "local_energy" not in df.columns:
        raise ValueError(f"{input_path} must contain a 'local_energy' column.")

    rows = blocking_analysis(df["local_energy"].to_numpy())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote blocking table to {output_path}")
    print(f"Conservative blocking error: {blocking_error(df['local_energy'].to_numpy()):.6g}")


if __name__ == "__main__":
    main()
