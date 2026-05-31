#!/usr/bin/env python3
"""Create compact LaTeX tables from selected CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_csv_or_none(input_path: str | Path) -> pd.DataFrame | None:
    path = Path(input_path)

    if not path.exists():
        print(f"Skipping missing input: {path}")
        return None

    if path.stat().st_size == 0:
        print(f"Skipping empty input: {path}")
        return None

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"Skipping unreadable empty CSV: {path}")
        return None

    if df.empty:
        print(f"Skipping header-only input: {path}")
        return None

    return df


def write_latex(df: pd.DataFrame, output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(df.to_latex(index=False, float_format="%.6g"), encoding="utf-8")
    print(f"Wrote {out}")


def final_row_table(input_path: str, output_path: str, label: str) -> None:
    df = read_csv_or_none(input_path)
    if df is None:
        return

    cols = ["iteration", "energy_mean", "energy_error", "acceptance_rate", "gradient_norm"]
    cols = [col for col in cols if col in df.columns]

    row = df[cols].tail(1).copy()
    row.insert(0, "run", label)

    write_latex(row, output_path)


def combined_final_energy_table() -> None:
    runs = [
        ("Part b, brute force, 1p1d", "results/selected/part_b_1p1d_2h_bruteforce.csv", 0.5),
        ("Part b, brute force, 2p2d", "results/selected/part_b_2p2d_no_interaction_bruteforce.csv", 2.0),
        ("Part c, importance, 1p1d", "results/selected/part_c_1p1d_2h_importance.csv", 0.5),
        ("Part c, importance, 2p2d", "results/selected/part_c_2p2d_no_interaction_importance.csv", 2.0),
        ("Part e, interacting", "results/selected/part_e_2p2d_interacting.csv", 3.0),
        ("Part e, interacting long", "results/selected/part_e_2p2d_interacting_long.csv", 3.0),
    ]

    rows = []

    for label, path, exact in runs:
        df = read_csv_or_none(path)
        if df is None:
            continue

        last = df.tail(1).iloc[0]
        energy = float(last["energy_mean"])

        rows.append(
            {
                "run": label,
                "energy": energy,
                "error": float(last.get("energy_error", float("nan"))),
                "exact": exact,
                "abs_error": abs(energy - exact),
                "acceptance": float(last.get("acceptance_rate", float("nan"))),
            }
        )

    if rows:
        write_latex(pd.DataFrame(rows), "results/tables/final_energy_summary_table.tex")


def sweep_table(input_path: str, output_path: str, columns: list[str]) -> None:
    df = read_csv_or_none(input_path)
    if df is None:
        return

    cols = [col for col in columns if col in df.columns]
    if not cols:
        print(f"Skipping {input_path}: none of the requested columns were found")
        return

    write_latex(df[cols], output_path)


def blocking_table(input_path: str, output_path: str) -> None:
    df = read_csv_or_none(input_path)
    if df is None:
        return

    write_latex(df, output_path)


def main() -> None:
    final_row_table(
        "results/selected/part_b_1p1d_2h_bruteforce.csv",
        "results/tables/part_b_non_interacting_table.tex",
        "Part b, 1p1d",
    )

    final_row_table(
        "results/selected/part_e_2p2d_interacting_long.csv",
        "results/tables/part_e_interacting_table.tex",
        "Part e, 2p2d interacting long",
    )

    combined_final_energy_table()

    blocking_table(
        "results/selected/part_d_blocking_importance.csv",
        "results/tables/part_d_blocking_table.tex",
    )

    blocking_table(
        "results/selected/part_e_blocking_interacting_long.csv",
        "results/tables/part_e_blocking_table.tex",
    )

    sweep_columns = [
        "optimization_learning_rate",
        "model_num_hidden",
        "sampler_time_step",
        "final_energy",
        "final_error",
        "acceptance_rate",
    ]

    sweep_table(
        "results/selected/part_b_learning_rate_sweep.csv",
        "results/tables/part_b_learning_rate_sweep_table.tex",
        sweep_columns,
    )

    sweep_table(
        "results/selected/part_b_hidden_units_sweep.csv",
        "results/tables/part_b_hidden_units_sweep_table.tex",
        sweep_columns,
    )

    sweep_table(
        "results/selected/part_e_learning_rate_sweep.csv",
        "results/tables/part_e_learning_rate_sweep_table.tex",
        sweep_columns,
    )

    sweep_table(
        "results/selected/part_e_hidden_units_sweep.csv",
        "results/tables/part_e_hidden_units_sweep_table.tex",
        sweep_columns,
    )

    sweep_table(
        "results/selected/part_e_time_step_sweep.csv",
        "results/tables/part_e_time_step_sweep_table.tex",
        sweep_columns,
    )


if __name__ == "__main__":
    main()
