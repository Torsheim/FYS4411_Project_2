#!/usr/bin/env python3
"""Generate report-ready figures from selected result files.

The figures are generated at approximately one-column width with 10 pt labels,
ticks, legends, and annotations, matching the report body text when included
with width=\\columnwidth in LaTeX.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from nqs_vmc.plotting.plot_blocking import plot_blocking
from nqs_vmc.plotting.plot_convergence import plot_energy_convergence
from nqs_vmc.plotting.plot_parameter_sweeps import plot_parameter_sweep


def csv_has_data(path: str | Path) -> bool:
    path = Path(path)

    if not path.exists():
        print(f"Skipping missing input: {path}")
        return False

    if path.stat().st_size == 0:
        print(f"Skipping empty input: {path}")
        return False

    try:
        df = pd.read_csv(path, nrows=2)
    except pd.errors.EmptyDataError:
        print(f"Skipping unreadable empty CSV: {path}")
        return False

    if df.empty:
        print(f"Skipping header-only input: {path}")
        return False

    return True


def maybe_plot(
    func: Callable[..., Any],
    input_path: str,
    output_path: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    if not csv_has_data(input_path):
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        func(input_path, output_path, *args, **kwargs)
    except KeyError as exc:
        print(f"Skipping {input_path}: {exc}")
        return
    except ValueError as exc:
        print(f"Skipping {input_path}: {exc}")
        return

    print(f"Wrote {output_path}")


def main() -> None:
    maybe_plot(
        plot_energy_convergence,
        "results/selected/part_b_1p1d_2h_bruteforce.csv",
        "results/figures/part_b_energy_convergence_1p1d.pdf",
        exact_energy=0.5,
    )

    maybe_plot(
        plot_energy_convergence,
        "results/selected/part_b_2p2d_no_interaction_bruteforce.csv",
        "results/figures/part_b_energy_convergence_2p2d.pdf",
        exact_energy=2.0,
    )

    maybe_plot(
        plot_energy_convergence,
        "results/selected/part_c_1p1d_2h_importance.csv",
        "results/figures/part_c_energy_convergence_1p1d_importance.pdf",
        exact_energy=0.5,
    )

    maybe_plot(
        plot_energy_convergence,
        "results/selected/part_c_2p2d_no_interaction_importance.csv",
        "results/figures/part_c_energy_convergence_2p2d_importance.pdf",
        exact_energy=2.0,
    )

    maybe_plot(
        plot_energy_convergence,
        "results/selected/part_e_2p2d_interacting.csv",
        "results/figures/part_e_energy_convergence_interacting.pdf",
        exact_energy=3.0,
    )

    maybe_plot(
        plot_energy_convergence,
        "results/selected/part_e_2p2d_interacting_long.csv",
        "results/figures/part_e_energy_convergence_interacting_long.pdf",
        exact_energy=3.0,
    )

    maybe_plot(
        plot_energy_convergence,
        "results/selected/part_e_2p2d_interacting_final.csv",
        "results/figures/part_e_energy_convergence_interacting_final.pdf",
        exact_energy=3.0,
    )

    maybe_plot(
        plot_blocking,
        "results/selected/part_d_blocking_importance.csv",
        "results/figures/part_d_blocking_importance.pdf",
    )

    maybe_plot(
        plot_blocking,
        "results/selected/part_e_blocking_interacting_long.csv",
        "results/figures/part_e_blocking_interacting_long.pdf",
    )

    maybe_plot(
        plot_blocking,
        "results/selected/part_e_blocking_interacting_final.csv",
        "results/figures/part_e_blocking_interacting_final.pdf",
    )

    maybe_plot(
        plot_parameter_sweep,
        "results/selected/part_b_learning_rate_sweep.csv",
        "results/figures/part_b_learning_rate_sweep.pdf",
        x_column="optimization_learning_rate",
        exact_energy=0.5,
        title="Part b: learning-rate sweep",
    )

    maybe_plot(
        plot_parameter_sweep,
        "results/selected/part_b_hidden_units_sweep.csv",
        "results/figures/part_b_hidden_units_sweep.pdf",
        x_column="model_num_hidden",
        exact_energy=0.5,
        title="Part b: hidden-unit sweep",
    )

    maybe_plot(
        plot_parameter_sweep,
        "results/selected/part_e_learning_rate_sweep.csv",
        "results/figures/part_e_learning_rate_sweep.pdf",
        x_column="optimization_learning_rate",
        exact_energy=3.0,
        title="Part e: learning-rate sweep",
    )

    maybe_plot(
        plot_parameter_sweep,
        "results/selected/part_e_hidden_units_sweep.csv",
        "results/figures/part_e_hidden_units_sweep.pdf",
        x_column="model_num_hidden",
        exact_energy=3.0,
        title="Part e: hidden-unit sweep",
    )

    maybe_plot(
        plot_parameter_sweep,
        "results/selected/part_e_time_step_sweep.csv",
        "results/figures/part_e_time_step_sweep.pdf",
        x_column="sampler_time_step",
        exact_energy=3.0,
        title="Part e: time-step sweep",
    )


if __name__ == "__main__":
    main()
