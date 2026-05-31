"""Parameter-sweep plots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .style import disable_scientific_offset, make_column_figure, pretty_label, save_report_figure


def plot_parameter_sweep(
    csv_path: str | Path,
    output_path: str | Path,
    x_column: str,
    y_column: str = "final_energy",
    exact_energy: float | None = None,
    title: str | None = None,
) -> None:
    """Plot a parameter sweep result file."""
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"{csv_path} contains no rows.")

    if x_column not in df.columns:
        raise KeyError(f"Column {x_column!r} not found. Available columns: {list(df.columns)}")

    if y_column not in df.columns:
        raise KeyError(f"Column {y_column!r} not found. Available columns: {list(df.columns)}")

    df = df.sort_values(x_column)

    fig, ax = make_column_figure()

    yerr = None
    if y_column == "final_energy" and "final_error" in df.columns:
        yerr = df["final_error"]

    ax.errorbar(
        df[x_column],
        df[y_column],
        yerr=yerr,
        marker="o",
        linewidth=0.95,
        elinewidth=0.65,
        capsize=1.3,
    )

    if exact_energy is not None:
        ax.axhline(
            exact_energy,
            linestyle="--",
            linewidth=0.95,
            label=f"Exact: {exact_energy:g}",
        )
        ax.legend(loc="best", frameon=True, borderpad=0.35, handlelength=1.7)

    if title:
        ax.set_title(title)

    ax.set_xlabel(pretty_label(x_column))
    ax.set_ylabel(pretty_label(y_column))
    disable_scientific_offset(ax)
    ax.grid(True, alpha=0.30)

    save_report_figure(fig, output_path)
