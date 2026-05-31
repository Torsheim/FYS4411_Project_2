"""Sampling-comparison plots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .style import disable_scientific_offset, make_column_figure, pretty_label, save_report_figure


def plot_sampling_comparison(
    csv_path: str | Path,
    output_path: str | Path,
    x_column: str = "sampler",
    y_column: str = "final_energy",
    exact_energy: float | None = None,
    title: str = "Sampling comparison",
) -> None:
    """Plot a comparison between samplers."""
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"{csv_path} contains no rows.")

    if x_column not in df.columns:
        raise KeyError(f"Column {x_column!r} not found. Available columns: {list(df.columns)}")

    if y_column not in df.columns:
        raise KeyError(f"Column {y_column!r} not found. Available columns: {list(df.columns)}")

    fig, ax = make_column_figure()

    yerr = df["final_error"] if y_column == "final_energy" and "final_error" in df.columns else None

    ax.errorbar(
        df[x_column].astype(str),
        df[y_column],
        yerr=yerr,
        marker="o",
        linewidth=0.0,
        elinewidth=0.7,
        capsize=1.5,
    )

    if exact_energy is not None:
        ax.axhline(exact_energy, linestyle="--", linewidth=1.0, label=f"Exact: {exact_energy:g}")
        ax.legend(frameon=True)

    ax.set_xlabel(pretty_label(x_column))
    ax.set_ylabel(pretty_label(y_column))
    ax.set_title(title)

    disable_scientific_offset(ax)
    ax.grid(True, axis="y", alpha=0.35)

    save_report_figure(fig, output_path)
