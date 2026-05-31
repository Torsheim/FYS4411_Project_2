"""Blocking-analysis plots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .style import make_column_figure, pretty_label, save_report_figure


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    lower_to_original = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    raise KeyError(f"None of the columns {candidates} were found. Available columns: {list(df.columns)}")


def plot_blocking(csv_path: str | Path, output_path: str | Path, title: str = "Blocking analysis") -> None:
    """Plot estimated standard error against block size."""
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"{csv_path} contains no rows.")

    x_col = _find_column(df, ["block_size", "block size", "block"])
    y_col = _find_column(df, ["std_error", "standard_error", "error", "estimated_standard_error"])

    fig, ax = make_column_figure()

    ax.plot(df[x_col], df[y_col], marker="o", linewidth=1.0)

    if (df[x_col] > 0).all():
        ax.set_xscale("log", base=2)

    ax.set_xlabel(pretty_label("block_size"))
    ax.set_ylabel(pretty_label("std_error"))
    ax.set_title(title)

    ax.grid(True, alpha=0.35)

    save_report_figure(fig, output_path)
