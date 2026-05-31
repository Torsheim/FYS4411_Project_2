"""Energy-convergence plots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .style import disable_scientific_offset, make_column_figure, pretty_label, save_report_figure


def plot_energy_convergence(
    csv_path: str | Path,
    output_path: str | Path,
    exact_energy: float | None = None,
    title: str | None = None,
) -> None:
    """Plot energy as a function of optimization iteration."""
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"{csv_path} contains no rows.")

    fig, ax = make_column_figure()

    yerr = df["energy_error"] if "energy_error" in df.columns else None
    ax.errorbar(
        df["iteration"],
        df["energy_mean"],
        yerr=yerr,
        marker="o",
        linewidth=1.0,
        elinewidth=0.7,
        capsize=1.5,
    )

    if exact_energy is not None:
        ax.axhline(exact_energy, linestyle="--", linewidth=1.0, label=f"Exact: {exact_energy:g}")
        ax.legend()

    if title:
        ax.set_title(title)

    ax.set_xlabel(pretty_label("iteration"))
    ax.set_ylabel(pretty_label("energy_mean"))
    disable_scientific_offset(ax)
    ax.grid(True, alpha=0.35)

    save_report_figure(fig, output_path)
