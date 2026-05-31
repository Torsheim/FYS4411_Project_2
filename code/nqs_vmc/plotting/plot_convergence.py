"""Energy-convergence plots for the report.

For report figures we do not show an error bar on every optimization
iteration. Dense error bars make long runs difficult to read. Statistical
uncertainties are instead reported in tables and blocking plots.

For short runs, we show the raw energy with modest markers.
For long runs, we show the raw energy as a thin background curve and a
rolling mean as the main convergence guide.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .style import disable_scientific_offset, make_column_figure, pretty_label, save_report_figure


def _rolling_window(n_points: int) -> int:
    """Choose a reasonable rolling-average window for long convergence plots."""
    if n_points < 150:
        return 1
    return max(10, min(30, n_points // 30))


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

    if "iteration" not in df.columns:
        raise KeyError(f"Column 'iteration' not found in {csv_path}. Columns: {list(df.columns)}")

    if "energy_mean" not in df.columns:
        raise KeyError(f"Column 'energy_mean' not found in {csv_path}. Columns: {list(df.columns)}")

    x = df["iteration"]
    y = df["energy_mean"]
    n_points = len(df)

    fig, ax = make_column_figure()

    window = _rolling_window(n_points)

    if window > 1:
        rolling = y.rolling(window=window, center=True, min_periods=1).mean()

        ax.plot(
            x,
            y,
            linewidth=0.55,
            alpha=0.35,
            label="Raw energy",
        )
        ax.plot(
            x,
            rolling,
            linewidth=1.15,
            label=f"{window}-step mean",
        )
    else:
        markevery = max(1, n_points // 45)
        ax.plot(
            x,
            y,
            marker="o",
            markevery=markevery,
            linewidth=0.95,
            markersize=3.0,
            label="Energy",
        )

    if exact_energy is not None:
        ax.axhline(
            exact_energy,
            linestyle="--",
            linewidth=0.95,
            label=f"Exact: {exact_energy:g}",
        )

    if title:
        ax.set_title(title)

    ax.set_xlabel(pretty_label("iteration"))
    ax.set_ylabel(pretty_label("energy_mean"))
    disable_scientific_offset(ax)
    ax.grid(True, alpha=0.30)
    ax.legend(loc="best", frameon=True, borderpad=0.35, handlelength=1.7)

    save_report_figure(fig, output_path)
