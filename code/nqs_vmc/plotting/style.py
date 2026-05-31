"""Shared plotting style for the FYS4411 Project 2 report.

The report body text is 10 pt. The figures are generated at approximately
one-column width, so when included in LaTeX with width=\\columnwidth the
axis labels, tick labels, legends, and annotations appear at the same size
as the report text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib as mpl
import matplotlib.pyplot as plt

REPORT_FONT_SIZE_PT: Final[int] = 10
TITLE_FONT_SIZE_PT: Final[int] = 11

# Approximate one-column width for a two-column report.
# Use figures in LaTeX with \includegraphics[width=\columnwidth]{...}.
COLUMN_WIDTH_IN: Final[float] = 3.35
COLUMN_HEIGHT_IN: Final[float] = 2.50

FULL_WIDTH_IN: Final[float] = 6.95
FULL_HEIGHT_IN: Final[float] = 4.10


def set_report_plot_style() -> None:
    """Set Matplotlib defaults to match the report typography."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": REPORT_FONT_SIZE_PT,
            "axes.labelsize": REPORT_FONT_SIZE_PT,
            "axes.titlesize": TITLE_FONT_SIZE_PT,
            "xtick.labelsize": REPORT_FONT_SIZE_PT,
            "ytick.labelsize": REPORT_FONT_SIZE_PT,
            "legend.fontsize": REPORT_FONT_SIZE_PT,
            "figure.titlesize": TITLE_FONT_SIZE_PT,
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "lines.linewidth": 1.1,
            "lines.markersize": 4.0,
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def make_column_figure():
    """Create a one-column figure with report-size fonts."""
    set_report_plot_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, COLUMN_HEIGHT_IN), constrained_layout=True)
    return fig, ax


def make_full_width_figure():
    """Create a full-width figure for use with figure* if needed."""
    set_report_plot_style()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, FULL_HEIGHT_IN), constrained_layout=True)
    return fig, ax


def save_report_figure(fig, output_path: str | Path) -> None:
    """Save a figure without tight scaling.

    Avoiding bbox_inches='tight' helps keep the physical PDF size fixed,
    which helps preserve the intended font size in LaTeX.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def pretty_label(name: str) -> str:
    """Convert column names into readable axis labels."""
    replacements = {
        "iteration": "Iteration",
        "energy_mean": "Energy [a.u.]",
        "energy_error": "Standard error [a.u.]",
        "final_energy": "Final energy [a.u.]",
        "final_error": "Final standard error [a.u.]",
        "acceptance_rate": "Acceptance rate",
        "optimization_learning_rate": "Learning rate",
        "model_num_hidden": "Hidden units",
        "sampler_time_step": "Time step",
        "block_size": "Block size",
        "std_error": "Estimated standard error",
        "standard_error": "Estimated standard error",
        "error": "Estimated standard error",
    }
    return replacements.get(name, name.replace("_", " ").capitalize())


def disable_scientific_offset(ax) -> None:
    """Avoid small '+5e-1' offset text in zoomed energy plots."""
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
