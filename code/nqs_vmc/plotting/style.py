"""Shared plotting style for the FYS4411 Project 2 report.

All figure text is set to 10 pt, matching the report body text.
The figures are generated at one-column width and should be included
in LaTeX as:

    \includegraphics[width=\columnwidth]{figures/name.pdf}

Do not include these figures with width=\textwidth, because that rescales
the text inside the plot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib as mpl
import matplotlib.pyplot as plt

REPORT_FONT_SIZE_PT: Final[int] = 10

# One-column width for the two-column article layout.
# This matches typical \columnwidth for a 10 pt two-column article.
COLUMN_WIDTH_IN: Final[float] = 3.35
COLUMN_HEIGHT_IN: Final[float] = 2.45


def set_report_plot_style() -> None:
    """Set Matplotlib defaults to match the LaTeX report text."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "font.size": REPORT_FONT_SIZE_PT,
            "axes.labelsize": REPORT_FONT_SIZE_PT,
            "axes.titlesize": REPORT_FONT_SIZE_PT,
            "xtick.labelsize": REPORT_FONT_SIZE_PT,
            "ytick.labelsize": REPORT_FONT_SIZE_PT,
            "legend.fontsize": REPORT_FONT_SIZE_PT,
            "figure.titlesize": REPORT_FONT_SIZE_PT,
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.5,
            "grid.linewidth": 0.45,
            "legend.frameon": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def make_column_figure():
    """Create a one-column figure with report-size fonts."""
    set_report_plot_style()
    fig, ax = plt.subplots(
        figsize=(COLUMN_WIDTH_IN, COLUMN_HEIGHT_IN),
        constrained_layout=True,
    )
    return fig, ax


def save_report_figure(fig, output_path: str | Path) -> None:
    """Save the figure with a fixed physical size.

    We intentionally do not use bbox_inches='tight', because that changes
    the physical PDF bounding box and leads to inconsistent font scaling
    when LaTeX includes the figure.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def pretty_label(name: str) -> str:
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
    """Avoid offset labels such as '+5e-1' in zoomed energy plots."""
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
