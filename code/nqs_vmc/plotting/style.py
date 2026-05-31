"""Shared plotting style for the FYS4411 Project 2 report.

The report uses a two-column A4 layout with

    left margin  = 1.55 cm
    right margin = 1.55 cm
    columnsep    = 0.70 cm

so the column width is

    (21.0 - 1.55 - 1.55 - 0.70)/2 = 8.60 cm.

All figures are therefore generated with physical width 8.60 cm. In LaTeX,
include them as

    \\includegraphics[width=\\columnwidth]{figures/name.pdf}

or, equivalently, just

    \\includegraphics{figures/name.pdf}

if the PDF file is not otherwise scaled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib as mpl
import matplotlib.pyplot as plt

# Match the report body text.
REPORT_FONT_SIZE_PT: Final[int] = 10

# Exact column width from the report geometry.
CM_PER_INCH: Final[float] = 2.54
COLUMN_WIDTH_CM: Final[float] = 8.60
COLUMN_HEIGHT_CM: Final[float] = 6.15

COLUMN_WIDTH_IN: Final[float] = COLUMN_WIDTH_CM / CM_PER_INCH
COLUMN_HEIGHT_IN: Final[float] = COLUMN_HEIGHT_CM / CM_PER_INCH


def set_report_plot_style() -> None:
    """Set Matplotlib defaults to match the LaTeX report."""
    mpl.rcParams.update(
        {
            # Use a serif font close to LaTeX.
            "font.family": "serif",
            "font.serif": [
                "Latin Modern Roman",
                "CMU Serif",
                "Computer Modern Roman",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",

            # Text sizes. Everything inside plots is 10 pt.
            "font.size": REPORT_FONT_SIZE_PT,
            "axes.labelsize": REPORT_FONT_SIZE_PT,
            "axes.titlesize": REPORT_FONT_SIZE_PT,
            "xtick.labelsize": REPORT_FONT_SIZE_PT,
            "ytick.labelsize": REPORT_FONT_SIZE_PT,
            "legend.fontsize": REPORT_FONT_SIZE_PT,
            "figure.titlesize": REPORT_FONT_SIZE_PT,

            # Lines and markers, kept modest for one-column figures.
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.minor.width": 0.55,
            "ytick.minor.width": 0.55,
            "lines.linewidth": 0.95,
            "lines.markersize": 3.3,
            "grid.linewidth": 0.40,

            # PDF settings.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def make_column_figure():
    """Create one figure with exact report-column width."""
    set_report_plot_style()
    fig, ax = plt.subplots(
        figsize=(COLUMN_WIDTH_IN, COLUMN_HEIGHT_IN),
        constrained_layout=False,
    )

    # Fixed margins. This keeps all generated PDFs the same physical size.
    fig.subplots_adjust(
        left=0.19,
        right=0.97,
        bottom=0.19,
        top=0.96,
    )
    return fig, ax


def save_report_figure(fig, output_path: str | Path) -> None:
    """Save a figure with fixed physical dimensions.

    Important: do not use bbox_inches='tight'. It changes the PDF bounding box
    and then LaTeX rescales different plots differently.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def pretty_label(name: str) -> str:
    """Convert internal column names to readable axis labels."""
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
    """Avoid offset text such as '+5e-1' on zoomed energy axes."""
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
