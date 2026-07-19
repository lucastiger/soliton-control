"""Plotting and visualization helper functions.

This file will include reusable plotting utilities for trajectories, spectra,
state transitions, and training diagnostics.
"""

import matplotlib as mpl


def apply_pub_style() -> None:
    """Publication-quality defaults shared by the analysis figures.

    150+ dpi, readable axis/legend fonts, light grid. Call once before
    building a figure so every analysis script renders consistently.
    """
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.2,
        "figure.constrained_layout.use": True,
    })
