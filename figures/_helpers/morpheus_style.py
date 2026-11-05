from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

SKILL_ROOT = Path(__file__).resolve().parents[1]
STYLE_FILE = SKILL_ROOT / "assets" / "morpheus.mplstyle"

# Okabe-Ito; stable default categorical palette.
OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
]

# A muted categorical extension for >8 classes. Keep mappings stable once assigned.
MUTED_EXTENDED = [
    "#332288", "#88CCEE", "#44AA99", "#117733", "#999933",
    "#DDCC77", "#CC6677", "#882255", "#AA4499", "#661100",
    "#6699CC", "#AA4466", "#4477AA", "#228833", "#EE6677",
    "#BBBBBB",
]


def apply_morpheus_style(*, scienceplots: bool = True, nature_like: bool = True) -> None:
    """Apply consistent publication styling.

    SciencePlots is optional. The local mplstyle is always the final override.
    """
    if scienceplots:
        try:
            import scienceplots  # noqa: F401
            styles = ["science", "no-latex"]
            if nature_like:
                styles.insert(1, "nature")
            plt.style.use(styles)
        except Exception:
            pass
    plt.style.use(str(STYLE_FILE))
    mpl.rcParams["axes.prop_cycle"] = cycler(color=OKABE_ITO)


def stable_category_colors(categories: Iterable[str], existing: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return deterministic colors without remapping existing labels."""
    existing = dict(existing or {})
    palette = OKABE_ITO + MUTED_EXTENDED
    result = dict(existing)
    for cat in sorted({str(x) for x in categories}):
        if cat not in result:
            result[cat] = palette[len(result) % len(palette)]
    return result


def outside_legend(ax, *, side: str = "right", title: str | None = None, ncol: int = 1):
    """Place an axes legend outside the plotting region.

    Axes.legend integrates well with constrained layout. The caller should use
    layout='constrained' and increase figure width for large legends.
    """
    if side == "right":
        return ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            frameon=False,
            title=title,
            ncol=ncol,
        )
    if side == "bottom":
        return ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            borderaxespad=0.0,
            frameon=False,
            title=title,
            ncol=ncol,
        )
    raise ValueError(f"Unsupported legend side: {side}")


def clean_axis(ax, *, despine: bool = True, grid: bool = False) -> None:
    if despine:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.grid(grid, axis="y", linewidth=0.4, alpha=0.25)
