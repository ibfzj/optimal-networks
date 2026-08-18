"""Figure style for Nature Communications submission.

The rules encoded here come from the journal's "Guide to formatting articles"
(figures section, p. 4) and the artwork guidelines it links to:

  * figures are supplied as individual vector files with editable text,
    which requires embedded TrueType rather than matplotlib's default Type 3
  * sans-serif throughout, preferably Helvetica or Arial
  * text no larger than 7 pt and no smaller than 5 pt
  * RGB colour
  * 88 mm for a single column, 180 mm for a double column
  * no figure larger than a single A4 page, 260 x 179 mm

Because point sizes are only meaningful once the figure is at its final
physical size, every figure built with this style must set its figsize in
millimetres via `mm()` and must not be saved with bbox_inches="tight",
which silently rescales the result.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Physical limits from the artwork guidelines, in millimetres.
SINGLE_COLUMN_MM = 88.0
DOUBLE_COLUMN_MM = 180.0
MAX_WIDTH_MM = 179.0
MAX_HEIGHT_MM = 260.0

# Text sizes, in points. The guidelines allow 5-7 pt.
#
# Note on the lower bound: matplotlib renders mathtext sub- and superscripts at
# exactly 0.7x the base size. Any label containing one therefore has to sit at
# the 7 pt maximum, otherwise its subscripts drop below the 5 pt minimum (a 6 pt
# base yields 4.2 pt). That is why the legend is at 7 pt rather than 6 pt: the
# panel b legend entries carry subscripts. Tick labels are plain numerals with
# no subscripts, so they can stay at 6 pt and keep some visual hierarchy.
# 7 pt is the journal's maximum, so everything that has to be read sits at it.
# Drop FONT_TICK back to 6.0 if the tick labels ever crowd a narrow panel.
FONT_LABEL = 7.0
FONT_TICK = 7.0
FONT_LEGEND = 7.0
FONT_PANEL = 7.0
FONT_ANNOT = 6.0

# Colourblind-safe pair used for the two competing branches throughout.
COLOR_BROKEN = "#003366"   # navy
COLOR_SYMMETRIC = "#E69F00"  # orange

MM_PER_INCH = 25.4


def mm(*values):
    """Convert millimetres to inches, for use as a matplotlib figsize."""
    inches = tuple(v / MM_PER_INCH for v in values)
    return inches[0] if len(inches) == 1 else inches


def check_size(width_mm, height_mm):
    """Raise if a figure would exceed the single-A4-page limit."""
    if width_mm > MAX_WIDTH_MM:
        raise ValueError(f"width {width_mm} mm exceeds the {MAX_WIDTH_MM} mm limit")
    if height_mm > MAX_HEIGHT_MM:
        raise ValueError(f"height {height_mm} mm exceeds the {MAX_HEIGHT_MM} mm limit")


# Helvetica and Arial are both requested by the guidelines. Mathtext is set to
# stixsans rather than a custom Helvetica fontset because the figures are
# dominated by Greek symbols (gamma, beta, kappa, mu) and neither Helvetica nor
# Arial carries a complete Greek range in matplotlib's mathtext machinery.
RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    "font.family": "sans-serif",
    # Arial leads rather than Helvetica because the journal accepts either and
    # only Arial ships a real bold face here: macOS registers Helvetica.ttc at
    # weight 400 only, so fontweight="bold" on the panel letters silently falls
    # back to the regular face and the labels come out not bold.
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",

    "font.size": FONT_LABEL,
    "axes.labelsize": FONT_LABEL,
    "axes.titlesize": FONT_LABEL,
    "legend.fontsize": FONT_LEGEND,
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
    "figure.titlesize": FONT_LABEL,

    "axes.linewidth": 0.5,
    "lines.linewidth": 0.8,
    "patch.linewidth": 0.5,
    "grid.linewidth": 0.4,

    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "xtick.minor.size": 1.2,
    "ytick.minor.size": 1.2,
    "xtick.major.pad": 1.8,
    "ytick.major.pad": 1.8,
    "axes.labelpad": 1.8,

    # frameless everywhere: legends are placed in empty regions, and a box
    # would be the only inconsistency against the panels that have none
    "legend.frameon": False,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "0.5",
    "legend.borderpad": 0.3,
    "legend.labelspacing": 0.25,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.3,
    "legend.columnspacing": 1.0,

    "figure.dpi": 200,
    "savefig.dpi": 600,
    "savefig.facecolor": "white",
    "savefig.transparent": False,
}


def apply():
    """Apply the Nature Communications style to the global rcParams."""
    mpl.rcParams.update(RC)


def style(**overrides):
    """Return a context manager applying the style, for use in a with-block."""
    params = dict(RC)
    params.update(overrides)
    return plt.rc_context(params)


def panel_label(fig, letter, x, y, fontsize=FONT_PANEL):
    """Place a panel letter in figure coordinates.

    Nature Communications labels panels with bare bold lowercase letters, so no
    parentheses are added here. Placing the labels in figure coordinates rather
    than per-axes coordinates keeps them aligned across panels whose y-axis
    labels have different widths.
    """
    return fig.text(x, y, letter, fontsize=fontsize, fontweight="bold",
                    ha="left", va="top")


def save(fig, path):
    """Save a figure as a vector PDF at exactly its declared physical size.

    bbox_inches is deliberately not passed: "tight" would crop to the drawn
    content and change the final width away from the intended column width.
    """
    fig.savefig(path, format="pdf")
    return path
