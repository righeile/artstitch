"""Figures in the lab's publication style.

The house style, taken from the lab's own ephys module
(``working_stuff/Processing Scripts/voltage_clamp_processing.py``): a value
axis 1.2 in long and 0.25 in to a category (:data:`AXES_LENGTH_MM`,
:attr:`PlotSpec.bar_mm`), Arial, 7/8/9 pt, no top or right spine,
transparent background, and ``svg.fonttype='none'`` so every label is still
live text in Illustrator.  It is *opt-in*: importing this module leaves
``matplotlib.rcParams`` alone, because changing them at import would silently
restyle every unrelated plot in the same notebook session.  Use
:func:`publication_style` around the figures that want it.

:data:`VECTOR_TEXT` is the one exception, and it *is* applied at import --
see the note on it.

On top of that sit the three figures these tools actually produce:

:func:`group_plot`
    Groups side by side -- bars, boxes, violins or a beeswarm -- with **every
    data point drawn on top**.  The points are the figure.  With four animals a
    side, a bar and an error bar cannot tell you whether a difference is four
    animals agreeing or one animal carrying the group, and that is the only
    thing a reader needs to know.
:func:`xy_plot`
    Mean ± SEM against a continuous x, one line per group.  An IV curve, a
    dose-response, a time course.
:func:`trace_plot`
    Averaged traces with a shaded SEM band and an L-shaped scale bar instead of
    axes, which is how a current trace is shown.

and the thing all three can carry:

:func:`annotate_p`
    Significance brackets **on the figure**.  Hand it the table
    :func:`labkit.stats.compare` returned and it draws the comparison over the
    groups it belongs to, stacking brackets so they never collide and never
    sit on the data.  The number printed is the number in the table -- there is
    no second formatting path for anyone to get wrong.
"""

from __future__ import annotations

import contextlib
import inspect
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from . import theme

__all__ = [
    "PUBLICATION_RCPARAMS", "use_publication_style", "publication_style",
    "add_scale_bars", "save_figure", "hide_axes",
    "annotate_p", "group_plot", "xy_plot", "trace_plot",
    "PlotSpec", "figure_from_spec", "comparisons_for", "annotate_from_spec",
    "PALETTES", "FONTS", "VECTOR_TEXT", "TIGHT_CROP",
    "MM", "jitter", "AXES_LENGTH_MM", "BOX_LW",
    "series_names", "order_series", "reorder_data",
]

SMALL, MEDIUM, LARGE = 7, 8, 9

#: How much of one category a bar, box or violin fills.  Named because
#: :func:`figure_from_spec` sizes the panel from it when ``PlotSpec.bar_mm``
#: asks for a bar of a given printed width: the two have to be the same number
#: or the bar comes out the wrong size, silently and by a constant factor.
BAR_WIDTH = 0.62

#: Every line of a box -- edge, whiskers, caps, median.  One weight, which is
#: what seaborn's ``linewidth=0.7`` gives the lab notebook's boxes.
BOX_LW = 0.7

#: How long the value axis is, in mm -- the data area's height on an upright
#: plot, its width on a turned one.  A bar's height is read against its
#: neighbours and against the next panel, so it has to be the same length on
#: the page whatever the figure is.  Stretched to the figure instead, a plot of
#: two groups and a plot of seven carry two y scales and nothing says so.
#:
#: 30.48 mm is 1.2 in, the number the lab's own ephys notebook draws at
#: (``figsize=(0.25*18, 1.2)`` in ``ClC-7 2025.ipynb``, via
#: ``working_stuff/Processing Scripts/voltage_clamp_processing.py``).  Set on
#: the figures rather than derived.  ``PlotSpec.panel_height_mm`` is the step
#: family's panel and keeps its own 38.1 -- a sweep of fourteen voltage steps
#: is not a bar chart and does not read short.
#:
#: ``PlotSpec.axes_height_mm``/``axes_width_mm`` override it.  The *category*
#: axis is never pinned from here -- ``bar_mm`` sizes that one.
AXES_LENGTH_MM = 30.48

#: How to ask matplotlib's boxplot and violinplot to lie on their side.
#: ``vert=False`` was deprecated in 3.11 for ``orientation='horizontal'``,
#: which only exists from 3.10, and these apps ask for ``matplotlib>=3.7`` --
#: so read which one this matplotlib has, once, rather than at every call.
_SIDEWAYS: dict = (
    {"orientation": "horizontal"}
    if "orientation" in inspect.signature(plt.Axes.boxplot).parameters
    else {"vert": False}
)


def _is_horizontal(spec) -> bool:
    """Whether a spec turns its panel on its side.

    ``'xy'`` and ``'trace'`` have no categories to turn, so they never do.
    """
    return bool(getattr(spec, "horizontal", False)) and spec.kind not in ("xy", "trace")


#: One millimeter, in the inches matplotlib measures figures in.  Figure widths
#: are specified in mm here because journals specify them in mm.
MM = 1.0 / 25.4


#: How text is written into a vector file.  These three are applied globally,
#: at import, which nothing else here does, because they are the only settings
#: in this module that are not style: they change how a letter is *encoded* in
#: an .svg or .pdf, never how the figure looks.
#:
#: They have to be global because matplotlib reads them when the file is
#: *written*, not when the artists are made -- unlike a font size, which each
#: Text bakes in as it is created.  So a figure drawn inside
#: :func:`publication_style` and saved after the block, which is what every one
#: of these apps does, was saved under matplotlib's default ``'path'``: every
#: label came out as outlines, the file had no ``<text>`` in it at all, and
#: nobody could tell, because outlined letters look exactly like letters until
#: someone opens the figure to change one.
#:
#: 42 is TrueType.  The default, Type 3, is rejected outright by most journals.
VECTOR_TEXT: dict = {
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
mpl.rcParams.update(VECTOR_TEXT)


#: How much white is left round a figure, and how the crop is made to agree
#: with the size that was asked for.
#:
#: Global for exactly the reason :data:`VECTOR_TEXT` is: ``savefig.*`` is read
#: when the file is *written*, and every one of these apps writes it outside
#: the style block, so the value sitting in :data:`PUBLICATION_RCPARAMS` never
#: reached a single saved file.  They all got matplotlib's default 0.1 inch,
#: which put three millimeters of margin round a figure asked for at a
#: journal's column width: 89 mm went out as 92.  Production scales that back
#: to the column, and the 7 pt type checked against the journal's minimum
#: arrives at 6.8 pt.
#:
#: **The two numbers have to be equal.**  Constrained layout reserves ``w_pad``
#: of blank round the labels; saving with ``bbox_inches='tight'`` crops that
#: blank off and then adds ``pad_inches`` back.  Equal, they cancel and the
#: file is the figure size exactly -- 89 mm asked, 89.00 mm written, at every
#: width and every set of labels.  Unequal, every figure is out by twice the
#: difference: matplotlib's own defaults are 0.0417 against 0.1, which is
#: where the three millimeters came from.
#:
#: 0.02 in is half a millimeter, which is the tight end of sensible.  Not
#: zero, because a tight bbox is measured from text extents and clips the odd
#: descender or tick end.
TIGHT_CROP: dict = {
    "savefig.pad_inches": 0.02,
    "figure.constrained_layout.w_pad": 0.02,
    "figure.constrained_layout.h_pad": 0.02,
}
mpl.rcParams.update(TIGHT_CROP)


#: The fonts a figure may be set in.  Arial is the default because it is what
#: the journals ask for -- Nature, Cell and the physiology journals all name
#: Arial or Helvetica, and a figure submitted in anything else is a figure
#: somebody re-sets.  PT Sans is here because it is what the interfaces are
#: drawn in, so a figure going into a talk or a poster beside them can match;
#: PT Mono is for the figure whose labels are sequences, well numbers or
#: filenames, where a fixed width is the reason.  The whole stack is written
#: into an exported SVG by ``svg.fonttype='none'``, so the fallbacks are what
#: a collaborator without the first font actually gets.
FONTS: dict[str, list[str]] = {
    "arial":   ["Arial", "Helvetica", "DejaVu Sans"],
    "pt-sans": ["PT Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "pt-mono": ["PT Mono", "DejaVu Sans Mono", "monospace"],
}


#: The lab's conventions, as an explicit dict rather than an import-time global.
PUBLICATION_RCPARAMS: dict = {
    "lines.linewidth": 0.2,
    "lines.markersize": 1,
    "lines.color": theme.PRINT["ink"],
    "axes.prop_cycle": mpl.cycler(color=theme.SERIES),
    "font.family": FONTS["arial"],
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.edgecolor": theme.PRINT["ink"],
    "axes.labelcolor": theme.PRINT["ink"],
    "axes.grid": False,

    # Tight furniture.  matplotlib's gaps are sized for a six-inch figure on a
    # screen; on an 85 mm column they cost more width than some of the data.
    # `titlepad` is the lab notebook's 10 rather than the 3 that was here: a
    # panel title in these figures is a label standing above the plot, not a
    # caption sitting on it.
    "axes.titlepad": 10,
    "axes.labelpad": 2,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,

    # Line weights inside what the journals ask for: Nature 0.25-1 pt, Cell
    # 0.5-1.5, JGP 0.5-1.  matplotlib's 0.8 default sits at the top of all
    # three, and a frame heavier than the trace inside it is backwards.
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "legend.loc": "best",
    "legend.frameon": False,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.3,
    "legend.handlelength": 1.0,
    "legend.handleheight": 0.5,
    "legend.borderpad": 0.3,
    "legend.borderaxespad": 0.3,
    "legend.columnspacing": 1.0,
    "legend.labelcolor": theme.PRINT["ink"],
    "legend.markerscale": 1,
    "legend.scatterpoints": 1,
    "legend.numpoints": 1,
    "figure.figsize": (2, 2),
    "xtick.color": theme.PRINT["ink"],
    "ytick.color": theme.PRINT["ink"],
    "xtick.labelcolor": theme.PRINT["ink"],
    "ytick.labelcolor": theme.PRINT["ink"],
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.bottom": True,
    "ytick.left": True,
    # Kept here too, so the dict is a complete description of the style.
    **VECTOR_TEXT,
    **TIGHT_CROP,
    "font.size": SMALL,
    "axes.titlesize": MEDIUM,
    "axes.labelsize": MEDIUM,
    "xtick.labelsize": SMALL,
    "ytick.labelsize": SMALL,
    "legend.fontsize": SMALL,
    "figure.titlesize": LARGE,
    "savefig.bbox": "tight",
    "savefig.transparent": True,
    "savefig.dpi": 300,
}


def use_publication_style() -> None:
    """Apply the style globally.  Prefer :func:`publication_style` instead."""
    mpl.rcParams.update(PUBLICATION_RCPARAMS)


@contextlib.contextmanager
def publication_style(font_pt: float | None = None, font: str | None = None,
                      **overrides):
    """Apply the publication style for the duration of the block only.

    >>> with publication_style(figure__figsize=(3, 2)):
    ...     fig, ax = plt.subplots()

    ``font_pt`` scales the whole 7/8/9 ladder at once, which is what the size
    control in every app's plot panel is wired to.  ``font`` names a key of
    :data:`FONTS`.  Other keys may use dotted rcParam names or double
    underscores.
    """
    params = dict(PUBLICATION_RCPARAMS)
    if font:
        params["font.family"] = FONTS.get(str(font).lower(), FONTS["arial"])
    if font_pt:
        scale = float(font_pt) / MEDIUM
        params.update({
            "font.size": SMALL * scale,
            "axes.titlesize": MEDIUM * scale,
            "axes.labelsize": MEDIUM * scale,
            "xtick.labelsize": SMALL * scale,
            "ytick.labelsize": SMALL * scale,
            "legend.fontsize": SMALL * scale,
            "figure.titlesize": LARGE * scale,
        })
    params.update({k.replace("__", "."): v for k, v in overrides.items()})
    with mpl.rc_context(params):
        yield


def hide_axes(ax) -> None:
    """Remove the frame entirely, for trace panels that carry scale bars."""
    ax.set_axis_off()


# ==========================================================================
# Significance brackets
# ==========================================================================

def _span_of(ax, label, positions: dict) -> float | None:
    """Position of a group label on its own axis, tolerating int/str mismatches."""
    if label in positions:
        return float(positions[label])
    text = str(label)
    for key, value in positions.items():
        if str(key) == text:
            return float(value)
    return None


def _data_ceiling(ax, low: float, high: float, horizontal: bool = False) -> float:
    """The furthest thing already drawn between two group positions, in data units.

    "Furthest" is up on an ordinary panel and to the right on a horizontal one:
    the categories and the values swap axes together, so one index pair runs
    both rather than a second copy of the scan.

    Scanned off the artists rather than taken from the limits so a bracket
    clears the *data* and not just the axes box -- with a generous limit those
    are very different heights, and a bracket floating a third of the panel
    above the tallest bar looks like a mistake.

    Bars, points, error bars, scatter and violins are all covered because the
    group figure can be drawn as any of them.
    """
    cat, val = (1, 0) if horizontal else (0, 1)
    pad = 1e-9
    low, high = min(low, high) - pad, max(low, high) + pad
    top = -np.inf

    for patch in ax.patches:                       # bars
        try:
            box = (patch.get_x(), patch.get_width(),
                   patch.get_y(), patch.get_height())
        except AttributeError:
            continue
        # A bar sits on the category axis and reaches along the value one; barh
        # swaps which pair of the four is which.
        near, thick = (box[2], box[3]) if horizontal else (box[0], box[1])
        far = (box[0] + box[1]) if horizontal else (box[2] + box[3])
        if near + thick >= low and near <= high and np.isfinite(far):
            top = max(top, far)

    for line in ax.lines:                          # points, error-bar caps, joins
        data = (np.asarray(line.get_xdata(), float),
                np.asarray(line.get_ydata(), float))
        cats, vals = data[cat], data[val]
        if cats.size == 0 or cats.size != vals.size:
            continue
        inside = (cats >= low) & (cats <= high) & np.isfinite(vals)
        if inside.any():
            top = max(top, float(np.max(vals[inside])))

    for collection in ax.collections:              # scatter, violin bodies, fills
        try:
            offsets = np.asarray(collection.get_offsets(), float)
        except Exception:
            offsets = np.empty((0, 2))
        if offsets.ndim == 2 and offsets.shape[0]:
            inside = ((offsets[:, cat] >= low) & (offsets[:, cat] <= high)
                      & np.isfinite(offsets[:, val]))
            if inside.any():
                top = max(top, float(np.max(offsets[inside, val])))
        for path in getattr(collection, "get_paths", lambda: [])():
            vertices = path.vertices
            if vertices.size == 0:
                continue
            inside = ((vertices[:, cat] >= low) & (vertices[:, cat] <= high)
                      & np.isfinite(vertices[:, val]))
            if inside.any():
                top = max(top, float(np.max(vertices[inside, val])))

    if not np.isfinite(top):
        top = float((ax.get_xlim() if horizontal else ax.get_ylim())[1])
    return top


def annotate_p(
    ax,
    comparisons: Sequence[dict] | "Any",
    positions: dict | None = None,
    *,
    style: str = "auto",
    only_significant: bool = False,
    alpha: float = 0.05,
    use_corrected: bool = True,
    step: float = 0.085,
    gap: float = 0.035,
    tip: float = 0.018,
    linewidth: float = 0.8,
    fontsize: float | None = None,
    colour: str | None = None,
    dim_ns: bool = True,
    align: str = "stagger",
    headroom: float = 0.04,
    horizontal: bool | None = None,
) -> list[dict]:
    """Draw significance brackets over the groups they compare.

    ``comparisons``
        Either the table :func:`labkit.stats.compare` returned, or a list of
        ``{'a', 'b', 'p'}`` dicts.  Passing the table is the intended route --
        the number printed is then the corrected p-value from the table, so a
        figure and its statistics section cannot drift apart.
    ``positions``
        ``{group label: x}``.  Omitted, the x tick labels are used, which is
        right for anything :func:`group_plot` drew.
    ``style``
        ``'auto'`` (a number, floored at ``p < 0.0001``), ``'exact'``,
        ``'stars'`` or ``'both'``.  See :func:`labkit.stats.format_p`.
    ``align``
        ``'stagger'`` starts each bracket just above the data it spans, which
        is compact.  ``'level'`` puts every bracket at the same base height,
        which reads more evenly when the groups are of similar size.
    ``horizontal``
        Categories down the y axis, brackets reaching right instead of up.
        Left ``None`` it is read back off the axes, which :func:`group_plot`
        marks -- so a caller that passes the table it always passed gets
        brackets the right way round without knowing about this argument.

    Brackets are stacked: a bracket is raised until it sits above every bracket
    it overlaps, so a three-group figure with all three pairs annotated comes
    out as a readable ladder rather than three lines on top of each other.
    The y-limit is then extended to fit them, because a bracket clipped by the
    axes is worse than no bracket.

    Returns the comparisons it actually drew, each with the ``y`` it was drawn
    at, so a caller can put something else beside them.
    """
    from .stats import brackets as _brackets, format_p

    if horizontal is None:
        horizontal = getattr(ax, "_labkit_horizontal", False)
    horizontal = bool(horizontal)
    # A horizontal panel is the same picture turned a quarter turn: the
    # categories run down y and the ladder climbs along x.  One flag through
    # the whole function rather than a second copy of it -- ``_pair`` puts a
    # (category, value) pair the way round this panel draws them, and
    # ``value`` names the axis the brackets are stacked along.
    value = "x" if horizontal else "y"
    along = 0 if horizontal else 1

    def _pair(category, magnitude):
        return (magnitude, category) if horizontal else (category, magnitude)

    # --- normalize the input to a list of dicts ---------------------------
    if comparisons is None:
        return []
    if hasattr(comparisons, "columns"):                       # a DataFrame
        items = _brackets(comparisons, use_corrected=use_corrected, style=style,
                          only_significant=only_significant, alpha=alpha)
    else:
        items = []
        for entry in comparisons:
            p = float(entry.get("p", entry.get("p_value", np.nan)))
            if not np.isfinite(p):
                continue
            significant = p < alpha
            if only_significant and not significant:
                continue
            items.append({
                "a": str(entry.get("a", entry.get("group_a"))),
                "b": str(entry.get("b", entry.get("group_b"))),
                "p": p,
                "text": entry.get("text") or format_p(
                    p, style=style, show_ns=not only_significant),
                "significant": significant,
            })
    if not items:
        return []

    # --- where each group sits on its own axis ----------------------------
    if positions is None:
        ticks = ax.get_yticks() if horizontal else ax.get_xticks()
        texts = ax.get_yticklabels() if horizontal else ax.get_xticklabels()
        positions = {
            text.get_text(): tick
            for tick, text in zip(ticks, texts)
            if text.get_text()
        }
    placeable = []
    for entry in items:
        xa, xb = _span_of(ax, entry["a"], positions), _span_of(ax, entry["b"], positions)
        if xa is None or xb is None:
            continue                    # a comparison whose groups are not on this panel
        placeable.append({**entry, "x1": min(xa, xb), "x2": max(xa, xb)})
    if not placeable:
        return []

    # Short spans first: an adjacent pair should end up underneath the pair
    # that reaches across the whole axis, not the other way round.
    placeable.sort(key=lambda e: (e["x2"] - e["x1"], e["x1"]))

    colour = colour or theme.PRINT["rule"]
    fontsize = fontsize or mpl.rcParams["font.size"]

    # The layout is worked out in axes fractions, because a fixed data-unit
    # step between rows of the ladder would collapse at the top of a log axis.
    # It is then converted back to data units *before* anything is drawn, and
    # drawn in data coordinates -- an earlier version left the brackets pinned
    # to axes fractions, so extending the y-limit to make room moved the data
    # down and left the brackets sitting outside the frame.
    to_axes = ax.transData + ax.transAxes.inverted()
    from_axes = ax.transAxes + ax.transData.inverted()

    def _axes_y(magnitude: float) -> float:
        return float(to_axes.transform(_pair(0.0, magnitude))[along])

    def _data_y(fraction: float) -> float:
        return float(from_axes.transform(_pair(0.0, fraction))[along])

    # Every ceiling is measured before the first bracket exists.  Measuring as
    # we go would let each bracket see the one below it as data and raise
    # itself off that, so the ladder would climb twice as fast as it needs to.
    ceilings = {
        id(entry): _axes_y(_data_ceiling(ax, entry["x1"], entry["x2"], horizontal))
        for entry in placeable
    }
    common_base = max(ceilings.values()) if align == "level" else 0.0

    placed: list[tuple[float, float, float]] = []   # (x1, x2, y) already laid out
    layout: list[tuple[dict, float]] = []
    highest = 0.0

    for entry in placeable:
        base = common_base if align == "level" else ceilings[id(entry)]
        y = base + gap
        # Raise until clear of every bracket whose span this one overlaps.  The
        # span is padded so two brackets that merely touch at a shared group
        # still get separate rows.
        margin = 0.02 * max(1.0, abs(entry["x2"] - entry["x1"]))
        while any(
            not (entry["x2"] + margin < px1 or entry["x1"] - margin > px2)
            and abs(py - y) < step * 0.9
            for px1, px2, py in placed
        ):
            y += step
        placed.append((entry["x1"], entry["x2"], y))
        layout.append((entry, y))
        highest = max(highest, y + step * 0.55)

    # Convert while the old limits are still in force.  ``from_axes`` is a live
    # transform, not a snapshot: read it after ``set_ylim`` and every position
    # would be re-measured against the limits it had just been used to choose.
    resolved = [
        (entry, _data_y(y), _data_y(y - tip), _data_y(y + tip * 0.45))
        for entry, y in layout
    ]
    ceiling_data = _data_y(highest + headroom)

    low, high = getattr(ax, f"get_{value}lim")()
    inverted = high < low
    if highest + headroom > 1.0:
        getattr(ax, f"set_{value}lim")(
            *((ceiling_data, high) if inverted else (low, ceiling_data)))

    drawn: list[dict] = []
    for entry, y, tip_y, text_y in resolved:
        faded = dim_ns and not entry["significant"]
        line_colour = theme.PRINT["muted"] if faded else colour

        ax.plot(
            *_pair([entry["x1"], entry["x1"], entry["x2"], entry["x2"]],
                   [tip_y, y, y, tip_y]),
            clip_on=False, zorder=10,
            lw=linewidth, color=line_colour, solid_capstyle="butt",
        )
        if entry["text"]:
            ax.text(
                *_pair((entry["x1"] + entry["x2"]) / 2.0, text_y), entry["text"],
                clip_on=False, zorder=10,
                ha="left" if horizontal else "center",
                va="center" if horizontal else "bottom", color=line_colour,
                fontsize=fontsize * (0.92 if faded else 1.0),
            )
        drawn.append({**entry, "y": y})
    return drawn


# ==========================================================================
# The three figures
# ==========================================================================

def jitter(n: int, width: float, seed: int = 0) -> np.ndarray:
    """Spread overlapping points sideways, reproducibly.

    Reproducibly matters: a figure that moves its dots every time it is redrawn
    cannot be compared with the copy already pasted into a draft.
    """
    if n <= 1:
        return np.zeros(max(n, 0))
    rng = np.random.default_rng(seed)
    return (rng.random(n) - 0.5) * width


def _beeswarm(values: np.ndarray, width: float, n_bins: int = 24) -> np.ndarray:
    """Offsets that lay points out sideways instead of on top of each other.

    Cheap binning rather than a real force layout: at the n these experiments
    run at, the difference is invisible and the cost is not.
    """
    values = np.asarray(values, float)
    if values.size <= 1:
        return np.zeros(values.size)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.size)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return jitter(values.size, width, seed=1)
    bins = np.clip(((values - lo) / (hi - lo) * (n_bins - 1)).astype(int), 0, n_bins - 1)
    offsets = np.zeros(values.size)
    for b in np.unique(bins):
        idx = np.flatnonzero(bins == b)
        k = idx.size
        # -w/2 .. +w/2, densest bin using the full width
        spread = np.linspace(-0.5, 0.5, k) if k > 1 else np.array([0.0])
        offsets[idx] = spread * width * min(1.0, k / 5.0 + 0.35)
    return offsets


def _error_of(values: np.ndarray, error: str) -> float:
    values = values[np.isfinite(values)]
    if error == "none" or values.size < 2:
        return 0.0
    sd = float(np.std(values, ddof=1))
    if error == "sd":
        return sd
    if error == "sem":
        return sd / np.sqrt(values.size)
    if error in ("ci", "ci95"):
        import scipy.stats as ss
        return float(ss.t.ppf(0.975, values.size - 1)) * sd / np.sqrt(values.size)
    raise ValueError(f"Unknown error bar {error!r}: sd, sem, ci or none")


def _centre_of(values: np.ndarray, centre: str) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.median(values)) if centre == "median" else float(np.mean(values))


def group_plot(
    groups: "dict[str, Sequence[float]] | Any",
    *,
    value: str | None = None,
    group: str | None = None,
    kind: str = "bar",
    centre: str = "mean",
    error: str = "sem",
    points: str | bool = "jitter",
    paired: "dict[str, Sequence] | None" = None,
    palette: Sequence[str] | None = None,
    ax=None,
    ylabel: str = "",
    xlabel: str = "",
    title: str = "",
    order: Sequence[str] | None = None,
    bar_width: float = BAR_WIDTH,
    point_size: float = 2.5,
    show_n: bool = True,
    log_y: bool = False,
    zero_line: bool = True,
    horizontal: bool = False,
    seed: int = 0,
):
    """Groups side by side, with every data point on top of them.

    ``groups`` is ``{label: values}``, or a tidy DataFrame together with
    ``value=`` and ``group=``.

    ``kind``
        ``'bar'``, ``'box'``, ``'violin'`` or ``'none'`` (points only, with a
        line at the center).  ``'none'`` is the honest default for small n and
        is what ``'bar'`` degrades to when a group has fewer than three values.
    ``points``
        ``'jitter'``, ``'swarm'``, ``'strip'`` (no offset) or ``False``.
    ``paired``
        ``{label: unit ids}`` parallel to ``groups``.  Given, the same unit is
        joined by a thin line across the groups, which is the whole argument
        for a paired test made visible.
    ``horizontal``
        The same panel turned a quarter turn: groups down the y axis, bars
        reaching right, first group at the top.  ``ylabel`` follows the values
        round to the x axis and ``xlabel`` follows the groups to the y one, so
        a panel keeps its labels when it is turned.  It is what long group
        names want -- ``R201C`` and ``rescue`` have a whole column to
        themselves instead of 6.35 mm of tick.

    Returns ``(ax, positions)`` -- the positions being ``{label: x}``, ready to
    hand to :func:`annotate_p`.
    """
    # --- accept a tidy frame as readily as a dict -------------------------
    if hasattr(groups, "columns"):
        if not value or not group:
            raise ValueError("a DataFrame needs value= and group= naming its columns")
        import pandas as pd
        frame = groups
        levels = order or [str(v) for v in pd.unique(frame[group].dropna())]
        data = {
            level: np.asarray(
                pd.to_numeric(frame.loc[frame[group].astype(str) == level, value],
                              errors="coerce"), float)
            for level in levels
        }
        if paired is None and value and group:
            paired = None
    else:
        data = {str(k): np.asarray(v, dtype=float).ravel() for k, v in groups.items()}
        levels = list(order) if order else list(data)
    data = {k: data[k][np.isfinite(data[k])] for k in levels if k in data}
    levels = [k for k in levels if k in data]
    if not levels:
        raise ValueError("nothing to plot: every group is empty")

    if ax is None:
        _, ax = plt.subplots()
    # Marked on the axes so :func:`annotate_p` can read it back: a caller that
    # passes the table it has always passed gets its brackets turned too.
    ax._labkit_horizontal = bool(horizontal)
    cat, val = ("y", "x") if horizontal else ("x", "y")

    def _pair(category, magnitude):
        """A (category, value) pair, the way round this panel draws them."""
        return (magnitude, category) if horizontal else (category, magnitude)

    colours = list(palette) if palette else theme.SERIES
    positions = {label: float(i) for i, label in enumerate(levels)}

    # A bar is a claim about a mean, and a mean of two numbers is not one.
    if kind == "bar" and min((len(v) for v in data.values()), default=0) < 3:
        kind = "none"

    # --- the summary shape ------------------------------------------------
    for i, label in enumerate(levels):
        values = data[label]
        colour = colours[i % len(colours)]
        x = positions[label]
        height = _centre_of(values, centre)
        err = _error_of(values, error)

        if kind == "bar":
            (ax.barh if horizontal else ax.bar)(
                x, height, **{("height" if horizontal else "width"): bar_width},
                color=colour, alpha=0.75, edgecolor=colour, linewidth=0.8,
                zorder=2)
            if err:
                ax.errorbar(*_pair(x, height), **{f"{val}err": err},
                            fmt="none", capsize=2.5,
                            elinewidth=0.9, capthick=0.9,
                            ecolor=theme.PRINT["ink"], zorder=4)
        elif kind == "box":
            # Black, like every other box in these apps (`PlotStyle.box_edge_colour`
            # in pHerret, which is where the house style was settled).  A box is
            # line art and line art is black; the series colour lives on the
            # points drawn on top, which is the only place it has to be read.
            # Drawn in the colour instead, the same data gives a different
            # picture depending on which app drew it -- and two such panels end
            # up side by side in one figure.
            ink = theme.PRINT["ink"]
            box = ax.boxplot([values], positions=[x], widths=bar_width,
                             **(_SIDEWAYS if horizontal else {}),
                             patch_artist=True, showfliers=False, zorder=2,
                             medianprops={"color": ink, "lw": BOX_LW},
                             whiskerprops={"color": ink, "lw": BOX_LW},
                             capprops={"color": ink, "lw": BOX_LW})
            for patch in box["boxes"]:
                # Outline only.  A filled box competes with the points drawn on
                # top of it for the same ink, and the points are the reason the
                # figure exists -- with three or four animals a side, what the
                # reader needs to see is where each one fell, not a tinted
                # rectangle in front of them.
                patch.set(facecolor="none", edgecolor=ink, linewidth=BOX_LW)
        elif kind == "violin":
            if values.size >= 3 and np.ptp(values) > 0:
                parts = ax.violinplot([values], positions=[x], widths=bar_width,
                                      **(_SIDEWAYS if horizontal else {}),
                                      showextrema=False, showmedians=False)
                for body in parts["bodies"]:
                    body.set(facecolor=colour, alpha=0.30, edgecolor=colour, linewidth=0.8)
            ax.plot(*_pair([x - bar_width / 3, x + bar_width / 3], [height] * 2),
                    color=theme.PRINT["ink"], lw=1.0, zorder=4, solid_capstyle="butt")
            if err:
                ax.errorbar(*_pair(x, height), **{f"{val}err": err},
                            fmt="none", elinewidth=0.9,
                            capsize=2.0, capthick=0.9,
                            ecolor=theme.PRINT["ink"], zorder=4)
        else:                                            # 'none': the center only
            # The line is the summary shape, so it carries the series colour the
            # way a bar's fill or a violin's body does.  The error bar does not:
            # it is a whisker, and whiskers are black in all four kinds -- it
            # was the series colour here alone, which made one plot kind quietly
            # unlike its neighbours.
            ax.plot(*_pair([x - bar_width / 2.4, x + bar_width / 2.4], [height] * 2),
                    color=colour, lw=1.4, zorder=4, solid_capstyle="butt")
            if err:
                ax.errorbar(*_pair(x, height), **{f"{val}err": err},
                            fmt="none", elinewidth=0.9, capsize=2.5, capthick=0.9,
                            ecolor=theme.PRINT["ink"], zorder=4)

    # --- the points, which are the point ----------------------------------
    offsets: dict[str, np.ndarray] = {}
    if points:
        for i, label in enumerate(levels):
            values = data[label]
            width = bar_width * 0.55
            if points == "swarm":
                dx = _beeswarm(values, width)
            elif points == "strip":
                dx = np.zeros(values.size)
            else:
                dx = jitter(values.size, width, seed=seed + i * 97)
            offsets[label] = positions[label] + dx
            # Filled in the group's colour with a hairline white halo, which
            # is what separates two dots that land on each other -- the lab
            # notebook's `edgecolor='w', linewidth=0.25`.  The box behind them
            # is an outline, so a filled dot covers nothing.
            ax.plot(*_pair(offsets[label], values), "o", ms=point_size,
                    mfc=colours[i % len(colours)],
                    mec=theme.PRINT["point_edge"],
                    mew=0.25, lw=0, zorder=6, clip_on=False)

    # --- join the pairs ---------------------------------------------------
    if paired and points:
        keyed: dict[str, dict[Any, float]] = {}
        for label in levels:
            ids = list(paired.get(label, []))
            keyed[label] = {}
            for n, unit in enumerate(ids):
                if n < len(offsets.get(label, [])) and n < len(data[label]):
                    keyed[label][unit] = n
        for left, right in zip(levels[:-1], levels[1:]):
            shared = set(keyed[left]) & set(keyed[right])
            for unit in shared:
                il, ir = keyed[left][unit], keyed[right][unit]
                ax.plot(*_pair([offsets[left][il], offsets[right][ir]],
                               [data[left][il], data[right][ir]]),
                        color=theme.PRINT["muted"], lw=0.4, alpha=0.7, zorder=5)

    # --- frame ------------------------------------------------------------
    getattr(ax, f"set_{cat}ticks")(list(positions.values()))
    labels = [
        f"{label}\n(n={len(data[label])})" if show_n else label for label in levels
    ]
    getattr(ax, f"set_{cat}ticklabels")(labels)
    edge = 0.5 + bar_width * 0.1
    if horizontal:
        # Reversed, so the first group is at the top: a list of groups reads
        # down the page, and drawing it upside down is the one thing everybody
        # notices about a horizontal bar chart somebody else made.
        ax.set_ylim(len(levels) - 1 + edge, -edge)
    else:
        ax.set_xlim(-edge, len(levels) - 1 + edge)
    if log_y:
        getattr(ax, f"set_{val}scale")("log")
    elif zero_line and kind in ("bar", "none"):
        low, high = getattr(ax, f"get_{val}lim")()
        if low < 0 < high:
            (ax.axvline if horizontal else ax.axhline)(
                0, color=theme.PRINT["grid"], lw=0.6, zorder=1)
    # The labels follow their own data round the corner, so turning a panel
    # does not mean retyping them.
    if ylabel:
        getattr(ax, f"set_{val}label")(ylabel)
    if xlabel:
        getattr(ax, f"set_{cat}label")(xlabel)
    if title:
        ax.set_title(title)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax, positions


def xy_plot(
    groups: Sequence[dict],
    *,
    ax=None,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    palette: Sequence[str] | None = None,
    error: str = "band",
    markers: "bool | Sequence[str]" = True,
    marker_size: float = 2.6,
    line_width: float = 1.0,
    line_styles: Sequence[str] | None = None,
    join: bool = True,
    zero_lines: bool = True,
    reference_y: float | None = None,
    legend: bool = True,
):
    """Mean ± SEM against a continuous x, one line per group.

    An IV curve is the archetype, which is why the axes cross at the origin by
    default: for a current-voltage relation the origin is a real place on the
    plot and the reversal potential is read off it.

    Each entry of ``groups`` is ``{'label', 'x', 'mean', 'sem'}`` and may carry
    ``'n'``.  ``error`` is ``'band'``, ``'bars'``, ``'both'`` or ``'none'``.
    ``markers`` and ``line_styles`` are cycled across the curves, so groups
    stay apart in a figure printed in gray.
    """
    if ax is None:
        _, ax = plt.subplots()
    colours = list(palette) if palette else theme.SERIES
    shapes = ["o"] if markers is True else ([] if markers is False else list(markers))
    dashes = list(line_styles) if line_styles else ["-"]

    for i, entry in enumerate(groups):
        colour = colours[i % len(colours)]
        x = np.asarray(entry["x"], float)
        mean = np.asarray(entry["mean"], float)
        sem = np.asarray(entry.get("sem", np.zeros_like(mean)), float)
        good = np.isfinite(x) & np.isfinite(mean)
        if not good.any():
            continue
        n = entry.get("n")
        n_max = int(np.nanmax(np.asarray(n, float))) if n is not None and len(np.atleast_1d(n)) else None
        label = entry.get("label", f"group {i + 1}")
        if n_max:
            label = f"{label} (n={n_max})"

        if error in ("band", "both"):
            band = np.where(np.isfinite(sem), sem, 0.0)
            ax.fill_between(x[good], (mean - band)[good], (mean + band)[good],
                            color=colour, alpha=0.18, linewidth=0, zorder=2)
        if error in ("bars", "both"):
            ax.errorbar(x[good], mean[good], yerr=np.where(np.isfinite(sem), sem, 0.0)[good],
                        fmt="none", ecolor=colour, elinewidth=0.7, capsize=1.6,
                        capthick=0.7, zorder=3)
        if join:
            ax.plot(x[good], mean[good], dashes[i % len(dashes)], color=colour,
                    lw=line_width, label=label, zorder=4)
        if shapes:
            ax.plot(x[good], mean[good], shapes[i % len(shapes)], ms=marker_size, color=colour,
                    mfc=colour, mew=0, zorder=5, ls="none",
                    label=None if join else label)

    if zero_lines:
        low, high = ax.get_ylim()
        if low < 0 < high:
            ax.axhline(0, color=theme.PRINT["grid"], lw=0.6, zorder=1)
        low, high = ax.get_xlim()
        if low < 0 < high:
            ax.axvline(0, color=theme.PRINT["grid"], lw=0.6, zorder=1)
    if reference_y is not None:
        ax.axhline(reference_y, color=theme.PRINT["grid"], lw=0.6, ls=(0, (2, 3)), zorder=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if legend and len(groups) > 1:
        ax.legend()
    return ax


def trace_plot(
    traces: Sequence[dict],
    *,
    ax=None,
    palette: Sequence[str] | None = None,
    band: bool = True,
    error: str = "band",
    line_width: float = 0.7,
    line_styles: Sequence[str] | None = None,
    scale_bars: bool = True,
    xlabel: str = "",
    ylabel: str = "",
    x_unit: str = "ms",
    y_unit: str = "pA/pF",
    x_length: float | None = None,
    y_length: float | None = None,
    legend: bool = True,
    title: str = "",
):
    """Averaged traces with a shaded SEM band and scale bars instead of axes.

    A current trace is shown with scale bars because the absolute time on the
    x axis is an artifact of when the protocol started, and a labeled axis
    invites the reader to take it seriously.
    """
    if ax is None:
        _, ax = plt.subplots()
    colours = list(palette) if palette else theme.SERIES
    dashes = list(line_styles) if line_styles else ["-"]

    for i, entry in enumerate(traces):
        colour = colours[i % len(colours)]
        x = np.asarray(entry["x"], float)
        mean = np.asarray(entry["mean"], float)
        sem = np.asarray(entry.get("sem", np.zeros_like(mean)), float)
        good = np.isfinite(x) & np.isfinite(mean)
        if not good.any():
            continue
        spread = np.where(np.isfinite(sem), sem, 0.0)
        if band and error in ("band", "both"):
            ax.fill_between(x[good], (mean - spread)[good], (mean + spread)[good],
                            color=colour, alpha=0.16, linewidth=0, zorder=2)
        if band and error in ("bars", "both"):
            # A bar per sample would be a smear; a handful along the trace is
            # readable and says the same thing.
            step = max(1, int(good.sum() // 12))
            picked = np.flatnonzero(good)[::step]
            ax.errorbar(x[picked], mean[picked], yerr=spread[picked], fmt="none",
                        ecolor=colour, elinewidth=0.6, capsize=1.2, capthick=0.6, zorder=3)
        label = entry.get("label", f"trace {i + 1}")
        if entry.get("n"):
            label = f"{label} (n={int(entry['n'])})"
        ax.plot(x[good], mean[good], dashes[i % len(dashes)], color=colour,
                lw=line_width, label=label, zorder=3)

    if title:
        ax.set_title(title)
    if legend and len(traces) > 1:
        ax.legend()
    if scale_bars:
        hide_axes(ax)
        add_scale_bars(ax, x_length=x_length, y_length=y_length,
                       x_unit=x_unit, y_unit=y_unit)
    else:
        # Axes instead of scale bars: then they need labels like any other.
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return ax


def _panel_grid(n: int, cols: int = 0) -> tuple[int, int]:
    """Rows and columns for ``n`` panels.

    Automatically a near-square grid -- 2 panels as 2 x 1, 6 as 3 x 2, 9 as
    3 x 3 -- with no wholly empty rows, so five groups get 3 x 2 rather than a
    3 x 3 grid with four blank panels.
    """
    if n <= 0:
        return 1, 1
    if cols and cols > 0:
        return int(np.ceil(n / cols)), int(cols)
    root = int(np.ceil(np.sqrt(n)))
    columns = root - 1 if root > 1 and n <= root * (root - 1) else root
    return int(np.ceil(n / columns)), columns


def _step_colours(n: int, palette: str) -> list:
    """One color per step, in voltage order: a ramp, because the order is the protocol."""
    from matplotlib.colors import LinearSegmentedColormap

    ramps = {
        "red-grey": LinearSegmentedColormap.from_list("rg", ["#b7bdc5", "#5f6772", "#c62f34"]),
        "greys": plt.get_cmap("Greys"),
        "reds": plt.get_cmap("Reds"),
        "colourblind": plt.get_cmap("viridis"),
    }
    cmap = ramps.get(palette, ramps["red-grey"])
    if palette in ("greys", "reds"):
        return [cmap(0.35 + 0.6 * i / max(n - 1, 1)) for i in range(n)]
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _family_colours(panels: Sequence[dict], spec: "PlotSpec") -> list[list]:
    """One color per step of every panel, in panel order.

    Coloring by voltage says what the protocol did; coloring by a field says
    what the cells were, and is the one that survives a figure with fourteen
    steps in it -- the ramp is unreadable at that count, and the panel titles
    already carry the voltages. Black is for the figure whose whole point is
    the shape of the current, where color is one more thing to explain.

    A field's color is fixed across the whole figure, not per panel, so two
    panels of the same construct match wherever they land in the grid.
    """
    by = (spec.family_colour or "voltage").strip()
    if by in ("", "voltage"):
        return [_step_colours(len(panel["sweeps"]), spec.palette) for panel in panels]
    if by == "black":
        return [[theme.PRINT["ink"]] * len(panel["sweeps"]) for panel in panels]

    def value_of(panel):
        return str((panel.get("keys") or {}).get(by, "") or "")

    wheel = PALETTES.get(spec.palette) or PALETTES["red-grey"]
    order = sorted({value_of(panel) for panel in panels})
    colour_of = {value: wheel[i % len(wheel)] for i, value in enumerate(order)}
    return [[colour_of[value_of(panel)]] * len(panel["sweeps"]) for panel in panels]


def family_plot(panels: Sequence[dict], spec: "PlotSpec", unit: str = "pA/pF"):
    """Averaged step families, one panel per group, every step overlaid.

    Each panel is ``{'label', 'n', 'time_ms', 'sweeps': [{'voltage_mv',
    'mean', 'sem'}]}``, with a ``'keys'`` of the values it was grouped by if
    the traces are to be colored by one. Panels share one y range so their
    currents can be compared by eye, which is also why the scale bar is drawn
    once, on the first. Returns ``(fig, axes)``.

    The publication style is applied here, so a caller neither needs to wrap
    this nor should: the axes size is settled after the style block exits,
    where the saving happens, and a block around this call would put the
    two back on opposite sides of it.
    """
    with publication_style(font_pt=spec.font_pt, font=spec.font):
        rows, cols = _panel_grid(len(panels), spec.panel_cols)
        width_in = max(spec.panel_width_mm, 8.0) * MM * cols
        height_in = max(spec.panel_height_mm, 8.0) * MM * rows
        fig, grid = plt.subplots(rows, cols, figsize=(width_in, height_in), squeeze=False,
                                 layout="constrained")
        axes = list(grid.ravel())

        x_limits = _limits(spec.xlim)
        y_limits = _limits(spec.ylim)
        if y_limits is None:
            lows, highs = [], []
            for panel in panels:
                time = np.asarray(panel["time_ms"], float)
                window = np.ones(time.size, bool) if x_limits is None else \
                    (time >= x_limits[0]) & (time <= x_limits[1])
                for sweep in panel["sweeps"]:
                    values = np.asarray(sweep["mean"], float)[window]
                    values = values[np.isfinite(values)]
                    if values.size:
                        # Percentiles, not extremes: a capacitive spike a few
                        # samples wide would otherwise set every panel's scale.
                        lows.append(np.percentile(values, 0.5))
                        highs.append(np.percentile(values, 99.5))
            if lows:
                low, high = min(lows), max(highs)
                pad = 0.06 * (high - low or 1.0)
                y_limits = (low - pad, high + pad)

        scale_bars = spec.spines in ("auto", "none")
        per_panel = _family_colours(panels, spec)
        for index, (ax, panel) in enumerate(zip(axes, panels)):
            time = np.asarray(panel["time_ms"], float)
            colours = per_panel[index]
            for colour, sweep in zip(colours, panel["sweeps"]):
                mean = np.asarray(sweep["mean"], float)
                good = np.isfinite(mean) & np.isfinite(time)
                if spec.family_spread:
                    sem = np.nan_to_num(np.asarray(sweep.get("sem", np.zeros_like(mean)), float))
                    ax.fill_between(time[good], (mean - sem)[good], (mean + sem)[good],
                                    color=colour, alpha=0.15, lw=0)
                ax.plot(time[good], mean[good], color=colour, lw=spec.line_width * 0.6)
            if x_limits:
                ax.set_xlim(*x_limits)
            if y_limits:
                ax.set_ylim(*y_limits)
            low, high = ax.get_xlim()
            ax.hlines(0, low, high, linestyles=(0, (1, 2)), lw=0.6, color=theme.PRINT["ink"])
            ax.set_title(f"{panel['label']},\nn = {panel['n']}", fontsize=spec.font_pt, pad=2)
            if scale_bars:
                # The lab's convention: no axes, and one scale bar for the lot,
                # which the shared y range makes true of every panel.
                ax.set_axis_off()
                if index == 0:
                    add_scale_bars(ax, x_length=spec.scale_x_s or None, y_length=spec.scale_y or None,
                                   x_unit="s", y_unit=unit, x_scale=1000.0,
                                   fontsize=spec.font_pt * 0.9)
            else:
                row, col = divmod(index, cols)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(spec.spines == "box")
                if row < rows - 1 and index + cols < len(panels):
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel(spec.xlabel or "Time (ms)")
                if col:
                    ax.tick_params(labelleft=False)
                else:
                    ax.set_ylabel(spec.ylabel or unit)
        for ax in axes[len(panels):]:
            ax.set_axis_off()
    # Outside the style block on purpose.  Constrained layout settles the
    # axes box at draw time and reads the tick label size from the rcParams
    # in force *then*, so a figure fitted under the publication style and
    # saved under the caller's comes out a couple of percent off the size it
    # was asked for -- silently, because fit_axes never measures again after
    # its last resize.  Fit where the saving happens and the number lands.
    fit_axes(fig, spec.axes_width_mm, spec.axes_height_mm)
    return fig, axes


# ==========================================================================
# Scale bars ("forks")
# ==========================================================================

_UNIT_LABELS = {
    "uA": r"$\mathdefault{\mu}$A",
    "uV": r"$\mathdefault{\mu}$V",
    "us": r"$\mathdefault{\mu}$s",
    "um": r"$\mathdefault{\mu}$m",
    "uM": r"$\mathdefault{\mu}$M",
}


def _nice(value: float) -> float:
    """Round to a 1/2/5 x 10^n value, for auto-sized scale bars."""
    if not np.isfinite(value) or value <= 0:
        return 1.0
    exponent = np.floor(np.log10(value))
    mantissa = value / (10.0**exponent)
    for candidate in (1.0, 2.0, 5.0, 10.0):
        if mantissa <= candidate:
            return float(candidate * 10.0**exponent)
    return float(10.0 ** (exponent + 1))


def add_scale_bars(
    ax,
    x_length: float | None = None,
    y_length: float | None = None,
    x_unit: str = "s",
    y_unit: str = "pA/pF",
    x_scale: float = 1.0,
    position: tuple[float, float] = (0.03, 0.05),
    linewidth: float = 1.0,
    color: str | None = None,
    fontsize: float | None = None,
):
    """Draw an L-shaped scale bar instead of axes.

    ``x_length`` and ``y_length`` are in data units of the *labeled* quantity;
    ``x_scale`` converts the label unit into x-data units (e.g. pass the
    sampling rate when x is in samples but the label is in seconds).  Either
    length may be ``None`` to be chosen automatically from the current limits.

    Returns the ``(x_length, y_length)`` actually drawn.
    """
    color = color or theme.PRINT["ink"]
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x_span, y_span = xlim[1] - xlim[0], ylim[1] - ylim[0]

    if y_length is None:
        y_length = _nice(abs(y_span) / 5.0)
    if x_length is None:
        x_length = _nice(abs(x_span) / x_scale / 5.0)

    x0 = xlim[0] + position[0] * x_span
    y0 = ylim[0] + position[1] * y_span
    x_data_length = x_length * x_scale

    ax.vlines(x0, ymin=y0, ymax=y0 + y_length, linewidth=linewidth, color=color, clip_on=False)
    ax.hlines(y0, xmin=x0, xmax=x0 + x_data_length, linewidth=linewidth, color=color, clip_on=False)

    kw = {"color": color, "clip_on": False}
    if fontsize is not None:
        kw["fontsize"] = fontsize

    x_label = f"{_fmt(x_length)} {_UNIT_LABELS.get(x_unit, x_unit)}"
    y_label = f"{_fmt(y_length)} {_UNIT_LABELS.get(y_unit, y_unit)}"
    ax.text(x0 + x_data_length / 2.0, y0 - 0.03 * y_span, x_label, ha="center", va="top", **kw)
    ax.text(x0 - 0.015 * x_span, y0 + y_length / 2.0, y_label, ha="right", va="center",
            rotation=90, **kw)
    return x_length, y_length


def _fmt(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


def save_figure(
    figure,
    path: str | os.PathLike,
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 300,
    transparent: bool = True,
) -> list[Path]:
    """Save a figure in several formats, creating parent directories.

    Always both a raster to look at and a vector to edit, because the figure
    that goes in the paper is never the first one produced.  Never depends on
    the current working directory and never touches the pyplot global figure.

    The vector one is PDF.  It is what journals ask for, it is one file rather
    than a file plus whatever fonts the reader happens to have, and the text
    stays text: ``pdf.fonttype: 42`` in the rcParams above embeds the TrueType
    outlines instead of drawing the glyphs as paths, so a label can still be
    edited in Illustrator -- the same promise ``svg.fonttype: 'none'`` makes
    for SVG.  Pass ``formats=("png", "svg")`` where an SVG is wanted.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        target = path.with_suffix(f".{ext.lstrip('.')}")
        figure.savefig(target, bbox_inches="tight", dpi=dpi, transparent=transparent)
        written.append(target)
    return written


# ==========================================================================
# The control state the interfaces send
# ==========================================================================

@dataclass
class PlotSpec:
    """Everything the plot panel in any of these apps can set.

    One dataclass, one JSON shape, one set of controls -- so "error bars: SEM"
    means the same thing and sits in the same place whether you are looking at
    vesicles, aortas or currents.  The web apps post this, and
    :func:`figure_from_spec` turns it into a figure.
    """

    kind: str = "bar"              # bar | box | violin | none | xy | trace
    centre: str = "mean"           # mean | median
    error: str = "sem"             # sem | sd | ci | none
    points: str = "jitter"         # jitter | swarm | strip | none
    palette: str = "red-grey"      # a key into PALETTES
    #: Per-series color, ``{"KO": "#c62f34"}``, overriding the palette for the
    #: names it mentions.  "KO is this red" is a decision about a paper, not
    #: about a palette, and it has to survive a figure gaining a group.
    colours: dict[str, str] = field(default_factory=dict)
    #: The sequence the conditions are drawn in, ``["WT", "P716L", "Y715C"]``.
    #: Empty leaves them in the order the data arrived, which for most callers
    #: is alphabetical -- and alphabetical puts the wild type after the mutants.
    #: Named by name, like :attr:`colours`, and for the same reason: an order
    #: has to survive the figure gaining a condition.  :func:`order_series` says
    #: what happens to a condition this does not mention.
    order: list[str] = field(default_factory=list)
    width_mm: float = 85.0         # single column
    height_mm: float = 0.0         # 0 = derive from width
    #: The plotting area of one panel, labels *outside* it -- see
    #: :func:`fit_axes`.  0 leaves the size to ``width_mm``/``height_mm``.
    axes_width_mm: float = 0.0
    axes_height_mm: float = 0.0
    #: The printed width of one bar, box or violin, in mm.  The panel is sized
    #: from it and the number of groups, so a bar is the same size on the page
    #: whether there are two of them or seven -- which is the thing a reader
    #: compares across a figure, and the thing that goes wrong when a panel is
    #: instead stretched to fill a column.
    #:
    #: 3.94 mm puts a category at 6.35 mm, which is 0.25 in -- the pitch the
    #: lab's ephys notebook draws at (``figsize=(0.25*18, 1.2)``), so a panel
    #: from an app and a panel from that notebook sit side by side at one
    #: scale.  It also clears the ``(n = 5)`` under a group name (5.33 mm at
    #: 7 pt), which 3 mm did not.  Two groups make a panel 12.7 mm wide, so
    #: six stand in one journal column.  Names longer than about five
    #: characters -- ``rescue`` -- still want the x rotation.
    #:
    #: ``width_mm`` becomes a ceiling rather than the width.  0 hands the
    #: width back to it; ``axes_width_mm``, being explicit, wins over both.
    #: ``'xy'`` and ``'trace'`` have no categories and ignore this.
    bar_mm: float = 3.94
    #: Turn the panel a quarter turn: groups down the y axis, bars reaching
    #: right, first group at the top.  Everything else is the same figure --
    #: the value label follows the values to the x axis, ``log_y`` and the
    #: reference lines follow them too, and the significance brackets stack to
    #: the right instead of upward.
    #:
    #: The sizing follows as well, with the two axes swapped: :attr:`bar_mm`
    #: now sets the panel's *height* -- one bar is the same thickness on the
    #: page whatever the group count -- and :attr:`axes_width_mm` pins the
    #: value axis the way :attr:`axes_height_mm` does upright.  The ceiling
    #: moves with it, from ``width_mm`` to ``height_mm``; 0 there means the
    #: panel grows with the groups, which is the point of turning it.
    #:
    #: Which is what long names want: ``R201C`` and ``rescue`` get a whole
    #: column to themselves instead of the 6.35 mm of tick a 3.94 mm bar
    #: leaves them.  ``'xy'`` and ``'trace'`` have no categories and ignore it.
    horizontal: bool = False
    font_pt: float = 8.0
    font: str = "arial"            # a key into FONTS
    log_y: bool = False
    show_n: bool = True
    legend: bool = True
    title: str = ""
    ylabel: str = ""
    xlabel: str = ""

    # --- axes -------------------------------------------------------------
    #: What the size boxes are typed in. The stored numbers stay millimeters.
    size_unit: str = "mm"          # mm | in
    xlim: str = ""                 # "min, max"; empty leaves it to the data
    ylim: str = ""
    #: Tick positions: a list ("0, 20, 40"), or "step 20" for regular ones.
    xticks: str = ""
    yticks: str = ""
    #: Text to put at those ticks instead of the numbers.
    xtick_labels: str = ""
    ytick_labels: str = ""
    xtick_rotation: float = 0.0
    ytick_rotation: float = 0.0
    tick_direction: str = "out"    # out | in | inout
    minor_ticks: bool = False
    grid: str = "none"             # none | y | x | both
    #: ``"auto"`` keeps each kind's own convention -- scale bars on a trace,
    #: left and bottom elsewhere. The rest override it: a full box, axes
    #: crossing at zero, or none at all (scale bars).
    spines: str = "auto"           # auto | lb | box | zero | none
    log_x: bool = False
    #: Dotted horizontal lines, e.g. a control level: "0, -30".
    ref_lines: str = ""
    legend_loc: str = "best"
    dpi: int = 300

    # --- curves, for xy and trace ----------------------------------------
    #: How the spread is drawn: a shaded band, error bars, both, or nothing.
    error_style: str = "band"      # band | bars | both | none
    #: Marker codes cycled across curves, e.g. "o,s,^". Empty draws none.
    markers: str = "o"
    marker_size: float = 2.6
    line_width: float = 1.0
    #: Line styles cycled across curves, e.g. "-,--,:".
    line_styles: str = "-"
    #: Join the points of a curve at all.
    join: bool = True
    #: Size of the individual points on a group plot.
    point_size: float = 2.5

    # --- step families (one panel per group) ------------------------------
    #: Size of each panel, in mm; the figure is as many of them as there are.
    panel_width_mm: float = 25.4
    panel_height_mm: float = 38.1      # 1 x 1.5 in, matching PANEL_HEIGHT_IN
    #: Panels per row; 0 lays them out as near a square as the count allows.
    panel_cols: int = 0
    #: Scale bar lengths: seconds, and the current unit. 0 picks round ones.
    scale_x_s: float = 0.0
    scale_y: float = 0.0
    #: Shade each step's SEM. Off by default: fourteen overlapping bands hide
    #: the currents they describe.
    family_spread: bool = False
    #: What sets a trace's color. ``"voltage"`` ramps across the steps of a
    #: panel, ``"black"`` draws them all in ink, and any other value is read as
    #: a grouping field ("construct", "condition", ...): every step of a panel
    #: then takes the color of that field's value, shared across panels.
    family_colour: str = "voltage"   # voltage | black | <field name>

    # --- p-values on the figure ------------------------------------------
    show_p: bool = True
    p_style: str = "auto"          # auto | exact | stars | both
    p_test: str = "mann-whitney"
    p_correction: str = "fdr_bh"   # fdr_bh | holm | bonferroni | none
    p_pairs: str = "all"           # all | adjacent | <control level name>
    p_only_significant: bool = False
    p_use_corrected: bool = True
    p_align: str = "stagger"       # stagger | level
    p_alpha: float = 0.05
    p_pair_on: str = ""            # column identifying what is paired

    #: What a panel is, in the absence of anything asking for something else:
    #: 1.5 inches tall.  It used to be 0.72 of the width, which tied the height
    #: to a number that no longer sets the width -- with :attr:`bar_mm` sizing
    #: the panel from its bars, a default bar chart came out 0.85 x 2.41 in, a
    #: sliver.  It also meant a two-column figure was 5 inches tall for one
    #: panel.  A height has nothing to do with how wide a figure is.
    PANEL_HEIGHT_IN = 1.5

    def figsize(self) -> tuple[float, float]:
        width = max(self.width_mm, 25.0) * MM
        height = (self.height_mm * MM) if self.height_mm else self.PANEL_HEIGHT_IN
        return (width, height)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict | None) -> "PlotSpec":
        """Build from whatever the interface posted, ignoring anything extra.

        Deliberately forgiving: a stale browser tab sending a key this version
        dropped should get a figure, not a 500.
        """
        raw = dict(raw or {})
        fields = {f for f in cls.__dataclass_fields__}
        known = {k: v for k, v in raw.items() if k in fields}
        # Checkboxes arrive as "true"/"on"/1 depending on the app.
        for key, current in list(known.items()):
            default = cls.__dataclass_fields__[key].default
            if isinstance(default, bool):
                known[key] = str(current).lower() in ("1", "true", "on", "yes")
            elif isinstance(default, bool):
                pass
            elif isinstance(default, int):
                try:
                    known[key] = int(float(current))
                except (TypeError, ValueError):
                    known[key] = default
            elif isinstance(default, float):
                try:
                    known[key] = float(current)
                except (TypeError, ValueError):
                    known[key] = default
        spec = cls(**known)
        spec.colours = {str(k): str(v) for k, v in spec.colours.items()} \
            if isinstance(spec.colours, dict) else {}
        if spec.points in ("none", "false", "False"):
            spec.points = ""
        return spec


#: Named palettes the controls offer.  ``red-gray`` is the house one and the
#: reason the rest exist is that a figure with five conditions needs more than
#: two colors, not that anyone wanted a theme picker.
PALETTES: dict[str, list[str]] = {
    "red-grey": theme.SERIES,
    "greys": ["#2b2f36", "#5f6772", "#8d95a0", "#b7bdc5", "#d9dde1"],
    "reds": ["#7a1c20", "#a8272c", "#c62f34", "#dd6266", "#eb9c9e"],
    "colourblind": ["#5f6772", "#c62f34", "#0072b2", "#e69f00",
                    "#009e73", "#cc79a7", "#56b4e9", "#111111"],
}


def series_names(data) -> list[str]:
    """The series labels, in the order a figure will draw them.

    ``{label: values}`` for the group kinds, ``[{'label': ...}, ...]`` for the
    curve kinds, a plain list of labels for a caller that already knows them.
    """
    if isinstance(data, dict):
        return [str(k) for k in data]
    return [str(d.get("label", "") if isinstance(d, dict) else d) for d in data]


def order_series(names, order) -> list[str]:
    """``names`` put in the sequence ``order`` asks for, keeping the strangers.

    The sequence a reader needs is almost never alphabetical: ``WT``, ``P716L``,
    ``Y715C`` is the order of the mutation along the protein, and sorted() puts
    the wild type last.  So the order is the user's to state, and this is the one
    rule that applies it everywhere -- the figure, the colors, the brackets and
    the legend all read the same list.

    A name ``order`` does not mention keeps its place *after* the ones that are
    mentioned, rather than disappearing: a saved order outlives the dataset it
    was typed for, and the commonest thing to happen next is a new condition.
    Dropping it silently would be a figure missing a genotype that nobody asked
    to hide.  A name in ``order`` that is not in the data is simply skipped.
    """
    names = [str(n) for n in names]
    if not order:
        return names
    want = [str(o) for o in order]
    seen = set()
    first = [n for n in want if n in names and not (n in seen or seen.add(n))]
    return first + [n for n in names if n not in seen]


def reorder_data(data, order):
    """``data`` in the sequence :func:`order_series` gives, whatever shape it is.

    ``{label: values}`` for the group kinds and ``[{'label': ...}, ...]`` for
    the curve kinds, which is every shape :func:`figure_from_spec` takes.  Doing
    it to the data once, before anything reads it, is what keeps the order out
    of the drawing code: the colors are assigned by walking the series
    (:func:`colours_for`), the positions by drawing them, and the brackets by
    reading the positions back, so all three follow a reordered input without
    knowing an order exists.  Passing an order to the drawing instead would have
    left :func:`colours_for` walking the old sequence and quietly handed every
    series its neighbour's color.
    """
    if not order:
        return data
    want = order_series(series_names(data), order)
    if isinstance(data, dict):
        return {k: data[k] for k in want if k in data}
    at = {}
    for item in data:
        at.setdefault(str(item.get("label", "") if isinstance(item, dict) else item),
                      []).append(item)
    return [item for name in want for item in at.get(name, [])]


def colours_for(spec: "PlotSpec | dict", data) -> list[str]:
    """One color per series: the named palette, with the spec's overrides on top.

    ``spec.colours`` is what the color boxes in the panel write, and it is
    keyed by the *name* of the series rather than its position -- so a scheme
    saved for a two-genotype figure still puts the KO in the same red when a
    third genotype appears in the middle of the list, which an index-keyed one
    would not.  Names it does not mention keep the palette's color.
    """
    spec = spec if isinstance(spec, PlotSpec) else PlotSpec.from_dict(spec)
    base = PALETTES.get(spec.palette, theme.SERIES)
    names = series_names(data)
    if not names:
        return list(base)
    return [str(spec.colours.get(name) or base[i % len(base)])
            for i, name in enumerate(names)]


def fit_axes(fig, width_mm: float = 0.0, height_mm: float = 0.0,
             passes: int = 6) -> tuple[float, float]:
    """Grow the canvas until the *plotting area* is the size asked for, in mm.

    ``figsize`` is the whole picture, and the axes box is placed inside it in
    canvas *fractions* -- so the title, the tick labels and the axis label eat
    into the panel from a fixed number of inches away.  That bite is 40% of a
    small panel and 20% of a large one, which means the same "85 mm" buys a
    different amount of plot every time the labels change, and two panels that
    have to sit side by side on a page do not match.

    This pins the data area instead and lets the canvas be whatever the labels
    need.  Constrained layout only decides the axes box at draw time, so the
    only way to ask for a size is to draw, measure, and hand back a canvas
    larger by the shortfall; it settles in two or three passes.  Returns the
    figure size it settled on.  Either argument left at 0 leaves that
    dimension to ``figsize``.
    """
    want_w, want_h = max(width_mm, 0.0) * MM, max(height_mm, 0.0) * MM
    size = tuple(float(v) for v in fig.get_size_inches())
    if want_w <= 0.0 and want_h <= 0.0:
        return size
    panels = [a for a in fig.axes if a.get_visible()
              and getattr(a, "get_subplotspec", lambda: None)() is not None]
    if not panels:
        return size
    nrow, ncol = panels[0].get_subplotspec().get_gridspec().get_geometry()
    for _ in range(max(1, passes)):
        fig.draw_without_rendering()
        box = panels[0].get_window_extent().transformed(
            fig.dpi_scale_trans.inverted())
        wide, high = fig.get_size_inches()
        if box.width <= 0.0 or box.height <= 0.0:
            # The canvas is smaller than its own labels, so there is nothing
            # to measure and no shortfall to read off.  Make room and look
            # again rather than stepping on a number that is not there.
            fig.set_size_inches(wide * 1.6, high * 1.6)
            continue
        dw = ncol * (want_w - box.width) if want_w > 0.0 else 0.0
        dh = nrow * (want_h - box.height) if want_h > 0.0 else 0.0
        size = (max(wide + dw, 0.3), max(high + dh, 0.3))
        fig.set_size_inches(*size)
        if abs(dw) < 0.008 and abs(dh) < 0.008:      # a fifth of a millimeter
            break
    return size


def comparisons_for(
    spec: "PlotSpec | dict",
    data: "dict[str, Sequence[float]]",
    pair_on: dict | None = None,
):
    """The comparison table one :class:`PlotSpec` asks for, or ``None``.

    Every app that draws brackets needs exactly this -- run the test the
    controls name, correct the way the controls name, and hand the result to
    :func:`annotate_p`.  Doing it in one place is what stops "correction: none"
    meaning something subtly different in three codebases.
    """
    from . import stats as _stats

    spec = spec if isinstance(spec, PlotSpec) else PlotSpec.from_dict(spec)
    data = {k: v for k, v in data.items() if len(v)}
    if not spec.show_p or len(data) < 2:
        return None

    # There is no such thing as an uncorrected family, so "none" is honoured by
    # copying the raw p across rather than by choosing a different method --
    # anything else would report a corrected number under a "none" setting.
    correction = spec.p_correction if spec.p_correction != "none" else "bonferroni"
    table = _stats.compare_arrays(
        data, how=spec.p_pairs, test=spec.p_test, alpha=spec.p_alpha,
        correction=correction,
        units=pair_on if (spec.p_pair_on and pair_on) else None,
    )
    if spec.p_correction == "none" and len(table):
        table["p_corrected"] = table["p_value"]
        table["significant"] = table["p_value"] < spec.p_alpha
        table["stars"] = [_stats.stars(p) for p in table["p_value"]]
    return table


def annotate_from_spec(ax, spec: "PlotSpec | dict", table, positions=None) -> list[dict]:
    """:func:`annotate_p` with the arguments one :class:`PlotSpec` implies."""
    spec = spec if isinstance(spec, PlotSpec) else PlotSpec.from_dict(spec)
    if not spec.show_p or table is None or not len(table):
        return []
    return annotate_p(
        ax, table, positions,
        style=spec.p_style, only_significant=spec.p_only_significant,
        alpha=spec.p_alpha, use_corrected=spec.p_use_corrected, align=spec.p_align,
    )


def _numbers(text: str) -> list[float]:
    """The numbers in a comma- or space-separated string; anything else ignored."""
    out = []
    for part in str(text or "").replace(";", ",").split(","):
        for token in part.split():
            try:
                out.append(float(token))
            except ValueError:
                continue
    return out


def _limits(text: str) -> tuple[float, float] | None:
    values = _numbers(text)
    return (values[0], values[1]) if len(values) >= 2 else None


def _labels(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def _cycle(text: str, fallback: str) -> list[str]:
    items = [part.strip() for part in str(text or "").split(",") if part.strip()]
    return items or [fallback]


def _apply_ticks(ax, axis: str, positions: str, labels: str, rotation: float) -> None:
    values = _numbers(positions)
    names = _labels(labels)
    setter = ax.set_xticks if axis == "x" else ax.set_yticks
    text_of = ax.set_xticklabels if axis == "x" else ax.set_yticklabels
    stepped = str(positions or "").strip().lower().startswith("step")
    if stepped and values:
        locator = mticker.MultipleLocator(values[0])
        (ax.xaxis if axis == "x" else ax.yaxis).set_major_locator(locator)
    elif values:
        setter(values)
    if names:
        # Only replace the text when it can line up with the ticks; silently
        # relabelling the wrong ones would be worse than ignoring the request.
        current = ax.get_xticks() if axis == "x" else ax.get_yticks()
        if len(names) == len(current):
            text_of(names)
    if rotation:
        if axis == "x" and abs(rotation) > 45:
            # "Label\n(n=6)" turns with the label, so the second line ends up
            # beside the first and lands on the next group's. Upright labels
            # stack; turned ones run on.
            ticks = [t.get_text().replace("\n(n=", " (n=") for t in ax.get_xticklabels()]
            if any("\n" not in t for t in ticks):
                ax.set_xticklabels(ticks)
        ax.tick_params(axis=axis, rotation=rotation)


def style_axes(ax, spec: "PlotSpec") -> None:
    """Everything the spec says about the axes themselves.

    Applied after the data are drawn, so limits and ticks the user asked for
    win over whatever the data suggested.
    """
    # "log y" and the reference lines are about the *values*, which a
    # horizontal panel keeps on x.  Following the data round is the whole
    # contract of the flag: nothing about a turned panel should be retyped.
    value_axis = "x" if _is_horizontal(spec) else "y"
    if spec.log_x:
        ax.set_xscale("log")
    if spec.log_y and getattr(ax, f"get_{value_axis}scale")() != "log":
        getattr(ax, f"set_{value_axis}scale")("log")

    _apply_ticks(ax, "x", spec.xticks, spec.xtick_labels, spec.xtick_rotation)
    _apply_ticks(ax, "y", spec.yticks, spec.ytick_labels, spec.ytick_rotation)
    ax.tick_params(which="both", direction=spec.tick_direction or "out")
    if spec.minor_ticks:
        ax.minorticks_on()
    else:
        ax.minorticks_off()

    if spec.grid in ("x", "y", "both"):
        ax.grid(True, axis=spec.grid, color=theme.PRINT["grid"], lw=0.5, alpha=0.7)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)

    for value in _numbers(spec.ref_lines):
        (ax.axvline if value_axis == "x" else ax.axhline)(
            value, color=theme.PRINT["grid"], lw=0.6, ls=(0, (2, 3)), zorder=1)

    if spec.spines == "box":
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(True)
    elif spec.spines == "none":
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
    elif spec.spines == "zero":
        # An IV curve's origin is a real place on the plot: the axes cross at
        # it and the reversal potential is read off the crossing.
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_position("zero")
    else:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    if spec.spines == "zero":
        # With the axes crossing in the middle of the data, a centered axis
        # label lands on the plot. The lab's own script put them at the ends.
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), loc="right")
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), loc="top")
            ax.yaxis.label.set_rotation(0)
            ax.yaxis.label.set_horizontalalignment("left")
        # The tick labels at zero would sit on top of the crossing itself.
        for axis in (ax.xaxis, ax.yaxis):
            ticks = [t for t in axis.get_majorticklocs() if t != 0]
            axis.set_ticks(ticks)

    # Limits last: a bracket or a scale bar may have widened the view.
    x = _limits(spec.xlim)
    y = _limits(spec.ylim)
    if x:
        ax.set_xlim(*x)
    if y:
        ax.set_ylim(*y)
    legend = ax.get_legend()
    if legend is not None and spec.legend_loc and spec.legend_loc != "best":
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc=spec.legend_loc)


def figure_from_spec(
    spec: "PlotSpec | dict",
    data: "dict[str, Sequence[float]] | Sequence[dict]",
    *,
    paired: dict | None = None,
    comparisons=None,
    ax=None,
):
    """Draw the figure one :class:`PlotSpec` describes, p-values and all.

    ``data`` is ``{label: values}`` for the group kinds, or the list of
    ``{'label', 'x', 'mean', 'sem'}`` dicts for ``'xy'`` and ``'trace'``.
    ``comparisons`` overrides the statistics; left ``None`` for a group plot,
    the comparison is run from ``spec`` so the controls alone are enough.

    Returns ``(fig, ax, table)`` where *table* is the comparison table that was
    annotated, or ``None``.
    """
    spec = spec if isinstance(spec, PlotSpec) else PlotSpec.from_dict(spec)
    # First, so that everything below -- the colors, the drawing order, the
    # positions the brackets are hung from -- reads one sequence.
    data = reorder_data(data, spec.order)
    palette = colours_for(spec, data)
    table = None

    with publication_style(font_pt=spec.font_pt, font=spec.font,
                           figure__figsize=spec.figsize()):
        own_figure = ax is None
        if own_figure:
            fig, ax = plt.subplots(figsize=spec.figsize(), layout="constrained")
        else:
            fig = ax.figure

        if spec.kind == "xy":
            xy_plot(data, ax=ax, palette=palette, xlabel=spec.xlabel,
                    ylabel=spec.ylabel, title=spec.title, legend=spec.legend,
                    error=spec.error_style if spec.error != "none" else "none",
                    markers=_cycle(spec.markers, "") if spec.markers else [],
                    marker_size=spec.marker_size, line_width=spec.line_width,
                    line_styles=_cycle(spec.line_styles, "-"), join=spec.join,
                    # Spines at zero draw the crossing themselves.
                    zero_lines=spec.spines != "zero")
        elif spec.kind == "trace":
            trace_plot(data, ax=ax, palette=palette, title=spec.title,
                       legend=spec.legend, band=spec.error != "none",
                       error=spec.error_style if spec.error != "none" else "none",
                       line_width=spec.line_width,
                       line_styles=_cycle(spec.line_styles, "-"),
                       scale_bars=spec.spines in ("auto", "none"),
                       xlabel=spec.xlabel, ylabel=spec.ylabel)
        else:
            ax, positions = group_plot(
                data, kind=spec.kind, centre=spec.centre, error=spec.error,
                points=spec.points or False, paired=paired, palette=palette,
                ax=ax, ylabel=spec.ylabel, xlabel=spec.xlabel, title=spec.title,
                show_n=spec.show_n, log_y=spec.log_y, point_size=spec.point_size,
                horizontal=_is_horizontal(spec),
            )
            if spec.show_p:
                table = (comparisons_for(spec, data, pair_on=paired)
                         if comparisons is None else comparisons)
                annotate_from_spec(ax, spec, table, positions)
        style_axes(ax, spec)
    # Last, and outside the style block on purpose: it reads the layout that
    # everything above just settled, and that layout is settled again at draw
    # time -- out here, under the caller's rcParams, which is where the figure
    # gets saved.  Fitted inside the block, a bar panel asked for 2.00 in came
    # out 2.08.  See check_plot_sizing.py, which measures it from out here.
    if own_figure:
        # ``bar_mm`` sizes the axis the categories sit on, which is x upright
        # and y turned; the other axis keeps its own pinned number, and the
        # figure dimension across the categories is the ceiling.  One index
        # rather than two copies of the fit.
        sideways = _is_horizontal(spec)
        across = 1 if sideways else 0
        want = [spec.axes_width_mm, spec.axes_height_mm]
        ceiling = spec.height_mm if sideways else spec.width_mm
        # The value axis is a fixed length unless somebody typed one, for the
        # reason in AXES_LENGTH_MM.  'xy' and 'trace' have no categories, so
        # `across` is only ever the orientation there: the vertical axis.
        #
        # ``height_mm`` is a typed *figure* height and outranks the default --
        # it defaults to 0, so a number there was asked for.  ``width_mm`` is
        # not the same thing: it is always set, and documented as a column
        # ceiling rather than a target, so a turned panel still gets the pin.
        typed = spec.height_mm > 0 if not sideways else False
        if not want[1 - across] and not typed:
            want[1 - across] = AXES_LENGTH_MM
        if not want[across] and spec.bar_mm > 0 and spec.kind not in ("xy", "trace"):
            # The categories span that axis, so one bar is BAR_WIDTH of
            # (span / n).  Read the span rather than counting the groups: it
            # already carries whatever limit the spec asked for.
            limits = ax.get_ylim() if sideways else ax.get_xlim()
            want[across] = spec.bar_mm * abs(float(np.diff(limits)[0])) / BAR_WIDTH
        settled = fit_axes(fig, *want)
        # A column width is a ceiling, not a target.  Two groups at 6 mm a bar
        # is a panel a third of a column wide, and three of those side by side
        # are the figure; twenty groups would run off the page instead, so the
        # bars give way rather than the column.  Looped, because taking the
        # width off the panel puts some of it back as furniture: the tick
        # labels crowd and ask for more room than they did at the first fit.
        #
        # Turned, the ceiling is ``height_mm``, which is 0 unless somebody
        # typed one -- so a horizontal panel grows down the page with its
        # groups until it is told not to, which is the reason to turn it.
        for _ in range(3):
            over = settled[across] / MM - ceiling
            if not (want[across] and ceiling > 0 and over > 0.01):
                break
            want[across] = max(want[across] - over, 5.0)
            settled = fit_axes(fig, *want)
    return fig, ax, table
