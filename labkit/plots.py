"""Figures in the lab's publication style.

The house style, taken from the lab's own ephys module
(``working_stuff/Processing Scripts/voltage_clamp_processing.py``) and sized
as the lab's current figures draw: a value axis 23.6 mm long and 4.5 mm to a
category (:data:`AXES_LENGTH_MM`, :attr:`PlotSpec.bar_mm`), Arial, 6/7/8 pt, no top or right spine,
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
import functools
import inspect
import math
import os
import re
import warnings
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
    "PlotSpec", "figure_from_spec", "facet_figure", "hold_layout", "comparisons_for", "annotate_from_spec",
    "PALETTES", "FONTS", "VECTOR_TEXT", "TIGHT_CROP",
    "MM", "jitter", "AXES_LENGTH_MM", "CATEGORY_MM", "BOX_LW",
    "series_names", "order_series", "reorder_data",
]

#: 8 pt titles and axis titles; 7 pt ticks, categories, keys, n, scale bars and
#: protocol levels; 6 pt p values and stat notes -- nothing outside JCI's 6-8 pt.
TINY, SMALL, MEDIUM, LARGE = 6, 7, 8, 8

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
#: 23.6 mm is 0.93 in, the length the lab's current figures draw at -- the
#: ClC-7 2025 figure sets it in ephys, pHerret and purifig, panel by panel, and
#: purifig's quant panels default to it.  It was 30.48 mm, 1.2 in, the lab's
#: ephys notebook's ``figsize=(0.25*18, 1.2)``.  Set on the figures rather
#: than derived.  ``PlotSpec.panel_height_mm`` is the step
#: family's panel and keeps its own size -- a sweep of voltage steps is not a
#: bar chart and is read by its shape, not against a value axis.
#:
#: ``PlotSpec.axes_height_mm``/``axes_width_mm`` override it.  The *category*
#: axis is never pinned from here -- ``bar_mm`` sizes that one.
AXES_LENGTH_MM = 23.6

#: 4.5 mm to a category: ``PlotSpec.bar_mm`` over :data:`BAR_WIDTH`, the house
#: pitch.  The beeswarm needs it to know how wide a marker is in the category
#: units it lays points out in.
CATEGORY_MM = 4.5

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

    ``font_pt`` scales the whole 6/7/8 ladder at once, which is what the size
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
    gap: float = 3.0,
    tip: float = 2.0,
    linewidth: float = 0.8,
    fontsize: float | None = None,
    colour: str | None = None,
    dim_ns: bool = True,
    align: str = "stagger",
    outside: bool = False,
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
        ``'auto'`` (the exact number), ``'exact'``,
        ``'stars'`` or ``'both'``.  See :func:`labkit.stats.format_p`.
    ``align``
        ``'stagger'`` starts each bracket just above the data it spans, which
        is compact.  ``'level'`` puts every bracket at the same base height,
        which reads more evenly when the groups are of similar size.
    ``outside``
        Stack the brackets past the far edge of the axes and leave the value
        limit alone.  For a panel whose data area is pinned to a typed size:
        room made inside it comes out of the data, and five brackets on a
        0.93 in panel would leave the data a third of it.  ``align`` has no
        say there.
    ``horizontal``
        Categories down the y axis, brackets reaching right instead of up.
        Left ``None`` it is read back off the axes, which :func:`group_plot`
        marks -- so a caller that passes the table it always passed gets
        brackets the right way round without knowing about this argument.
    ``gap``
        Points from the data (or the edge) to the first bracket.  ``tip`` is
        kept for old callers and ignored: the lines have no ends.

    Against one control -- every comparison sharing a group -- there is no
    bracket at all: each p stands over (turned, beside) the other group.

    Brackets are stacked: a bracket takes the lowest row of the ladder clear
    of every bracket it overlaps, so a three-group figure with all three pairs
    annotated comes out as a readable ladder rather than three lines on top of
    each other.  A row is one label deep -- the ends of the bracket above, the
    text, a hairline -- so no label is crossed by the bracket over it; turned,
    the labels run along their brackets, which keeps a row one line deep
    there too.  The rows are in points, not axes fractions, because every
    caller pins its panel to a size *after* this runs: a fraction of the axes
    as they are now is a different distance on the page by the time the
    figure is saved.  Inside, the value limit is left to the data: the
    brackets stand on their groups and run past the end of the axes where
    they must (user, 2026-11-03), so the data keep the whole axis.  Nothing
    is clipped, and the layout makes room for whatever runs past.

    Returns the comparisons it actually drew, each with its ``row`` and the
    ``base`` it stands on inside (data units; 1.0, the edge, when outside).
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import text_to_path
    from matplotlib.transforms import (Affine2D, ScaledTranslation,
                                       blended_transform_factory)
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

    # Against one control -- every comparison shares one group, as "vs WT"
    # does -- each p goes over the other group on its own, with no line: the
    # control is said once, by the figure.  Otherwise a plain line between
    # the two groups, without ends.  Either way nothing hangs down to cross
    # the data.
    tip = 0.0
    shared = set.intersection(*({e["a"], e["b"]} for e in placeable))
    control = next(iter(shared)) if len(shared) == 1 else None
    if control is not None:
        spots = sorted(set(float(v) for v in positions.values()))
        half = 0.4 * (min(np.diff(spots)) if len(spots) > 1 else 1.0)
        for entry in placeable:
            other = entry["b"] if entry["a"] == control else entry["a"]
            at = _span_of(ax, other, positions)
            # Half a category either side, so the label clears the jittered
            # points of its group, not just the middle of it.
            entry["x1"], entry["x2"] = at - half, at + half

    # Short spans first: an adjacent pair should end up underneath the pair
    # that reaches across the whole axis, not the other way round.
    placeable.sort(key=lambda e: (e["x2"] - e["x1"], e["x1"]))

    colour = colour or theme.PRINT["rule"]
    fontsize = fontsize or mpl.rcParams["font.size"] * TINY / SMALL
    # One row of the ladder: the ends of the bracket above, a hairline, a line
    # of text, a hairline.
    lift, pad = 0.8, 1.0
    line = text_to_path.get_text_width_height_descent(
        "lp", FontProperties(size=fontsize), False)[1]
    if control is not None:
        # A label with no line stands beside its group (turned) or over it
        # reading upwards (upright, where "p = 0.001" is wider than its
        # 4.5 mm category), so a row is as deep as the longest one is long.
        line = max(text_to_path.get_text_width_height_descent(
            e["text"] or "", FontProperties(size=fontsize), False)[0] for e in placeable)
    step = tip + lift + line + pad

    # Every ceiling is measured before the first bracket exists.  Measuring as
    # we go would let each bracket see the one below it as data and raise
    # itself off that, so the ladder would climb twice as fast as it needs to.
    ceilings = {
        id(entry): _data_ceiling(ax, entry["x1"], entry["x2"], horizontal)
        for entry in placeable
    }
    common = max(ceilings.values())

    # A base is in data units and a row in points, so how far apart two bases
    # are on the page depends on the size the panel ends up.  A bracket
    # therefore stands on the highest base of anything it overlaps, and slots
    # *under* one only when the two share a base -- the one case where
    # "underneath" holds at every size.
    placed: list[tuple[dict, float, int]] = []
    for entry in placeable:
        base = 1.0 if outside else common if align == "level" else ceilings[id(entry)]
        # The span is padded so two brackets that merely touch at a shared
        # group still get separate rows.
        margin = 0.02 * max(1.0, abs(entry["x2"] - entry["x1"]))
        under = [(b, r) for e, b, r in placed
                 if not (entry["x2"] + margin < e["x1"] or entry["x1"] - margin > e["x2"])]
        base = max([base] + [b for b, _ in under])
        row = 0
        while any(row == r or (row < r and b != base) for b, r in under):
            row += 1
        placed.append((entry, base, row))

    # Each bracket is drawn in a frame of its own: the category in data units,
    # the magnitude in points from its base.  The transforms are live, so the
    # ladder keeps its shape whatever size the panel is pinned to afterwards.
    fig = ax.figure
    points = Affine2D().scale(1 / 72) + fig.dpi_scale_trans

    def _frame(base: float, anchor, across=ax.transData):
        up = points + ScaledTranslation(*_pair(0.0, base), anchor)
        return blended_transform_factory(*_pair(across, up))

    drawn: list[dict] = []
    marks: list[list] = []
    for entry, base, row in placed:
        faded = dim_ns and not entry["significant"]
        line_colour = theme.PRINT["muted"] if faded else colour
        bar = gap + row * step
        # Drawn in its frame from the start: plotted in data units, the
        # points would be taken for data and stretch the autoscaled limit.
        frame = _frame(1.0, ax.transAxes) if outside else _frame(base, ax.transData)
        mark = [] if control is not None else ax.plot(
            *_pair([entry["x1"], entry["x1"], entry["x2"], entry["x2"]],
                   [bar - tip, bar, bar, bar - tip]),
            transform=frame, clip_on=False, zorder=10,
            lw=linewidth, color=line_colour, solid_capstyle="butt",
        )
        if entry["text"]:
            # Turned, the label runs along its line.  Across it, a row would
            # have to be as deep as the label is long -- which a label with
            # no line, beside its own group, is.
            mark.append(ax.text(
                *_pair((entry["x1"] + entry["x2"]) / 2.0, bar + lift), entry["text"],
                transform=frame, clip_on=False, zorder=10,
                rotation=90 if horizontal == (control is None) else 0,
                ha="left" if horizontal else "center",
                va="center" if horizontal else "bottom", color=line_colour,
                fontsize=fontsize,             # grey says n.s.; smaller would go under 6 pt
            ))
            # spread_labels may slide it off a spine or a box, but only along
            # its line or outward: below, it would leave its bracket.
            mark[-1]._labkit_out = 0.0 if horizontal else np.pi / 2
            if control is None:
                mark[-1]._labkit_line = mark[0]     # and the line rises with it
            if align == "level" and control is None and not horizontal and not outside:
                mark[-1]._labkit_tier = line + pad  # tier_level_labels: how far up a text line is
        marks.append(mark)
        drawn.append({**entry, "base": base, "row": row})
    depth = gap + max(row for *_, row in placed) * step + lift + line

    def _anchor(out: bool) -> None:
        """Stand the ladder on the data, or every bracket of it on the edge."""
        for mark, (_, base, _) in zip(marks, placed):
            frame = _frame(1.0, ax.transAxes) if out else _frame(base, ax.transData)
            for artist in mark:
                artist.set_transform(frame)
        ladder = getattr(ax, "_labkit_ladder", None)
        if out and ladder is None:
            # Nothing on the axes says the ladder is there, so a title would
            # sit on it.  An empty child axes its size does: a title clears
            # the child axes, and tight and constrained layout make room.
            ladder = ax.inset_axes(_pair(0.0, 0.0) + _pair(1.0, depth),
                                   transform=_frame(1.0, ax.transAxes, ax.transAxes))
            ladder.set_axis_off()
            ax._labkit_ladder = ladder
        elif not out and ladder is not None:
            ladder.remove()
            del ax._labkit_ladder

    if outside:
        _anchor(True)
        return drawn

    # Run past the end of the axes, nothing on them says the brackets are
    # there, so a title would sit on them.  An empty child axes over them
    # does: a title clears the child axes, and the layout makes room.
    flat = [artist for mark in marks for artist in mark]
    if flat:
        from matplotlib.transforms import Bbox
        cover = ax.inset_axes((0, 0, 1, 1))
        cover.set_axis_off()
        cover.set_axes_locator(lambda _a, renderer: Bbox.union(
            [ax.bbox] + [a.get_window_extent(renderer) for a in flat]
        ).transformed(fig.transFigure.inverted()))
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


def _beeswarm(values: np.ndarray, width: float, span: float,
              point_size: float = 2.5) -> np.ndarray:
    """Offsets that lay points out sideways instead of on top of each other.

    Whether two points overlap is a physical question -- a marker is a fixed
    size on paper -- so the test is made in millimetres: ``span`` is the range
    the value axis will cover along its :data:`AXES_LENGTH_MM`, which says how
    many markers stack up it, and :data:`CATEGORY_MM` says how many fit across
    a category.  Points closer together than one marker are neighbours; each
    takes the column nearest the middle that its neighbours have left.

    Binning the values on their own range instead -- what this did until
    2026-11-04 -- put one point in almost every bin at the n these experiments
    run at, and a bin holding one point was given offset zero.  Fourteen
    normally distributed values came back as fourteen zeros: a strip plot.

    ``span`` is measured in whatever scale the axis is drawn in, so a log axis
    passes log values here and gets even spacing in the logarithm.

    ponytail: the value axis is taken to be the house :data:`AXES_LENGTH_MM`,
    because the figure is sized after it is drawn and the real length is not
    known yet here.  A panel given a longer one is spread a little more
    generously than it strictly needs, never less.  Thread the length through
    from ``PlotSpec`` if that ever shows.
    """
    values = np.asarray(values, float)
    if values.size <= 1 or not np.isfinite(span) or span <= 0:
        # One point, or every value the same: nothing to sort by, so scatter.
        return np.zeros(values.size) if values.size <= 1 else jitter(values.size, width, seed=1)
    mark = point_size / 72 * 25.4                     # marker diameter, mm
    gap = span * mark / AXES_LENGTH_MM                # ... in value units
    step = mark / CATEGORY_MM                         # ... in category units
    # Odd, so one column sits on the middle, and only as many as fit inside
    # the lane: rounded up, the outer ones stood in the next group's.  The
    # house bar has room for three at 2.5 pt.
    cols = int(width / step + 1e-9) + 1
    cols = max(1, cols if cols % 2 else cols - 1)
    lattice = (np.arange(cols) - (cols - 1) / 2) * step
    lattice = lattice[np.argsort(np.abs(lattice), kind="stable")]
    offsets = np.zeros(values.size)
    placed: list[tuple[float, float]] = []
    for i in np.argsort(values, kind="stable"):
        value = float(values[i])
        if not np.isfinite(value):
            continue
        near = [o for v, o in placed if abs(v - value) < gap]
        free = [o for o in lattice if all(abs(o - t) >= step - 1e-9 for t in near)]
        # ponytail: a pile deeper than the lane is wide reuses its emptiest
        # column rather than spilling into the next group. Widen the lane if
        # that shows.
        offsets[i] = free[0] if free else max(
            lattice, key=lambda o: min(abs(o - t) for t in near))
        placed.append((value, offsets[i]))
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
    markers: Sequence[str] = (),
    rename: "dict[str, str] | None" = None,
):
    """Groups side by side, with every data point on top of them.

    ``groups`` is ``{label: values}``, or a tidy DataFrame together with
    ``value=`` and ``group=``.

    ``kind``
        ``'bar'``, ``'box'``, ``'violin'`` or ``'none'`` (the estimate as a dot
        on its interval, as pHerret's estimate panel draws it, beside the
        points).  Histograms are :func:`hist_plot`.
        ``'none'`` is the honest default for small n and
        is what ``'bar'`` degrades to when a group has fewer than three values.
    ``points``
        ``'jitter'``, ``'swarm'``, ``'strip'`` (no offset) or ``False``.
    ``paired``
        ``{label: unit ids}`` parallel to ``groups``.  Given, the same unit is
        joined by a thin line across the groups, which is the whole argument
        for a paired test made visible.
    ``markers``
        Point shapes, cycled across the second part of two-level names: with
        ``("o", "^")`` every "/ 0 min" is a circle and every "/ 3 min" a
        triangle.  Plain names keep the circle.  Each shape is sized to the
        circle's area (:func:`_ms`).
    ``rename``
        ``{name: printed name}`` for the labels only, a whole name or one part
        of a two-level one; everything else still goes by the name.
    ``horizontal``
        The same panel turned a quarter turn: groups down the y axis, bars
        reaching right, first group at the top.  ``ylabel`` follows the values
        round to the x axis and ``xlabel`` follows the groups to the y one, so
        a panel keeps its labels when it is turned.  It is what long group
        names want -- ``R201C`` and ``rescue`` have a whole column to
        themselves instead of 4.5 mm of tick.

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
        else:                                            # 'none': the estimate only
            # pHerret's estimate panel (`plot_effect_forest`): the interval a
            # thick round-ended line and the center a larger dot on it, both in
            # the group's colour, the dot in a white halo.  Not a whisker -- it
            # is the estimate itself, so it has no caps and is not black line
            # art; Line2D, not an errorbar collection.  With them drawn the cells
            # and the estimate share the lane 0.17 of a row either side of its
            # middle, as pHerret's two levels do, rather than one on the other.
            at = x + 0.17 if points else x
            if err:
                ax.plot(*_pair([at, at], [height - err, height + err]), color=colour,
                        lw=2.0, solid_capstyle="round", zorder=7, clip_on=False)
            ax.plot(*_pair(at, height), "o", ms=point_size * 1.45, mfc=colour,
                    mec=theme.PRINT["point_edge"], mew=0.5, lw=0, zorder=8,
                    clip_on=False, gid="estimate")

    # --- the points, which are the point ----------------------------------
    offsets: dict[str, np.ndarray] = {}
    tiers = _two_level(levels)
    inner = list(dict.fromkeys(b for _, b in tiers)) if tiers else []
    if points:
        # The swarm asks how close two points may be before they touch, which
        # only the range of the whole axis can answer -- not one group's.
        seen = np.concatenate([np.asarray(data[lab], float) for lab in levels])
        seen = np.log10(seen) if log_y else seen
        seen = seen[np.isfinite(seen)]
        lo, hi = (float(seen.min()), float(seen.max())) if seen.size else (0.0, 0.0)
        if kind == "bar" and not log_y:
            lo, hi = min(lo, 0.0), max(hi, 0.0)       # a bar is drawn from zero
        span = hi - lo
        for i, label in enumerate(levels):
            values = data[label]
            # The points spread as wide as the bar: at the house 4.5 mm
            # category that is three markers abreast, and 0.55 of it was one.
            width = bar_width
            # Beside an estimate (`kind='none'`) the cells keep to their half
            # of the lane.
            lane = positions[label] - 0.17 if kind == "none" else positions[label]
            if kind == "none":
                width = 0.24
            if points == "swarm":
                dx = _beeswarm(np.log10(values) if log_y else values,
                               width, span, point_size)
            elif points == "strip":
                dx = np.zeros(values.size)
            else:
                dx = jitter(values.size, width, seed=seed + i * 97)
            offsets[label] = lane + dx
            # Filled in the group's colour with a hairline white halo, which
            # is what separates two dots that land on each other -- the lab
            # notebook's `edgecolor='w', linewidth=0.25`.  The box behind them
            # is an outline, so a filled dot covers nothing.
            shape = markers[inner.index(tiers[i][1]) % len(markers)] if tiers and markers else "o"
            colour = colours[i % len(colours)]
            filled = _area_of(shape) > 0           # an x or a + has no face to fill
            ax.plot(*_pair(offsets[label], values), shape, ms=_ms(shape, point_size),
                    mfc=colour, mec=theme.PRINT["point_edge"] if filled else colour,
                    mew=0.25 if filled else 0.8, lw=0, zorder=6, clip_on=False)

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
    names = [_shown(b, rename) for _, b in tiers] if tiers else [_shown(label, rename) for label in levels]
    # "0 min", "3 min" under 4.5 mm of tick: the numbers on the ticks and the
    # unit once, at the head of the row.
    unit = None
    if tiers and not horizontal:
        heads = [name.rsplit(" ", 1) for name in names]
        if all(len(h) == 2 and _is_number(h[0]) for h in heads) and len({h[1] for h in heads}) == 1:
            unit, names = heads[0][1], [h[0] for h in heads]
    labels = [
        f"{name}\n(n={len(data[label])})" if show_n else name
        for name, label in zip(names, levels)
    ]
    getattr(ax, f"set_{cat}ticklabels")(labels)
    if tiers:
        _outer_labels(ax, [(_shown(a, rename), b) for a, b in tiers], horizontal)
    if unit:
        # On the first number's baseline, 3 pt to its left.  The label is looked
        # up when drawing: a spine set at 0 makes new tick labels, and one held
        # from now stays where the axis used to be.
        size = ax.get_xticklabels()[0].get_fontsize()

        def baseline(renderer):
            from matplotlib.transforms import Bbox
            first = ax.get_xticklabels()[0]
            # Its first line's baseline as Text.draw puts it, from the font's
            # own ascent: worked out from "lp" it was 0.1 pt high, and hung from
            # the label's box 1.9 pt low in the SVG.
            _, lines, _ = first._get_layout(renderer)
            y = first.get_transform().transform(first.get_unitless_position())[1] + lines[0][2][1]
            x = first.get_window_extent(renderer).x0
            return Bbox([[x, y], [x, y]])

        # One of the tick labels, so the pass that keeps free text clear of
        # them leaves it be: it took the 3 pt gap for a touch and moved it down.
        ax.annotate(unit, xy=(0, 0), xycoords=baseline, xytext=(-3, 0), textcoords="offset points",
                    ha="right", va="baseline", annotation_clip=False,
                    fontsize=size)._labkit_tick = True
    edge = 0.5 + bar_width * 0.1
    if horizontal:
        # Reversed, so the first group is at the top: a list of groups reads
        # down the page, and drawing it upside down is the one thing everybody
        # notices about a horizontal bar chart somebody else made.
        ax.set_ylim(len(levels) - 1 + edge, -edge)
    else:
        ax.set_xlim(-edge, len(levels) - 1 + edge)
    if not log_y:
        _more_ticks(getattr(ax, f"{val}axis"))
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


def _more_ticks(axis) -> None:
    """Matplotlib's own locator, but never fewer than four ticks.

    It spaces ticks by the axis length, so a 20 mm value axis -- a small
    turned panel -- got two, which is a scale nobody can read off.  Long
    axes already get more and are left as they were.
    """
    axis.set_major_locator(mticker.MaxNLocator(
        # 4 among the steps: 0-1250 on a short axis is 0, 400, 800, 1200,
        # where the next step down, 250, would crowd six labels in.
        nbins="auto", steps=[1, 2, 2.5, 4, 5, 10], min_n_ticks=4))


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _two_level(levels) -> list[tuple[str, str]] | None:
    """``"WT / 0 min"`` as ``("WT", "0 min")``, when every name has two parts.

    ``None`` unless all of them split the same way and each first part is one
    run of neighbors, at least one of them more than a single group: only then
    is there anything to write once.
    """
    for sep in (" / ", ", "):
        parts = [str(label).split(sep) for label in levels]
        if not all(len(p) == 2 and all(x.strip() for x in p) for p in parts):
            continue
        outer = [p[0].strip() for p in parts]
        runs = [o for i, o in enumerate(outer) if i == 0 or outer[i - 1] != o]
        if len(runs) == len(set(runs)) and len(runs) < len(outer):
            return [(p[0].strip(), p[1].strip()) for p in parts]
    return None


def _outer_labels(ax, tiers, horizontal: bool) -> None:
    """Each first part once, under a line across its run of tick labels.

    Hung from the drawn tick labels rather than a fixed offset, so the n
    line, a rotation or a turned panel move it with them.
    """
    from matplotlib.transforms import Bbox

    texts = ax.get_yticklabels() if horizontal else ax.get_xticklabels()
    start = 0
    for i in range(1, len(tiers) + 1):
        if i < len(tiers) and tiers[i][0] == tiers[start][0]:
            continue
        run = texts[start:i]

        def under(renderer, span=(start, i)):
            # The labels drawn now, not the ones above: a spine set at 0
            # replaces them, and a held one stays where the axis used to be.
            live = (ax.get_yticklabels() if horizontal else ax.get_xticklabels())[slice(*span)]
            box = Bbox.union([t.get_window_extent(renderer) for t in live])
            gap = renderer.points_to_pixels(1.5)
            return box.translated(-gap, 0) if horizontal else box.translated(0, -gap)

        a, b = ((0, 0), (0, 1)) if horizontal else ((0, 0), (1, 0))
        ax.annotate("", xy=a, xycoords=under, xytext=b, textcoords=under,
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", lw=0.5, shrinkA=0, shrinkB=0,
                                    color=theme.PRINT["ink"]))
        ax.annotate(tiers[start][0], xy=(0, 0.5) if horizontal else (0.5, 0),
                    xycoords=under, xytext=(-1.5, 0) if horizontal else (0, -1.5),
                    textcoords="offset points", annotation_clip=False,
                    ha="right" if horizontal else "center",
                    va="center" if horizontal else "top",
                    rotation=90 if horizontal else 0,
                    fontsize=texts[start].get_fontsize() if run else None)
        start = i


def hist_plot(
    data: "dict[str, Sequence[float]]",
    *,
    ax=None,
    palette: Sequence[str] | None = None,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    show_n: bool = True,
    horizontal: bool = True,
    notes: "dict[str, str] | None" = None,
    rename: "dict[str, str] | None" = None,
):
    """Every group's histogram overlaid on one value axis.

    One set of bins for all of them, and each as the fraction of its own
    values, so a group of 15 cells and one of 8 compare bar for bar.  Outlined
    in the group's colour over a light fill of it, so where two overlap both
    edges still show.  ``horizontal`` keeps the value axis where a turned
    group plot has it, along x; upright it runs up y, as it does there.
    ``ylabel`` names the values and ``xlabel`` the fractions, following the
    same two axes round as they do in :func:`group_plot`.  ``notes`` adds a
    line to a group's legend entry -- its p against the control.
    """
    if ax is None:
        _, ax = plt.subplots()
    colours = list(palette) if palette else theme.SERIES
    clean = {k: np.asarray(v, float)[np.isfinite(np.asarray(v, float))] for k, v in data.items()}
    pooled = np.concatenate(list(clean.values()) or [np.zeros(0)])
    edges = np.histogram_bin_edges(pooled, bins="auto") if pooled.size else np.zeros(2)
    if len(edges) > 31:
        edges = np.linspace(edges[0], edges[-1], 31)
    turn = "vertical" if horizontal else "horizontal"
    for i, (label, values) in enumerate(clean.items()):
        colour = colours[i % len(colours)]
        share = np.histogram(values, edges)[0] / max(values.size, 1)
        name = f"{_shown(label, rename)} (n={values.size})" if show_n else _shown(label, rename)
        if notes and notes.get(label):
            name += f", {notes[label]}"
        ax.stairs(share, edges, orientation=turn, fill=True, color=colour,
                  alpha=0.15, lw=0, zorder=2)
        ax.stairs(share, edges, orientation=turn, color=colour, lw=1.0,
                  zorder=3, label=name)
    value, count = ("x", "y") if horizontal else ("y", "x")
    getattr(ax, f"set_{count}lim")(0, None)
    _more_ticks(getattr(ax, f"{value}axis"))
    getattr(ax, f"set_{value}label")(ylabel)
    getattr(ax, f"set_{count}label")(xlabel or "fraction")
    if title:
        ax.set_title(title)
    if clean:
        # Beside the axes, not in them: overlaid histograms leave no empty
        # corner to put a key in.
        ax.legend(frameon=False,
                  handlelength=1.2, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  borderaxespad=0.0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


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
    show_n: bool = True,
    rename: "dict[str, str] | None" = None,
):
    """Mean ± SEM against a continuous x, one line per group.

    An IV curve is the archetype, which is why the axes cross at the origin by
    default: for a current-voltage relation the origin is a real place on the
    plot and the reversal potential is read off it.

    Each entry of ``groups`` is ``{'label', 'x', 'mean', 'sem'}`` and may carry
    ``'n'``, and its own ``'marker'`` (``''`` for none) and ``'join'``, which
    win over the cycles: data as points with its fit as a line.  ``error`` is ``'band'``, ``'bars'``, ``'both'`` or ``'none'``.
    ``markers`` and ``line_styles`` are cycled across the curves, so groups
    stay apart in a figure printed in gray.  ``show_n`` and ``rename`` are as
    in :func:`hist_plot`.
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
        label = _shown(entry.get("label", f"group {i + 1}"), rename)
        if n_max and show_n:
            label = f"{label} (n={n_max})"

        shape = entry.get("marker", shapes[i % len(shapes)] if shapes else "")
        joined = entry.get("join", join)

        band = np.where(np.isfinite(sem), sem, 0.0)
        # No spread given is no band, rather than one of no width -- or a cap
        # drawn on every point of a curve that has no error to show.
        spread = band[good].any()
        if spread and error in ("band", "both"):
            ax.fill_between(x[good], (mean - band)[good], (mean + band)[good],
                            color=colour, alpha=0.18, linewidth=0, zorder=2)
        if spread and error in ("bars", "both"):
            # No caps on a curve: at every point of a dense I/V they read as
            # a second set of markers.
            ax.errorbar(x[good], mean[good], yerr=band[good],
                        fmt="none", ecolor=colour, elinewidth=0.7, capsize=0, zorder=3)
        if joined:
            ax.plot(x[good], mean[good], dashes[i % len(dashes)], color=colour,
                    lw=line_width, label=label, zorder=4)
        if shape:
            ax.plot(x[good], mean[good], shape, ms=_ms(shape, marker_size), color=colour,
                    mfc=colour, mew=0 if _area_of(shape) > 0 else 0.8 * line_width,
                    zorder=5, ls="none", label=None if joined else label)

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
    # The values only: a voltage axis at -100, 0, 100 is the convention, and
    # with the axis at zero current its labels sit among the curves.
    _more_ticks(ax.yaxis)
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
    rename: "dict[str, str] | None" = None,
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
        shown = band and spread[good].any()      # none given: no band of no width
        if shown and error in ("band", "both"):
            ax.fill_between(x[good], (mean - spread)[good], (mean + spread)[good],
                            color=colour, alpha=0.16, linewidth=0, zorder=2)
        if shown and error in ("bars", "both"):
            # A bar per sample would be a smear; a handful along the trace is
            # readable and says the same thing.
            step = max(1, int(good.sum() // 12))
            picked = np.flatnonzero(good)[::step]
            ax.errorbar(x[picked], mean[picked], yerr=spread[picked], fmt="none",
                        ecolor=colour, elinewidth=0.6, capsize=0, zorder=3)
        label = _shown(entry.get("label", f"trace {i + 1}"), rename)
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
    by = (spec.family_colour or "black").strip()
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


#: Gaps between the panels of a step family, in mm.  Panels that differ only
#: in their last key -- one construct at 0 and 3 min -- are a matched pair and
#: sit almost touching; a new construct gets a gap that says so.
FAMILY_GAP_IN_MM = 0.8
FAMILY_GAP_OUT_MM = 2.5
FAMILY_ROW_GAP_MM = 1.0
#: The protocol drawn under a family: as wide as a panel, this tall -- room
#: for its 0 mV and holding labels between the highest and lowest step.
FAMILY_PROTOCOL_MM = 10.0


def _active_window(panels: Sequence[dict], pad: float = 0.04) -> tuple[float, float] | None:
    """The stretch of the sweep where something happens, in ms.

    The holding before the first step and after the tail has relaxed is the
    same flat line in every sweep: on a published family it is most of the
    width and says nothing.  Kept is where the sweeps differ from each other
    -- or, for a panel of one sweep, from its own start -- by more than that
    panel's own noise and 3 % of its own excursion, plus a little either side
    so the step can be seen to start from somewhere.  One window for every
    panel, since they share a time axis.  None when nothing stands out.

    Each panel against itself, because against the figure's largest panel 3 %
    of a small figure's top sits *under* a quiet panel's own baseline scatter,
    and that panel then reports the whole sweep as active -- which is how a
    3 s td family came to be drawn untrimmed (2026-11-04).
    """
    found = []
    for panel in panels:
        time = np.asarray(panel["time_ms"], float)
        means = [np.asarray(s["mean"], float)[:time.size] for s in panel["sweeps"]]
        means = [m for m in means if m.size == time.size]
        if not means or time.size < 3:
            continue
        stack = np.vstack(means)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if len(means) > 1:
                signal = np.nanmax(stack, 0) - np.nanmin(stack, 0)
            else:
                signal = np.abs(stack[0] - np.nanmedian(stack[0][:max(3, time.size // 50)]))
        # One sample over the line -- a switch click, one sweep's glitch
        # surviving the mean -- is not something happening, and since the
        # window runs out to the outermost crossing, a single one of them
        # opened a 3 s sweep to its full width.  A 5-sample median drops it;
        # a step edge is hundreds of ms wide and does not move.
        signal = np.median(np.lib.stride_tricks.sliding_window_view(
            np.pad(np.nan_to_num(signal), 2, mode="edge"), 5), axis=-1)
        found.append((time, signal))
    if not found:
        return None
    low, high = np.inf, -np.inf
    for time, signal in found:
        base = float(np.median(signal))          # the holding: most of a sweep
        floor = 6 * float(np.median(np.abs(signal - base)))   # 6 MAD, ~4 sigma
        rise = float(np.percentile(signal, 99.5)) - base
        if not rise > floor:
            continue              # noise only: nothing for a window to hold on
        on = np.flatnonzero(signal > base + max(floor, 0.03 * rise))
        if on.size:
            low, high = min(low, time[on[0]]), max(high, time[on[-1]])
    if not high > low:
        return None
    first = min(float(np.nanmin(t)) for t, _ in found)
    last = max(float(np.nanmax(t)) for t, _ in found)
    margin = pad * (high - low)
    return max(low - margin, first), min(high + margin, last)


def _family_groups(panels: Sequence[dict]) -> list[list[int]]:
    """Runs of panels that share every key but the last: matched pairs."""
    groups: list[list[int]] = []
    last = object()
    for index, panel in enumerate(panels):
        keys = list((panel.get("keys") or {}).values())
        outer = tuple(keys[:-1]) if len(keys) > 1 else (index,)
        if groups and outer == last:
            groups[-1].append(index)
        else:
            groups.append([index])
        last = outer
    return groups


def _family_canvas(panels: Sequence[dict], spec: "PlotSpec", cols: int, name_lines: int,
                   group_titles: Sequence[str] | None = None, protocol_mm: float = 0.0,
                   protocol_left_mm: float = 0.0):
    """Axes placed in mm: a matched pair close, the next group a gap away.

    Rows are filled group by group, and a group that does not fit on what is
    left of a row starts the next one rather than being split -- unless it is
    wider than a row on its own.  ``group_titles``, one per group, are set
    once over the whole group, above the panels' own titles -- ``name_lines``
    of them at the figure's size over the n at ``spec.n_pt``.  No layout
    engine: every position is known, so nothing moves at save time.  Returns
    ``(fig, axes, places)`` with each panel's (row, column).  With
    ``protocol_mm``, one more axes is appended to ``axes``: that tall, a panel
    wide, under the first column, a line clear of the scale bar's label, and
    ``protocol_left_mm`` in from the edge for the labels on its left.
    """
    line_mm = spec.font_pt * 1.2 / 72 * 25.4
    n_mm = spec.n_pt * 1.2 / 72 * 25.4
    title_mm = name_lines * line_mm + n_mm + 0.8
    width = spec.axes_width_mm if spec.axes_width_mm > 0 else max(spec.panel_width_mm, 8.0)
    # The panel size is the traces' own: a title that wraps adds above them
    # rather than squeezing them.
    height = spec.axes_height_mm if spec.axes_height_mm > 0 else max(spec.panel_height_mm, 8.0)
    rows: list[list[tuple[int, float]]] = [[]]     # (panel, left edge in mm)
    group_of: dict[int, int] = {}
    used = 0
    for g, group in enumerate(_family_groups(panels)):
        group_of.update((index, g) for index in group)
        if rows[-1] and used + len(group) > cols:
            rows.append([])
            used = 0
        for k, index in enumerate(group):
            if rows[-1] and used >= cols:
                rows.append([])
                used = 0
            row = rows[-1]
            left = 0.0 if not row else row[-1][1] + width + (
                FAMILY_GAP_IN_MM if k else FAMILY_GAP_OUT_MM)
            row.append((index, left))
            used += 1
    fig_w = max(row[-1][1] + width for row in rows if row)
    fig_h = len(rows) * (title_mm + height) + (len(rows) - 1) * FAMILY_ROW_GAP_MM
    below = protocol_mm + FAMILY_ROW_GAP_MM + line_mm if protocol_mm else 0.0
    fig_h += below
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))
    axes, places = [None] * len(panels), [None] * len(panels)
    for r, row in enumerate(rows):
        bottom = fig_h - r * (title_mm + height + FAMILY_ROW_GAP_MM) - title_mm - height
        for c, (index, left) in enumerate(row):
            axes[index] = fig.add_axes((left / fig_w, bottom / fig_h, width / fig_w, height / fig_h))
            places[index] = (r, c)
        if group_titles:
            spans: dict[int, list[float]] = {}
            for index, left in row:
                span = spans.setdefault(group_of[index], [left, left + width])
                span[1] = left + width
            # Over the panel titles, which take the lines below it.
            top = bottom + height + 0.7 + n_mm + (name_lines - 1) * line_mm
            for g, (left, right) in spans.items():
                fig.text((left + right) / 2 / fig_w, top / fig_h, group_titles[g],
                         ha="center", va="bottom", fontsize=spec.font_pt)
    if protocol_mm:
        axes.append(fig.add_axes((protocol_left_mm / fig_w, 0.0, width / fig_w, protocol_mm / fig_h)))
    return fig, axes, places


def _family_xlim(spec: "PlotSpec", panels: Sequence[dict]) -> tuple[float, float] | None:
    """The time window: ``spec.xlim`` with both ends, else the active stretch,
    else the whole sweep -- the protocol needs one to be drawn on."""
    x_limits = _limits(spec.xlim)
    if x_limits and None not in x_limits:
        return x_limits
    found = _active_window(panels)
    if found is None:
        times = [t for p in panels for t in (np.nanmin(p["time_ms"]), np.nanmax(p["time_ms"]))
                 if len(p["time_ms"])]
        found = (float(min(times)), float(max(times))) if times else None
    return found


def _family_range(panels: Sequence[dict], x_limits) -> tuple[float, float] | None:
    """The y range the panels' traces take inside the window, padded 6 %."""
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
    if not lows:
        return None
    low, high = min(lows), max(highs)
    pad = 0.06 * (high - low or 1.0)
    return low - pad, high + pad


def _family_traces(ax, panel: dict, colours, x_limits, y_limits, spec: "PlotSpec", lw: float):
    """One panel's mean sweeps, its limits and its dotted zero line."""
    time = np.asarray(panel["time_ms"], float)
    for colour, sweep in zip(colours, panel["sweeps"]):
        mean = np.asarray(sweep["mean"], float)
        good = np.isfinite(mean) & np.isfinite(time)
        if x_limits:
            good &= (time >= x_limits[0]) & (time <= x_limits[1])
        if spec.family_spread:
            sem = np.nan_to_num(np.asarray(sweep.get("sem", np.zeros_like(mean)), float))
            ax.fill_between(time[good], (mean - sem)[good], (mean + sem)[good],
                            color=colour, alpha=0.15, lw=0)
        ax.plot(time[good], mean[good], color=colour, lw=lw)
    if x_limits:
        ax.set_xlim(*x_limits)
    if y_limits:
        ax.set_ylim(*y_limits)
    low, high = ax.get_xlim()
    # The zero line is a reference, so it stays under the data's weight.
    ax.hlines(0, low, high, linestyles=(0, (1, 2)), lw=lw, color=theme.PRINT["ink"])


def _mv(level: float) -> str:
    return "0 mV" if round(level) == 0 else f"{level:+.0f} mV".replace("-", "−")


def _protocol_labels(command: Sequence[dict], xlim) -> list[tuple[float, float, str]]:
    """Where a protocol is labeled: ``(ms, mV, side)``, side one of left,
    right, above, below.

    A stretch whose level changes from sweep to sweep is the step, labeled at
    its highest level over it and its lowest under it.  A stretch at one
    level -- the holding, the tail -- is labeled once per level: the first
    on the left, later ones at their right end.  Stretches outside the time
    window are not drawn, so not labeled either.  Timing is the first sweep's.
    """
    corners = command[0]["time_ms"]
    low_x, high_x = xlim
    out, done, first = [], [], True
    for j in range(len(corners) // 2):
        a, b = corners[2 * j], corners[2 * j + 1]
        if b <= low_x or a >= high_x:
            continue
        a, b = max(a, low_x), min(b, high_x)
        levels = [step["mv"][2 * j] for step in command]
        top, bottom = max(levels), min(levels)
        if top - bottom > 0.5:
            out += [((a + b) / 2, top, "above"), ((a + b) / 2, bottom, "below")]
            done += [top, bottom]
        elif all(abs(top - d) > 0.5 for d in done):
            out.append((a, top, "left") if first else (b, top, "right"))
            done.append(top)
        first = False
    return out


def _text_mm(text: str, pt: float) -> float:
    """A label's width in mm at ``pt``, in the font in force."""
    from matplotlib.textpath import TextPath
    return TextPath((0, 0), text, size=pt).get_extents().width / 72 * 25.4


def _protocol_left_mm(command, spec: "PlotSpec", xlim) -> float:
    """Room the labels on a protocol's left need, in mm."""
    if not command or xlim is None:
        return 0.0
    left = [_text_mm(_mv(v), spec.font_pt * SMALL / MEDIUM) for _, v, side in _protocol_labels(command, xlim)
            if side == "left"]
    # Ink width plus the 2 pt offset and the glyphs' side bearings.
    return max(left) + 1.5 if left else 0.0


def _draw_protocol(ax, command: Sequence[dict], xlim, spec: "PlotSpec", lw: float):
    """The protocol on the traces' time axis, its levels labeled, a dotted 0 mV."""
    ink = theme.PRINT["ink"]
    for step in command:
        ax.plot(step["time_ms"], step["mv"], color=ink, lw=lw)
    ax.set_xlim(xlim)
    levels = [v for step in command for v in step["mv"]]
    top, bottom = max(levels), min(levels)
    ax.set_ylim(bottom - 0.05 * (top - bottom or 1), top + 0.05 * (top - bottom or 1))
    ax.set_axis_off()
    labels = _protocol_labels(command, xlim)
    if bottom < 0 < top and all(abs(v) > 0.5 for _, v, _ in labels):
        ax.hlines(0, *xlim, linestyles=(0, (1, 2)), lw=lw, color=ink)
        labels.append((xlim[1], 0.0, "right"))
    place = {"left": ((-2, 0), "right", "center"), "right": ((2, 0), "left", "center"),
             "above": ((0, 1.5), "center", "bottom"), "below": ((0, -1.5), "center", "top")}
    for x, level, side in labels:
        offset, ha, va = place[side]
        ax.annotate(_mv(level), (x, level), xytext=offset, textcoords="offset points",
                    ha=ha, va=va, fontsize=spec.font_pt * SMALL / MEDIUM, annotation_clip=False)


def _family_layout_plot(panels: Sequence[dict], spec: "PlotSpec", unit: str):
    """Step families placed by name, as ``spec.family_layout`` lays them out.

    Each name is a construct: its panels side by side, its name (in its
    color from ``spec.colours``) and n centered over them.  What tells the
    panels of one construct apart -- "pH 7.4", "pH 4.5" -- is said once, over
    the top construct of each column.  Every row is only as tall as its own
    currents at the one scale all rows share, so the scale bar holds for each
    and a row of flat traces takes almost no room.  A typed ``spec.ylim``
    sets that scale -- the panel height is what its whole range takes -- so
    figures given the same range and height draw a current the same length;
    a row still takes only its own currents' height.  "protocol" draws the
    protocol at a panel's width and time scale, under the scale bar, in a
    place that runs down past the rows that leave it empty; a name not
    found leaves its place empty.  Returns ``(fig, axes)``, the axes of the
    panels drawn.
    """
    groups = _family_groups(panels)
    keys = [list((p.get("keys") or {}).items()) for p in panels]
    renames = _renames(spec.rename)

    def head(g):
        first = keys[g[0]]
        return ", ".join(str(v) for _, v in first[:-1]) if len(first) > 1 else str(panels[g[0]]["label"])

    found = {}
    for g in groups:
        found.setdefault(head(g), g)
        found.setdefault(_shown(head(g), renames), g)
    layout = []
    for row in re.split(r"[;\n]", spec.family_layout):
        if row.strip():
            layout.append([("protocol" if name.strip().lower() == "protocol" else found.get(name.strip()))
                           for name in row.split("|")])
    drawn = [i for row in layout for g in row if isinstance(g, list) for i in g]
    if not drawn:
        raise ValueError("None of the names in the layout is a panel here: "
                         + ", ".join(sorted({head(g) for g in groups})))
    command = next((panels[i]["command"] for i in drawn if panels[i].get("command")), None)

    def column_name(i):
        key, value = keys[i][-1] if keys[i] else ("", panels[i]["label"])
        value = str(value)
        if value in renames:            # "74 = 161 mM": said as the rename says
            return renames[value]
        # A bare number needs its quantity: "7.4" is "pH 7.4".
        text = f"{key} {value}" if key and re.fullmatch(r"[-+]?\d+(\.\d+)?", value) else value
        return _shown(text, renames)

    with publication_style(font_pt=spec.font_pt, font=spec.font):
        line_mm = spec.font_pt * 1.2 / 72 * 25.4
        n_mm = spec.n_pt * 1.2 / 72 * 25.4
        width = spec.axes_width_mm if spec.axes_width_mm > 0 else max(spec.panel_width_mm, 8.0)
        height = spec.axes_height_mm if spec.axes_height_mm > 0 else max(spec.panel_height_mm, 8.0)
        per = max(len(g) for row in layout for g in row if isinstance(g, list))
        slot = per * width + (per - 1) * FAMILY_GAP_IN_MM
        x_limits = _family_xlim(spec, [panels[i] for i in drawn])
        fixed = _limits(spec.ylim)
        fixed = fixed if fixed and None not in fixed else None
        whole = fixed or _family_range([panels[i] for i in drawn], x_limits) or (-1.0, 1.0)
        scale = height / (whole[1] - whole[0])          # mm per unit, every row

        # Each row: its y range, the traces' height, the titles over them.
        top_of = {}
        for r, row in enumerate(layout):
            for k, g in enumerate(row):
                if isinstance(g, list):
                    top_of.setdefault(k, r)
        ranges, data_h, title_h = [], [], []
        for r, row in enumerate(layout):
            members = [panels[i] for g in row if isinstance(g, list) for i in g]
            span = _family_range(members, x_limits) if members else None
            if span and fixed:          # a typed range sets the scale and clips
                span = (max(span[0], fixed[0]), min(span[1], fixed[1]))
            if span is None or span[1] <= span[0]:
                ranges.append(None), data_h.append(0.0), title_h.append(0.0)
                continue
            h = max((span[1] - span[0]) * scale, 1.5)
            middle = (span[0] + span[1]) / 2
            ranges.append((middle - h / scale / 2, middle + h / scale / 2))
            data_h.append(h)
            header = any(top_of.get(k) == r for k, g in enumerate(row) if isinstance(g, list))
            title_h.append(line_mm * (1 + header) + n_mm + 0.8)
        gaps = [FAMILY_ROW_GAP_MM] * len(layout)

        # The protocol's place: its row down through the rows that leave it empty.
        where = next(((r, row.index("protocol")) for r, row in enumerate(layout) if "protocol" in row), None)
        y_len = spec.scale_y or _nice((whole[1] - whole[0]) / 5.0)
        # The bar's labels, 0.75 of the type: time 3.5 pt under it, the current
        # upright 2 pt left of it -- a label longer than the bar overhangs
        # both its ends, so that much is kept clear above and below.
        bar_pt = spec.font_pt * SMALL / MEDIUM
        bar_line = bar_pt * 1.2 / 72 * 25.4
        gap_mm = 2 / 72 * 25.4
        over = max(0.0, (_text_mm(f"{_fmt(y_len)} {unit}", bar_pt) - y_len * scale) / 2)
        bar_block = over + y_len * scale + max(over, 3.5 / 72 * 25.4 + bar_line)
        bar_inset = gap_mm + bar_line + 0.3        # the upright label's width, left of the bar
        if where and command:
            r0, k0 = where
            last = r0
            while last + 1 < len(layout) and not (len(layout[last + 1]) > k0 and layout[last + 1][k0] is not None):
                last += 1
            bar_mm = bar_block + 0.5                    # the bar and its labels, a gap
            need = bar_mm + 2 * line_mm + FAMILY_PROTOCOL_MM
            have = sum(title_h[r] + data_h[r] for r in range(r0, last + 1)) + sum(gaps[r0:last])
            gaps[last] += max(need - have, 0.0)
        else:
            where = None

        tops, y = [], 0.0
        for r in range(len(layout)):
            tops.append(y)
            y += title_h[r] + data_h[r] + gaps[r]
        fig_h = y - FAMILY_ROW_GAP_MM       # the last gap's extra, if any, is the protocol's
        if not where:
            # No protocol to stand over: the scale bar goes under the last row,
            # at the left, its time label below it.
            bar_top = fig_h + FAMILY_ROW_GAP_MM + over
            fig_h = bar_top - over + bar_block
        fig_w = max(len(row) * slot + (len(row) - 1) * FAMILY_GAP_OUT_MM for row in layout)
        fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))

        def box(left, top, w, h):     # mm from the top-left, to figure fractions
            return (left / fig_w, (fig_h - top - h) / fig_h, w / fig_w, h / fig_h)

        def say(x, top, text, **kw):
            return fig.text(x / fig_w, (fig_h - top) / fig_h, text, ha="center", va="baseline", **kw)

        trace_lw = spec.line_width * 0.2
        colours = _family_colours(panels, spec)
        axes = {}
        for r, row in enumerate(layout):
            for k, g in enumerate(row):
                if not isinstance(g, list):
                    continue
                left = k * (slot + FAMILY_GAP_OUT_MM)
                axes_top = tops[r] + title_h[r]
                for j, i in enumerate(g):
                    x0 = left + j * (width + FAMILY_GAP_IN_MM)
                    ax = fig.add_axes(box(x0, axes_top, width, data_h[r]))
                    _family_traces(ax, panels[i], colours[i], x_limits, ranges[r], spec, trace_lw)
                    ax.set_axis_off()
                    axes[i] = ax
                    if top_of.get(k) == r:
                        say(x0 + width / 2, axes_top - 0.7 - n_mm - line_mm, column_name(i),
                            fontsize=spec.font_pt)
                middle = left + (len(g) * width + (len(g) - 1) * FAMILY_GAP_IN_MM) / 2
                ns = sorted({int(panels[i]["n"]) for i in g})
                say(middle, axes_top - 0.7, "n = " + "/".join(map(str, ns)), fontsize=spec.n_pt)
                name = head(g)
                outline(say(middle, axes_top - 0.7 - n_mm, _shown(name, renames),
                            fontsize=spec.font_pt, fontweight="bold",
                            color=(spec.colours or {}).get(name, theme.PRINT["ink"])))

        first = axes[drawn[0]]
        if where:
            r0, k0 = where
            left = k0 * (slot + FAMILY_GAP_OUT_MM)
            top = tops[r0] + bar_mm + line_mm
            pad = max(_protocol_left_mm(command, spec, x_limits), 9.0)
            proto = fig.add_axes(box(left + pad, top, width, FAMILY_PROTOCOL_MM))
            _draw_protocol(proto, command, x_limits, spec, trace_lw)
            bar_left, bar_top = left + bar_inset, tops[r0] + over
        else:
            bar_left = bar_inset
        # The scale bar at the traces' scale, in axes exactly its height:
        # taller ones stuck out over the figure's edge and the saved figure
        # grew a blank band to hold them.
        bar = fig.add_axes(box(bar_left, bar_top, width, y_len * scale))
        bar.set_xlim(x_limits or first.get_xlim())
        bar.set_ylim(0, y_len)
        bar.set_axis_off()
        add_scale_bars(bar, x_length=spec.scale_x_s or None, y_length=y_len, x_unit="s",
                       y_unit=unit, x_scale=1000.0, position=(0.0, 0.0), fontsize=bar_pt,
                       y_label_side="outside")
    return fig, [axes[i] for i in drawn]


def family_plot(panels: Sequence[dict], spec: "PlotSpec", unit: str = "pA/pF"):
    """Averaged step families, one panel per group, every step overlaid.

    Each panel is ``{'label', 'n', 'time_ms', 'sweeps': [{'voltage_mv',
    'mean', 'sem'}]}``, with a ``'keys'`` of the values it was grouped by if
    the traces are to be colored by one. Panels share one y range so their
    currents can be compared by eye, which is also why the scale bar is drawn
    once, on the first. Returns ``(fig, axes)``.

    Drawn for print, so as compact as it reads: the holding either side of
    the protocol is cut (:func:`_active_window`) unless ``spec.xlim`` says
    otherwise, each title puts its n on a line of its own, and with scale bars
    -- the lab's convention -- panels of a matched pair nearly touch while
    the next construct sits a gap away (:func:`_family_canvas`).

    ``spec.family_protocol`` draws the protocol under the first column, a
    panel wide and on its time axis, so the time scale bar serves it too: the
    first panel's ``'command'``, ``[{'voltage_mv', 'time_ms', 'mv'}]`` corner
    points, labeled as :func:`_protocol_labels` says.

    ``spec.family_layout`` places the constructs by name instead:
    :func:`_family_layout_plot`.

    The publication style is applied here, so a caller neither needs to wrap
    this nor should: the axes size is settled after the style block exits,
    where the saving happens, and a block around this call would put the
    two back on opposite sides of it.
    """
    if spec.order:
        # By the panel's name or, for "WT / 74", its construct: the order is
        # typed once, by construct, and a pair keeps its own sequence.
        rank = {str(o): i for i, o in enumerate(spec.order)}

        def place(panel):
            head = next(iter((panel.get("keys") or {}).values()), panel["label"])
            return rank.get(str(panel["label"]), rank.get(str(head), len(rank)))
        panels = sorted(panels, key=place)
    scale_bars = spec.spines in ("auto", "none")
    if spec.family_layout.strip() and scale_bars:
        return _family_layout_plot(panels, spec, unit)
    with publication_style(font_pt=spec.font_pt, font=spec.font):
        rows, cols = _panel_grid(len(panels), spec.panel_cols)
        # A matched pair says its construct once, over both panels, and each
        # panel only what differs -- "0 min" over "n = 6" -- so the panels can be
        # as narrow as their traces.
        groups = _family_groups(panels)
        paired = scale_bars and any(len(g) > 1 for g in groups)
        heads = None
        names = [panel["label"] for panel in panels]
        if paired:
            keys = [list((panel.get("keys") or {}).values()) for panel in panels]
            heads = [", ".join(str(v) for v in keys[g[0]][:-1]) for g in groups]
            names = [str(k[-1]) if k else panel["label"] for k, panel in zip(keys, panels)]
        # The n on a line of its own, smaller: a name and its n side by side
        # are wider than a 20 mm panel and run into the next title.
        # ponytail: under the scale-bar canvas only; the axes-and-ticks layout
        # has no row for it.
        command = next((p["command"] for p in panels if p.get("command")), None) \
            if spec.family_protocol and scale_bars else None
        if scale_bars:
            fig, axes, places = _family_canvas(
                panels, spec, cols, 1 + bool(paired), heads,
                protocol_mm=FAMILY_PROTOCOL_MM if command else 0.0,
                protocol_left_mm=_protocol_left_mm(command, spec, _family_xlim(spec, panels))
                if command else 0.0)
            protocol_ax = axes.pop() if command else None
        else:
            width_in = max(spec.panel_width_mm, 8.0) * MM * cols
            height_in = max(spec.panel_height_mm, 8.0) * MM * rows
            fig, grid = plt.subplots(rows, cols, figsize=(width_in, height_in), squeeze=False,
                                     layout="constrained")
            fig.get_layout_engine().set(w_pad=1 / 72, h_pad=1 / 72, wspace=0, hspace=0)
            axes = list(grid.ravel())
            places = [divmod(i, cols) for i in range(len(axes))]

        x_limits = _family_xlim(spec, panels)
        y_limits = _limits(spec.ylim)
        y_limits = y_limits if y_limits and None not in y_limits else _family_range(panels, x_limits)

        # A family draws a dozen sweeps over each other, so its traces are drawn
        # a fifth of the house weight -- any heavier and the overlap fills in.
        trace_lw = spec.line_width * 0.2
        per_panel = _family_colours(panels, spec)
        for index, (ax, panel) in enumerate(zip(axes, panels)):
            _family_traces(ax, panel, per_panel[index], x_limits, y_limits, spec, trace_lw)
            # Both on baselines: the n 2 pt over the traces, the name a line over it.
            outline(ax.set_title(_shown(names[index], _renames(spec.rename)),
                                 fontsize=spec.font_pt, pad=2 + spec.n_pt * 1.2,
                                 fontweight="bold",
                                 color=(spec.colours or {}).get(names[index], theme.PRINT["ink"])))
            ax.annotate(f"n = {panel['n']}", (0.5, 1), xycoords="axes fraction", xytext=(0, 2),
                        textcoords="offset points", ha="center", va="baseline",
                        fontsize=spec.n_pt)
            if scale_bars:
                # The lab's convention: no axes, and one scale bar for the lot,
                # which the shared y range makes true of every panel.
                ax.set_axis_off()
                if index == 0:
                    add_scale_bars(ax, x_length=spec.scale_x_s or None, y_length=spec.scale_y or None,
                                   x_unit="s", y_unit=unit, x_scale=1000.0,
                                   fontsize=spec.font_pt * SMALL / MEDIUM)
            else:
                row, col = places[index]
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
        if command:
            _draw_protocol(protocol_ax, command, axes[0].get_xlim(), spec, trace_lw)
    if not scale_bars:
        # Outside the style block on purpose.  Constrained layout settles the
        # axes box at draw time and reads the tick label size from the
        # rcParams in force *then*, so a figure fitted under the publication
        # style and saved under the caller's comes out a couple of percent
        # off the size it was asked for -- silently, because fit_axes never
        # measures again after its last resize.  Fit where the saving happens
        # and the number lands.  (The scale-bar canvas is placed in mm and
        # has nothing to fit.)
        fit_axes(fig, spec.axes_width_mm, spec.axes_height_mm)
        if spec.axes_width_mm > 0 or spec.axes_height_mm > 0:
            hold_layout(fig)
    spread_labels(fig)
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
    y_label_side: str = "left",
):
    """Draw an L-shaped scale bar instead of axes.

    ``y_label_side="right"`` writes the y label level, to the right of the
    bar, for a bar shorter than its label; ``"outside"`` upright, left of the
    bar, 2 pt off it, and the time 3.5 pt under it.

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
    if y_label_side == "outside":
        # Gaps in points, not in the axes' units: a bar's own axes are only
        # as tall as the bar, so a fraction of them put the label on it.
        ax.annotate(x_label, (x0 + x_data_length / 2.0, y0), xytext=(0, -3.5), textcoords="offset points",
                    ha="center", va="top", annotation_clip=False, **kw)
        ax.annotate(y_label, (x0, y0 + y_length / 2.0), xytext=(-2, 0), textcoords="offset points",
                    ha="right", va="center", rotation=90, annotation_clip=False, **kw)
        return x_length, y_length
    ax.text(x0 + x_data_length / 2.0, y0 - 0.03 * y_span, x_label, ha="center", va="top", **kw)
    if y_label_side == "right":
        ax.annotate(y_label, (x0, y0 + y_length / 2.0), xytext=(3, 0), textcoords="offset points",
                    ha="left", va="center", annotation_clip=False, **kw)
    else:
        ax.text(x0 - 0.015 * x_span, y0 + y_length / 2.0, y_label, ha="right", va="center",
                rotation=90, **kw)
    return x_length, y_length


def _fmt(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


def outline(text, color: str = "#000000", width_pt: float = 0.1):
    """Give a label a thin outline, the way the variant names are set in the
    lab's figures: bold, in the variant's colour, 0.1 pt of black around it.

    SVG only, and the label stays text: matplotlib's path effects would draw
    it as glyph outlines under ``svg.fonttype: 'none'``, so this puts a
    ``stroke`` in the ``<text>`` element's style instead.  Returns the text.
    """
    text._labkit_outline = (color, width_pt)
    return text


def _outline_svg_text():
    from matplotlib.backends.backend_svg import RendererSVG
    plain = RendererSVG.draw_text
    if getattr(plain, "_labkit_outline", False):
        return

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        line = getattr(mtext, "_labkit_outline", None)
        if not line:
            return plain(self, gc, x, y, s, prop, angle, ismath, mtext)
        element = self.writer.element

        def outlined(tag, *args, attrib={}, **extra):
            if tag == "text" and "style" in attrib:
                attrib = {**attrib, "style": attrib["style"]
                          + f"; stroke: {line[0]}; stroke-width: {line[1]:g}"}
            return element(tag, *args, attrib=attrib, **extra)

        self.writer.element = outlined
        try:
            return plain(self, gc, x, y, s, prop, angle, ismath, mtext)
        finally:
            del self.writer.element

    draw_text._labkit_outline = True
    RendererSVG.draw_text = draw_text


_outline_svg_text()


def svg_scripts(svg: str) -> str:
    """One ``<text>`` per mathtext label, so an editor reads a subscript as one.

    Under ``svg.fonttype: 'none'`` matplotlib writes ``I$_{140}$/C`` as a
    ``<tspan>`` per *character*, each with its own x and y.  Illustrator and
    omnifig take a tspan with a y of its own for a new line, so the label
    arrives as a dozen one-letter texts.  This joins the characters back into
    runs and writes each run after the first with a ``dy`` and no x -- the
    shape every editor writes a subscript in, and the one omnifig reads back
    as a lowered run of the same text.

    ponytail: only text whose tspans change size (a script) is touched, and a
    space is put back where the glyphs stand a quarter of a size apart.
    """
    import re

    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextToPath

    span = re.compile(r'<tspan x="([-\d.]+)" y="([-\d.]+)" style="font-size: ([\d.]+)px;'
                      r' font-family: \'([^\']*)\'">(.*?)</tspan>', re.S)

    def join(block):
        spans = [(float(x), float(y), float(z), f, c) for x, y, z, f, c in span.findall(block.group(2))]
        if len({z for _, _, z, _, _ in spans}) < 2:
            return block.group(0)               # no script in it
        # matplotlib writes the baseline first and the scripts after it; the
        # eye, and an editor, read left to right.
        spans.sort(key=lambda t: t[0])
        first = spans[0][0]
        runs, text, last = [], "", None
        for x, y, z, face, ch in spans:
            if last and (y, z) == last[:2]:
                # A gap wider than the glyph before it was a space: mathtext
                # drops them, so the label spells one "\ ".
                width = TextToPath().get_text_width_height_descent(
                    last[4], FontProperties(family=last[3], size=z), False)[0]
                if x - last[2] - width > 0.15 * z:
                    text += " "
            elif last:
                runs.append((last[:2], last[3], text)); text = ""
            text += ch
            last = (y, z, x, face, ch)
        runs.append((last[:2], last[3], text))
        body, y0 = "", None
        for (y, z), face, t in runs:
            # mathtext draws scripts at 0.7x (5.6 pt under an 8 pt title); JCI's floor is 6.
            style = f"font-size: {max(z, 6.0):g}px; font-family: '{face}'"
            if y0 is None:
                body += f'<tspan x="{first:g}" y="{y:g}" style="{style}">{t}</tspan>'
            else:
                body += f'<tspan dy="{y - y0:.4g}" style="{style}">{t}</tspan>'
            y0 = y
        return block.group(1) + body + "</text>"

    return re.sub(r"(<text[^>]*>)\s*((?:<tspan.*?</tspan>\s*)+)</text>", join, svg, flags=re.S)


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
    #: 2.79 mm puts a category at 4.5 mm, the pitch the lab's current figures
    #: draw at -- the ClC-7 2025 figure in ephys, pHerret and purifig, whose
    #: quant panels default to the same bar -- so a panel from labkit and one
    #: from those apps sit side by side at one scale.  It was 3.94 mm, a
    #: 6.35 mm category, the ephys notebook's 0.25 in.  Two groups make a panel
    #: 9 mm wide.  An ``(n = 5)`` under a group name (5.33 mm at 7 pt) and
    #: names longer than about three characters want the x rotation or
    #: :attr:`horizontal`.
    #:
    #: ``width_mm`` becomes a ceiling rather than the width.  0 hands the
    #: width back to it; ``axes_width_mm``, being explicit, wins over both.
    #: ``'xy'`` and ``'trace'`` have no categories and ignore this.
    bar_mm: float = 2.79
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
    #: column to themselves instead of the 4.5 mm of tick a 2.79 mm bar
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
    #: Names as printed, ``"74 = 161, loCl = 41"``: a whole name, or one part
    #: of a two-level ``"WT / 74"``.  Only the text changes -- colors, order
    #: and the p-value pairs still name the groups as the data does, so the
    #: printed name can change without any of them being set again.
    rename: str = ""

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
    spines: str = "auto"           # auto | lb | box | zero | x0 | b | none
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
    #: Size of each panel's traces, in mm; the figure is as many of them as
    #: there are, plus their titles.  20 mm is a family at print size.
    panel_width_mm: float = 20.0
    panel_height_mm: float = 20.0
    #: Panels per row; 0 lays them out as near a square as the count allows.
    panel_cols: int = 0
    #: Scale bar lengths: seconds, and the current unit. 0 picks round ones.
    scale_x_s: float = 0.0
    scale_y: float = 0.0
    #: Size of the n under each panel's title, in pt.
    n_pt: float = 7.0
    #: A trace's scale bars are in its own data's units, which these name:
    #: ``scale_x`` is the time bar in ``x_unit`` and ``scale_y`` the value bar
    #: in ``y_unit``.  0 picks round ones, as for a family.
    scale_x: float = 0.0
    x_unit: str = "ms"
    y_unit: str = "pA/pF"
    #: Shade each step's SEM. Off by default: fourteen overlapping bands hide
    #: the currents they describe.
    family_spread: bool = False
    #: Draw the voltage protocol under a family of traces, a panel wide.
    family_protocol: bool = False
    #: Where each construct goes, by name: rows split by ";" or new lines,
    #: places in a row by "|", "protocol" for the protocol.  Empty: a grid.
    family_layout: str = ""
    #: What sets a trace's color. ``"voltage"`` ramps across the steps of a
    #: panel, ``"black"`` draws them all in ink, and any other value is read as
    #: a grouping field ("construct", "condition", ...): every step of a panel
    #: then takes the color of that field's value, shared across panels.
    # Black unless asked: a step family is one cell's protocol, and the ramp
    # says what the axis already does.  Every ClC-7 screen set it by hand.
    family_colour: str = "black"     # black | voltage | <field name>

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
        if not spec.n_pt > 0:
            spec.n_pt = cls.n_pt
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

    The sequence a reader needs is almost never alphabetical: ``WT``, ``Y715C``,
    ``P716L`` is the order of the mutation along the protein, and sorted() puts
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
        with warnings.catch_warnings(record=True) as said:
            warnings.simplefilter("always")
            fig.draw_without_rendering()
        if wrap_labels(fig, want_w, want_h):
            continue                   # drawn again with the label on two lines
        box = panels[0].get_window_extent().transformed(
            fig.dpi_scale_trans.inverted())
        wide, high = fig.get_size_inches()
        if box.width <= 0.0 or box.height <= 0.0:
            # The canvas is smaller than its own labels, so there is nothing
            # to measure and no shortfall to read off.  Make room and look
            # again rather than stepping on a number that is not there.
            fig.set_size_inches(wide * 1.6, high * 1.6)
            continue
        if any("collapsed to zero" in str(w.message) for w in said):
            # Constrained layout that gives up says so and leaves the axes
            # where they were: a size, but not one the labels fit around.
            # What they need is on the page all the same -- a ladder of
            # brackets taller than the canvas, say -- so give it that.
            need = fig.get_tightbbox()
            grown = (max(wide, need.width * 1.1), max(high, need.height * 1.1))
            if grown != (wide, high):
                fig.set_size_inches(*grown)
                continue
        # A bracket ladder is laid out in points, so the room it makes in the
        # value limit, or past it, depends on how long the axis will be: ask
        # it for the size being pinned, not the size just drawn, or the two
        # chase each other.  Whatever it moves needs one more look.
        moved = [getattr(ax, "_labkit_make_room", lambda *_: False)(want_w, want_h)
                 for ax in panels]
        dw = ncol * (want_w - box.width) if want_w > 0.0 else 0.0
        dh = nrow * (want_h - box.height) if want_h > 0.0 else 0.0
        fig.set_size_inches(max(wide + dw, 0.3), max(high + dh, 0.3))
        if abs(dw) < 0.008 and abs(dh) < 0.008 and not any(moved):  # a fifth of a mm
            break
    return tuple(float(v) for v in fig.get_size_inches())


def hold_layout(fig) -> None:
    """Keep the layout :func:`fit_axes` settled, whatever the file is saved at.

    Constrained layout runs again at every draw, and a save is a draw at the
    *file's* resolution: 72 dpi for SVG and PDF, the asked-for dpi for a PNG.
    The labels measure a little differently there, so the layout settles on
    other axes -- measured 2026-10-27, three groups fitted at 19.86 mm saved at
    20.58 in all three formats, a category 6.59 mm wide instead of 6.35.  The
    value axis held only because it is the one dimension nothing crowds.
    Nothing warns: the canvas is the size it was, only the plot inside it moved.

    So, once the fit is final: one more draw, so the axes sit in the canvas
    size the last fit pass set, and then no layout engine, so no save can move
    them again.  An artist added afterwards is not laid out around -- which is
    the point -- but ``bbox_inches='tight'`` still widens the file to hold it.

    Only for a figure whose axes were pinned.  Where the *figure* size is the
    promise -- ``bar_mm=0`` and a typed width and height -- it is the save-time
    layout that keeps it: :data:`TIGHT_CROP` cancels the blank it reserves, so
    the file is exactly the size asked.  Held, the labels measure narrower at
    72 dpi and the file comes out 0.15 mm short.  So the axes are held when
    they are what was asked for, and the canvas when it is.
    """
    fig.draw_without_rendering()
    fig.set_layout_engine("none")


def tier_level_labels(fig, pad_pt: float = 0.75) -> int:
    """In a one-row ladder (``align="level"``), lift every other label a text
    line when neighbours would overlap; the brackets stay in their row.

    Measured unhinted, at the size the page prints: at screen dpi Arial's
    hinted widths come out ~2 pt short, so labels that touch on paper pass
    :func:`spread_labels` as clear.  Run once the figure has its final size.
    Returns how many labels went up.
    """
    from matplotlib.textpath import TextToPath
    from matplotlib.transforms import ScaledTranslation

    measure = TextToPath().get_text_width_height_descent
    lifted = 0
    for ax in fig.axes:
        rows: dict = {}
        for t in ax.texts:
            if getattr(t, "_labkit_tier", None) and t.get_visible():
                rows.setdefault(round(t.get_position()[1], 6), []).append(t)
        for row in rows.values():
            def span(t):
                x = t.get_transform().transform(t.get_position())[0] * 72 / fig.dpi
                w = measure(t.get_text(), t.get_fontproperties(), False)[0]
                return x - w / 2, x + w / 2
            row.sort(key=lambda t: span(t)[0])
            if all(span(a)[1] + pad_pt <= span(b)[0] for a, b in zip(row, row[1:])):
                continue
            for t in row[1::2]:
                t.set_transform(t.get_transform()
                                + ScaledTranslation(0.0, t._labkit_tier / 72, fig.dpi_scale_trans))
                lifted += 1
    return lifted


def spread_labels(fig, pad_pt: float = 0.75, reach_pt: float = 24.0) -> int:
    """Move the labels drawn on the axes off each other and off the tick and
    axis labels, each by the least that clears it, in points.

    A scale bar's turned label touching its time label, a p-value standing on
    a tick label: each drawing places its own text well, but nothing looks at
    what the others put down.  Tick labels, axis labels and titles stay where
    they are; the free text on the axes (``ax.texts``, annotations included)
    is looked at in the order it was drawn, and one that touches anything
    already down is tried at growing distances, the way away from what it
    touches first, then fifteen others.  In points, so the move is the
    same at whatever size the panel is saved.  Returns how many were moved.
    A bracket's label (annotate_p) only slides along its line or outward,
    and its line rises with it, so a label pushed up keeps its bracket.

    What is drawn counts too: a label is not to cross a line, a spine, the
    outline of a box or bar, or a point.  Only the outline -- a label wholly
    inside a shape, a count in a Venn region, is there on purpose -- and a
    heatmap's cells not at all.
    """
    from matplotlib.collections import PathCollection, QuadMesh
    from matplotlib.path import Path
    from matplotlib.text import Annotation, Text
    from matplotlib.transforms import Bbox, ScaledTranslation

    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    pad = pad_pt * px

    def box(t):
        b = t.get_window_extent(renderer)
        return (b.x0 - pad, b.y0 - pad, b.x1 + pad, b.y1 + pad)

    def hits(a, b):
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

    free = [t for ax in fig.axes for t in ax.texts if t.get_visible() and t.get_text().strip()
            and not getattr(t, "_labkit_tick", False)]
    # A tick label is a Text whether it is drawn or not: one past the view, or
    # on an axes turned off, is not there to touch.
    unseen = set(map(id, free))
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            shown = set(axis._update_ticks()) if ax.axison else set()
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if tick not in shown:
                    unseen |= {id(tick.label1), id(tick.label2)}
            if not ax.axison:
                unseen.add(id(axis.label))
    down = [box(t) for t in fig.findobj(Text)
            if id(t) not in unseen and t.get_visible() and t.get_text().strip()]
    # The drawing, in pixels: (path, whether its outline counts, a point's
    # radius, the box it is clipped to).  The clip matters: a bar on the lower
    # part of a broken axis runs, undrawn, right up through the upper one.
    shapes = []

    def clip_of(artist):
        return artist.get_clip_box() if artist.get_clip_on() else None

    none = (None, "", " ", "None", "none")
    # A bracket's line goes where its label goes, outward: it is the label's
    # to keep clear, not something to keep clear of.
    riders = {id(getattr(t, "_labkit_line", None)) for t in free}
    for ax in fig.axes:
        if ax.axison:
            for spine in ax.spines.values():
                if spine.get_visible():
                    e = spine.get_window_extent(renderer)
                    down.append((e.x0, e.y0, e.x1, e.y1))
        for line in ax.lines:
            if line.get_visible() and id(line) not in riders:
                dot = line.get_markersize() * px / 2 if line.get_marker() not in none else 0.0
                shapes.append((line.get_transform().transform_path(line.get_path()),
                               line.get_linestyle() not in none, dot, clip_of(line)))
        for c in ax.collections:
            if not c.get_visible() or isinstance(c, QuadMesh):
                continue
            if isinstance(c, PathCollection) and len(c.get_offsets()):
                sizes = c.get_sizes()
                dot = (np.sqrt(sizes.max()) if len(sizes) else 6.0) * px / 2
                shapes.append((Path(c.get_offset_transform().transform(c.get_offsets())), False,
                               dot, clip_of(c)))
            else:
                shapes += [(c.get_transform().transform_path(q), True, 0.0, clip_of(c))
                           for q in c.get_paths()]
        for patch in ax.patches:
            if patch.get_visible():
                shapes.append((patch.get_transform().transform_path(patch.get_path()), True, 0.0,
                               clip_of(patch)))

    def crossed(r):
        """The parts of the drawing a box crosses, as points to step away from."""
        out = []
        for path, edge, dot, clip in shapes:
            if clip is None:
                seen = r
            else:
                seen = (max(r[0], clip.x0), max(r[1], clip.y0), min(r[2], clip.x1), min(r[3], clip.y1))
                if seen[0] >= seen[2] or seen[1] >= seen[3]:
                    continue
            v = path.vertices
            near = v[(v[:, 0] > seen[0] - dot) & (v[:, 0] < seen[2] + dot)
                     & (v[:, 1] > seen[1] - dot) & (v[:, 1] < seen[3] + dot)]
            if dot and len(near):
                out.append(near.mean(axis=0))
            elif edge and len(v) > 1 and path.intersects_bbox(Bbox.from_extents(*seen), filled=False):
                out.append(near.mean(axis=0) if len(near) else v.mean(axis=0))
        return out

    def clear(r):
        return not any(hits(r, d) for d in down) and not crossed(r)

    def shifted(r, dx, dy):
        return (r[0] + dx * px, r[1] + dy * px, r[2] + dx * px, r[3] + dy * px)

    moved = 0
    for t in free:
        b = box(t)
        line = getattr(t, "_labkit_line", None)
        lb = box(line) if line is not None else None
        out = getattr(t, "_labkit_out", None)     # a bracket's label: not back across it

        def rides(dx, dy):                          # the line's share of a move: outward only
            return (dx, 0.0) if out == 0.0 else (0.0, dy)

        touched = [((d[0] + d[2]) / 2, (d[1] + d[3]) / 2) for d in down
                   if hits(b, d) or (lb and hits(lb, d))] + crossed(b)
        if touched:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            ox, oy = np.mean(touched, axis=0)
            away = np.arctan2(cy - oy, cx - ox) if (ox, oy) != (cx, cy) else np.pi / 2
            ways = [away] + [away + s * k * np.pi / 8 for k in range(1, 9) for s in (1, -1)][:-1]
            if out is not None:
                ways = [a for a in ways if np.cos(a - out) > -1e-9]
            # Its line along, a bracket's label can go further and still read.
            reach = reach_pt * (2 if line is not None else 1)
            best = next(((dx, dy) for r in np.arange(0.5, reach + 0.01, 0.5) for a in ways
                         for dx, dy in [(r * np.cos(a), r * np.sin(a))]
                         if clear(shifted(b, dx, dy))
                         and (lb is None or clear(shifted(lb, *rides(dx, dy))))), None)
            if best is not None:
                dx, dy = best
                if isinstance(t, Annotation) and t.anncoords == "offset points":
                    t.xyann = (t.xyann[0] + dx, t.xyann[1] + dy)
                elif isinstance(t, Annotation):
                    best = None       # placed in some other frame: left where it is
                else:
                    t.set_transform(t.get_transform()
                                    + ScaledTranslation(dx / 72, dy / 72, fig.dpi_scale_trans))
                if best is not None:
                    b = shifted(b, dx, dy)
                    moved += 1
                    if line is not None:
                        lx, ly = rides(dx, dy)
                        lb = shifted(lb, lx, ly)
                        line.set_transform(line.get_transform()
                                           + ScaledTranslation(lx / 72, ly / 72, fig.dpi_scale_trans))
        down.append(b)
        if lb is not None:
            down.append(lb)
    return moved


_LEGEND_CORNERS = ("upper right", "upper left", "lower right", "lower left",
                   "center right", "center left", "upper center", "lower center", "center")


def place_legends(fig, pad_pt: float = 0.75) -> int:
    """Keep each key off what its panel draws; how many moved.

    :func:`spread_labels`' rule for a key: it is not to cross a line, a point,
    the outline of a box or bar, or a text.  A key placed only by ``loc`` (its
    anchor its own axes) goes to the first corner of its panel where it crosses
    none of them, its own spot first.  With no such corner -- a narrow
    distribution panel is curve from side to side -- it tries its top right
    corner pushed out into the gap beside the panel (no further than the next
    panel's y axis, or the widest such gap in the figure for a row's last
    panel): the space over a distribution's tail plus the gap often hold it.
    Failing that it goes above the panel, its left edge on the axes', and the
    title rises over it.  A key a plot
    anchored somewhere itself (beside the axes, in an empty quadrant) is left
    where it was put.  Call it once the panels have their final size, then lay
    the figure out again: a key moved above takes room the layout gave nothing.
    """
    from matplotlib.collections import PathCollection, QuadMesh
    from matplotlib.path import Path as MPath
    from matplotlib.transforms import Bbox, ScaledTranslation

    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    pad = pad_pt * px
    none = (None, "", " ", "None", "none")
    moved = 0

    def room_right(ax):
        # Up to the y axis of the next panel in the row, None with none.
        box, near = ax.get_window_extent(renderer), None
        for other in fig.axes:
            ob = other.get_window_extent(renderer)
            if other is ax or not other.get_visible() or ob.x0 <= box.x1 \
                    or ob.y1 <= box.y0 or ob.y0 >= box.y1:
                continue
            yb = other.yaxis.get_tightbbox(renderer) if other.yaxis.get_visible() else None
            edge = min(ob.x0, yb.x0) if yb is not None and yb.width > 0 else ob.x0
            near = edge if near is None else min(near, edge)
        return None if near is None else max(0.0, near - box.x1 - pad)

    gaps = [r for a in fig.axes if a.get_visible() for r in [room_right(a)] if r is not None]
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None or not leg.get_visible() or leg.get_bbox_to_anchor() is not ax.bbox:
            continue
        shapes = []                                   # (path in pixels, outline counts, point radius)
        for line in ax.lines:
            if line.get_visible():
                dot = line.get_markersize() * px / 2 if line.get_marker() not in none else 0.0
                shapes.append((line.get_transform().transform_path(line.get_path()),
                               line.get_linestyle() not in none, dot))
        for c in ax.collections:
            if not c.get_visible() or isinstance(c, QuadMesh):
                continue
            if isinstance(c, PathCollection) and len(c.get_offsets()):
                sizes = c.get_sizes()
                shapes.append((MPath(c.get_offset_transform().transform(c.get_offsets())), False,
                               (np.sqrt(sizes.max()) if len(sizes) else 6.0) * px / 2))
            else:
                shapes += [(c.get_transform().transform_path(q), True, 0.0) for q in c.get_paths()]
        for patch in ax.patches:
            if patch.get_visible():
                shapes.append((patch.get_transform().transform_path(patch.get_path()), True, 0.0))
        texts = [t.get_window_extent(renderer) for t in ax.texts
                 if t.get_visible() and t.get_text().strip()]

        def clear(b):
            r = Bbox.from_extents(b.x0 - pad, b.y0 - pad, b.x1 + pad, b.y1 + pad)
            if any(r.overlaps(t) for t in texts):
                return False
            for path, edge, dot in shapes:
                v = path.vertices
                if dot and len(v) and np.any((v[:, 0] > r.x0 - dot) & (v[:, 0] < r.x1 + dot)
                                             & (v[:, 1] > r.y0 - dot) & (v[:, 1] < r.y1 + dot)):
                    return False
                if edge and len(v) > 1 and path.intersects_bbox(r, filled=False):
                    return False
            return True

        if clear(leg.get_window_extent(renderer)):
            continue
        moved += 1
        for corner in _LEGEND_CORNERS:
            leg.set_loc(corner)
            if clear(leg.get_window_extent(renderer)):
                break
        else:
            room = room_right(ax)
            # Between two panels it stays out of the layout, which would
            # otherwise push them apart to make the room it already has; the
            # last one's key is in it, so the canvas grows to hold it.
            leg.set_in_layout(room is None)
            room = max(gaps, default=0.0) if room is None else room
            leg.set_loc("upper right")
            for dx in np.arange(1.0, room / px + 0.5, 1.0):
                leg.set_bbox_to_anchor((1.0, 1.0), transform=ax.transAxes
                                       + ScaledTranslation(dx / 72, 0.0, fig.dpi_scale_trans))
                if clear(leg.get_window_extent(renderer)):
                    break
            else:
                leg.set_bbox_to_anchor(ax.bbox)
                leg.set_in_layout(True)
                room = None
            if room is not None:
                continue
            # Above, clear of whatever else the panel puts on its top edge: a
            # second scale, top ticks.  In points, which a resize leaves alone.
            # The y axis does not count: its top number pokes above the
            # spine on the first panel only, and the keys of a row would not
            # line up.
            title = ax.title.get_text()
            top = max([ax.get_window_extent(renderer).y1]
                      + [a.get_tightbbox(renderer).y1 for a in [ax.xaxis, *ax.child_axes]
                         if a.get_visible() and a.get_tightbbox(renderer) is not None])
            lift = (top - ax.get_window_extent(renderer).y1) / px + 1.0
            leg.set_loc("lower left")
            leg.borderaxespad = 0.0
            leg.set_bbox_to_anchor((0.0, 1.0), transform=ax.transAxes
                                   + ScaledTranslation(0.0, lift / 72, fig.dpi_scale_trans))
            if title:
                high = leg.get_window_extent(renderer).height / px
                ax.set_title(title, pad=lift + high + mpl.rcParams["axes.titlepad"] / 2,
                             fontproperties=ax.title.get_fontproperties().copy(),
                             color=ax.title.get_color())
    return moved


def wrap_labels(fig, width_in: float = 0.0, height_in: float = 0.0) -> int:
    """Put an axis label longer than its axis on two lines; how many were.

    "p90 vesicle cross-sectional area, relative to WT" under a 19 mm axis
    stood out 17 mm on each side, and the panel was as wide as its label
    rather than its plot -- room the panel beside it could have had.  Broken
    at the space nearest its middle, so the two lines come out even, or before
    a unit in parentheses when that is not far from it.  Against
    the axes size about to be pinned (`width_in`/`height_in`, inches) where
    there is one, else the size drawn now.  A label already on two lines, or
    one word, is left as it is.
    """
    renderer = fig.canvas.get_renderer()
    wrapped = 0
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        box = ax.get_window_extent(renderer)
        room = (width_in * fig.dpi or box.width, height_in * fig.dpi or box.height)
        for label, along in ((ax.xaxis.label, 0), (ax.yaxis.label, 1)):
            text = label.get_text()
            if not label.get_visible() or "\n" in text or " " not in text.strip():
                continue
            ext = label.get_window_extent(renderer)
            if (ext.width, ext.height)[along] <= room[along]:
                continue
            text = text.strip()
            # A unit in parentheses goes down whole: "Vesicle area / (µm²)",
            # not "Vesicle / area (µm²)".
            cut = min((i for i, c in enumerate(text) if c == " "),
                      key=lambda i: abs(2 * i - len(text))
                      - (len(text) / 3 if text[i + 1:i + 2] == "(" else 0))
            label.set_text(text[:cut] + "\n" + text[cut + 1:])
            wrapped += 1
    return wrapped


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
    return _as_asked(table, spec)


def curve_comparisons_for(spec: "PlotSpec | dict", data: "Sequence[dict]"):
    """:func:`comparisons_for` for curves: each pair of them as a whole.

    Reads the per-cell ``'curves'`` an entry carries -- ``{'cell_ids',
    'values'}``, a row per cell along its own ``'x'`` or else the entry's --
    and hands them to
    :func:`labkit.stats.compare_curves`; the mean and SEM a curve is drawn from
    cannot say which points came from one cell.  ``p_test`` does not apply:
    there is one test for a curve, and ``show_p`` neither: nothing of it is
    drawn.  ``None`` with fewer than two curves carrying cells.
    """
    from . import stats as _stats

    spec = spec if isinstance(spec, PlotSpec) else PlotSpec.from_dict(spec)
    curves = {str(e.get("label")): {"x": e.get("x", []), **e["curves"]}
              for e in data if e.get("curves")}
    if len(curves) < 2:
        return None
    table = _stats.compare_curves(
        curves, how=spec.p_pairs, alpha=spec.p_alpha,
        correction=spec.p_correction if spec.p_correction != "none" else "bonferroni")
    return _as_asked(table, spec)


def _as_asked(table, spec: "PlotSpec"):
    """``table`` with "correction: none" honored by copying the raw p across."""
    from . import stats as _stats

    if spec.p_correction == "none" and len(table):
        table["p_corrected"] = table["p_value"]
        table["significant"] = table["p_value"] < spec.p_alpha
        table["stars"] = [_stats.stars(p) for p in table["p_value"]]
    return table


def _legend_notes(table, spec: "PlotSpec", order: Sequence[str], rename=None) -> dict[str, str]:
    """The p-values a legend carries, ``{label: "p = 0.01"}``.

    Each p once, on the entry of whichever group of the pair is listed later --
    against one control that is never the control, so it reads "p = ..." alone;
    otherwise it names the partner.
    """
    from .stats import brackets as _brackets

    items = _brackets(table, use_corrected=spec.p_use_corrected, style=spec.p_style,
                      alpha=spec.p_alpha, only_significant=spec.p_only_significant) \
        if table is not None and len(table) else []
    rank = {k: i for i, k in enumerate(order)}
    shared = set.intersection(*({e["a"], e["b"]} for e in items)) if items else set()
    notes: dict[str, str] = {}
    for e in items:
        me, other = sorted((e["a"], e["b"]), key=lambda k: -rank.get(k, 0))
        text = e["text"] if len(shared) == 1 else f"{e['text']} vs {_shown(other, rename)}"
        notes[me] = "; ".join(filter(None, [notes.get(me), text]))
    return notes


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


def _limits(text: str) -> tuple[float | None, float | None] | None:
    """``"min, max"``; either side left empty stays with the data, so
    ``"0,"`` starts the axis at zero and lets the top find itself."""
    values = _numbers(text)
    if len(values) >= 2:
        return values[0], values[1]
    sides = str(text or "").replace(";", ",").split(",")
    if len(sides) == 2 and len(values) == 1:
        low, high = (_numbers(side) for side in sides)
        return (low[0] if low else None), (high[0] if high else None)
    return None


def _labels(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def _cycle(text: str, fallback: str) -> list[str]:
    items = [part.strip() for part in str(text or "").split(",") if part.strip()]
    return items or [fallback]


def _renames(text: str) -> dict[str, str]:
    """``"74 = 161, loCl = 41"`` as ``{"74": "161", "loCl": "41"}``.  A part
    with no ``=`` is ignored rather than guessed at.  A comma inside brackets
    is part of a name -- ``50uM PI(3,5)P2 = PI(3,5)P2`` -- not a break."""
    out = {}
    for part in re.split(r"[;\n]|,(?![^()]*\))", str(text or "")):
        old, sep, new = part.partition("=")
        if sep and old.strip():
            out[old.strip()] = new.strip()
    return out


def _shown(label, rename: "dict[str, str] | None") -> str:
    """A name as printed: renamed whole when :attr:`PlotSpec.rename` names it,
    otherwise part by part, so ``74 = 161`` makes ``WT / 74`` ``WT / 161``."""
    label = str(label)
    if not rename:
        return label
    if label in rename:
        return rename[label]
    for sep in (" / ", ", "):
        if sep in label:
            return sep.join(rename.get(part.strip(), part.strip()) for part in label.split(sep))
    return label


@functools.lru_cache(maxsize=None)
def _area_of(shape: str) -> float:
    """The area a marker covers at ``ms=1``; 0 for the line shapes, x and +."""
    from matplotlib.markers import MarkerStyle

    style = MarkerStyle(shape)
    if not style.is_filled():
        return 0.0
    area = 0.0
    for poly in style.get_path().to_polygons(style.get_transform()):
        x, y = poly[:, 0], poly[:, 1]
        area += 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    return area


def _ms(shape: str, size: float) -> float:
    """The ``ms`` that gives ``shape`` the area of a circle ``size`` across.

    At one ``ms`` a diamond covers 27 % more than the circle beside it and a
    triangle 36 % less, so "3 min as diamonds" also read as "3 min bigger".
    The line shapes have no area and keep the size.
    """
    area = _area_of(shape)
    return size * math.sqrt(_area_of("o") / area) if area > 0 else size


def _place_legend(ax, spec: "PlotSpec") -> None:
    """Where the key goes, and how long its lines are."""
    if ax.get_legend() is None:
        return
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    where: dict = {}
    if spec.legend_loc and spec.legend_loc != "best":
        where = {"loc": spec.legend_loc}
    elif spec.spines == "zero" and spec.kind == "xy":
        # The axes cross in the middle, and an outward current leaves the
        # quadrant left of zero voltage and above zero current empty: the key
        # goes there, its right edge clear of the current axis's numbers.  In
        # points, which resizing the panel leaves alone.  A key wider than the
        # quadrant runs on past the end of the voltage axis, not over them.
        from matplotlib.transforms import ScaledTranslation

        fig = ax.figure
        renderer = fig.canvas.get_renderer()
        widths = [t.get_window_extent(renderer).width
                  for t in ax.yaxis.get_ticklabels() if t.get_visible() and t.get_text()]
        tick = mpl.rcParams["ytick.major.size"] if spec.tick_direction != "in" else 0.0
        clear = max(widths, default=0.0) * 72 / fig.dpi + tick + mpl.rcParams["ytick.major.pad"] + 2.0
        low, high = ax.get_xlim()
        zero = (0 - low) / (high - low) if low < 0 < high else 0.0
        where = {"loc": "upper right", "bbox_to_anchor": (zero, 1.0), "borderaxespad": 0.0,
                 "bbox_transform": ax.transAxes + ScaledTranslation(-clear / 72, 0, fig.dpi_scale_trans),
                 "borderpad": 0.0}
    elif spec.spines == "zero":
        # An axis through the middle leaves no empty corner inside.
        where = {"loc": "lower left", "bbox_to_anchor": (1.02, 0.0), "borderaxespad": 0.0}
    if spec.kind == "xy":
        # Lines as long as the panel can spare: on a 25 mm I/V a full-length
        # handle was a fifth of the axis.  Full length from 40 mm of axes up.
        wide = spec.axes_width_mm or 0.75 * spec.width_mm
        k = min(1.0, max(0.5, wide / 40.0))
        where.update({key: mpl.rcParams[f"legend.{key}"] * k
                      for key in ("handlelength", "handletextpad", "labelspacing", "columnspacing")})
    if where:
        legend = ax.legend(handles, labels, **where)
        if "bbox_transform" in where:
            # A key taller than the room above zero current -- inward currents
            # push zero up -- would sit on the voltage label and the curves:
            # it goes into columns, wider past the end of the axis instead,
            # but never wider than the axes (six in a row made a panel 84 mm
            # wide).  Past that, legend_loc puts it somewhere else.
            renderer = ax.figure.canvas.get_renderer()
            low, high = ax.get_ylim()
            bottom = ax.transAxes.transform((0, (0 - low) / (high - low) if low < 0 < high else 0))[1]
            if ax.xaxis.label.get_text():
                bottom = max(bottom, ax.xaxis.label.get_window_extent(renderer).y1)
            room = ax.get_window_extent(renderer).width
            for ncols in range(2, len(handles) + 1):
                if legend.get_window_extent(renderer).y0 >= bottom:
                    break
                if ax.legend(handles, labels, ncols=ncols, **where).get_window_extent(renderer).width > room:
                    ax.legend(handles, labels, ncols=ncols - 1, **where)
                    break
                legend = ax.get_legend()


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


def _zero_axis_labels(ax) -> None:
    """With the axes crossing at zero: V over the left end of its axis, I
    upright beside the top of its own -- where zero is in the current limits,
    so whatever moves the limits calls this again."""
    def at_zero(low, high):
        return (0 - low) / (high - low) if low < 0 < high else 0.0
    gap = 0.02
    if ax.get_xlabel():
        ax.xaxis.label.set(ha="left", va="bottom")
        ax.xaxis.set_label_coords(0.0, at_zero(*ax.get_ylim()) + gap)
    if ax.get_ylabel():
        # Aligned before the quarter turn (rotation_mode "anchor"): the
        # end of the text at the top of the axis, the text to its right.
        ax.yaxis.label.set(rotation=90, rotation_mode="anchor", ha="right", va="top")
        ax.yaxis.set_label_coords(at_zero(*ax.get_xlim()) + gap, 1.0)


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
    elif spec.spines == "b":
        # A panel beside another on the same value range: its y axis and
        # ticks, but the numbers and the label are the left panel's.
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", which="both", labelleft=False)
        ax.set_ylabel("")
    elif spec.spines == "x0":
        # The voltage axis at zero current, the current axis left where it
        # is: a reversal potential is read where a curve crosses the axis.
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_position(("data", 0))
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), loc="right")
    else:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    if spec.spines == "zero":
        # With the axes crossing in the middle of the data, a centered axis
        # label lands on the plot.  Laid out as the lab's notebook draws an
        # IV curve: V at the left end, over its axis; I upright beside the
        # top of its axis, on the right.  The key is :func:`_place_legend`'s.  The tick labels at zero would sit on top of the crossing itself.
        # set_ticks widens the view to every tick it is given, including the
        # ones just outside it, so the limits are put back afterwards.
        limits = ax.get_xlim(), ax.get_ylim()
        # The notebook's voltage ticks, every 40 mV across -120..140: an IV
        # curve is read off its axis, and two labels do not read.  Ticks the
        # spec names win: on a small panel every 40 mV runs together.
        if not spec.xticks:
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 4, 5, 10]))
        for axis in (ax.xaxis, ax.yaxis):
            ticks = [t for t in axis.get_majorticklocs() if t != 0]
            axis.set_ticks(ticks)
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        _zero_axis_labels(ax)

    # Limits last: a bracket or a scale bar may have widened the view.
    x = _limits(spec.xlim)
    y = _limits(spec.ylim)
    if x:
        ax.set_xlim(*x)
    if y:
        ax.set_ylim(*y)
    _place_legend(ax, spec)


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
    rename = _renames(spec.rename)
    table = None

    with publication_style(font_pt=spec.font_pt, font=spec.font,
                           figure__figsize=spec.figsize()):
        own_figure = ax is None
        if own_figure:
            fig, ax = plt.subplots(figsize=spec.figsize(), layout="constrained")
        else:
            fig = ax.figure

        if spec.kind == "xy":
            # The curve test goes in the table beside the figure, never on it.
            table = curve_comparisons_for(spec, data) if comparisons is None else comparisons
            xy_plot(data, ax=ax, palette=palette, xlabel=spec.xlabel,
                    ylabel=spec.ylabel, title=spec.title, legend=spec.legend,
                    show_n=spec.show_n, rename=rename,
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
                       x_length=spec.scale_x or None, y_length=spec.scale_y or None,
                       x_unit=spec.x_unit, y_unit=spec.y_unit,
                       xlabel=spec.xlabel, ylabel=spec.ylabel, rename=rename)
        elif spec.kind == "hist":
            notes = {}
            if spec.show_p:
                table = (comparisons_for(spec, data, pair_on=paired)
                         if comparisons is None else comparisons)
                notes = _legend_notes(table, spec, list(data), rename)
            hist_plot(data, ax=ax, palette=palette, xlabel=spec.xlabel,
                      ylabel=spec.ylabel, title=spec.title, show_n=spec.show_n,
                      horizontal=_is_horizontal(spec), notes=notes, rename=rename)
        else:
            ax, positions = group_plot(
                data, kind=spec.kind, centre=spec.centre, error=spec.error,
                points=spec.points or False, paired=paired, palette=palette,
                ax=ax, ylabel=spec.ylabel, xlabel=spec.xlabel, title=spec.title,
                show_n=spec.show_n, log_y=spec.log_y, point_size=spec.point_size,
                markers=_cycle(spec.markers, "o") if spec.markers else (),
                horizontal=_is_horizontal(spec), rename=rename,
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
        if not want[across] and spec.kind == "hist":
            want[across] = 20.0                  # the fractions: no categories to count
        if not want[across] and spec.bar_mm > 0 and spec.kind not in ("xy", "trace", "hist"):
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
        # A typed axes size wins over the ceiling, as PlotSpec documents:
        # squeezing it would undo the very number somebody asked for.
        pinned = (spec.axes_height_mm if sideways else spec.axes_width_mm) > 0
        for _ in range(3 * (not pinned)):
            over = settled[across] / MM - ceiling
            if not (want[across] and ceiling > 0 and over > 0.01):
                break
            want[across] = max(want[across] - over, 5.0)
            settled = fit_axes(fig, *want)
        if any(want) and spec.kind == "xy" and spec.spines == "zero":
            # How many columns the key needs above zero depends on how tall
            # the axes are, and they are only that now.  In the style block,
            # which the key's fonts were set in.
            with publication_style(font_pt=spec.font_pt, font=spec.font):
                _place_legend(ax, spec)
            fit_axes(fig, *want)
        if any(want):
            hold_layout(fig)
        tier_level_labels(fig)
        spread_labels(fig)      # a caller's axes: the caller's figure is not done yet
    return fig, ax, table


def facet_figure(spec: "PlotSpec | dict", panels, *, ax_titles: bool = True):
    """One spec drawn as panels side by side, like a seaborn FacetGrid row.

    ``panels`` is ``[(title, data, paired, comparisons), ...]``, each panel's
    data shaped as :func:`figure_from_spec` takes it.  Every panel is drawn
    with exactly the sizes the first would get alone -- the same data area,
    the same category width -- so the row reads as one figure.

    Group plots share the value axis: one set of limits, and the label and
    tick labels only on the first panel.  'xy' and 'trace' keep their own
    axes (an I/V at pH 4.5 is not on the scale of one at 7.4) and share the
    key instead: only the first panel keeps it, at the very left of the row.

    Returns ``(fig, axes, tables)``.
    """
    spec = spec if isinstance(spec, PlotSpec) else PlotSpec.from_dict(spec)
    rename = _renames(spec.rename)
    # The size one panel settles on alone, measured rather than recomputed,
    # so a facet can never drift from the single panel it replaces.
    first, ax0, _ = figure_from_spec(spec, panels[0][1], paired=panels[0][2],
                                     comparisons=panels[0][3])
    box = ax0.get_window_extent().transformed(first.dpi_scale_trans.inverted())
    w_mm, h_mm = box.width / MM, box.height / MM
    plt.close(first)

    shared = spec.kind not in ("xy", "trace")
    with publication_style(font_pt=spec.font_pt, font=spec.font):
        fig, axes = plt.subplots(1, len(panels), figsize=spec.figsize(),
                                 layout="constrained", squeeze=False)
        axes = list(axes[0])
        tables = []
        for ax, (title, data, paired, comparisons) in zip(axes, panels):
            _, _, table = figure_from_spec(spec, data, paired=paired,
                                           comparisons=comparisons, ax=ax)
            tables.append(table)
            if ax_titles and title:
                ax.set_title(_shown(title, rename))
        if shared:
            value = "x" if getattr(axes[0], "_labkit_horizontal", False) else "y"
            lims = [getattr(a, f"get_{value}lim")() for a in axes]
            low, high = min(min(l) for l in lims), max(max(l) for l in lims)
            for ax in axes:
                getattr(ax, f"set_{value}lim")(low, high)
                if ax is not axes[0]:
                    getattr(ax, f"share{value}")(axes[0])
                    ax.tick_params(axis=value, **{"labelleft" if value == "y" else "labelbottom": False})
                    getattr(ax, f"set_{value}label")("")
        else:
            # ponytail: the first panel's key stands for all; a construct
            # missing from it would go unlabelled -- merge handles if that bites.
            for ax in axes[1:]:
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
    want = (w_mm, h_mm)
    fit_axes(fig, *want)
    if spec.kind == "xy" and spec.spines == "zero" and axes[0].get_legend() is not None:
        with publication_style(font_pt=spec.font_pt, font=spec.font):
            _place_legend(axes[0], spec)
        fit_axes(fig, *want)
    hold_layout(fig)
    if not shared and len(axes) > 1:
        _share_value_scale(fig, axes, spec)
        if spec.kind == "xy" and spec.spines == "zero" and axes[0].get_legend() is not None:
            with publication_style(font_pt=spec.font_pt, font=spec.font):
                _place_legend(axes[0], spec)      # again: its panel may be shorter now
    tier_level_labels(fig)
    spread_labels(fig)
    return fig, axes, tables


def _share_value_scale(fig, axes, spec: "PlotSpec") -> None:
    """Panels side by side on one scale: the same size, each axis as long in
    units as the longest of its kind, but each over its own data.

    So the same current is the same length everywhere while one panel runs
    -100 to 100 and the next 0 to 200.  Tight: the span is the widest data
    range, error bars and all, plus 3 % either side, and a narrower range
    gets the room left over split evenly either side.  Both axes -- they are
    the same size, so a shared span is a shared scale -- except one the spec
    gives limits for.
    """
    for axis, get, put, typed in (("y", "intervaly", "set_ylim", spec.ylim),
                                  ("x", "intervalx", "set_xlim", spec.xlim)):
        if _limits(typed):
            continue
        data = [tuple(getattr(ax.dataLim, get)) for ax in axes]
        if not all(np.isfinite(v) for d in data for v in d):
            continue
        widest = max(hi - lo for lo, hi in data)
        span = widest * 1.06 or 1.0
        for ax, (lo, hi) in zip(axes, data):
            middle = (lo + hi) / 2
            getattr(ax, put)(middle - span / 2, middle + span / 2)
    if spec.spines == "zero":
        for ax in axes:
            _zero_axis_labels(ax)
