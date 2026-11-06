"""The plot kinds past labkit's group and curve plots, in the same house style.

Heatmaps, cumulative distributions, stacked and percent bars, fitted curves,
survival curves, volcano plots, rasters with their PSTH, UpSet and Venn
diagrams -- and two things any curve or group plot can ask for: a value axis
broken across a gap, and a second value axis on the right.

One entry point, :func:`figure`, which takes a spec dict as :class:`plots.
PlotSpec` does (palette, colors, order, labels, font, sizes) plus the keys each
kind reads, and returns ``(fig, ax, table)`` as :func:`plots.figure_from_spec`
does: ``ax`` is the panel a figure is anchored on.

The new kinds take their data as spreadsheet columns, ``[[name, values], ...]``
(or ``{name: values}``), a cell being a number, text, or None for a blank --
which is what a sheet pasted into omnifig, or an app's table, already is.  The
two modifiers, ``break_y`` and ``right``, take whatever figure_from_spec takes
for the kind they modify.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from . import plots as P
from . import theme
from .stats import format_p

KINDS = ("heatmap", "ecdf", "stacked", "percent", "fit", "survival", "volcano",
         "raster", "upset", "venn")
MODELS = ("hill", "boltzmann", "exp", "linear")
DOWN = "#0072b2"                    # the colorblind palette's blue
FAINT = "#c3c8ce"


def wants(spec: dict) -> bool:
    """Whether a spec is one for :func:`figure` rather than figure_from_spec."""
    spec = spec or {}
    return spec.get("kind") in KINDS or bool(spec.get("break_y")) or bool(spec.get("right"))


def figure(spec: dict, data):
    """Draw the figure a spec describes: ``(fig, ax, table)``."""
    raw = dict(spec or {})
    s = P.PlotSpec.from_dict(raw)
    kind = raw.get("kind") or "bar"
    with P.publication_style(font_pt=s.font_pt, font=s.font):
        if kind in KINDS:
            cols = _columns(data)
            fig, ax, table, axes, want = DRAW[kind](cols, s, raw)
        elif raw.get("break_y"):
            fig, ax, table, axes, want = _broken(data, s, raw)
        else:
            fig, ax, table, axes, want = _second_axis(data, s, raw)
    _fit(fig, axes, *want)
    return fig, ax, table


# ----------------------------------------------------------------- helpers ---

def _columns(data) -> list[tuple[str, list]]:
    items = data.items() if isinstance(data, dict) else data
    return [(str(k), list(v)) for k, v in items]


def _num(v) -> float:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return math.nan


def _nums(vals) -> np.ndarray:
    return np.array([_num(v) for v in vals], dtype=float)


def _texty(vals) -> bool:
    """A column of names rather than numbers."""
    cells = [v for v in vals if v is not None and str(v).strip()]
    return bool(cells) and sum(math.isnan(_num(v)) for v in cells) > len(cells) / 2


def _named(cols):
    """A first column of names, and the numeric columns after it."""
    if cols and _texty(cols[0][1]):
        names = [str(v).strip() if v is not None else "" for v in cols[0][1]]
        return names, cols[1:]
    n = max((len(v) for _, v in cols), default=0)
    return [str(i + 1) for i in range(n)], cols


def _palette(s, names) -> list[str]:
    base = P.PALETTES.get(s.palette, theme.SERIES)
    return [str(s.colours.get(n) or base[i % len(base)]) for i, n in enumerate(names)]


def _order(names, s):
    return P.order_series(names, s.order) if s.order else list(names)


def _flag(raw, key, default=False) -> bool:
    v = raw.get(key, default)
    return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "on", "yes")


def _float(raw, key, default):
    v = _num(raw.get(key, default))
    return default if math.isnan(v) else v


def _p(s, p) -> str:
    return format_p(p, style=s.p_style if s.p_style != "stars" else "auto")


def _sizes(s, w_mm, h_mm):
    """The data area asked for: the spec's typed sizes over the kind's own."""
    return (s.axes_width_mm or w_mm, s.axes_height_mm or h_mm)


def _new(s, rows=1, cols=1, **kw):
    w, h = s.figsize()
    return plt.subplots(rows, cols, figsize=(w, h), layout="constrained", **kw)


def _labels(ax, s, xlabel="", ylabel=""):
    ax.set_xlabel(s.xlabel or xlabel)
    ax.set_ylabel(s.ylabel or ylabel)
    if s.title:
        ax.set_title(s.title)


def _fit(fig, axes, want_w_mm, want_h_mm, passes=6):
    """Grow the canvas until the span of ``axes`` is the size asked, in mm --
    :func:`plots.fit_axes` for a panel made of several axes -- then hold it."""
    want_w, want_h = want_w_mm * P.MM, want_h_mm * P.MM
    if want_w <= 0 and want_h <= 0:
        return
    for _ in range(passes):
        with warnings.catch_warnings(record=True) as said:
            warnings.simplefilter("always")
            fig.draw_without_rendering()
        wide, high = fig.get_size_inches()
        if any("collapsed to zero" in str(w.message) for w in said):
            # Smaller than its own labels, so laid out not at all: give it
            # what they need and look again (as plots.fit_axes does).
            need = fig.get_tightbbox()
            fig.set_size_inches(max(wide, need.width * 1.1) * 1.3, max(high, need.height * 1.1) * 1.3)
            continue
        inv = fig.dpi_scale_trans.inverted()
        boxes = [a.get_window_extent().transformed(inv) for a in axes]
        span_w = max(b.x1 for b in boxes) - min(b.x0 for b in boxes)
        span_h = max(b.y1 for b in boxes) - min(b.y0 for b in boxes)
        if span_w <= 0 or span_h <= 0:
            fig.set_size_inches(wide * 1.6, high * 1.6)
            continue
        dw = want_w - span_w if want_w > 0 else 0.0
        dh = want_h - span_h if want_h > 0 else 0.0
        fig.set_size_inches(max(wide + dw, 0.3), max(high + dh, 0.3))
        if abs(dw) < 0.008 and abs(dh) < 0.008:
            break
    P.hold_layout(fig)


# ----------------------------------------------------------------- heatmap ---

def heatmap(cols, s, raw):
    """Rows named by a first column of names, a column of the sheet each.
    ``cluster``: none | rows | cols | both; ``center``: the value the diverging
    map is centred on (a matrix crossing zero centres on 0 unasked);
    ``annotate`` writes each value in its cell; ``cell_mm`` sizes a cell;
    ``cmap`` names any matplotlib map; the color bar is labeled by ``ylabel``."""
    rows, numeric = _named(cols)
    names = [n for n, _ in numeric]
    m = np.array([_nums(v).tolist() + [math.nan] * (len(rows) - len(v)) for _, v in numeric]).T
    if not m.size:
        raise ValueError("a heatmap wants a column of names, then a column of numbers each")
    m = m[:len(rows)]
    how = str(raw.get("cluster") or "none")
    ri, ci = np.arange(m.shape[0]), np.arange(m.shape[1])
    if how in ("rows", "both") and m.shape[0] > 2:
        ri = _leaves(m)
    if how in ("cols", "both") and m.shape[1] > 2:
        ci = _leaves(m.T)
    m, rows, names = m[ri][:, ci], [rows[i] for i in ri], [names[i] for i in ci]
    lo, hi = np.nanmin(m), np.nanmax(m)
    center = raw.get("center")
    if center in (None, "") and lo < 0 < hi:
        center = 0.0
    kw = {}
    if center not in (None, ""):
        c = float(center)
        reach = max(abs(hi - c), abs(c - lo)) or 1.0
        kw = {"vmin": c - reach, "vmax": c + reach}
    # 64 steps: no eye tells them from a smooth map, and the color bar stays
    # 64 shapes rather than a picture.
    cmap = plt.get_cmap(raw.get("cmap") or ("RdBu_r" if kw else "viridis"), 64)
    fig, ax = _new(s)
    # Cells as shapes, not a picture: a figure editor can recolor or pick one.
    im = ax.pcolormesh(np.arange(m.shape[1] + 1) - 0.5, np.arange(m.shape[0] + 1) - 0.5,
                       np.ma.masked_invalid(m), cmap=cmap, **kw)
    ax.invert_yaxis()
    ax.set_xticks(range(len(names)), names, rotation=90)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    if _flag(raw, "annotate"):
        norm = im.norm
        for (i, j), v in np.ndenumerate(m):
            if not math.isnan(v):
                dark = norm(v) > 0.6 if not kw else abs(norm(v) - 0.5) > 0.3
                ax.text(j, i, f"{v:.3g}", ha="center", va="center",
                        fontsize=plt.rcParams["font.size"] * 0.8,
                        color="white" if dark else "#111111")
    bar = fig.colorbar(im, ax=ax, fraction=0.05, aspect=15, pad=0.02)
    bar.outline.set_visible(False)
    bar.solids.set_rasterized(False)
    bar.set_label(s.ylabel)
    if s.title:
        ax.set_title(s.title)
    cell = _float(raw, "cell_mm", 4.0)
    return fig, ax, None, [ax], _sizes(s, cell * m.shape[1], cell * m.shape[0])


def _leaves(m):
    """The order hierarchical clustering puts the rows of a matrix in."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    filled = np.where(np.isnan(m), np.nanmean(m), m)
    return leaves_list(linkage(filled, "average", metric="euclidean"))


# --------------------------------------------------------------------- ECDF ---

def ecdf(cols, s, raw):
    """A cumulative distribution per column -- amplitudes, intervals -- each
    against the first by a two-sample Kolmogorov-Smirnov test."""
    from scipy.stats import ks_2samp
    groups = {n: v[~np.isnan(v)] for n, v in ((n, _nums(v)) for n, v in cols)}
    groups = {n: np.sort(groups[n]) for n in _order(list(groups), s) if groups[n].size}
    if not groups:
        raise ValueError("no numbers to plot")
    names = list(groups)
    colors = _palette(s, names)
    fig, ax = _new(s)
    first = groups[names[0]]
    table = []
    for name, v, c in zip(names, groups.values(), colors):
        y = np.arange(1, v.size + 1) / v.size
        label = f"{name} (n = {v.size})" if s.show_n else name
        if s.show_p and name != names[0] and first.size and v.size:
            p = ks_2samp(first, v).pvalue
            table.append({"a": names[0], "b": name, "test": "Kolmogorov-Smirnov", "p": float(p)})
            label += ", " + _p(s, p)
        ax.step(np.r_[v[0], v], np.r_[0, y], where="post", color=c, label=label,
                lw=s.line_width or 1.0)
    ax.set_ylim(0, 1.02)
    _labels(ax, s, ylabel="Cumulative fraction")
    if s.legend:
        ax.legend(frameon=False, loc="lower right", handlelength=1.2)
    P.style_axes(ax, s)
    return fig, ax, table, [ax], _sizes(s, P.AXES_LENGTH_MM, P.AXES_LENGTH_MM)


# ---------------------------------------------------------- stacked/percent ---

def stacked(cols, s, raw, percent=False):
    """A bar per row (named by a first column of names), stacked by the counts
    in each column after it; ``percent`` draws each as shares of its total.
    Rows against columns are tested by Fisher's exact test for two of each,
    and chi-square otherwise."""
    rows, numeric = _named(cols)
    outs = [n for n, _ in numeric]
    m = np.nan_to_num(np.array([_nums(v)[:len(rows)] for _, v in numeric]).T)
    keep = m.sum(axis=1) > 0
    rows, m = [r for r, k in zip(rows, keep) if k], m[keep]
    if not len(rows) or not outs:
        raise ValueError("a column of group names, then a column of counts for each outcome")
    order = _order(rows, s)
    m = m[[rows.index(r) for r in order]]
    rows = order
    shown = m / m.sum(axis=1, keepdims=True) * 100 if percent else m
    colors = _palette(s, outs)
    fig, ax = _new(s)
    x = np.arange(len(rows))
    base = np.zeros(len(rows))
    for j, (name, c) in enumerate(zip(outs, colors)):
        ax.bar(x, shown[:, j], P.BAR_WIDTH, bottom=base, color=c, label=name,
               edgecolor="white", linewidth=0.4)
        base += shown[:, j]
    ax.set_xticks(x, rows)
    if s.show_n:
        for i, t in enumerate(m.sum(axis=1)):
            ax.annotate(str(int(t)), (i, base[i]), xytext=(0, 1.5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=plt.rcParams["font.size"] * 0.85)
    ax.set_xlim(-0.5, len(rows) - 0.5)
    if percent:
        ax.set_ylim(0, 100)
    _labels(ax, s, ylabel="%" if percent else "Count")
    table = []
    if s.show_p and len(rows) > 1 and len(outs) > 1:
        from scipy.stats import chi2_contingency, fisher_exact
        if m.shape == (2, 2):
            p, test = fisher_exact(m).pvalue, "Fisher's exact"
        else:
            p, test = chi2_contingency(m).pvalue, "chi-square"
        table.append({"test": test, "p": float(p)})
        ax.set_title(_p(s, p), fontsize=plt.rcParams["font.size"],
                     pad=plt.rcParams["font.size"] * (1.8 if s.show_n else 0.6))
    if s.legend:
        hs, ls = ax.get_legend_handles_labels()
        ax.legend(hs[::-1], ls[::-1], frameon=False, loc="upper left",
                  bbox_to_anchor=(1.0, 1.0), handlelength=0.8, handleheight=0.8)
    P.style_axes(ax, s)
    return fig, ax, table, [ax], _sizes(s, s.bar_mm * len(rows) / P.BAR_WIDTH, P.AXES_LENGTH_MM)


# ---------------------------------------------------------------- fit curves ---

def _curves(cols):
    """x first, then a column per curve; a column named for the one before it
    plus "sem" is its spread (as omnifig's curves read them)."""
    if len(cols) < 2:
        raise ValueError("an x column first, then a column for each curve")
    x = _nums(cols[0][1])
    out = []
    for name, vals in cols[1:]:
        mm = re.fullmatch(r"(.*?)[\s_]*sem", name, re.I)
        if mm and out and mm.group(1) in ("", out[-1]["label"]):
            out[-1]["sem"] = _nums(vals)
        else:
            out.append({"label": name, "x": x, "y": _nums(vals)})
    return out


def _model(name):
    if name == "hill":
        f = lambda x, bottom, top, ec50, n: bottom + (top - bottom) / (1 + (ec50 / np.maximum(x, 1e-300)) ** n)  # noqa: E731
        guess = lambda x, y: [y.min(), y.max(), np.exp(np.mean(np.log(x[x > 0]))) if (x > 0).any() else 1.0, 1.0]  # noqa: E731
        say = lambda p: f"EC50 = {p[2]:.3g}, n = {p[3]:.2g}"  # noqa: E731
    elif name == "boltzmann":
        f = lambda x, bottom, top, half, k: bottom + (top - bottom) / (1 + np.exp((half - x) / k))  # noqa: E731
        guess = lambda x, y: [y.min(), y.max(), x[np.argmin(abs(y - (y.min() + y.max()) / 2))],  # noqa: E731
                              (x.max() - x.min()) / 10 * (1 if y[np.argmax(x)] > y[np.argmin(x)] else -1)]
        say = lambda p: f"V½ = {p[2]:.3g}, k = {p[3]:.2g}"  # noqa: E731
    elif name == "exp":
        f = lambda x, a, tau, c: a * np.exp(-(x - x0[0]) / tau) + c  # noqa: E731
        x0 = [0.0]
        guess = lambda x, y: (x0.__setitem__(0, x.min()) or  # noqa: E731
                              [y[np.argmin(x)] - y[np.argmax(x)], (x.max() - x.min()) / 3, y[np.argmax(x)]])
        say = lambda p: f"τ = {p[1]:.3g}"  # noqa: E731
    elif name == "linear":
        f = lambda x, a, b: a + b * x  # noqa: E731
        guess = lambda x, y: [0.0, 1.0]  # noqa: E731
        say = lambda p: f"slope = {p[1]:.3g}"  # noqa: E731
    else:
        raise ValueError(f"no model {name!r}: one of {', '.join(MODELS)}")
    return f, guess, say


def fit(cols, s, raw):
    """Points with a fitted curve through each, its parameters in the legend.
    ``model``: hill (dose-response, x on a log axis), boltzmann (activation,
    inactivation), exp (a decay from the first x), linear (with r²).  A curve's
    "sem" column weights its fit and draws its error bars."""
    from scipy.optimize import curve_fit
    name = str(raw.get("model") or "hill")
    f, guess, say = _model(name)
    curves = _curves(cols)
    curves = [next(c for c in curves if c["label"] == n) for n in _order([c["label"] for c in curves], s)]
    colors = _palette(s, [c["label"] for c in curves])
    fig, ax = _new(s)
    table = []
    log_x = name == "hill" and _flag(raw, "log_x", True)
    for c, col in zip(curves, colors):
        ok = ~np.isnan(c["x"]) & ~np.isnan(c["y"])
        x, y = c["x"][ok], c["y"][ok]
        sem = c.get("sem")
        sem = sem[ok] if sem is not None else None
        ax.errorbar(x, y, yerr=sem, fmt="o", ms=s.marker_size or 2.6, color=col,
                    elinewidth=0.6, capsize=1.2, lw=0)
        label = c["label"]
        if x.size > len(guess(x, y)):
            try:
                w = sem if sem is not None and np.all(sem > 0) else None
                p, _ = curve_fit(f, x, y, p0=guess(x, y), sigma=w, maxfev=20000)
                xs = (np.geomspace(x[x > 0].min(), x.max(), 200) if log_x and (x > 0).any()
                      else np.linspace(x.min(), x.max(), 200))
                ax.plot(xs, f(xs, *p), color=col, lw=s.line_width or 1.0)
                text = say(p)
                if name == "linear":
                    r2 = 1 - np.sum((y - f(x, *p)) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-300)
                    text += f", r² = {r2:.2f}"
                label = f"{label}: {text}"
                table.append({"label": c["label"], "model": name, "params": [float(v) for v in p]})
            except (RuntimeError, ValueError) as exc:
                label = f"{label}: no fit ({str(exc).split('.')[0]})"
        ax.plot([], [], "o-", color=col, ms=s.marker_size or 2.6, lw=s.line_width or 1.0, label=label)
    if log_x:
        ax.set_xscale("log")
    _labels(ax, s)
    if s.legend:
        ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0,
                  handlelength=1.4)
    P.style_axes(ax, s)
    return fig, ax, table, [ax], _sizes(s, P.AXES_LENGTH_MM * 1.25, P.AXES_LENGTH_MM)


# ----------------------------------------------------------------- survival ---

def _censored(t, e):
    from scipy.stats import CensoredData
    return CensoredData(uncensored=t[e], right=t[~e])


def survival(cols, s, raw):
    """Kaplan-Meier curves: a column of times per group, each optionally
    followed by one named for it plus "event" (1 the event, 0 censored) or
    "censored" (the other way round); without one, every time is an event.
    Censored times are ticked, the numbers at risk written under the axis, and
    each group set against the first by the log-rank test."""
    from scipy.stats import ecdf as _ecdf, logrank
    groups = {}
    for name, vals in cols:
        mm = re.fullmatch(r"(.*?)[\s_]*(event|events|status|censored|censor)", name, re.I)
        if mm and groups and mm.group(1) in ("", list(groups)[-1]):
            last = list(groups)[-1]
            flag = _nums(vals)[:groups[last][0].size] > 0
            groups[last] = (groups[last][0], ~flag if mm.group(2).lower().startswith("censor") else flag)
        else:
            t = _nums(vals)
            groups[name] = (t, np.ones(t.size, bool))
    clean = {}
    for name in _order(list(groups), s):
        t, e = groups[name]
        ok = ~np.isnan(t)
        if ok.any():
            clean[name] = (t[ok], e[:t.size][ok])
    if not clean:
        raise ValueError("a column of times for each group")
    names = list(clean)
    colors = _palette(s, names)
    fig, ax = _new(s)
    table = []
    first = _censored(*clean[names[0]])
    for name, col in zip(names, colors):
        t, e = clean[name]
        sf = _ecdf(_censored(t, e)).sf
        xs, ps = np.r_[0, sf.quantiles], np.r_[1, sf.probabilities]
        label = name
        if s.show_p and name != names[0]:
            p = logrank(first, _censored(t, e)).pvalue
            table.append({"a": names[0], "b": name, "test": "log-rank", "p": float(p)})
            label += ", " + _p(s, p)
        ax.step(xs, ps, where="post", color=col, lw=s.line_width or 1.0, label=label)
        cut = t[~e]
        if cut.size:
            ax.plot(cut, sf.evaluate(cut), "|", color=col, ms=3, mew=0.6)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    _labels(ax, s, "Time", "Fraction surviving")
    if s.legend:
        ax.legend(frameon=False, loc="lower left", handlelength=1.2)
    P.style_axes(ax, s)
    if _flag(raw, "at_risk", True):
        fig.draw_without_rendering()
        ticks = [v for v in ax.get_xticks() if ax.get_xlim()[0] <= v <= ax.get_xlim()[1]]
        ax.set_xticks(ticks)
        size = plt.rcParams["xtick.labelsize"]
        line = size * 1.25
        top = -(plt.rcParams["axes.labelsize"] * 1.6 + size * 2.4)
        ax.annotate("At risk", (0, 0), xycoords="axes fraction", xytext=(0, top + line),
                    textcoords="offset points", fontsize=size, ha="right", va="top")
        for i, (name, col) in enumerate(zip(names, colors)):
            t = clean[name][0]
            for v in ticks:
                ax.annotate(str(int((t >= v).sum())), (v, 0), xycoords=("data", "axes fraction"),
                            xytext=(0, top - i * line), textcoords="offset points",
                            fontsize=size, ha="center", va="top", color=col)
    return fig, ax, table, [ax], _sizes(s, P.AXES_LENGTH_MM * 1.5, P.AXES_LENGTH_MM)


# ------------------------------------------------------------------ volcano ---

def volcano(cols, s, raw):
    """A first column of names, then the log2 fold change and the p-value (a
    column named p, padj, q or fdr is taken for p wherever it stands).  Past
    ``fc_cut`` either way and under ``p_cut`` is up (red) or down (blue); the
    ``label_top`` smallest p of those are named."""
    names, numeric = _named(cols)
    if len(numeric) < 2:
        raise ValueError("names, then the log2 fold change and the p-value")
    pi = next((i for i, (n, _) in enumerate(numeric) if re.search(r"\b(p|padj|q|fdr|p.?val\w*)\b", n, re.I)), 1)
    fi = next(i for i in range(len(numeric)) if i != pi)
    fc, p = _nums(numeric[fi][1]), _nums(numeric[pi][1])
    n = min(fc.size, p.size, len(names))
    fc, p, names = fc[:n], p[:n], names[:n]
    ok = ~np.isnan(fc) & ~np.isnan(p) & (p > 0)
    fc, p, names = fc[ok], p[ok], [m for m, k in zip(names, ok) if k]
    y = -np.log10(p)
    fc_cut, p_cut = _float(raw, "fc_cut", 1.0), _float(raw, "p_cut", 0.05)
    up, down = (fc >= fc_cut) & (p < p_cut), (fc <= -fc_cut) & (p < p_cut)
    up_c = s.colours.get("up") or theme.RED
    down_c = s.colours.get("down") or DOWN
    fig, ax = _new(s)
    ms = (s.point_size or 4) * 0.8
    rest = ~(up | down)
    ax.scatter(fc[rest], y[rest], s=ms, color=FAINT, lw=0, label=f"ns ({rest.sum()})")
    ax.scatter(fc[up], y[up], s=ms, color=up_c, lw=0, label=f"up ({up.sum()})")
    ax.scatter(fc[down], y[down], s=ms, color=down_c, lw=0, label=f"down ({down.sum()})")
    for v in (-fc_cut, fc_cut):
        ax.axvline(v, color=theme.GREY, lw=0.5, ls=(0, (2, 2)))
    ax.axhline(-math.log10(p_cut), color=theme.GREY, lw=0.5, ls=(0, (2, 2)))
    top = int(_float(raw, "label_top", 10))
    hits = np.where(up | down)[0]
    size = plt.rcParams["font.size"] * 0.85
    _place_labels(ax, [(names[i], fc[i], y[i]) for i in hits[np.argsort(p[hits])][:top]], size)
    _labels(ax, s, "log$_2$ fold change", "−log$_{10}$ p")
    if s.legend:
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), handletextpad=0.1,
                  markerscale=1.5)
    P.style_axes(ax, s)
    return fig, ax, None, [ax], _sizes(s, P.AXES_LENGTH_MM * 1.5, P.AXES_LENGTH_MM * 1.5)


def _place_labels(ax, items, size):
    """Names beside their points, each moved up or down off the ones placed
    before it, in points; one with nowhere to go is left out.
    ponytail: greedy and in label order, not a proper repel."""
    fig = ax.figure
    fig.draw_without_rendering()
    to_pt = 72.0 / fig.dpi
    placed = []
    for name, x, y in items:
        px, py = (v * to_pt for v in ax.transData.transform((x, y)))
        w, h = len(name) * size * 0.62, size * 1.1
        for dy in (2, -h - 2, h + 2, -2 * h - 2, 2 * h + 2):
            box = (px + 2, py + dy, px + 2 + w, py + dy + h)
            if all(box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3] for b in placed):
                placed.append(box)
                ax.annotate(name, (x, y), xytext=(2, dy), textcoords="offset points", fontsize=size,
                            arrowprops=None if abs(dy) < h else dict(arrowstyle="-", lw=0.4, color=theme.GREY,
                                                                     shrinkA=0, shrinkB=1))
                break


# ------------------------------------------------------------ raster + PSTH ---

def raster(cols, s, raw):
    """A row of ticks per column of event times -- a trial, a cell -- over the
    PSTH of each group: columns named "WT 1", "WT 2" are the group "WT".  The
    PSTH is the mean rate in ``bin`` wide bins (picked if not given), in Hz
    for times in ``x_unit`` s or ms; a line marks time 0 where it shows."""
    trials = [(n, v[~np.isnan(v)]) for n, v in ((n, _nums(v)) for n, v in cols)]
    if not trials:
        raise ValueError("a column of event times for each trial")
    group_of = lambda n: re.sub(r"[\s_#-]*\d+$", "", n) or n  # noqa: E731
    groups = _order(list(dict.fromkeys(group_of(n) for n, _ in trials)), s)
    trials.sort(key=lambda t: groups.index(group_of(t[0])))
    colors = dict(zip(groups, _palette(s, groups)))
    allv = np.concatenate([v for _, v in trials]) if trials else np.array([0.0])
    lo, hi = (float(allv.min()), float(allv.max())) if allv.size else (0.0, 1.0)
    width = _float(raw, "bin", 0.0) or P._nice((hi - lo) / 40 or 1.0)
    edges = np.arange(math.floor(lo / width) * width, hi + width, width)
    per_s = 1000.0 if (raw.get("x_unit") or "ms") == "ms" else 1.0
    fig, (top, bottom) = _new(s, 2, 1, sharex=True, height_ratios=[1.6, 1])
    top.eventplot([v for _, v in trials], colors=[colors[group_of(n)] for n, _ in trials],
                  linelengths=0.8, linewidths=0.6)
    top.set_ylim(len(trials) - 0.5, -0.5)
    top.set_yticks([])
    top.set_ylabel("Trial")
    top.tick_params(labelbottom=False, bottom=False)
    top.spines["bottom"].set_visible(False)
    for g in groups:
        vs = [v for n, v in trials if group_of(n) == g]
        counts = np.histogram(np.concatenate(vs), edges)[0] if vs else np.zeros(edges.size - 1)
        rate = counts / max(len(vs), 1) / (width / per_s)
        bottom.stairs(rate, edges, color=colors[g], lw=s.line_width or 1.0,
                      label=g if len(groups) > 1 else None)
    for a in (top, bottom):
        if edges[0] < 0 < edges[-1]:
            a.axvline(0, color=theme.GREY, lw=0.5, ls=(0, (2, 2)))
    bottom.set_xlabel(s.xlabel or f"Time ({raw.get('x_unit') or 'ms'})")
    bottom.set_ylabel(s.ylabel or "Rate (Hz)")
    if s.title:
        top.set_title(s.title)
    if s.legend and len(groups) > 1:
        bottom.legend(frameon=False, loc="upper right", handlelength=1.0)
    return fig, bottom, None, [top, bottom], _sizes(s, P.AXES_LENGTH_MM * 1.5, P.AXES_LENGTH_MM * 1.6)


# ------------------------------------------------------------- UpSet / Venn ---

def _sets(cols, s):
    sets = {}
    for name, vals in cols:
        items = {str(v).strip() for v in vals if v is not None and str(v).strip()}
        items = {i[:-2] if i.endswith(".0") else i for i in items}
        if items:
            sets[name] = items
    names = _order(list(sets), s)
    if len(names) < 2:
        raise ValueError("two or more columns, each the members of a set")
    return names, [sets[n] for n in names]


def upset(cols, s, raw):
    """Each column the members of a set.  The bars are the sizes of each
    exclusive intersection, largest first (``max_bars`` of them), the dots
    under them which sets that is, the bars on the left each set's size."""
    names, sets = _sets(cols, s)
    every = set().union(*sets)
    combos = {}
    for item in every:
        key = tuple(item in st for st in sets)
        combos[key] = combos.get(key, 0) + 1
    shown = sorted(combos.items(), key=lambda kv: (-kv[1], kv[0]))[:int(_float(raw, "max_bars", 20))]
    k, n = len(shown), len(names)
    fig, axs = _new(s, 2, 2, width_ratios=[1, max(k, 1) * 0.35 + 0.3], height_ratios=[1.6, n * 0.3 + 0.2])
    corner, bars, sizes, dots = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    corner.set_axis_off()
    ink = s.colours.get("bars") or "#2b2f36"
    x = np.arange(k)
    bars.bar(x, [c for _, c in shown], 0.6, color=ink)
    for i, (_, c) in enumerate(shown):
        bars.text(i, c, str(c), ha="center", va="bottom", fontsize=plt.rcParams["font.size"] * 0.85)
    bars.set_xlim(-0.6, k - 0.4)
    bars.set_xticks([])
    bars.spines["bottom"].set_visible(False)
    bars.set_ylabel(s.ylabel or "Intersection size")
    for i, (key, _) in enumerate(shown):
        on = [j for j, b in enumerate(key) if b]
        dots.scatter([i] * n, range(n), s=14, color=FAINT, lw=0)
        dots.scatter([i] * len(on), on, s=14, color=ink, lw=0)
        if len(on) > 1:
            dots.plot([i, i], [min(on), max(on)], color=ink, lw=1.0)
    dots.set_xlim(-0.6, k - 0.4)
    dots.set_ylim(n - 0.5, -0.5)
    dots.set_axis_off()
    colors = _palette(s, names)
    sizes.barh(range(n), [len(st) for st in sets], 0.6, color=colors)
    sizes.set_ylim(n - 0.5, -0.5)
    sizes.invert_xaxis()
    sizes.yaxis.tick_right()
    sizes.set_yticks(range(n), names)
    sizes.tick_params(axis="y", length=0)
    sizes.spines["left"].set_visible(False)
    sizes.spines["right"].set_visible(False)
    sizes.set_xlabel(s.xlabel or "Set size")
    if s.title:
        bars.set_title(s.title)
    return fig, bars, None, [bars, dots], _sizes(s, max(k, 1) * 4.0, P.AXES_LENGTH_MM + n * 3.0)


def venn(cols, s, raw):
    """Two or three columns, each the members of a set, as overlapping circles
    with the count in each region.  More sets than three want :func:`upset`."""
    from matplotlib.patches import Circle
    names, sets = _sets(cols, s)
    if len(sets) > 3:
        raise ValueError("a Venn diagram holds two or three sets; more read as an UpSet plot")
    colors = _palette(s, names)
    fig, ax = _new(s)
    ax.set_aspect("equal")
    ax.set_axis_off()
    r = 1.0
    if len(sets) == 2:
        centers = [(-0.5, 0), (0.5, 0)]
        a, b = sets
        spots = {(1, 0): (-0.95, 0), (0, 1): (0.95, 0), (1, 1): (0, 0)}
        counts = {(1, 0): len(a - b), (0, 1): len(b - a), (1, 1): len(a & b)}
        name_at = [(-0.5, 1.12, "center"), (0.5, 1.12, "center")]
    else:
        centers = [(-0.55, 0.32), (0.55, 0.32), (0, -0.62)]
        every = set().union(*sets)
        counts = {}
        for item in every:
            key = tuple(int(item in st) for st in sets)
            counts[key] = counts.get(key, 0) + 1
        spots = {(1, 0, 0): (-1.0, 0.55), (0, 1, 0): (1.0, 0.55), (0, 0, 1): (0, -1.15),
                 (1, 1, 0): (0, 0.72), (1, 0, 1): (-0.62, -0.38), (0, 1, 1): (0.62, -0.38),
                 (1, 1, 1): (0, 0.02)}
        name_at = [(-1.25, 1.4, "right"), (1.25, 1.4, "left"), (0, -1.78, "center")]
    for (cx, cy), c in zip(centers, colors):
        ax.add_patch(Circle((cx, cy), r, facecolor=c, alpha=0.28, edgecolor=c, lw=0.8))
    for key, at in spots.items():
        ax.text(*at, str(counts.get(key, 0)), ha="center", va="center")
    for (nx, ny, ha), name, c, st in zip(name_at, names, colors, sets):
        ax.text(nx, ny, f"{name} ({len(st)})" if s.show_n else name, ha=ha, va="center",
                color=c, fontsize=plt.rcParams["axes.labelsize"])
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.95 if len(sets) == 3 else -1.15, 1.55)
    if s.title:
        ax.set_title(s.title)
    w = P.AXES_LENGTH_MM * 1.3
    return fig, ax, None, [ax], _sizes(s, w, w * (3.5 if len(sets) == 3 else 2.7) / 3.5)


DRAW = {"heatmap": heatmap, "ecdf": ecdf, "stacked": stacked,
        "percent": lambda c, s, r: stacked(c, s, r, percent=True), "fit": fit,
        "survival": survival, "volcano": volcano, "raster": raster,
        "upset": upset, "venn": venn}


# ------------------------------------------------- broken and second axes ---

def _broken(data, s, raw):
    """Any figure_from_spec kind with the values between ``break_y`` = [low,
    high] cut out: one tall bar no longer flattens the rest.  The upper part
    gets ``break_share`` of the height (a third unless said).  The
    significance brackets hang on the outer edge of the upper part: standing
    on their data, a pair of short groups would put theirs inside the break."""
    lo, hi = sorted(float(v) for v in raw["break_y"])
    share = min(max(_float(raw, "break_share", 1 / 3), 0.1), 0.9)
    fig, (top, bottom) = _new(s, 2, 1, sharex=True, height_ratios=[share, 1 - share])
    data = P.reorder_data(data, s.order)
    P.figure_from_spec(replace(s, xlabel="", ylabel="", show_p=False), data, ax=top)
    P.figure_from_spec(replace(s, show_p=False, title="", ylabel=""), data, ax=bottom)
    table = None
    # Points and error bars go unclipped so the axes do not cut them; across
    # a break that puts each half's data on the other.
    for a in (top, bottom):
        for c in a.lines + a.collections + a.patches:
            c.set_clip_on(True)
    top.set_ylim(hi, max(top.get_ylim()[1], hi * 1.05 if hi > 0 else hi + 1))
    ns = list(top.texts) if s.show_n else []    # before the p labels join them
    if s.show_p and s.kind not in ("xy", "trace", "hist"):
        table = P.comparisons_for(s, data)
        if table is not None and len(table):
            # Group i sits at x = i, as group_plot put it: the upper half
            # shares x and has no tick labels to find them by.
            at = {k: i for i, k in enumerate(data)}
            P.annotate_p(top, table, at, style=s.p_style, only_significant=s.p_only_significant,
                         alpha=s.p_alpha, use_corrected=s.p_use_corrected, outside=True)
    bottom.set_ylim(min(bottom.get_ylim()[0], 0 if lo > 0 else lo - 1), lo)
    top.spines["bottom"].set_visible(False)
    top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    for t in ns:
        if t.get_position()[1] < hi:
            t.set_visible(False)
    bottom.spines["top"].set_visible(False)
    top.legend_ and bottom.legend_ and bottom.legend_.remove()
    # The two slashes across the value axis, in points so a resize keeps them.
    d = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=5, linestyle="none", color="k",
             mec="k", mew=0.6, clip_on=False)
    top.plot([0], [0], transform=top.transAxes, **d)
    bottom.plot([0], [1], transform=bottom.transAxes, **d)
    fig.supylabel(s.ylabel, fontsize=plt.rcParams["axes.labelsize"])
    kind = s.kind
    if kind in ("xy", "trace"):
        w = s.axes_width_mm or P.AXES_LENGTH_MM * 1.25
    else:
        n = len(data)
        w = s.axes_width_mm or s.bar_mm * n / P.BAR_WIDTH
    return fig, bottom, table, [top, bottom], (w, s.axes_height_mm or P.AXES_LENGTH_MM)


def _second_axis(data, s, raw):
    """Curves (``xy`` or ``trace``) with those named in ``right`` drawn
    against a second value axis on the right, labeled by ``ylabel2``."""
    right = set(raw["right"] if isinstance(raw["right"], list) else
                [r.strip() for r in str(raw["right"]).split(",") if r.strip()])
    data = P.reorder_data(data, s.order)
    colors = P.colours_for(s, data)
    left = [(d, c) for d, c in zip(data, colors) if d["label"] not in right]
    rhs = [(d, c) for d, c in zip(data, colors) if d["label"] in right]
    if not rhs:
        raise ValueError("none of the curves is called " + ", ".join(sorted(right)))
    fig, ax = _new(s)
    other = ax.twinx()
    for a, part in ((ax, left), (other, rhs)):
        if part:
            P.xy_plot([d for d, _ in part], ax=a, palette=[c for _, c in part],
                      error=s.error_style if s.error != "none" else "none",
                      markers=bool(s.markers) if s.kind == "xy" else False,
                      join=s.join, line_width=s.line_width, zero_lines=False, legend=False)
    ax.set_xlabel(s.xlabel)
    ax.set_ylabel(s.ylabel, color=left[0][1] if len(left) == 1 else None)
    other.set_ylabel(str(raw.get("ylabel2") or ""), color=rhs[0][1] if len(rhs) == 1 else None)
    other.spines["right"].set_visible(True)
    other.spines["top"].set_visible(False)
    if s.title:
        ax.set_title(s.title)
    if s.legend:
        hs = [h for a in (ax, other) for h in a.get_legend_handles_labels()[0]]
        ls = [lab for a in (ax, other) for lab in a.get_legend_handles_labels()[1]]
        if hs:
            ax.legend(hs, ls, frameon=False, loc="best")
    P.style_axes(ax, s)
    return fig, ax, None, [ax], _sizes(s, P.AXES_LENGTH_MM * 1.25, P.AXES_LENGTH_MM)
