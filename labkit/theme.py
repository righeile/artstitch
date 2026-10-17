"""The one place any color in any of these apps is decided.

Everything here is red-on-gray: gray carries the interface, red means *you*
(selection, the active tab, the primary action, the thing being edited).  The
rule is worth stating because it is what keeps the accent readable -- if red
also meant "bad" or "channel 2" it would stop meaning "here".

Two palettes live here, and they are deliberately not the same:

``SCREEN``
    The interface, dark by default.  Tuned for a monitor in a dim microscope
    room.
``PRINT``
    What figures are drawn in.  Tuned for white paper: gray first, red second,
    and every series distinguishable after a grayscale photocopy, because
    reviewers still print things.

``css()`` renders SCREEN into the custom properties every app's stylesheet
reads, so a change here reaches all six interfaces, and ``PRINT`` feeds
``labkit.plots``, so it reaches every figure.
"""

from __future__ import annotations

__all__ = [
    "SCREEN", "SCREEN_LIGHT", "PRINT", "SERIES", "SERIES_NAMES",
    "css", "series_colour", "GREY", "RED",
]


# --------------------------------------------------------------------------
# Screen: the interface
# --------------------------------------------------------------------------

#: Dark theme.  The red is Radix red-9, the one that stays legible at 11px.
SCREEN: dict[str, str] = {
    # surfaces, back to front
    "bg":        "#1a1d21",   # the window
    "panel":     "#22262b",   # side panels, toolbars, headers
    "panel2":    "#2a2f36",   # raised things on a panel: buttons, tab strip
    "field":     "#15181c",   # inset things: inputs, readouts, lists, canvas gutter
    "canvas":    "#101215",   # the stage an image or figure sits on
    "raise":     "#333941",   # hover on a raised surface
    "line":      "#3a4048",   # every border
    "edge":      "#4b525c",   # border on hover / focus-adjacent

    # text
    "ink":       "#e2e6eb",   # body text
    "dim":       "#949ba5",   # labels, captions, units
    "faint":     "#6d747d",   # disabled, placeholder, "nothing here yet"

    # the accent, and only the accent
    "accent":      "#e5484d",  # selection, active tab, links, progress
    "accent-deep": "#8f3034",  # filled primary buttons
    "accent-hot":  "#a83a3f",  # those buttons, hovered
    "accent-edge": "#b8474d",  # their border
    "accent-wash": "rgba(229,72,77,.12)",   # a selected row's background
    "accent-veil": "rgba(229,72,77,.30)",   # a drop target

    # status.  Deliberately few: red is spoken for, so "bad" is a hotter red
    # that only ever appears as text or a thin rule, never as a fill.
    "ok":     "#4cb782",
    "warn":   "#d99a4e",
    "bad":    "#ff6b6b",
    "on-accent": "#ffffff",   # text on a filled accent surface
    "on-status": "#14171a",   # text on a filled ok/warn/bad surface

    # a neutral ramp for "quality" overlays on canvases, gray -> red, so that
    # "nothing wrong" is calm and only trouble is colored
    "q-good":    "#79818c",
    "q-warn":    "#c8735f",
    "q-bad":     "#ff5d5d",
    "q-unknown": "#565d67",
    "sel":       "#f2f5f8",   # what *you* selected: white, not red -- that is
                              # a statement about you, not about the object

    "radius":   "5px",
    "radius-l": "8px",
    # PT Sans and PT Serif are one family, drawn to the same metrics and
    # weights, so prose set in the serif sits beside a sans control without
    # either looking borrowed. macOS ships both; the rest of each stack is for
    # a machine that does not have them.
    "font":     '"PT Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
    "serif":    '"PT Serif", Georgia, "Times New Roman", Times, serif',
    "mono":     "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
}

#: Light theme.  Same roles, same red; only the surfaces and text invert.  It
#: exists because ephys was written light and someone may still want it that
#: way at a desk by a window -- not as a separate design.
SCREEN_LIGHT: dict[str, str] = dict(
    SCREEN,
    bg="#ffffff",
    panel="#f4f5f7",
    panel2="#e9ebee",
    field="#ffffff",
    canvas="#f0f1f3",
    **{"raise": "#dfe2e6"},
    line="#d6dade",
    edge="#b3b9c0",
    ink="#1a1d21",
    dim="#61686f",
    faint="#8b9198",
    accent="#c62f34",
    **{"accent-deep": "#c62f34",
       "accent-hot": "#a8272c",
       "accent-edge": "#a8272c",
       "accent-wash": "rgba(198,47,52,.10)",
       "accent-veil": "rgba(198,47,52,.22)"},
    ok="#1a7f4f",
    warn="#9a6206",
    bad="#c0322b",
    **{"on-accent": "#ffffff", "on-status": "#ffffff"},
    **{"q-good": "#8b929b", "q-warn": "#b4573f", "q-bad": "#d33a3a",
       "q-unknown": "#a8aeb5"},
    sel="#1a1d21",
)


# --------------------------------------------------------------------------
# Print: figures
# --------------------------------------------------------------------------

GREY = "#5f6772"
RED = "#c62f34"

#: Series colors for figures, in order.  Gray first and red second is not
#: decoration: a two-group figure is the common case, and control-gray against
#: mutant-red needs no legend to read.  The rest are chosen to stay separable
#: in grayscale (their luminances are roughly 45/40/55/65/35/50 %), because
#: journals still print in black and white and a figure that collapses there
#: is a figure that gets a query.
SERIES: list[str] = [
    GREY,       # control / wild-type
    RED,        # the thing you are testing
    "#2c7fb8",  # blue
    "#c8935f",  # tan
    "#7d5ba6",  # violet
    "#3f8f6b",  # green
    "#1f2933",  # near-black
    "#d1495b",  # rose
]

SERIES_NAMES = ["grey", "red", "blue", "tan", "violet", "green", "black", "rose"]

#: The rest of the ink in a figure.  Black, because the lab's own figures are
#: drawn in black (`mpl.rcParams['axes.edgecolor'] = 'k'` and the rest of the
#: block in `voltage_clamp_processing.py`), and a panel from an app has to sit
#: beside one from that notebook without the axes being a different colour.
PRINT: dict[str, str] = {
    "ink":      "#000000",   # axes, ticks, labels, annotation text
    "rule":     "#000000",   # significance brackets
    "grid":     "#c9ced4",
    "muted":    "#6b7480",   # footnotes, "n = " captions
    # A point is filled in its group's colour; this is the hairline halo round
    # it, which is what separates two dots that land on each other.
    "point_edge": "#ffffff", # the halo
    "point_face": "#ffffff", # kept for callers that draw an open circle
    "warn":     "#8c1d18",   # a caption that needs to be read
    "paper":    "#ffffff",
}


def series_colour(index: int) -> str:
    """Color for the *index*-th group, cycling."""
    return SERIES[index % len(SERIES)]


# --------------------------------------------------------------------------

def _decl(tokens: dict[str, str]) -> str:
    return "\n".join(f"  --{name}: {value};" for name, value in tokens.items())


def css() -> str:
    """The theme as custom properties, for ``labkit.css`` to be built from.

    Dark is the bare ``:root`` because five of the six apps run dark and a
    microscope room is not a bright place.  ``data-theme`` on ``<html>``
    overrides it either way, so the toggle wins over the system setting.
    """
    return (
        "/* Generated by labkit/theme.py -- edit there, not here. */\n"
        # `color-scheme` is not a custom property: without it, a browser renders
        # native control chrome (select dropdowns, spinners, scrollbars) using
        # its OS light defaults regardless of what --bg/--ink say, which is how
        # a themed page ends up with a dropdown list nobody can read.
        ":root {\n  color-scheme: dark;\n" + _decl(SCREEN) + "\n}\n\n"
        ':root[data-theme="light"] {\n  color-scheme: light;\n' + _decl(SCREEN_LIGHT) + "\n}\n\n"
        "@media (prefers-color-scheme: light) {\n"
        '  :root:not([data-theme="dark"]) {\n'
        "    color-scheme: light;\n"
        + "\n".join("  " + line for line in _decl(SCREEN_LIGHT).splitlines())
        + "\n  }\n}\n"
    )
