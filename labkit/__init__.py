"""Shared parts of the lab's tools: one look, one file browser, one set of
plotting controls, one place p-values come from.

    from labkit import theme, plots, stats

* ``theme``   the red-on-gray palette, and the CSS every interface is built from
* ``plots``   the publication style, the three figures these tools make, and
              significance brackets
* ``stats``   the comparison engine the brackets read their numbers from
* ``browse``  filesystem listing, with adapters for Flask, FastAPI and stdlib
* ``names``   acquisition filenames taken apart: roles, schemes, the group label
* ``plate``   microplate wells, the layout saying what is in each, and a plate
              reader's one-file-per-channel fields
* ``schemes`` saved color schemes, shared by every app that draws a figure
* ``chemstation`` an Agilent ChemStation run folder: the traces, their peaks,
              and the fractions the collector cut
* ``codemap`` an app's own MAP.md, read back out of its routes, modules, panes
              and section banners

``python -m labkit.sync`` copies the built assets into each app, so the apps
stay standalone and runnable on a machine that has never heard of labkit.
"""

__version__ = "1.0.0"

from . import theme  # noqa: F401  (the palette is always wanted)

__all__ = ["theme", "plots", "stats", "browse", "names", "plate", "schemes",
           "chemstation", "codemap", "__version__"]


def __getattr__(name):
    # Imported lazily: `theme` must stay importable without matplotlib or
    # pandas installed, because the pure-web apps have neither and still
    # want the colors.
    if name in ("plots", "stats", "browse", "names", "plate", "schemes",
                "chemstation", "codemap"):
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(name)
