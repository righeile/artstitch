"""
Named color schemes, saved once and offered by every one of these apps.

A scheme is a flat ``{series name: "#rrggbb"}`` map -- exactly what
:attr:`labkit.plots.PlotSpec.colours` holds -- kept under
``~/.labkit/colour_schemes``.  The store is shared between the tools on
purpose rather than split per app: "the KO is this red" is a decision about a
paper, and a paper's panels come out of four of these programs.

Keyed by name, not by position, so a scheme saved for a two-genotype figure
still puts the KO in the same red when a third genotype turns up between them.

    >>> write("paper 1", {"WT": "#5f6772", "KO": "#c62f34"})   # doctest: +SKIP
    >>> read("paper 1")["KO"]                                  # doctest: +SKIP
    '#c62f34'
"""
from __future__ import annotations

import json
import re
from pathlib import Path

DIR = Path.home() / ".labkit" / "colour_schemes"

#: What an ``<input type="color">`` produces, and nothing else.  A scheme is
#: read back straight into a figure, so a junk value here is a batch of plots
#: that raises instead of drawing -- hours later, in someone else's app.
COLOUR = re.compile(r"#[0-9a-fA-F]{3,8}\Z")
_UNSAFE = re.compile(r"[^A-Za-z0-9 _.-]+")
MAX_KEYS = 200


def _file(name: str) -> Path:
    """The file a scheme name maps to, with the name made safe for one.

    The names arrive from a web request, so this is a trust boundary: strip
    everything that is not a plain character, and refuse what is left if it
    could still climb out of the directory.
    """
    slug = _UNSAFE.sub("_", str(name or "").strip())[:80].strip(". ")
    if not slug or slug in (".", ".."):
        raise ValueError("a scheme needs a name")
    return DIR / f"{slug}.json"


def clean(colours: dict | None) -> dict[str, str]:
    """The colors worth storing: string keys, hex values, nothing else."""
    out = {}
    for key, value in (colours or {}).items():
        value = str(value).strip()
        if str(key).strip() and COLOUR.match(value):
            out[str(key).strip()[:120]] = value
        if len(out) >= MAX_KEYS:
            break
    return out


def listing() -> dict:
    """Every saved scheme, name and contents, plus where they live."""
    try:
        files = sorted(DIR.glob("*.json"), key=lambda f: f.name.lower())
    except OSError:
        files = []
    schemes = {}
    for f in files:
        try:
            schemes[f.stem] = clean(json.loads(f.read_text()))
        except (OSError, ValueError):
            continue          # a hand-edited file should cost its own row, not the list
    return {"names": list(schemes), "schemes": schemes, "dir": str(DIR)}


def read(name: str) -> dict[str, str]:
    """One scheme, or ``{}`` if there is no such file."""
    try:
        return clean(json.loads(_file(name).read_text()))
    except (OSError, ValueError):
        return {}


def write(name: str, colours: dict | None) -> Path:
    """Save *colours* under *name*, replacing whatever was there."""
    path = _file(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(colours), indent=1, sort_keys=True))
    return path


def remove(name: str) -> bool:
    """Delete a scheme.  ``False`` if it was not there to begin with."""
    try:
        _file(name).unlink()
        return True
    except (OSError, ValueError):
        return False


def flask_blueprint(name: str = "labkit_schemes", url: str = "/api/colour_schemes"):
    """A Flask blueprint serving the store at *url*.

    ``GET url`` lists, ``GET/PUT/DELETE url/<name>`` works on one.
    """
    from flask import Blueprint, jsonify, request

    blueprint = Blueprint(name, __name__)

    @blueprint.route(url)
    def _list():
        return jsonify(listing())

    @blueprint.route(f"{url}/<path:scheme>", methods=["GET", "PUT", "DELETE"])
    def _one(scheme: str):
        try:
            if request.method == "GET":
                return jsonify({"name": scheme, "colours": read(scheme)})
            if request.method == "DELETE":
                return jsonify({"name": scheme, "deleted": remove(scheme)})
            body = request.get_json(silent=True) or {}
            written = write(scheme, body.get("colours") or body)
            return jsonify({"name": scheme, "colours": read(scheme),
                            "path": str(written)})
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except OSError as exc:
            return jsonify({"error": f"could not write the scheme: {exc}"}), 500

    return blueprint


if __name__ == "__main__":
    # The check: a name that tries to climb out of the directory does not, and
    # a color that is not one never reaches a figure.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        DIR = Path(tmp) / "schemes"
        assert _file("../../etc/passwd").parent == DIR
        assert _file(" paper 1 ") == DIR / "paper 1.json"
        for bad in ("", "  ", ".", "..", "..."):
            try:
                _file(bad)
            except ValueError:
                continue
            raise AssertionError(f"{bad!r} was accepted as a scheme name")
        for odd in ("/", "a/b", "~", "../x", "\\", "con:"):
            assert _file(odd).parent == DIR, f"{odd!r} landed outside {DIR}"
        assert clean({"KO": "#c62f34", "WT": "red", "": "#fff", 1: "#000"}) == {
            "KO": "#c62f34", "1": "#000"}
        write("paper 1", {"KO": "#c62f34", "WT": "nonsense"})
        assert read("paper 1") == {"KO": "#c62f34"}
        assert listing()["names"] == ["paper 1"]
        assert remove("paper 1") and not remove("paper 1")
    print("all good")
