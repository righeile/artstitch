"""One filesystem listing, and the three ways these apps have of serving it.

Every app had grown its own ``/api/browse``: one returned folders only, one
returned files only, one filtered by extension and one did not, one 400'd on a
missing path and one silently fell back to ``$HOME``.  The dialogs on top of
them therefore could not be the same dialog.  This is the listing all of them
now return, and :func:`flask_blueprint`, :func:`fastapi_router` and
:func:`handle_request` are the adapters, so the contract does not depend on
which web framework an app happened to be written against.

The listing is read-only and deliberately unsandboxed: these are local tools
that open the user's own microscope data, and a root confined to one directory
would just mean typing paths instead of clicking them.  Nothing here writes,
deletes or executes -- the only verb is "list".
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "listing", "shortcuts", "flask_blueprint", "fastapi_router",
    "handle_request", "DEFAULT_LIMIT",
]

#: Some directories genuinely hold tens of thousands of files (a folder of
#: single-plane TIFFs from an automated scope).  Sending all of them helps
#: nobody -- the browser would spend longer laying them out than the user would
#: spend looking.  The count of what was withheld is returned so the dialog can
#: say so rather than quietly showing a truncated folder.
DEFAULT_LIMIT = 1500


def _size_of(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _count_matching(directory: Path, extensions: Sequence[str] | None, cap: int = 400) -> int:
    """How many interesting files a subfolder holds, for the listing's badge.

    Capped, and never recursive: this runs once per subfolder of whatever is
    on screen, so on a parent with 200 children an uncapped recursive walk
    would stat the entire tree to draw one column of numbers.
    """
    if not extensions:
        return 0
    n = 0
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if n >= cap:
                    break
                if entry.is_file() and Path(entry.name).suffix.lower() in extensions:
                    n += 1
    except (PermissionError, OSError):
        return 0
    return n


def shortcuts() -> list[dict]:
    """The places worth one click: home, mounted volumes, the root."""
    places = [{"label": "Home", "path": str(Path.home()), "glyph": "⌂"}]
    for mount in ("/Volumes", "/media", "/mnt"):
        if Path(mount).is_dir():
            places.append({"label": "Volumes", "path": mount, "glyph": "▤"})
            break
    for name in ("Desktop", "Documents", "Downloads"):
        candidate = Path.home() / name
        if candidate.is_dir():
            places.append({"label": name, "path": str(candidate), "glyph": "▸"})
    places.append({"label": "Root", "path": "/", "glyph": "/"})
    return places


def listing(
    path: str | os.PathLike | None = None,
    extensions: Iterable[str] | None = None,
    show_hidden: bool = False,
    limit: int = DEFAULT_LIMIT,
    count_in_folders: bool = True,
) -> dict:
    """What is in one directory, in the shape every app's browser expects.

    ``extensions`` is the set of file suffixes worth showing (``('.czi',
    '.tif')``); ``None`` shows every file.  Passing a *file* path lists the
    folder containing it and marks it selected, so a browser opened on a
    remembered file lands somewhere useful instead of at ``$HOME``.

    Never raises for a bad path -- an unreadable or missing directory comes
    back as an empty listing with ``error`` set, because a browser that throws
    a 500 when you click a folder you cannot read is worse than one that says
    so in place.
    """
    extensions = {e.lower() if e.startswith(".") else f".{e.lower()}"
                  for e in (extensions or ())} or None

    requested = Path(str(path)).expanduser() if path else Path.home()
    selected = ""
    error = ""

    if requested.is_file():
        selected, requested = str(requested), requested.parent
    if not requested.is_dir():
        error = f"{requested} is not a folder"
        requested = Path.home()

    directories: list[dict] = []
    files: list[dict] = []
    truncated = 0
    try:
        entries = sorted(os.scandir(requested), key=lambda e: e.name.lower())
    except (PermissionError, OSError) as exc:
        entries, error = [], f"{type(exc).__name__}: {exc}"

    for entry in entries:
        if entry.name.startswith(".") and not show_hidden:
            continue
        full = Path(entry.path)
        try:
            is_dir = entry.is_dir()
        except OSError:
            continue
        if is_dir:
            directories.append({
                "name": entry.name,
                "path": entry.path,
                "n_files": _count_matching(full, extensions) if count_in_folders else 0,
            })
        else:
            suffix = full.suffix.lower()
            if extensions and suffix not in extensions:
                continue
            if len(files) >= limit:
                truncated += 1
                continue
            files.append({
                "name": entry.name,
                "path": entry.path,
                "size": _size_of(full),
                "ext": suffix,
            })

    parent = str(requested.parent)
    return {
        "path": str(requested),
        "parent": parent if parent != str(requested) else "",
        "home": str(Path.home()),
        "shortcuts": shortcuts(),
        "dirs": directories,
        "files": files,
        "n_files": len(files),
        # What the filter actually was.  The client may not have asked for one
        # -- the app's own list of readable formats lives on the server -- so
        # the dialog needs to be told in order to say what it is showing.
        "extensions": sorted(extensions) if extensions else [],
        "truncated": truncated,
        "selected": selected,
        "error": error,
    }


def _params(raw: dict, default_extensions: Iterable[str] | None = None) -> dict:
    """Pull the listing arguments out of a query string or JSON body.

    ``default_extensions`` is what the app reads when the client does not say.
    Letting the server hold that list is the point: MicroFigure's set of
    readable formats lives in ``imgio.py`` and would otherwise have to be
    restated, and kept in step, in JavaScript.
    """
    extensions = raw.get("exts") or raw.get("extensions") or None
    if isinstance(extensions, str):
        extensions = [e for e in extensions.replace(" ", "").split(",") if e]
    if not extensions:
        extensions = list(default_extensions) if default_extensions else None
    hidden = str(raw.get("hidden", "")).lower() in ("1", "true", "yes", "on")
    return {
        "path": raw.get("path") or None,
        "extensions": extensions,
        "show_hidden": hidden,
    }


# --------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------

def flask_blueprint(name: str = "labkit_browse", url: str = "/api/browse",
                    default_extensions: Iterable[str] | None = None):
    """A Flask blueprint serving the listing at *url*.

    Accepts GET or POST, so a front end can use either.
    """
    from flask import Blueprint, jsonify, request

    blueprint = Blueprint(name, __name__)

    @blueprint.route(url, methods=["GET", "POST"])
    def _browse():
        raw = dict(request.args)
        if request.method == "POST":
            raw.update(request.get_json(silent=True) or {})
        return jsonify(listing(**_params(raw, default_extensions)))

    return blueprint


def fastapi_router(url: str = "/api/browse",
                   default_extensions: Iterable[str] | None = None):
    """An APIRouter serving the listing at *url*, for the ephys app."""
    from fastapi import APIRouter, Body, Query

    router = APIRouter()

    @router.get(url)
    def _browse_get(path: str | None = Query(None), exts: str | None = Query(None),
                    hidden: bool = Query(False)):
        return listing(**_params({"path": path, "exts": exts, "hidden": hidden},
                                 default_extensions))

    @router.post(url)
    def _browse_post(body: dict = Body(default={})):
        return listing(**_params(body or {}, default_extensions))

    return router


def handle_request(path_and_query: str, body: bytes | None = None,
                   default_extensions: Iterable[str] | None = None) -> dict:
    """The listing for a stdlib ``http.server`` handler, for cryofig and the QC tool.

    Hand it ``self.path`` (and the POST body, if any); it parses the query
    itself so the handler stays a two-line branch.
    """
    import json
    from urllib.parse import parse_qs, urlparse

    raw = {k: v[0] for k, v in parse_qs(urlparse(path_and_query).query).items()}
    if body:
        try:
            raw.update(json.loads(body.decode("utf-8")) or {})
        except (ValueError, UnicodeDecodeError):
            pass
    return listing(**_params(raw, default_extensions))
