"""
Send to board from an app open on its own: the figure goes to the omnifig
already running, or to one started for it.

In an omnifig tab, ``LK.board`` hands the figure to the page that frames it
and nothing here runs.  Open on its own, the page has no omnifig to talk to
and cannot start one, so it asks its own server, which can: this module.

The omnifig is the one at ``OMNIFIG_URL`` when that is set -- started there
if nothing answers -- else the first on ports 8880-8899 whose page names it,
else one started on the first free port of those, with omnifig's own python,
the way its launcher starts it.  A start opens it in the browser, and so does
a figure that arrives while no omnifig window is open.  ``LABKIT_NO_BROWSER``
set, it never opens one (checks).

A figure carries a ``key`` when it is linked: the app sends it again under the
same key each time it changes, ``update`` set, and omnifig redraws what it
placed under that key where it stands.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

PORTS = range(8880, 8900)
MAX_SVG = 64 * 1024 * 1024
_LOCK = threading.Lock()        # one start at a time: two copies would fight for a port


def _omnifig_dir() -> Path | None:
    """omnifig's folder: a sibling of the app this copy of labkit lives in."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "omnifig" / "app.py").is_file():
            return parent / "omnifig"
    return None


def _answers(url: str) -> bool | None:
    """True when omnifig answers at ``url``, False for something else, None for nothing."""
    try:
        with urllib.request.urlopen(url + "/", timeout=1) as r:
            return "omnifig" in r.read(4096).decode("utf-8", "replace").lower()
    except urllib.error.HTTPError:
        return False
    except OSError:
        return None


def _free(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _start(port: int) -> str:
    where = _omnifig_dir()
    if where is None:
        raise RuntimeError("no omnifig folder beside this app to start one from")
    python = next((str(p) for p in (where.parent / ".venv-apps" / "bin" / "python",
                                    where / ".venv" / "bin" / "python") if p.is_file()), sys.executable)
    # Its own session, so it outlives the app that started it.
    subprocess.Popen([python, "app.py", "--port", str(port)], cwd=where, start_new_session=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        if _answers(url):
            return url
        time.sleep(0.3)
    raise RuntimeError(f"omnifig was started on port {port} but did not answer")


def _running() -> str | None:
    """The omnifig running, without starting one."""
    pinned = os.environ.get("OMNIFIG_URL", "").rstrip("/")
    if pinned:
        return pinned if _answers(pinned) else None
    return next((u for u in (f"http://127.0.0.1:{p}" for p in PORTS) if _answers(u)), None)


def find_or_start() -> tuple[str, bool]:
    """The omnifig to send to, and whether it was started just now."""
    with _LOCK:
        url = _running()
        if url:
            return url, False
        pinned = os.environ.get("OMNIFIG_URL", "").rstrip("/")
        if pinned:
            return _start(int(pinned.rsplit(":", 1)[1])), True
        port = next((p for p in PORTS if _free(p)), None)
        if port is None:
            raise RuntimeError("ports 8880-8899 are all taken, and none of them is omnifig")
        return _start(port), True


def send(svg: str, app: str, key: str | None = None, update: bool = False, source=None, open=None) -> dict:
    """The figure onto omnifig's board; an update only redraws what is linked to ``key``.
    ``source`` is the source data behind it, a path or a list of them, which
    omnifig keeps with the panel and saves by figure and panel letter.  A table
    an app serves rather than writes may be its address on this machine.
    ``open`` (JSON) is how the app gets back to the figure, kept with the panel."""
    if not isinstance(svg, str) or "<svg" not in svg[:4096]:
        raise ValueError("that is not an SVG")
    if len(svg) > MAX_SVG:
        raise ValueError("the figure is too large to send")
    url, started = find_or_start()
    body = json.dumps({"svg": svg, "app": str(app)[:80], "key": (str(key)[:400] if key else None),
                       "update": bool(update),
                       "source": [(str(p) if "://" in str(p) else os.path.abspath(os.path.expanduser(str(p))))[:1000]
                                  for p in ([source] if isinstance(source, str) else source or [])[:20] if p],
                       "open": open}).encode()
    req = urllib.request.Request(url + "/api/board/inbox", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            got = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise RuntimeError(f"the omnifig at {url} is too old to take a figure from outside its tabs")
        raise RuntimeError(f"omnifig said: {exc.read().decode('utf-8', 'replace')[:300]}")
    # An update waits for the window to show it; a new figure brings one up.
    opened = not update and (started or not got.get("seen")) and not os.environ.get("LABKIT_NO_BROWSER")
    if opened:
        webbrowser.open(url)
    return {"url": url, "started": started, "opened": opened, **got}


def linked(app: str, key: str) -> bool:
    """Whether the omnifig running has had ``key`` from ``app`` before, sent from
    any browser or by a script: then a change to that figure goes to it as an
    update.  Nothing running, nothing linked; none is started for the asking."""
    url = _running()
    if not url or not key:
        return False
    req = urllib.request.Request(url + "/api/board/latest", headers={"Content-Type": "application/json"},
                                 data=json.dumps({"keys": [str(app)[:80] + "\n" + str(key)[:400]]}).encode())
    try:
        # ponytail: the answer carries the drawing too; a revs-only answer if that ever weighs.
        with urllib.request.urlopen(req, timeout=10) as r:
            return bool(json.loads(r.read()))
    except (OSError, ValueError):
        return False


def flask_blueprint(app_name: str, url: str = "/api/board"):
    """The same routes as :func:`fastapi_router`, for a Flask app."""
    from flask import Blueprint, jsonify, request

    blueprint = Blueprint("labkit_board", __name__)

    @blueprint.route(url)
    def _here():
        key = request.args.get("key")
        return jsonify({"ok": True, **({"linked": linked(app_name, key)} if key else {})})

    @blueprint.route(url, methods=["POST"])
    def _send():
        body = request.get_json(silent=True) or {}
        try:
            return jsonify(send(body.get("svg"), app_name, body.get("key"), bool(body.get("update")), body.get("source"),
                                body.get("open")))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 502

    return blueprint


def fastapi_router(app_name: str, url: str = "/api/board"):
    """``GET`` says there is a way to the board, and with ``?key=`` whether
    omnifig has that figure already; ``POST {svg, key, update, source, open}`` sends."""
    from fastapi import APIRouter, Body, HTTPException

    router = APIRouter()

    @router.get(url)
    def _here(key: str | None = None):
        return {"ok": True, **({"linked": linked(app_name, key)} if key else {})}

    @router.post(url)
    def _send(body: dict = Body(default={})):
        try:
            return send(body.get("svg"), app_name, body.get("key"), bool(body.get("update")), body.get("source"), body.get("open"))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except RuntimeError as exc:
            raise HTTPException(502, str(exc))

    return router
