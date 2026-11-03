#!/usr/bin/env python3
"""ArtStitch -- the browser front end.

    python app.py              # then open http://127.0.0.1:8800

Seven pages: Scans (open a set), Stitch (align, cut, look), Straighten & crop, Patch (dust), Color, Export
(presets) and Batch (a folder of sets, unattended).  The work is done by tiles / align / compose / edges / patch /
color / export; ``session.py`` is one set and everything decided about it, ``batch.py`` is a folder of them.
This file holds the current session and the routes.  Nothing is ever written next to a scan except in
the output folder of a preset, and an output never replaces a scan.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
from importlib.metadata import entry_points
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

import batch as B
import color
import dust
import export
import tiles
from labkit import browse as LKB
from session import MODES, Session, clean, resolve_folder

HERE = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="/static")
app.json.sort_keys = False
app.register_blueprint(LKB.flask_blueprint(default_extensions=tiles.TIFF_EXTS))

STORE = tiles.TileStore()                      # decoded scans, shared by the editor and the batch
#: The session: one person, one machine, one open set -- a module-level dict is the whole store.
STATE: dict = {"session": None, "batch": {"rows": [], "job": None}, "dir": Path.home() / ".artstitch"}
JOBS: dict = {}
STARTED = time.time()
HEAVY = threading.Lock()                       # one stitch / export / batch at a time: they share RAM and cores

COMPRESSIONS = {
    "zopfli": ("Deflate, squeezed (Zopfli)", "smallest file that every program opens; slow to save"),
    "deflate": ("Deflate", "opens everywhere; a few percent larger than Zopfli, much faster"),
    "lzma": ("LZMA", "about 10 % smaller than Deflate; opened by libtiff programs, not by Photoshop"),
    "zstd": ("Zstandard", "about 8 % smaller than Deflate; opened by libtiff programs, not by Photoshop"),
    "lzw": ("LZW", "opens everywhere, larger than Deflate"),
    "none": ("none", "no compression at all"),
}
DEPTHS = {
    "auto": ("bits as the scans", "as deep as the deepest scan: 8 or 16 bits a channel, 32 if one is floating point"),
    "8": ("8 bits", "256 levels a channel, as most programs show them"),
    "16": ("16 bits", "65536 levels a channel: room to adjust further without banding; about twice the size"),
    "32": ("32 bits, float", "floating point, the light kept linear as in Photoshop's 32 bits; the color goes into "
                             "the layers, not onto adjustment layers"),
}


class Problem(Exception):
    """Something the person can fix: shown to them as it is."""


@app.errorhandler(Exception)
def oops(e):
    if isinstance(e, HTTPException):
        return e
    fixable = isinstance(e, (Problem, ValueError, KeyError, FileNotFoundError, RuntimeError, TypeError))
    if not fixable:
        traceback.print_exc()
    msg = str(e.args[0]) if isinstance(e, KeyError) and e.args else str(e)
    return jsonify({"error": msg if fixable else f"{type(e).__name__}: {e}"}), 400 if fixable else 500


def body() -> dict:
    return request.get_json(silent=True) or {}


def sess() -> Session:
    s = STATE["session"]
    if s is None:
        raise Problem("open a set of scans first")
    return s


def ready() -> Session:
    s = sess()
    if s.layout is None or s.plan is None:
        raise Problem("stitch the set first")
    return s


# ============================================================================ jobs


def start_job(fn, label: str) -> str:
    """Run *fn(progress, cancel)* on a worker thread (after any running job) and hand back a job id."""
    jid = uuid.uuid4().hex[:12]
    JOBS[jid] = {"id": jid, "label": label, "progress": 0.0, "message": "starting", "done": False,
                 "error": None, "result": None, "t0": time.time(), "cancel": False}
    for k in [k for k, j in JOBS.items() if j["done"] and time.time() - j["t0"] > 3600]:
        del JOBS[k]

    def progress(frac: float, msg: str = "") -> None:
        JOBS[jid]["progress"] = float(np.clip(frac, 0.0, 1.0))
        if msg:
            JOBS[jid]["message"] = msg

    def run() -> None:
        try:
            if not HEAVY.acquire(blocking=False):
                progress(0.0, "waiting for the running job to finish")
                HEAVY.acquire()
            try:
                JOBS[jid]["result"] = clean(fn(progress, lambda: JOBS[jid]["cancel"]))
                JOBS[jid]["progress"], JOBS[jid]["message"] = 1.0, "finished"
            finally:
                HEAVY.release()
        except export.Cancelled:
            JOBS[jid]["error"] = "canceled"
        except Exception as exc:                     # surfaced to the browser, not swallowed
            JOBS[jid]["error"] = f"{type(exc).__name__}: {exc}"
            JOBS[jid]["traceback"] = traceback.format_exc()
            traceback.print_exc()
        finally:
            JOBS[jid]["done"] = True
            JOBS[jid]["seconds"] = round(time.time() - JOBS[jid]["t0"], 1)

    threading.Thread(target=run, daemon=True).start()
    return jid


@app.route("/api/job/<jid>")
def job_status(jid: str):
    job = JOBS.get(jid)
    if job is None:
        return jsonify({"error": "no such job"}), 404
    out = {k: v for k, v in job.items() if k not in ("result", "traceback")}
    if job["done"] and job["error"] is None:
        out["result"] = job["result"]
    return jsonify(out)


@app.route("/api/job/<jid>/cancel", methods=["POST"])
def job_cancel(jid: str):
    if jid in JOBS:
        JOBS[jid]["cancel"] = True
    return jsonify({"ok": True})


# ============================================================================ basics


@app.route("/")
def index():
    if not ADDONS["scripts"] and not ADDONS["styles"]:
        return send_from_directory(str(HERE / "static"), "index.html")
    page = (HERE / "static" / "index.html").read_text(encoding="utf-8")
    links = "".join(f'<link rel="stylesheet" href="{html.escape(u)}">\n' for u in ADDONS["styles"])
    scripts = "".join(f'<script src="{html.escape(u)}"></script>\n' for u in ADDONS["scripts"])
    page = page.replace("</head>", links + "</head>", 1).replace("</body>", scripts + "</body>", 1)
    return Response(page, mimetype="text/html")


@app.route("/api/health")
def health():
    s = STATE["session"]
    return jsonify({"ok": True, "app": "artstitch", "uptime_s": round(time.time() - STARTED, 1),
                    "session": s.name if s else None, "addons": ADDONS["names"]})


@app.route("/api/constants")
def constants():
    """Every enumeration the controls offer, from Python, so there is no second copy of them in JavaScript."""
    return jsonify({
        "compressions": [{"value": k, "label": v[0], "help": v[1]} for k, v in COMPRESSIONS.items()],
        "depths": [{"value": k, "label": v[0], "help": v[1]} for k, v in DEPTHS.items()],
        "kinds": [{"value": "tiff", "label": "TIFF (full size, lossless)"},
                  {"value": "psd", "label": "Photoshop PSD (full size, layers)"}, {"value": "png", "label": "PNG"},
                  {"value": "jpg", "label": "JPEG"}, {"value": "pdf", "label": "PDF"}],
        "sizes": [{"value": "full", "label": "full size"}, {"value": "dpi", "label": "resolution (dpi)"},
                  {"value": "long", "label": "long side (px)"}, {"value": "mb", "label": "largest under (MB)"}],
        "overwrite": [{"value": "number", "label": "add a number (name_2)"},
                      {"value": "skip", "label": "skip what exists"}, {"value": "replace", "label": "replace"}],
        "placeholders": [{"key": "name", "help": "name of the set"}, {"key": "n", "help": "number of scans"},
                         {"key": "dpi", "help": "resolution of that output"}, {"key": "date", "help": "today"},
                         {"key": "preset", "help": "name of this preset"}],
        "extensions": {"tiff": list(tiles.TIFF_EXTS), "image": list(tiles.IMAGE_EXTS)},
        "default_preset": export.DEFAULT_PRESET,
    })


@app.route("/api/scan", methods=["POST"])
def scan():
    b = body()
    exts = tiles.IMAGE_EXTS if b.get("all_images") else tiles.TIFF_EXTS
    return jsonify({"sets": clean(B.scan(b.get("folder", ""), bool(b.get("recursive")), exts))})


# ============================================================================ presets


def _store(name: str) -> str:
    STATE["dir"].mkdir(parents=True, exist_ok=True)
    return str(STATE["dir"] / name)


@app.route("/api/presets", methods=["GET", "POST"])
def presets():
    if request.method == "POST":
        given = body().get("presets")
        if not isinstance(given, list) or not given:
            raise Problem("no presets given")
        names = [str(p.get("name", "")).strip() for p in given]
        if "" in names or len(set(n.lower() for n in names)) != len(names):
            raise Problem("every preset needs its own name")
        for n, p in zip(names, given):
            bad = export.check_preset(p)
            if bad:
                raise Problem(f"preset '{n}': " + "; ".join(bad))
        export.save_presets(_store("presets.json"), given)
    return jsonify({"presets": export.load_presets(_store("presets.json"))})


def _colours() -> dict:
    try:
        with open(_store("colours.json")) as f:
            got = json.load(f)
        return got if isinstance(got, dict) else {}
    except (OSError, ValueError):
        return {}


@app.route("/api/colours", methods=["GET", "POST", "DELETE"])
def colours():
    have = _colours()
    if request.method != "GET":
        b = body()
        name = str(b.get("name", "")).strip()
        if not name:
            raise Problem("give the color preset a name")
        if request.method == "DELETE":
            have.pop(name, None)
        else:
            color.build(b.get("adj") or {})
            have[name] = b.get("adj") or {}
        with open(_store("colours.json") + ".part", "w") as f:
            json.dump(have, f, indent=1)
        os.replace(_store("colours.json") + ".part", _store("colours.json"))
    return jsonify({"colours": have})


# ============================================================================ the open set


@app.route("/api/open", methods=["POST"])
def open_set():
    b = body()
    paths = b.get("paths") or []
    for p in paths:
        if not os.path.isfile(p):
            raise Problem(f"not a file: {p}")
    STATE["session"] = Session(paths, STORE, (b.get("name") or "").strip() or None)
    return jsonify(STATE["session"].summary())


@app.route("/api/session")
def session_summary():
    s = STATE["session"]
    return jsonify(s.summary() if s else {"open": False})


@app.route("/api/session/settings", methods=["POST"])
def session_settings():
    s, b = sess(), body()
    if "name" in b:
        if not str(b["name"]).strip():
            raise Problem("the set needs a name")
        s.name = str(b["name"]).strip()
    if "dpi" in b:
        s.dpi = float(b["dpi"]) if b["dpi"] else None
    if "max_deg" in b:
        s.max_deg = max(0.0, min(float(b["max_deg"]), 45.0))
    return jsonify(s.summary())


@app.route("/api/session/save", methods=["POST"])
def session_save():
    s, b = sess(), body()
    path = b.get("path") or os.path.join(resolve_folder(b.get("folder") or "stitched", s.folder), s.name + B.SESSION_EXT)
    return jsonify({"path": s.save(path)})


@app.route("/api/session/load", methods=["POST"])
def session_load():
    STATE["session"] = None                        # a failed load must not leave the previous set showing
    STATE["session"] = Session.load(body().get("path", ""), STORE)
    return jsonify(STATE["session"].summary())


# ============================================================================ stitching


@app.route("/api/align", methods=["POST"])
def align():
    s, b = sess(), body()
    poses = b.get("poses") or None
    if poses is not None and len(poses) != len(s.paths):
        raise Problem("one pose per scan, please")

    def fn(progress, cancel):
        s.auto(progress, poses, b.get("max_deg"))
        return s.summary()
    return jsonify({"job": start_job(fn, "stitching")})


@app.route("/api/straighten", methods=["POST"])
def straighten():
    s, b = ready(), body()

    def fn(progress, cancel):
        progress(0.1, "looking for the paper edges")
        if b.get("line"):
            s.level_line(*b["line"])
        else:
            s.straighten(angle_deg=b.get("angle_deg"), inset=b.get("inset"), keep=bool(b.get("keep")))
        return s.summary()
    return jsonify({"job": start_job(fn, "straightening")})


@app.route("/api/crop", methods=["POST"])
def crop():
    s = ready()
    s.set_crop(body().get("crop") or [])
    return jsonify(s.summary())


@app.route("/api/turn", methods=["POST"])
def turn():
    """The result a quarter turn further: {"by": 1} clockwise, -1 counterclockwise."""
    s = ready()
    s.turn(int(body().get("by", 1)))
    return jsonify(s.summary())


@app.route("/api/tile/<int:k>.jpg")
def tile_thumb(k: int):
    """A scan as a small picture, for placing scans by hand."""
    s = sess()
    if not 0 <= k < len(s.paths):
        raise Problem("no such scan")
    img = STORE.proxy(s.paths[k], max(1, int(request.args.get("f", 32))))
    img = (img / 257.0).astype(np.uint8) if img.dtype == np.uint16 else img
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), mimetype="image/jpeg", headers={"Cache-Control": "max-age=3600"})


# ============================================================================ looking


def _mode() -> str:
    mode = request.args.get("mode", "mosaic")
    if mode not in MODES:
        raise Problem("unknown view")
    return mode


@app.route("/api/view/info")
def view_info():
    s = ready()
    (ox, oy), (w, h), angle, pivot = s.view(_mode())
    return jsonify(clean({"origin": [ox, oy], "size": [w, h], "angle_deg": angle, "pivot": pivot, "ver": s.ver}))


@app.route("/api/view.png")
def view_png():
    """A window of a view, as an image: ``x0 y0 w h`` in view pixels, drawn at scale ``s``."""
    s, a = ready(), request.args
    x0, y0, w, h = (int(float(a[k])) for k in ("x0", "y0", "w", "h"))
    scale = float(a["s"])
    if not (w > 0 and h > 0 and 0 < scale <= 1.0) or w * h * scale * scale > 36e6:
        raise Problem("that window is too large or empty")
    mode = _mode()
    img, cov = s.window(mode, x0, y0, w, h, scale, raw=a.get("raw") == "1",
                        overlay=tuple(x for x in a.get("overlay", "").split(",") if x))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if a.get("fmt") == "jpg":
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        mime = "image/jpeg"
    else:
        if mode != "result" and cov.min() < 1:
            bgr = np.dstack([bgr, np.round(cov * 255).astype(np.uint8)])
        ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        mime = "image/png"
    return Response(buf.tobytes(), mimetype=mime, headers={"Cache-Control": "no-store"})


# ============================================================================ patches


@app.route("/api/patches")
def patches():
    return jsonify({"patches": ready().patch_view(request.args.get("mode", "result"))})


@app.route("/api/patch", methods=["POST"])
def patch_edit():
    """{"op": "add", "pts": [[x, y], ...], "r"} (view px) | "undo" | "remove", "id" | "source", "id", "part" (the
    stretch), "d" (null: choose again) | "clear": the summary, and the patches healed."""
    s, b = ready(), body()
    mode = b.get("mode", "result")
    s.edit_patches(str(b.get("op")), mode, pts=b.get("pts"), r=b.get("r"), pid=b.get("id"), d=b.get("d"),
                   part=b.get("part"), strokes=b.get("strokes"))
    return jsonify({**s.summary(), "patches": s.patch_view(mode)})


@app.route("/api/dust", methods=["GET", "POST"])
def dust_found():
    """GET: the specks the last search found, strongest first, in px of the result (``items`` null: not looked for
    since the picture last changed).  POST: look again, a job."""
    s = ready()
    if request.method == "POST":
        return jsonify({"job": start_job(lambda progress, cancel: {"found": s.find_dust(progress, cancel)},
                                         "looking for dust")})
    got = s.dust or {}
    return jsonify({"items": got.get("items"), "key": got.get("key"), "thumb": dust.THUMB, "cols": dust.COLS,
                    "large": dust.LARGE})


@app.route("/api/dust/sheet.jpg")
def dust_sheet():
    """The specks' pictures, color-adjusted, THUMB px each in rows of COLS, in the order of their ``i``."""
    s = ready()
    got = s.dust
    if got is None:
        raise Problem("no dust has been looked for")
    img = export.q8(s._adjust(got["sheet"].astype(np.float32) / 255.0))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(buf.tobytes(), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


# ============================================================================ color


@app.route("/api/adjust", methods=["POST"])
def adjust():
    sess().set_adj(body().get("adj") or {})
    return jsonify({"ok": True})


@app.route("/api/histogram")
def histogram():
    return jsonify({"hist": color.histogram(ready().fit("result", 1000)[0])})


@app.route("/api/autolevels", methods=["POST"])
def autolevels():
    b = body()
    return jsonify({"levels": ready().auto_levels(float(b.get("clip", 0.001)), bool(b.get("per_channel", True)))})


@app.route("/api/curve", methods=["POST"])
def curve():
    return jsonify({"y": color.curve_samples(body().get("pts") or [])})


@app.route("/api/pick", methods=["POST"])
def pick():
    b = body()
    return jsonify({"rgb": ready().pick("result", float(b["x"]), float(b["y"]))})


# ============================================================================ saving


@app.route("/api/plan", methods=["POST"])
def plan():
    """Sizes and file names the current preset would give this set."""
    s, preset = ready(), body().get("preset") or {}
    W, H = export.size_of(s.geom or s.default_geom())
    (_, dpi, bits), bad = s.meta(), export.check_preset(preset)
    return jsonify({"problems": bad, "size": [W, H], "dpi": dpi,
                    "outputs": [] if bad else export.describe(preset, s.name, len(s.paths), W, H, dpi, bits)})


@app.route("/api/export", methods=["POST"])
def export_now():
    s, b = ready(), body()
    preset, folder = b.get("preset") or {}, b.get("folder") or None
    bad = export.check_preset(preset)
    if bad:
        raise Problem("; ".join(bad))

    def fn(progress, cancel):
        outputs = s.export(preset, folder, progress, cancel)
        where = folder or resolve_folder(preset.get("folder", ""), s.folder)
        kept = s.save(os.path.join(where, s.name + B.SESSION_EXT))       # so the result can be reopened and adjusted
        return {"outputs": outputs, "session": kept}
    return jsonify({"job": start_job(fn, "saving")})


@app.route("/api/reveal", methods=["POST"])
def reveal():
    path = body().get("path", "")
    if not os.path.exists(path):
        raise Problem("that file is not there any more")
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", path], check=False)
    elif os.name == "nt":
        subprocess.run(["explorer", "/select,", os.path.normpath(path)], check=False)
    else:
        subprocess.run(["xdg-open", os.path.dirname(path)], check=False)
    return jsonify({"ok": True})


# ============================================================================ batch


def _row(r: dict) -> dict:
    return {**{k: v for k, v in r.items() if k != "files"}, "n_files": sum(f.get("include", True) for f in r["files"])}


@app.route("/api/batch/run", methods=["POST"])
def batch_run():
    b = body()
    if not b.get("sets"):
        raise Problem("no sets to process")
    preset = b.get("preset") or {}
    bad = export.check_preset(preset)
    if bad:
        raise Problem("; ".join(bad))
    job = JOBS.get(STATE["batch"]["job"] or "")
    if job is not None and not job["done"]:
        raise Problem("a batch is already running")
    rows = B.rows_from(b["sets"])
    root = b.get("root") or (rows[0]["folder"] if rows else ".")

    def fn(progress, cancel):
        tally = B.run(rows, preset, STORE, b.get("colour") or {}, progress, cancel)
        return {"tally": tally, "report": B.write_report(rows, root) if any(r["include"] for r in rows) else ""}
    STATE["batch"] = {"rows": rows, "job": start_job(fn, "batch")}
    return jsonify({"job": STATE["batch"]["job"]})


@app.route("/api/batch/status")
def batch_status():
    st = STATE["batch"]
    job = JOBS.get(st["job"] or "")
    return jsonify(clean({"rows": [_row(r) for r in st["rows"]], "job": st["job"],
                          "running": bool(job and not job["done"]),
                          "progress": job["progress"] if job else 0, "message": job["message"] if job else ""}))


@app.route("/api/batch/open", methods=["POST"])
def batch_open():
    rows = STATE["batch"]["rows"]
    i = int(body().get("index", -1))
    if not 0 <= i < len(rows) or not rows[i]["session"]:
        raise Problem("that set has no saved session (it did not finish)")
    STATE["session"] = None
    STATE["session"] = Session.load(rows[i]["session"], STORE)
    return jsonify(STATE["session"].summary())


# ============================================================================ add-ons

#: What the installed add-ons are called and what they add to the page, in the order they added it.
ADDONS: dict = {"names": [], "scripts": [], "styles": []}


class Host:
    """What an add-on is handed.  An add-on is a package of its own, installed into the same Python, that
    names a function under the entry-point group ``artstitch.addons``; that function is called once, at
    startup, with a Host.  It adds its routes to ``app`` (a blueprint keeps them together), runs long work
    through ``start_job`` so the page can follow it, and puts its own script and stylesheet on the page.
    ArtStitch itself knows nothing of any add-on: without one installed, nothing here does anything."""

    def __init__(self) -> None:
        self.app, self.start_job, self.Problem, self.body, self.state = app, start_job, Problem, body, STATE

    def add_script(self, url: str) -> None:
        ADDONS["scripts"].append(url)

    def add_style(self, url: str) -> None:
        ADDONS["styles"].append(url)


def load_addons() -> list[str]:
    """Call every installed add-on.  One that fails is reported and left out; ArtStitch carries on without it."""
    for ep in entry_points(group="artstitch.addons"):
        if ep.name in ADDONS["names"]:
            continue
        try:
            ep.load()(Host())
            ADDONS["names"].append(ep.name)
        except Exception:
            print(f"add-on {ep.name!r} left out: it failed to load", flush=True)
            traceback.print_exc()
    return ADDONS["names"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ArtStitch web app")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT") or 8800))
    parser.add_argument("--state", default=None, help="where presets are kept (default ~/.artstitch)")
    args = parser.parse_args(argv)
    if args.state:
        STATE["dir"] = Path(args.state).expanduser()
    names = load_addons()
    print(f"ArtStitch on http://127.0.0.1:{args.port}   (presets in {STATE['dir']})", flush=True)
    if names:
        print("add-ons: " + ", ".join(names), flush=True)
    app.run(host="127.0.0.1", port=args.port, threaded=True, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
