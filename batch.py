"""Batch: a folder of scans -> sets -> the same steps and the same export preset for every set, unattended.

A set that fails is reported and the batch goes on.  So is a set whose scans fall into groups that share
no content (two sheets under one name): nothing sensible can be saved, so its session is kept to open and
place them by hand.  A set that finishes with something worth a look (a seam that still disagrees, no paper
edges, a crop much smaller than the canvas) is marked *review* rather than *ok*.  Every finished set leaves
its session file next to its outputs, so "review" means "open it and look", not "start again".  With the
preset's overwrite policy set to *skip*, running the same folder again only does what is missing.
"""

from __future__ import annotations

import csv
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import color
import export
import tiles
from session import Session, _sliced, resolve_folder

STATES = ("queued", "running", "ok", "review", "failed", "skipped", "canceled")
SESSION_EXT = ".artstitch.json"
SKIP_DIRS = ("stitched",)                        # the default output folder is never a source


def scan(folder: str, recursive: bool = False, exts=tiles.TIFF_EXTS) -> list[dict]:
    """The sets of scans in a folder.  ``include`` says what a batch does unless told otherwise:
    sets of two or more scans, without files that look like an earlier result."""
    root = Path(os.path.expanduser(folder))
    if not root.is_dir():
        raise FileNotFoundError(f"not a folder: {folder}")
    found = root.rglob("*") if recursive else root.iterdir()
    paths = sorted(p for p in found if p.is_file() and p.suffix.lower() in exts and not p.name.startswith(".")
                   and not any(part in SKIP_DIRS for part in p.relative_to(root).parts[:-1]))
    sets = tiles.group_sets(paths)
    for s in sets:
        for f in s["files"]:
            f["include"] = "error" not in f and not f.get("suspect")
        s["mp"] = round(sum(f["w"] * f["h"] for f in s["files"] if "w" in f) / 1e6)
        s["include"] = sum(f["include"] for f in s["files"]) >= 2
    return sets


def rows_from(sets: list[dict]) -> list[dict]:
    """Batch rows from (possibly edited) sets.  Every key exists from the start, so a reader polling
    the rows while the batch runs never sees one grow."""
    return [{"name": s["name"], "folder": s["folder"], "files": s["files"], "include": bool(s.get("include", True)),
             "state": "queued" if s.get("include", True) else "skipped", "message": "", "seconds": 0.0,
             "outputs": [], "qc": [], "session": "", "error": ""} for s in sets]


def _done(session_path: str) -> bool:
    """A session file whose outputs are all still there: this set was finished on an earlier run."""
    try:
        with open(session_path) as f:
            outs = json.load(f).get("exports") or []
        return bool(outs) and all(os.path.exists(o["path"]) for o in outs)
    except (OSError, ValueError, KeyError):
        return False


def _one(row: dict, preset: dict, store, colour: dict, progress) -> None:
    files = [f["path"] for f in row["files"] if f.get("include", True)]
    out = resolve_folder(preset.get("folder", ""), row["folder"])
    keep = os.path.join(out, row["name"] + SESSION_EXT)
    if preset.get("overwrite") == "skip" and _done(keep):
        row["state"], row["message"] = "skipped", "already done"
        return
    def keep_session():
        row["session"] = s.save(export.target(keep, "number" if preset.get("overwrite", "number") == "number" else "replace", set()))

    s = Session(files, store, row["name"])
    s.auto(_sliced(progress, 0, 0.55))
    if s.report.get("disconnected"):                    # an export would be the pieces side by side: keep the session to open, say why
        keep_session()
        row["qc"] = s.qc()
        row["state"], row["error"] = "failed", next(q["msg"] for q in row["qc"] if q["level"] == "error")
        return
    if colour.get("mode") == "adj":
        s.set_adj(colour.get("adj") or {})
    elif colour.get("mode") == "auto":
        s.set_adj(s.auto_levels(float(colour.get("clip", 0.001))))
    s.export(preset, out, _sliced(progress, 0.55, 0.97))            # not cancellable: a stop waits for the set to finish
    keep_session()
    row["outputs"], row["qc"] = s.exports, s.qc()
    row["state"] = "review" if row["qc"] else "ok"


def run(rows: list[dict], preset: dict, store, colour: dict | None = None, progress=lambda p, m="": None,
        cancel=lambda: False) -> dict:
    """Process the included rows one after another.  Returns the tally by state.  ``cancel`` is looked at between
    sets: the set in progress is finished (minutes of work are not thrown away), the ones still queued are not run."""
    bad = export.check_preset(preset)
    if bad:
        raise ValueError("; ".join(bad))
    todo = [r for r in rows if r["include"]]
    for i, row in enumerate(todo):
        row["state"] = "queued"
    for i, row in enumerate(todo):
        if cancel():
            for r in todo[i:]:
                r["state"] = "canceled"
            break
        t0 = time.time()
        row["state"], row["message"] = "running", "starting"

        def step(p, m="", i=i, row=row):
            row["message"] = m
            progress((i + p) / len(todo), f"{row['name']} ({i + 1}/{len(todo)}): {m}")

        try:
            _one(row, preset, store, colour or {}, step)
        except Exception as e:                              # this set only: report it, go on to the next
            row["state"], row["error"] = "failed", f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            row["seconds"] = round(time.time() - t0, 1)
            for f in row["files"]:
                store.drop(f["path"])
    tally = {s: sum(r["state"] == s for r in rows) for s in STATES}
    progress(1.0, ", ".join(f"{n} {s}" for s, n in tally.items() if n and s != "queued") or "nothing to do")
    return tally


def write_report(rows: list[dict], folder: str) -> str:
    """One line per set, for the morning after a long batch."""
    path = os.path.join(folder, f"artstitch_batch_{datetime.now():%Y%m%d-%H%M}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "state", "seconds", "notes", "outputs", "session"])
        for r in rows:
            if r["include"]:
                w.writerow([r["name"], r["state"], r["seconds"], r["error"] or "; ".join(q["msg"] for q in r["qc"]),
                            "; ".join(o["path"] for o in r["outputs"]), r["session"]])
    return path
