"""One set of scans and everything decided about it: poses, seams, straightening, color.

A ``Session`` is what the editor works on and what a batch makes for every set.  It is saved as one
JSON file (the mesh and the seam labels come to a few hundred kilobytes; the pixels are not in it,
only the paths of the scans), so a batch result can be opened and adjusted later without stitching
it again.

Nothing here draws a page: the server asks for windows of a *view* and gets pixels.  Three views
share one coordinate system per view, in full-resolution pixels from the view's origin:

    mosaic   the stitched canvas as the poses put it (what the Stitch page shows)
    plane    the same, rotated by the straightening angle (where the crop is drawn)
    result   the crop window of the plane: the image that is exported
"""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

import color
import dust
import edges
import export
import patch
import tiles
from align import align_set, overlapping_pairs
from compose import Mosaic, Plan, plan_seams
from geometry import Frame, Layout, rot

FORMAT = 1
MODES = ("mosaic", "plane", "result")
SEAM_RGB, OUTLINE_RGB = (255, 0, 255), (0, 220, 255)
RESIDUAL_WARN, RESIDUAL_BAD = 2.5, 8.0        # px at full resolution: RMS of what the mesh could not remove
_quiet = lambda p, m="": None


def _sliced(progress, lo: float, hi: float):
    return lambda p, m="": progress(lo + (hi - lo) * float(p), m)


def jsonable(o):
    """``json.dumps(default=...)``: numpy scalars and arrays are what the engine hands back."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not JSON: {type(o).__name__}")


def clean(o):
    return json.loads(json.dumps(o, default=jsonable))


def resolve_folder(preset_folder: str, scans_folder: str) -> str:
    """Where a preset writes: absolute as given, relative to the folder of the scans."""
    f = os.path.expanduser(preset_folder or "")
    return f if os.path.isabs(f) else os.path.join(scans_folder, f)


class Session:
    def __init__(self, paths, store, name: str | None = None):
        self.paths = [str(p) for p in paths]
        if not self.paths:
            raise ValueError("no scans")
        self.store = store
        self.name = name or tiles.set_key(Path(self.paths[0]).name)[0]
        self.folder = str(Path(self.paths[0]).parent)
        self.layout = self.plan = self.geom = None
        self.report: dict = {}
        self.adj: dict = {}
        self.max_deg = 3.0
        self.trim_mm = 0.0                              # the bend where the sheet runs off the glass, cut before matching
        self.trimmed: dict[int, list[str]] = {}         # per scan, the sides trim_px last found running off the glass
        self.tilts: list[float] | None = None           # degrees the paper lay at on the glass, if the scanner said
        self.dpi: float | None = None                   # None: from the scans, else 1200
        self.exports: list[dict] = []
        self.patches: list[dict] = []                   # {id, pts, r, src}: mosaic px; replaced, never changed in place
        self._undo: list[list[dict]] = []               # the lists before each change, for undo
        self.dust = None                                # the last search {items, sheet, key}, result px; gone with its picture
        self.ver = 0                                   # bumps whenever pixels would change
        self.key = uuid.uuid4().hex[:10]               # this Session object, for URLs a browser may cache (scan thumbnails)
        self._pix = 0                                   # bumps when the unpatched pixels change: the render cache's key
        self._retouch = patch.Retouch()
        self._sides = None
        self._mos = self._mos_key = None
        self._cache: OrderedDict = OrderedDict()
        self._built = (None, None)                      # (adj as json, color.build of it)
        self._seam_pts = (None, None)
        self._lock = threading.RLock()
        self.info = [tiles.probe(p) for p in self.paths]

    # -- steps ---------------------------------------------------------------------------------

    def _touch(self) -> None:
        self.ver += 1
        with self._lock:
            self._pix += 1
            self._cache.clear()
            self.dust = None

    def trim_px(self) -> list[list[float]]:
        """``trim_mm`` as pixels of each scan, per side (left, right, top, bottom), by that scan's own
        resolution and never more than a tenth of it, so a number typed by mistake cannot eat the overlap
        the scans are matched on.

        Only the sides where the sheet runs off the glass are cut -- there the paper lifts and the picture
        bends with it.  A side that shows the lid beyond the paper had the artwork's own edge lying flat on
        the glass, and keeps every pixel.  A set of one scan is never cut: there is nothing it runs under.
        """
        out = []
        for k, p in enumerate(self.paths):
            d = (self.info[k] or {}).get("dpi")
            dpi = float(self.dpi or (d[0] if d else 1200.0))
            w, h = self.info[k]["w"], self.info[k]["h"]
            px = min(self.trim_mm * dpi / 25.4, 0.1 * min(w, h))
            off = edges.off_glass(self.store.proxy(p, 16)) if px > 0 and len(self.paths) > 1 else set()
            self.trimmed[k] = sorted(off)
            out.append([px if s in off else 0.0 for s in edges.SIDES])
        return out

    def align(self, progress=_quiet, poses=None, max_deg: float | None = None, seams: bool = True) -> None:
        """Poses and meshes (and, by default, the seams) from the scans; ``poses`` starts from the user's."""
        if max_deg is not None:
            self.max_deg = float(max_deg)
        lay, report = align_set(self.store, self.paths, _sliced(progress, 0, 0.8 if seams else 1),
                                poses=poses, max_deg=self.max_deg,
                                tilts=None if poses is not None else self.tilts, trim=self.trim_px())
        self.layout, self.report, self.plan, self.geom, self._sides = lay, clean(report), None, None, None
        self.patches, self._undo, self._retouch = [], [], patch.Retouch()   # they were put on pixels that may have moved
        self._touch()
        if seams:
            self.seams(_sliced(progress, 0.8, 1))

    def seams(self, progress=_quiet) -> None:
        if self.layout is None:
            raise RuntimeError("align first")
        self.plan = plan_seams(self.layout, self.store, self.paths, progress)
        self._touch()

    def mosaic(self) -> Mosaic:
        if self.layout is None or self.plan is None:
            raise RuntimeError("stitch first")
        key = (id(self.layout), id(self.plan))
        if self._mos_key != key:
            self._mos, self._mos_key = Mosaic(self.layout, self.store, self.paths, self.plan), key
        return self._mos

    def straighten(self, angle_deg: float | None = None, inset: float | None = None, keep: bool = False) -> dict:
        """Find the paper edges (or take ``angle_deg``) and the largest crop inside them.
        ``keep`` leaves the angle, and how it was found, alone and finds only the crop."""
        mos, prev = self.mosaic(), self.geom
        if keep and prev:
            angle_deg = prev["angle_deg"]
        if inset is None:
            inset = prev.get("inset", 12.0) if prev else 12.0
        r = edges.detect(mos.render, self.layout.extent(), inset=inset, angle_deg=angle_deg, sides=self._sides)
        self._sides = r.pop("_sides")
        if keep and prev:
            r.update({k: prev[k] for k in ("method", "confidence", "spread_deg", "line_strength")})
        r["inset"] = inset
        r["quarter"] = prev.get("quarter", 0) if prev else 0            # the edges are found upright; the result stays turned
        r["crop"] = export.turn_rect(r["crop"], r, r["quarter"])
        self.geom = clean(r)
        self._touch()
        return self.geom

    def turn(self, quarters: int) -> dict:
        """Turn the result by 90 degrees ``quarters`` times (clockwise when positive); the crop keeps its pixels."""
        if self.geom is None:
            self.geom = self.default_geom()
        g = self.geom
        self.geom = {**g, "quarter": (g.get("quarter", 0) + int(quarters)) % 4, "crop": export.turn_rect(g["crop"], g, int(quarters))}
        self._touch()
        return self.geom

    def default_geom(self) -> dict:
        """No straightening, the whole covered canvas: what an unstraightened export would be."""
        ext = self.layout.extent()
        o = np.floor(ext[0])
        span = ext[1] - o
        return {"angle_deg": 0.0, "method": "none", "confidence": "manual", "spread_deg": 0.0, "line_strength": 0.0,
                "pivot": (o + span / 2).tolist(), "crop": [int(o[0]), int(o[1]), int(o[0] + span[0]), int(o[1] + span[1])],
                "sides": {}, "inset": 12.0}

    def set_crop(self, crop) -> dict:
        """A crop chosen by hand, whole pixels, inside the plane the straightened mosaic fills."""
        if self.geom is None:
            self.geom = self.default_geom()
        (ox, oy), (w, h), _, _ = self.view("plane")
        x0, x1 = sorted(int(round(float(crop[i]))) for i in (0, 2))
        y0, y1 = sorted(int(round(float(crop[i]))) for i in (1, 3))
        x0, y0, x1, y1 = max(x0, int(ox)), max(y0, int(oy)), min(x1, int(ox) + w), min(y1, int(oy) + h)
        if x1 - x0 < 16 or y1 - y0 < 16:
            raise ValueError("the crop is too small")
        self.geom = {**self.geom, "crop": [x0, y0, x1, y1]}
        self._touch()
        return self.geom

    def level_line(self, p1, p2) -> dict:
        """Straighten so that the line the user drew (view px of the plane) comes out horizontal or vertical."""
        if self.geom is None:
            self.geom = self.default_geom()
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        if math.hypot(dx, dy) < 2:
            raise ValueError("draw a longer line")
        return self.straighten(angle_deg=self.geom["angle_deg"] + math.degrees(edges.tilt(math.atan2(dy, dx))))

    def set_adj(self, adj: dict) -> None:
        adj = clean(adj or {})
        color.build(adj)                                # a bad value fails here, not in the next preview
        self.adj = adj
        with self._lock:
            self._built = (None, None)

    def auto_levels(self, clip: float = 0.001, per_channel: bool = True) -> dict:
        return color.auto_levels(color.histogram(self.fit("result", 1000)[0]), clip, per_channel)

    def auto(self, progress=_quiet, poses=None, max_deg: float | None = None) -> None:
        """Everything a batch does before saving: stitch, cut, find the edges."""
        self.align(_sliced(progress, 0, 0.9), poses, max_deg)
        progress(0.9, "finding the paper edges")
        self.straighten()
        progress(1.0, "stitched")

    # -- patches -------------------------------------------------------------------------------------

    def _plane(self, mode: str) -> Frame:
        if mode not in ("plane", "result"):
            raise ValueError("patches go on the straightened picture")
        (ox, oy), _, angle, pivot = self.view(mode)
        return Frame(1.0, (ox, oy), math.radians(angle), pivot)

    def edit_patches(self, op: str, mode: str = "result", pts=None, r=None, pid=None, d=None, part=None,
                     strokes=None) -> None:
        """add (``pts``, ``r``; or ``strokes``, [{pts, r}], as one change), remove (``pid``), source (``pid``, stretch
        ``part``, ``d``: None to choose again), clear, or undo the last of these.  ``pts`` and ``d`` are px of view
        ``mode``; patches are kept in mosaic px, so they stay on their dust whatever the straightening does."""
        ps = list(self.patches)
        if op == "add":
            nid = max([p["id"] for p in ps], default=0)
            for s in [{"pts": pts, "r": r}] if strokes is None else strokes:
                r, Q = float(s["r"]), np.asarray(s["pts"], float).reshape(-1, 2)
                if not (0.5 <= r <= patch.MAX_R and len(Q) and np.isfinite(Q).all()):
                    raise ValueError(f"a patch needs points and a size of 1 to {2 * patch.MAX_R:.0f} px")
                if patch.area(Q, r) > patch.MAX_AREA:
                    raise ValueError("that stroke covers too much to patch: patches are for dust, specks and hairs")
                nid += 1
                ps.append({"id": nid, "pts": np.round(self._plane(mode).to_mosaic(Q), 3).tolist(), "r": r, "src": None})
        elif op == "undo":
            ps = self._undo.pop() if self._undo else ps
        elif op == "remove":
            ps = [p for p in ps if p["id"] != pid]
        elif op == "source":
            src, k = None if d is None else (np.asarray(d, float) @ self._plane(mode).A.T).tolist(), int(part or 0)

            def put(p):                                               # src: per stretch, trailing Nones left off
                if not 0 <= k < patch.count(p["pts"], p["r"]):
                    raise ValueError(f"patch {pid} has no stretch {k}")
                srcs = list(p.get("src") or []) + [None] * (k + 1)
                srcs[k] = src                                         # a view vector, onto the mosaic
                while srcs and srcs[-1] is None:
                    srcs.pop()
                return {**p, "src": srcs or None}
            ps = [put(p) if p["id"] == pid else p for p in ps]
        elif op == "clear":
            ps = []
        else:
            raise ValueError(f"no such patch operation: {op}")
        if op != "undo" and ps != self.patches:
            self._undo.append(self.patches)
        self.patches = ps
        self.ver += 1

    def _heals(self, geom: dict | None = None, specs=None):
        return self._retouch(self.mosaic(), geom or self.geom or self.default_geom(), self.patches if specs is None else specs)

    def _retoucher(self, geom: dict):
        """``retouch(y0, img)`` that puts the patches into a strip of the result of ``geom`` from row ``y0``, or None."""
        if not self.patches:
            return None
        flat, (cx, cy) = self._heals(geom)[1], geom["crop"][:2]
        return lambda y0, img: patch.apply(img, (cx, cy + y0), 1.0, flat)

    def find_dust(self, progress=_quiet, cancel=None) -> int:
        """Look for dust in the result as it is patched so far; kept in ``dust`` until the picture changes."""
        if self.layout is None:
            raise RuntimeError("stitch first")
        pix, geom = self._pix, json.loads(json.dumps(self.geom or self.default_geom()))
        items, sheet = dust.find(self.mosaic(), geom, self._retoucher(geom), progress, cancel)
        with self._lock:
            if self._pix == pix:                                      # else straightened or cropped meanwhile
                self.dust = {"items": items, "sheet": sheet, "key": uuid.uuid4().hex[:8]}
        return len(items)

    def patch_view(self, mode: str = "result") -> list[dict]:
        """The patches in px of view ``mode``, each with its stretches and the offset each one's copy came from
        (None: nowhere to copy from)."""
        specs = self.patches
        if not specs or self.layout is None:
            return []
        heals, _ = self._heals(specs=specs)
        fr, out = self._plane(mode), []
        for s, h in zip(specs, heals):
            P, src, ds = fr.to_output(np.asarray(s["pts"], float)), s.get("src") or [], h.ds or []
            parts = [{"pts": np.round(q, 2), "d": ds[k] if k < len(ds) else None, "auto": k >= len(src) or src[k] is None}
                     for k, q in enumerate(patch.pieces(P, patch.count(s["pts"], s["r"])))]
            out.append({"id": s["id"], "pts": np.round(P, 2), "r": s["r"], "parts": parts})
        return clean(out)

    # -- what is known about the result ---------------------------------------------------------------

    def meta(self) -> tuple[bytes | None, float, int]:
        """ICC profile and dpi carried into the exports (the anchor scan's), and the bits a channel of the deepest scan."""
        path = self.paths[self.layout.anchor if self.layout is not None else 0]
        icc, dpi = tiles.read_meta(path)
        return icc, float(self.dpi or (dpi[0] if dpi else 1200.0)), max(tiles.bits(p) for p in self.paths)

    def poses(self) -> list[dict]:
        lay = self.layout
        return [] if lay is None else [{"x": float(lay.o[k][0]), "y": float(lay.o[k][1]),
                                        "theta": math.degrees(lay.theta[k]), "anchor": k == lay.anchor}
                                       for k in range(len(lay.sizes))]

    def result_size(self) -> tuple[int, int] | None:
        return None if self.geom is None else export.size_of(self.geom)

    def qc(self) -> list[dict]:
        """What a person should look at before trusting this set: [{level: 'error'|'warn', msg}]."""
        out, r, n = [], self.report or {}, len(self.paths)

        def add(level, msg):
            out.append({"level": level, "msg": msg})

        if self.layout is None or self.plan is None:
            add("error", "not stitched yet")
            return out
        if r.get("disconnected"):
            add("error", "scans " + " | ".join(", ".join(str(k + 1) for k in g) for g in r["disconnected"])
                + " could not be matched to each other, in any quarter turn, alone or as groups: place them by hand")
        if r.get("turned"):
            by: dict = {}
            for k, deg in r["turned"].items():
                by.setdefault(deg, []).append(k)
            ref = self.layout.anchor + 1
            add("warn", "; ".join(f"scan{'s' if len(ks) > 1 else ''} {', '.join(ks)} lay turned {deg}°" for deg, ks in sorted(by.items()))
                + f" in the scanner against scan {ref}: the result stands the way scan {ref} does (turn it on Straighten & crop)")
        cut = [f"{k + 1} ({', '.join(v)})" for k, v in sorted(self.trimmed.items()) if v] if self.trim_mm else []
        if cut:
            add("ok", f"{self.trim_mm:g} mm cut off where the sheet ran off the glass: scan "
                      + "; scan ".join(cut))
        res = r.get("residual") or {}
        for pair, v in res.items():
            if v["rms"] > RESIDUAL_BAD:
                add("error", f"scans {pair} still disagree by {v['rms']:.1f} px after alignment")
            elif v["rms"] > RESIDUAL_WARN:
                add("warn", f"scans {pair} disagree by {v['rms']:.1f} px (RMS) after alignment")
        seen = {tuple(int(t) - 1 for t in k.split("-")) for k in res}
        for a, b in overlapping_pairs(self.layout):
            if (a, b) not in seen and n > 2:
                add("warn", f"scans {a + 1} and {b + 1} overlap but nothing could be matched between them")
        g = self.geom
        if g is None:
            add("warn", "straightening and crop not done yet")
        else:
            if g["method"] in ("none", "lines"):
                add("warn", "no paper edges found: check the angle and the crop by eye")
            elif g["method"] == "edges" and g["confidence"] == "low":
                add("warn", f"paper edges disagree by {g['spread_deg']:.2f}°: check the angle" if sum(v["found"] for v in g["sides"].values()) > 1
                    else "only one paper edge found: check the angle")
            x0, y0, x1, y1 = g["crop"]
            (_, _), (pw, ph), _, _ = self.view("plane")
            if (x1 - x0) * (y1 - y0) < 0.5 * pw * ph:
                add("warn", "the crop is less than half of the stitched area")
        return out

    def status(self) -> str:
        q = self.qc()
        return "error" if any(x["level"] == "error" for x in q) else "warn" if q else "ok"

    def summary(self) -> dict:
        lay = self.layout
        ext = None if lay is None else lay.extent()
        return clean({
            "name": self.name, "folder": self.folder, "paths": self.paths, "tiles": self.info,
            "aligned": lay is not None, "seamed": self.plan is not None, "poses": self.poses(),
            "canvas": None if ext is None else [int(math.ceil(ext[1][0] - math.floor(ext[0][0]))) + 1,
                                                int(math.ceil(ext[1][1] - math.floor(ext[0][1]))) + 1],
            "report": {k: v for k, v in self.report.items() if k != "edges"},
            "geom": self.geom, "result_size": self.result_size(), "adj": self.adj, "max_deg": self.max_deg,
            "trim_mm": self.trim_mm, "tilts": self.tilts,
            "dpi": self.dpi, "exports": self.exports, "ver": self.ver, "key": self.key, "open": True,
            "patched": len(self.patches), "undo": len(self._undo),
            "qc": self.qc() if lay is not None else [],
        })

    # -- views -----------------------------------------------------------------------------------------

    def view(self, mode: str):
        """((origin x, y), (w, h), angle_deg, pivot) of a view: its pixels are full-resolution ones."""
        ext = self.layout.extent()
        o = np.floor(ext[0])
        if mode == "mosaic":
            return tuple(o), tuple(int(v) + 1 for v in np.ceil(ext[1] - o)), 0.0, tuple(o)
        g = self.geom or self.default_geom()
        a, pivot = export.turn_of(g)
        angle = math.degrees(a)
        if mode == "result":
            x0, y0, x1, y1 = g["crop"]
            return (x0, y0), (int(x1 - x0), int(y1 - y0)), angle, tuple(pivot)
        corners = np.array([[ext[i][0], ext[j][1]] for i in (0, 1) for j in (0, 1)])
        q = pivot + (corners - pivot) @ rot(a)
        lo = np.floor(q.min(0))
        return tuple(lo), tuple(int(v) + 1 for v in np.ceil(q.max(0) - lo)), angle, tuple(pivot)

    def _seams(self) -> np.ndarray:
        if self._seam_pts[0] != id(self.plan):
            lab, pts = self.plan.labels, []
            for a, b, dx, dy in ((lab[:, :-1], lab[:, 1:], 0.5, 0.0), (lab[:-1], lab[1:], 0.0, 0.5)):
                y, x = np.nonzero((a != b) & (a != 255) & (b != 255))
                pts.append(np.c_[x + dx, y + dy])
            self._seam_pts = (id(self.plan), self.plan.ext0 + np.vstack(pts) * self.plan.f)
        return self._seam_pts[1]

    def _overlay(self, img: np.ndarray, frame: Frame, kinds) -> None:
        h, w = img.shape[:2]
        if "seams" in kinds and self.plan is not None:
            q = frame.to_output(self._seams())
            x, y = np.round(q[(q[:, 0] > 0) & (q[:, 0] < w - 2) & (q[:, 1] > 0) & (q[:, 1] < h - 2)]).astype(int).T
            for dy in (0, 1):
                for dx in (0, 1):
                    img[y + dy, x + dx] = SEAM_RGB
        if "outlines" in kinds:
            t = np.linspace(0, 1, 60)
            for k, (tw, th) in enumerate(self.layout.sizes):
                ring = np.concatenate([np.c_[t * (tw - 1), 0 * t], np.c_[0 * t + tw - 1, t * (th - 1)],
                                       np.c_[(1 - t) * (tw - 1), 0 * t + th - 1], np.c_[0 * t, (1 - t) * (th - 1)]])
                q = frame.to_output(self.layout.fwd(k, ring))
                cv2.polylines(img, [np.round(q).astype(np.int32)], True, OUTLINE_RGB, 2, cv2.LINE_AA)
                c = frame.to_output(self.layout.fwd(k, np.array([[tw / 2.0, th / 2.0]])))[0]
                if 0 <= c[0] < w and 0 <= c[1] < h:
                    cv2.putText(img, str(k + 1), (int(c[0]) - 10, int(c[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, OUTLINE_RGB, 3, cv2.LINE_AA)

    def _adjust(self, x: np.ndarray) -> np.ndarray:
        with self._lock:
            key = json.dumps(self.adj, sort_keys=True)
            if self._built[0] != key:
                self._built = (key, color.build(self.adj))
            built = self._built[1]
        return color.apply(x, built)

    def render(self, mode: str, x0: int, y0: int, w: int, h: int, scale: float):
        """Window (x0, y0, w, h) of a view, in view px, at ``scale``: (float RGB 0..1, coverage bool), unadjusted,
        patched (but for the mosaic)."""
        key = (self._pix, mode, x0, y0, w, h, round(scale, 6))
        (ox, oy), _, angle, pivot = self.view(mode)
        with self._lock:
            got = self._cache.get(key)
            if got is not None:
                self._cache.move_to_end(key)
        if got is None:
            fr = Frame(scale, (ox + x0, oy + y0), math.radians(angle), pivot)
            got = self.mosaic().render(fr, 0, 0, max(1, math.ceil(w * scale)), max(1, math.ceil(h * scale)))
            with self._lock:
                self._cache[key] = got
                while len(self._cache) > 8:
                    self._cache.popitem(last=False)
        if mode == "mosaic" or not self.patches:
            return got
        img = got[0].copy()
        patch.apply(img, (ox + x0, oy + y0), scale, self._heals()[1])
        return img, got[1]

    def window(self, mode: str, x0: int, y0: int, w: int, h: int, scale: float, raw: bool = False,
               overlay=()) -> tuple[np.ndarray, np.ndarray]:
        """A window ready to show: (uint8 RGB, coverage float 0..1)."""
        img, cov = self.render(mode, x0, y0, w, h, scale)
        out = export.q8(img if raw else self._adjust(img))
        if overlay:
            (ox, oy), _, angle, pivot = self.view(mode)
            self._overlay(out, Frame(scale, (ox + x0, oy + y0), math.radians(angle), pivot), overlay)
        return out, cov.astype(np.float32)

    def fit(self, mode: str, long_side: int = 1600):
        """The whole view at a scale that puts ``long_side`` px on its long side: (float RGB, coverage, scale)."""
        _, (w, h), _, _ = self.view(mode)
        s = min(1.0, long_side / max(w, h))
        img, cov = self.render(mode, 0, 0, w, h, s)
        return img, cov, s

    def pick(self, mode: str, x: float, y: float, r: int = 8) -> list[float]:
        """Mean color (unadjusted, 0..255) of a patch of ``2r`` px around a view point, at full resolution."""
        img, cov = self.render(mode, int(x) - r, int(y) - r, 2 * r, 2 * r, 1.0)
        m = cov if cov.any() else np.ones_like(cov)
        return (img[m].mean(0) * 255).tolist()

    # -- saving ------------------------------------------------------------------------------------------

    def export(self, preset: dict, folder: str | None = None, progress=_quiet, cancel=None) -> list[dict]:
        if self.geom is None:
            self.straighten()
        geom, adj = json.loads(json.dumps(self.geom)), json.loads(json.dumps(self.adj))
        icc, dpi, bits = self.meta()
        out = folder or resolve_folder(preset.get("folder", ""), self.folder)
        if self.patches:
            progress(0.0, "healing the patches")
        patched = None
        if self.patches:
            Y, X, V, _ = self._heals(geom)[1]
            patched = (Y - int(round(geom["crop"][1])), X - int(round(geom["crop"][0])), V)
        self.exports = clean(export.export_set(self.mosaic(), geom, adj, preset, self.name, out, icc, dpi, bits,
                                               protect=self.paths, progress=progress, cancel=cancel, patches=patched))
        return self.exports

    def to_dict(self) -> dict:
        return clean({"format": FORMAT, "name": self.name, "paths": self.paths, "max_deg": self.max_deg, "dpi": self.dpi,
                      "trim_mm": self.trim_mm, "tilts": self.tilts,
                      "layout": None if self.layout is None else self.layout.to_dict(), "report": self.report,
                      "plan": None if self.plan is None else self.plan.to_dict(), "geom": self.geom, "adj": self.adj,
                      "patches": self.patches, "exports": self.exports})

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path + ".part", "w") as f:
            json.dump(self.to_dict(), f)
        os.replace(path + ".part", path)
        return path

    @classmethod
    def load(cls, path: str, store) -> "Session":
        with open(path) as f:
            d = json.load(f)
        if d.get("format") != FORMAT:
            raise ValueError("not an ArtStitch session file")
        here = os.path.dirname(os.path.abspath(path))
        paths = [p if os.path.exists(p) else os.path.join(here, os.path.basename(p)) for p in d["paths"]]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError("scans not found: " + ", ".join(os.path.basename(p) for p in missing))
        s = cls(paths, store, d["name"])
        s.max_deg, s.dpi, s.report = d.get("max_deg", 3.0), d.get("dpi"), d.get("report") or {}
        s.trim_mm, s.tilts = float(d.get("trim_mm") or 0.0), d.get("tilts")
        s.layout = None if d.get("layout") is None else Layout.from_dict(d["layout"])
        s.plan = None if d.get("plan") is None else Plan.from_dict(d["plan"])
        s.geom, s.adj, s.exports = d.get("geom"), d.get("adj") or {}, d.get("exports") or []
        s.patches = d.get("patches") or []
        for p in s.patches:                   # one source for the whole stroke, as saved before stretches (2026-09-29)
            if p.get("src") and not isinstance(p["src"][0], (list, type(None))):
                p["src"] = [p["src"]] * patch.count(p["pts"], p["r"])
        return s
