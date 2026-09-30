"""Reading scans: probe a file, decode it once, keep the decoded tile in RAM.

A flatbed scan at 1200 dpi is ~120 MP.  Everything downstream (alignment,
seams, the strip renderer, zoomed previews) wants the same decoded pixels many
times over, so a ``TileStore`` decodes each file once and hands out block-mean
proxies at integer factors -- ``proxy(path, f)`` pixel ``i`` covers source
pixels ``[f*i, f*i + f)``, whose center is ``f*i + (f-1)/2``.
"""

from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image

import color

Image.MAX_IMAGE_PIXELS = None

TIFF_EXTS = (".tif", ".tiff")
IMAGE_EXTS = TIFF_EXTS + (".png", ".jpg", ".jpeg")


def ram_bytes() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):   # Windows has no sysconf
        return 16 << 30


def probe(path: str | Path) -> dict:
    """Header-only facts about one scan: size, dpi, ICC, compression."""
    p = Path(path)
    info = {"path": str(p), "name": p.name, "bytes": p.stat().st_size,
            "dpi": None, "icc": False, "alpha": False}
    if p.suffix.lower() in TIFF_EXTS:
        with tifffile.TiffFile(p) as tf:
            pg = tf.pages[0]
            h, w = pg.shape[:2]
            ch = pg.shape[2] if pg.ndim == 3 else 1
            info.update(w=int(w), h=int(h), channels=int(ch), dtype=str(pg.dtype),
                        compression=str(getattr(pg.compression, "name", pg.compression)),
                        alpha=bool(pg.extrasamples))
            if "InterColorProfile" in pg.tags:
                info["icc"] = True
            try:
                xr, yr = pg.resolution
                unit = getattr(pg.resolutionunit, "value", pg.resolutionunit)
                k = 2.54 if unit == 3 else 1.0
                if unit in (2, 3):
                    info["dpi"] = [float(xr) * k, float(yr) * k]
            except Exception:
                pass
    else:
        with Image.open(p) as im:
            info.update(w=im.width, h=im.height, channels=len(im.getbands()),
                        dtype="uint16" if _png16(p) else "uint8", compression=str(im.format), alpha="A" in im.getbands(),
                        icc=bool(im.info.get("icc_profile")))
            if im.info.get("dpi"):
                info["dpi"] = [float(v) for v in im.info["dpi"][:2]]
    return info


def read_meta(path: str | Path) -> tuple[bytes | None, tuple[float, float] | None]:
    """The ICC profile bytes and dpi of a scan, to be carried into the exports."""
    p = Path(path)
    icc, dpi = None, None
    if p.suffix.lower() in TIFF_EXTS:
        with tifffile.TiffFile(p) as tf:
            pg = tf.pages[0]
            if "InterColorProfile" in pg.tags:
                v = pg.tags["InterColorProfile"].value
                icc = bytes(v) if not isinstance(v, bytes) else v
    else:
        with Image.open(p) as im:
            icc = im.info.get("icc_profile")
    d = probe(p).get("dpi")
    if d:
        dpi = (d[0], d[1])
    return icc, dpi


@dataclass
class Tile:
    rgb: np.ndarray                       # (H, W, 3) uint8 or uint16, C-contiguous
    valid: np.ndarray | None = None       # (H, W) bool where the alpha says "nothing here"
    proxies: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.rgb.shape[:2]


def _png16(p: Path) -> bool:
    """A PNG of 16 bits a channel, which Pillow would read as 8."""
    with open(p, "rb") as f:
        h = f.read(26)
    return h[:8] == b"\x89PNG\r\n\x1a\n" and h[24] == 16


def bits(path: str | Path) -> int:
    """Bits a channel of a scan: 8, 16, or 32 (floating point, or integers of 32 bits)."""
    t = probe(path)["dtype"]
    return 8 if t == "uint8" else 16 if t == "uint16" else 32


def decode(path: str | Path) -> Tile:
    """Decode a scan to RGB, 8 or 16 bits a channel.  An alpha channel only ever means "no data here".  A scan of
    32 bits is brought to 16: floating point, as Photoshop's 32 bits, is light, linear, and is put back on the
    profile's tone curve."""
    p = Path(path)
    icc = None
    if p.suffix.lower() in TIFF_EXTS:
        with tifffile.TiffFile(p) as tf:
            a = tf.asarray(maxworkers=os.cpu_count() or 4)
            v = tf.pages[0].tags.get("InterColorProfile")
            icc = v and bytes(v.value)
    elif _png16(p):
        a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        a = a if a.ndim == 2 else a[..., [2, 1, 0, 3][:a.shape[2]]]        # BGR(A)
    else:
        with Image.open(p) as im:
            a = np.asarray(im.convert("RGBA" if "A" in im.getbands() else "RGB"))
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[2] > 8:   # planar (C, H, W)
        a = np.moveaxis(a, 0, -1)
    if a.dtype.kind == "f":
        # ponytail: 16 bits (1/65535) are finer than any scanner, and above white is clipped; float tiles if HDR scans come
        x, T, out = np.linspace(0.0, 1.0, color.N), color.tone(icc), np.empty(a.shape, np.uint16)
        for c in range(a.shape[2]):                                     # a channel at a time: float64 of a whole scan
            out[..., c] = (np.interp(a[..., c], T[c], x) if c < 3 else np.clip(a[..., c], 0.0, 1.0)) * 65535.0 + 0.5
        a = out
    elif a.dtype.itemsize > 2:
        a = (a >> (8 * a.dtype.itemsize - 16)).astype(np.uint16)
    valid = None
    if a.shape[2] == 4:
        alpha = a[..., 3]
        if alpha.min() < alpha.max():
            valid = alpha > 0
        a = a[..., :3]
    return Tile(np.ascontiguousarray(a), valid)


def _block(img: np.ndarray, f: int) -> np.ndarray:
    """Exact f x f block mean (crops the ragged edge)."""
    if f == 1:
        return img
    h, w = img.shape[0] // f * f, img.shape[1] // f * f
    return cv2.resize(img[:h, :w], (w // f, h // f), interpolation=cv2.INTER_AREA)


class TileStore:
    """Decoded scans by path, evicted least-recently-used above a byte budget."""

    def __init__(self, budget: int | None = None):
        self.budget = budget or max(2 << 30, ram_bytes() // 2)
        self._tiles: OrderedDict[str, Tile] = OrderedDict()
        self._lock = threading.Lock()
        self._path_locks: dict[str, threading.Lock] = {}

    def get(self, path: str | Path) -> Tile:
        key = str(path)
        with self._lock:
            if key in self._tiles:
                self._tiles.move_to_end(key)
                return self._tiles[key]
            lock = self._path_locks.setdefault(key, threading.Lock())
        with lock:                                   # one decode per file, however many callers
            with self._lock:
                if key in self._tiles:
                    return self._tiles[key]
            tile = decode(key)
            with self._lock:
                self._tiles[key] = tile
                for k in list(self._tiles):                  # oldest first, never the tile just decoded
                    if k == key or self._used() <= self.budget:
                        break
                    del self._tiles[k]
            return tile

    def _used(self) -> int:
        return sum(t.rgb.nbytes for t in self._tiles.values())

    def drop(self, path: str | Path) -> None:
        with self._lock:
            self._tiles.pop(str(path), None)

    def proxy(self, path: str | Path, f: int, gray: bool = False) -> np.ndarray:
        """Block mean at 1/f.  RGB in the tile's own dtype, or float32 luma in 0..255.  Cached."""
        t = self.get(path)
        key = (f, gray)
        if key not in t.proxies:
            if gray:
                c = self.proxy(path, f).astype(np.float32)
                if t.rgb.dtype == np.uint16:
                    c /= 257.0
                t.proxies[key] = (c[..., 0] * 0.299 + c[..., 1] * 0.587 + c[..., 2] * 0.114).astype(np.float32)
            else:
                t.proxies[key] = np.ascontiguousarray(_block(t.rgb, f))
        return t.proxies[key]

    def proxy_valid(self, path: str | Path, f: int) -> np.ndarray | None:
        t = self.get(path)
        if t.valid is None:
            return None
        return _block(t.valid.astype(np.uint8), f) > 127


# --- grouping ----------------------------------------------------------------

_TILE_SUFFIX = re.compile(r"^(?P<stem>.+?)[ _\-](?P<n>\d{1,2})$")


def set_key(name: str) -> tuple[str, int]:
    """("Irchel_redsign 2.tiff") -> ("Irchel_redsign", 2); no number -> tile 0."""
    stem = Path(name).stem
    m = _TILE_SUFFIX.match(stem)
    return (m.group("stem"), int(m.group("n"))) if m else (stem, 0)


def group_sets(paths: list[str | Path]) -> list[dict]:
    """Group scans of one artwork by file name: "name", "name 1", "name 2" ... .

    Files that look like an earlier *result* (an area far above the rest of
    their set) come back with ``suspect=True`` so the batch leaves them out by
    default.
    """
    groups: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for p in paths:
        p = Path(p)
        stem, n = set_key(p.name)
        groups.setdefault((str(p.parent), stem), []).append((n, str(p)))
    sets = []
    for (folder, stem), members in sorted(groups.items()):
        members.sort()
        files = []
        for _, path in members:
            try:
                files.append(probe(path))
            except Exception as e:                    # unreadable file: report, do not abort the folder
                files.append({"path": path, "name": Path(path).name, "error": str(e)})
        areas = sorted(f["w"] * f["h"] for f in files if "w" in f)
        med = areas[len(areas) // 2] if areas else 0
        for f in files:
            f["suspect"] = bool("w" in f and len(areas) > 2 and f["w"] * f["h"] > 1.6 * med)
        sets.append({"name": stem, "folder": folder, "files": files})
    return sets


def list_images(folder: str | Path, exts=TIFF_EXTS) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("."))
