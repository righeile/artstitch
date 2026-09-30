"""Saving.  The final image is rendered once, strip by strip, and every output of a preset is fed from
those strips: the full-size TIFF is compressed and written as they arrive, a PSD's channels are
compressed into temporary files and put together at the end, the smaller versions are box-downscaled
into a mid-size image by an integer factor and finished (exact size, size cap, PNG, JPEG, PDF) when
the last strip is in.  Nothing is held at full size except what the TIFF writer is compressing, so a
300-megapixel result takes a few hundred megabytes.

Photoshop layers (a PSD, or a TIFF with ``layers``): each scan whole on a layer with a mask, the patches
on a layer above them, and the color adjustments as adjustment layers above those, in the order they
are applied here -- levels and curves per channel, then for all channels, then the hue/saturation/
light/contrast/temperature/tint matrix as a channel mixer.  At 32 bits a channel, which Photoshop keeps
as light, linear, the color is in the layers instead.  The flattened picture is stored as well, for
every other program.

TIFF and PSD are 8, 16 or 32 bits a channel (floating point); PNG, JPEG and PDF are 8.  TIFF is
lossless whatever the compression; the choices differ in size and in who can read them:

    zopfli   Deflate that Zopfli squeezed (about 4 % smaller than deflate, ~10x slower); any reader
    deflate  plain Deflate level 9; any reader
    lzma     about 10 % smaller than deflate; libtiff-based readers (not Photoshop)
    zstd     about 8 % smaller; libtiff-based readers (not Photoshop)
    lzw, none

All but ``none`` use TIFF's predictor (horizontal differencing; at 32 bits that of each byte of the
values), which takes a quarter off photographic data.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import struct
import tempfile
import time
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import date

import cv2
import imagecodecs
import numpy as np
import tifffile
from PIL import Image

import color
from geometry import Frame, rot

ROWS = 192                                    # strip height: a multiple of every box factor in K
K = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
EXT = {"tiff": (".tiff", ".tif"), "psd": (".psd", ".psb"), "png": (".png",), "jpg": (".jpg", ".jpeg"), "pdf": (".pdf",)}
FULL = ("tiff", "psd")                        # always full size; the rest take a size
COMPRESSIONS = ("zopfli", "deflate", "lzma", "zstd", "lzw", "none")

DEFAULT_PRESET = {
    "name": "Standard", "folder": "stitched", "overwrite": "number",
    "outputs": [{"kind": "tiff", "name": "{name}", "compression": "zopfli"},
                {"kind": "png", "name": "{name}_300dpi", "size": "dpi", "dpi": 300},
                {"kind": "png", "name": "{name}_small", "size": "mb", "mb": 10},
                {"kind": "jpg", "name": "{name}", "size": "dpi", "dpi": 300, "quality": 92}],
}


class Cancelled(Exception):
    pass


# -- geometry of the result ------------------------------------------------------------------

def turn_of(geom: dict) -> tuple[float, np.ndarray]:
    """(angle in radians, pivot) that put the mosaic into the straightened plane.  ``quarter`` clockwise turns of
    90 degrees (which way up the result is) come on top of the straightening angle, about the whole pixel nearest
    the pivot, so a turn alone moves whole pixels and resamples nothing."""
    a, p = math.radians(geom["angle_deg"]), np.asarray(geom["pivot"], float)
    q = int(geom.get("quarter", 0)) % 4
    if not q:
        return a, p
    c, t = np.round(p), -q * math.pi / 2          # X = p + R(a)(c + R(t)(Q - c) - p): one turn, about its fixed point
    return a + t, np.linalg.solve(np.eye(2) - rot(a + t), p + rot(a) @ (c - rot(t) @ c - p))


def turn_rect(rect, geom: dict, dq: int) -> list[int]:
    """A crop [x0, y0, x1, y1] (whole pixels) of the plane after ``dq`` more clockwise quarter turns: the same pixels."""
    c = np.round(np.asarray(geom["pivot"], float))
    x0, y0, x1, y1 = rect
    q = c + (np.array([[x0, y0], [x1 - 1, y1 - 1]], float) - c) @ np.round(rot(-dq * math.pi / 2))   # first and last pixel
    lo, hi = q.min(0), q.max(0)
    return [int(lo[0]), int(lo[1]), int(hi[0]) + 1, int(hi[1]) + 1]


def frame_of(geom: dict, scale: float = 1.0) -> Frame:
    """Output frame of the final image: crop window of the straightened plane, one pixel per mosaic pixel."""
    a, p = turn_of(geom)
    return Frame(scale, tuple(geom["crop"][:2]), a, p)


def size_of(geom: dict) -> tuple[int, int]:
    x0, y0, x1, y1 = geom["crop"]
    return int(math.floor(x1 - x0)), int(math.floor(y1 - y0))


def q8(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def q16(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 65535.0 + 0.5, 0, 65535).astype(np.uint16)


def quant(x: np.ndarray, bits: int) -> np.ndarray:
    """Float 0..1 as stored at ``bits`` a channel: 8 and 16 in whole levels, 32 as it is."""
    return q8(x) if bits == 8 else q16(x) if bits == 16 else x.astype(np.float32)


def strips(mos, geom: dict, rows: int = ROWS, cancel=None, each=None):
    """(y0, float32 RGB 0..1) strips of the final image, before any color adjustment.  ``each(y0, k, ...)`` gets
    every tile of each strip as it renders (``Mosaic.render``)."""
    W, H = size_of(geom)
    fr = frame_of(geom)
    for y0 in range(0, H, rows):
        if cancel and cancel():
            raise Cancelled()
        img, _ = mos.render(fr, 0, y0, W, min(rows, H - y0), each=each and (lambda *a: each(y0, *a)))
        yield y0, img


# -- TIFF ------------------------------------------------------------------------------------------

def _predict(s: np.ndarray) -> np.ndarray:
    """TIFF's predictor: horizontal differencing (2), or for floating point (3) that of each byte of the values."""
    s = np.ascontiguousarray(s)
    return imagecodecs.floatpred_encode(s, axis=-2) if s.dtype.kind == "f" else imagecodecs.delta_encode(s, axis=1)


_CODEC = {                                    # name -> (TIFF compression, strip encoder)
    "zopfli": ("adobe_deflate", lambda s: imagecodecs.zopfli_encode(_predict(s), level=1)),
    "deflate": ("adobe_deflate", lambda s: imagecodecs.zlib_encode(_predict(s), level=9)),
    "lzma": ("lzma", lambda s: imagecodecs.lzma_encode(_predict(s), level=6)),
    "zstd": ("zstd", lambda s: imagecodecs.zstd_encode(_predict(s), level=19)),
    "lzw": ("lzw", lambda s: imagecodecs.lzw_encode(_predict(s))),
    "none": (None, lambda s: np.ascontiguousarray(s).tobytes()),
}


def write_tiff(path: str, strip_iter, W: int, H: int, bits: int, comp: str, icc: bytes | None, dpi: float,
               note: str = "", workers: int | None = None, layered: int = 0) -> None:
    """Write strips (uint8, uint16 or float32 for ``bits`` 8, 16, 32; ROWS rows each) as a TIFF, compressed on
    ``workers`` threads as they arrive.
    ``layered``: at most this many bytes of Photoshop layers will follow -- a stand-in tag for ``add_tiff_layers``
    to point at them."""
    code, enc = _CODEC[comp]
    workers = workers or max(2, (os.cpu_count() or 4) - 1)

    def compressed():
        with ThreadPoolExecutor(workers) as ex:
            q: deque = deque()
            for s in strip_iter:
                q.append(ex.submit(enc, s))
                if len(q) >= 2 * workers:
                    yield q.popleft().result()
            while q:
                yield q.popleft().result()

    big = W * H * 3 * (bits // 8) + layered > 3.5e9                # raw size bounds the compressed one
    tags = ([(34675, 7, len(icc), icc, True)] if icc else []) + ([(37724, 7, len(PS_MAGIC), PS_MAGIC, True)] if layered else [])
    with tifffile.TiffWriter(path, bigtiff=big) as tw:
        tw.write(compressed(), shape=(H, W, 3), dtype={8: "uint8", 16: "uint16", 32: "float32"}[bits], photometric="rgb",
                 compression=code, predictor=comp != "none", rowsperstrip=ROWS, resolution=(dpi, dpi),
                 resolutionunit="INCH", software="ArtStitch", description=note or None, metadata=None,
                 extratags=tags or None)


def add_tiff_layers(path: str, write_layers, bits: int) -> None:
    """Append Photoshop's layer data block to a TIFF that ``write_tiff(layered=True)`` wrote, and point its
    ImageSourceData tag at it.  ``write_layers(f)`` writes the layer information."""
    with tifffile.TiffFile(path) as t:
        at, big, bo = t.pages[0].tags[37724].offset, t.is_bigtiff, t.byteorder
    with open(path, "r+b") as f:
        f.seek(0, 2)
        f.write(bytes(f.tell() & 1))                               # values start on a word boundary
        start = f.tell()
        f.write(PS_MAGIC + b"8BIM" + LAYERS_KEY[bits])
        with _sized(f, ">I", 4):
            write_layers(f)
        n = f.tell() - start
        f.seek(at + 4)
        f.write(struct.pack(bo + ("QQ" if big else "II"), n, start))


# -- Photoshop ---------------------------------------------------------------------------------------------

PS_MAGIC = b"Adobe Photoshop Document Data Block\0"
LAYERS_KEY = {8: b"Layr", 16: b"Lr16", 32: b"Lr32"}  # the block that holds the layers at each depth
MIXER = "Hue, saturation, light, contrast, temperature, tint"


def _live(f, p) -> bool:
    t = np.linspace(0.0, 1.0, 5)
    return f(t, p) is not t                     # color._levels and color._curve hand an identity back untouched


def ps_adjustments(adj: dict | None) -> list[tuple[str, bytes, bytes]]:
    """(name, key, data) of the Photoshop adjustment layers that redo ``adj`` on the unadjusted picture, bottom
    first.  Levels keep their points to 1/255 and gamma to 0.01; curves go through the same points on Photoshop's
    own spline; the matrix becomes a channel mixer in whole percent."""
    adj = adj or {}
    lv, cv = adj.get("levels") or {}, adj.get("curves") or {}
    b8 = lambda v: min(max(int(round(float(v) * 255)), 0), 255)

    def levl(ps):                               # composite, red, green, blue, then the unused records
        out = struct.pack(">H", 2)
        for p in ps + [None] * 25:
            ib, g, iw, ob, ow = p if _live(color._levels, p) else color.LEVELS_ID
            lo = min(b8(ib), 253)
            out += struct.pack(">5h", lo, max(b8(iw), lo + 2), b8(ob), b8(ow), round(min(max(g, 0.1), 9.99) * 100))
        return out

    def curv(chans):                            # {0: composite, 1: red, 2: green, 3: blue} -> control points
        pts = {}
        for i, p in chans.items():
            q = {}
            for a, b in sorted(p):
                q.setdefault(b8(a), b8(b))      # points closer than 1/255 are one point; both hold the ends flat
            pts[i] = sorted(q.items())
        one = lambda q: struct.pack(">H", len(q)) + b"".join(struct.pack(">2H", o, i) for i, o in q)
        return (struct.pack(">BHHH", 0, 1, 0, sum(1 << i for i in pts)) + b"".join(one(pts[i]) for i in sorted(pts)) +
                b"Crv " + struct.pack(">3H", 4, 0, len(pts)) + b"".join(struct.pack(">H", i) + one(pts[i]) for i in sorted(pts)))

    out = []
    if any(_live(color._levels, lv.get(c)) for c in color.CH):
        out.append(("Levels: red, green, blue", b"levl", levl([None] + [lv.get(c) for c in color.CH])))
    live = {i + 1: cv[c] for i, c in enumerate(color.CH) if _live(color._curve, cv.get(c))}
    if live:
        out.append(("Curves: red, green, blue", b"curv", curv(live)))
    if _live(color._levels, lv.get("rgb")):
        out.append(("Levels", b"levl", levl([lv["rgb"], None, None, None])))
    if _live(color._curve, cv.get("rgb")):
        out.append(("Curves", b"curv", curv({0: cv["rgb"]})))
    M = color.matrix(*(float(adj.get(k, 0)) for k in color.MATRIX_KEYS))
    if M is not None:
        pc = lambda v: int(np.clip(round(float(v) * 100), -200, 200))
        out.append((MIXER, b"mixr", struct.pack(">HH", 1, 0) + b"".join(
            struct.pack(">5h", pc(m[0]), pc(m[1]), pc(m[2]), 0, pc(m[3])) for m in M) + struct.pack(">5h", 40, 40, 20, 0, 0)))
    return out


@contextmanager
def _sized(f, fmt: str, pad: int = 1):
    """A length field of format ``fmt`` in front of what the block writes, padded to ``pad`` and filled in after."""
    at = f.tell()
    f.write(struct.pack(fmt, 0))
    yield
    n = f.tell() - at - struct.calcsize(fmt)
    f.write(bytes(-n % pad))
    end = f.tell()
    f.seek(at)
    f.write(struct.pack(fmt, n + -n % pad))
    f.seek(end)


def _block(key: bytes, data: bytes, pad: int = 2) -> bytes:
    data += bytes(-len(data) % pad)
    return b"8BIM" + key + struct.pack(">I", len(data)) + data


def _record(name: str, rect, chans, flags: int, blocks: bytes, psb: bool, mask=None) -> bytes:
    """One layer record; ``chans`` is [(channel id, bytes of its data with the compression code)], and ``mask`` the
    rectangle of the layer mask (channel -2), if it has one."""
    short = name.encode("ascii", "replace")[:255]
    short = bytes([len(short)]) + short
    extra = ((struct.pack(">I4iBB2x", 20, *mask, 0, 0) if mask else struct.pack(">I", 0)) +   # black outside it
             struct.pack(">I", 0) + short + bytes(-len(short) % 4) +                           # no blending ranges
             _block(b"luni", struct.pack(">I", len(name)) + name.encode("utf-16-be"), 4) + blocks)
    return (struct.pack(">4iH", *rect, len(chans)) + b"".join(struct.pack(">hQ" if psb else ">hI", c, n) for c, n in chans) +
            b"8BIMnorm" + struct.pack(">4BI", 255, 0, flags, 0, len(extra)) + extra)


def _copy(dst, src) -> None:
    src.seek(0)
    shutil.copyfileobj(src, dst, 1 << 24)


class Planes:
    """Full-size strips as Photoshop channel data, each channel compressed into an anonymous temporary file as the
    strips arrive: ``how`` is "zip" (with prediction) for a layer, "rle" for the flattened picture (Photoshop reads
    nothing else there), or "raw" for one of 32 bits.  The ``n`` channels are compressed on a thread each while the
    next strip renders."""

    def __init__(self, folder: str, how: str, psb: bool = False, n: int = 3):
        self.how, self.psb = how, psb
        self.files = [tempfile.TemporaryFile(dir=folder) for _ in range(n)]
        self.rows: list[list[int]] = [[] for _ in range(n)]    # RLE: bytes of every row
        self.z = [zlib.compressobj(6) for _ in range(n)]
        self.ex, self.busy, self.done = ThreadPoolExecutor(n), [], False

    def _one(self, c: int, a: np.ndarray) -> None:
        if self.how == "rle":
            rows = [imagecodecs.packbits_encode(r) for r in (a if a.dtype == np.uint8 else a.astype(">u2").view(np.uint8))]
            self.rows[c] += map(len, rows)
            self.files[c].write(b"".join(rows))
        elif self.how == "raw":
            self.files[c].write(a.astype(a.dtype.newbyteorder(">")).tobytes())
        elif a.dtype.kind == "f":                   # each row's bytes, most significant of every value first, differenced
            self.files[c].write(self.z[c].compress(imagecodecs.floatpred_encode(a, axis=-1).tobytes()))
        else:
            self.files[c].write(self.z[c].compress(imagecodecs.delta_encode(a, axis=1).astype(a.dtype.newbyteorder(">")).tobytes()))

    def __call__(self, s: np.ndarray) -> None:
        for b in self.busy:
            b.result()
        self.busy = [self.ex.submit(self._one, c, np.ascontiguousarray(s[..., c])) for c in range(len(self.files))]

    def close(self) -> list[int]:
        """Bytes of each channel's data (the layer's: with its 2-byte compression code)."""
        if not self.done:
            for b in self.busy:
                b.result()
            self.ex.shutdown()
            if self.how == "zip":
                for f, z in zip(self.files, self.z):
                    f.write(z.flush())
            self.done = True
        return [f.seek(0, 2) + 2 * (self.how == "zip") for f in self.files]

    def discard(self) -> None:
        self.ex.shutdown(cancel_futures=True)
        for f in self.files:
            f.close()

    def write_composite(self, f) -> None:
        self.close()
        f.write(struct.pack(">H", 0) if self.how == "raw" else
                struct.pack(">H", 1) + np.array(sum(self.rows, []), ">u4" if self.psb else ">u2").tobytes())
        for src in self.files:
            _copy(f, src)


class PixelLayer:
    """A layer of pixels over ``rect`` (top, left, bottom, right) of the picture.  Its channels -- ``ids``,
    Photoshop's: -1 transparency, 0-2 red, green, blue, -2 the mask -- are compressed as the strips go by."""

    def __init__(self, name: str, rect, ids, folder: str, bits: int):
        self.name, self.rect, self.ids, self.q = name, tuple(int(v) for v in rect), ids, lambda x: quant(x, bits)
        self.planes, self.at = Planes(folder, "zip", n=len(ids)), self.rect[0]      # the next row it wants

    def put(self, y: int, x: int, a: np.ndarray) -> None:
        """Pixels ``a`` (float 0..1, a channel each) with their top left at picture (x, y), below every row it has
        had: the rows in between stay empty, as do its columns beside them."""
        l, r = self.rect[1], self.rect[3]
        while self.at < y:
            n = min(ROWS, y - self.at)
            self.planes(self.q(np.zeros((n, r - l, len(self.ids)), np.float32)))
            self.at += n
        if len(a):
            row = np.zeros((len(a), r - l, len(self.ids)), np.float32)
            row[:, x - l:x - l + a.shape[1]] = a
            self.planes(self.q(row))
            self.at = y + len(a)

    def record(self, psb: bool) -> bytes:
        self.put(self.rect[2], self.rect[1], np.zeros((0, 0, len(self.ids))))                # empty to its bottom
        return _record(self.name, self.rect, list(zip(self.ids, self.planes.close())), 0x08, b"", psb,
                       self.rect if -2 in self.ids else None)


def layer_writer(layers: list[PixelLayer], adjustments, psb: bool = False):
    """``write(f)``: Photoshop's layer information -- the pixel layers, bottom first, and the adjustment layers
    above them -- as it stands in a PSD's layer section, a Layr/Lr16/Lr32 block of a TIFF."""
    def write(f):
        recs = [L.record(psb) for L in layers]
        recs += [_record(n, (0, 0, 0, 0), [(-1, 2), (0, 2), (1, 2), (2, 2)], 0x18, _block(k, d, 4 if k == b"curv" else 2), psb)
                 for n, k, d in adjustments]
        f.write(struct.pack(">h", len(recs)) + b"".join(recs))
        for L in layers:
            for src in L.planes.files:
                f.write(struct.pack(">H", 3))                   # ZIP with prediction
                _copy(f, src)
        f.write(bytes(8 * len(adjustments)))                   # four empty channels each: raw, no pixels
    return write


def write_psd(path: str, W: int, H: int, bits: int, icc: bytes | None, dpi: float, comp: Planes, layers=None,
              psb: bool = False) -> None:
    """A PSD (a PSB when ``psb``: over 30000 px a side or 2 GB) of the flattened picture ``comp``, with the layers
    that ``layers(f)`` writes (``layer_writer``) or none."""
    L = ">Q" if psb else ">I"
    res = b"8BIM" + struct.pack(">HHI", 0x03ED, 0, 16) + struct.pack(">IHHIHH", round(dpi * 65536), 1, 1, round(dpi * 65536), 1, 1)
    if icc:
        res += b"8BIM" + struct.pack(">HHI", 0x040F, 0, len(icc)) + icc + bytes(len(icc) & 1)
    with open(path, "wb") as f:
        f.write(b"8BPS" + struct.pack(">H6xHIIHH", 1 + psb, 3, H, W, bits, 3) + struct.pack(">II", 0, len(res)) + res)
        with _sized(f, L, 2):                                  # layer and mask information
            if layers is not None and bits == 8:
                with _sized(f, L, 4):
                    layers(f)
            else:
                f.write(struct.pack(L, 0))
            f.write(struct.pack(">I", 0))                      # no global layer mask
            if layers is not None and bits != 8:
                f.write(b"8BIM" + LAYERS_KEY[bits])
                with _sized(f, L, 4):
                    layers(f)
        comp.write_composite(f)


def psb_needed(W: int, H: int, bits: int) -> bool:
    """Photoshop's PSD stops at 30000 px a side and 2 GB.  Layered, a PSD holds the flattened picture and each scan
    whole: 1.93 times the picture's raw size on a watercolor of five scans, three to be safe."""
    return max(W, H) > 30000 or 3 * W * H * 3 * (bits // 8) > 2e9


# -- downscaled versions -----------------------------------------------------------------------------

class _Down:
    """Box-downscale by the integer ``k`` of a stream of full-size 8-bit strips."""

    def __init__(self, W: int, H: int, k: int):
        self.k, self.w = k, W // k
        self.img = np.empty((H // k, W // k, 3), np.uint8)

    def __call__(self, y0: int, s: np.ndarray) -> None:
        k, rows = self.k, s.shape[0] // self.k * self.k
        if rows:
            self.img[y0 // k:y0 // k + rows // k] = (s[:rows, :self.w * k] if k == 1 else
                                                     cv2.resize(s[:rows, :self.w * k], (self.w, rows // k), interpolation=cv2.INTER_AREA))


def _png(img: np.ndarray, icc, dpi: float, level: int = 9) -> bytes:
    b = io.BytesIO()
    Image.fromarray(img).save(b, "PNG", compress_level=level, dpi=(dpi, dpi), icc_profile=icc)
    return b.getvalue()


def _jpg(img: np.ndarray, icc, dpi: float, quality: int) -> bytes:
    b = io.BytesIO()
    Image.fromarray(img).save(b, "JPEG", quality=int(quality), subsampling=0, optimize=True, dpi=(dpi, dpi), icc_profile=icc)
    return b.getvalue()


def _pdf(img: np.ndarray, icc, dpi: float, quality: int | None = None, level: int = 9) -> bytes:
    """One page the picture's size at ``dpi``: the pixels as PNG's deflate stream (lossless) or as a JPEG."""
    h, w = img.shape[:2]
    if quality:
        data, how = _jpg(img, None, dpi, quality), b"/Filter /DCTDecode"
    else:
        png, data, i = _png(img, None, dpi, level), bytearray(), 8
        while i < len(png):                                     # the IDAT chunks are one zlib stream
            n, key = struct.unpack(">I4s", png[i:i + 8])
            if key == b"IDAT":
                data += png[i + 8:i + 8 + n]
            i += 12 + n
        how = b"/Filter /FlateDecode /DecodeParms << /Predictor 15 /Colors 3 /BitsPerComponent 8 /Columns %d >>" % w
    pw, ph = w * 72.0 / dpi, h * 72.0 / dpi
    stream = lambda d, s: b"<< %s /Length %d >>\nstream\n" % (d, len(s)) + bytes(s) + b"\nendstream"
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.3f %.3f] /Resources << /XObject << /Im0 5 0 R >> >> "
            b"/Contents 4 0 R >>" % (pw, ph),
            stream(b"", b"q %.3f 0 0 %.3f 0 0 cm /Im0 Do Q" % (pw, ph)),
            stream(b"/Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace %s /BitsPerComponent 8 %s"
                   % (w, h, b"[/ICCBased 6 0 R]" if icc else b"/DeviceRGB", how), data)]
    if icc:
        objs.append(stream(b"/N 3 /Alternate /DeviceRGB", icc))
    out, at = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"), []
    for k, o in enumerate(objs, 1):
        at.append(len(out))
        out += b"%d 0 obj\n" % k + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % a for a in at)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


def _scaled(mid: np.ndarray, W: int, H: int, s: float) -> np.ndarray:
    """The mid-size image (box factor k, so scale 1/k) brought down to scale ``s`` <= 1/k, untouched when equal."""
    w, h = (min(m, max(1, int(n * s + 1e-9))) for m, n in zip(mid.shape[1::-1], (W, H)))
    return mid if (w, h) == mid.shape[1::-1] else cv2.resize(mid, (w, h), interpolation=cv2.INTER_AREA)


def _capped(mid: np.ndarray, W: int, H: int, cap: int, encode, final):
    """Largest scale whose file stays within ``cap`` bytes: (image, bytes).  PNG size follows the area."""
    smax = 1.0 / (W // mid.shape[1])
    s = min(smax, math.sqrt(cap / (0.4 * 3 * W * H)))
    best = None
    for _ in range(7):
        img = _scaled(mid, W, H, s)
        n = len(encode(img))
        if n <= cap and (best is None or img.shape[1] > best[0].shape[1]):
            best = (img, n)
        if cap * 0.96 <= n <= cap or (n <= cap and s >= smax):
            break
        s = min(smax, s * math.sqrt(cap * 0.98 / n))
    if best is None:
        raise ValueError("cannot get under the size cap")
    data = final(best[0])
    return (best[0], data) if len(data) <= cap else (best[0], encode(best[0]))


def plan_size(o: dict, W: int, H: int, src_dpi: float) -> tuple[int, float | None]:
    """(box factor k of the mid-size image, scale of the result or None when a size cap decides it)."""
    if o["kind"] in FULL:
        return 1, 1.0
    mode = o.get("size", "dpi")
    if mode == "mb":                            # least share of raw a codec reaches: PNG never 25 %, JPEG 90 about 8 %
        least = 0.08 if o["kind"] == "jpg" or o.get("image") == "jpeg" else 0.25
        smax = min(1.0, math.sqrt(o["mb"] * 1e6 / (least * 3 * W * H)))
        return max(k for k in K if k <= 1 / smax + 1e-9), None
    s = {"full": 1.0, "dpi": o.get("dpi", 300) / src_dpi, "long": o.get("long", 2500) / max(W, H)}[mode]
    s = min(s, 1.0)
    return max(k for k in K if k <= 1 / s + 1e-9), s


# -- presets and file names ------------------------------------------------------------------------------

def file_name(o: dict, ctx: dict, psb: bool = False) -> str:
    s = re.sub(r"\{(\w+)\}", lambda m: str(ctx.get(m.group(1), m.group(0))), o.get("name") or "{name}")
    s = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", s).strip(" .") or ctx["name"]
    s = s if s.lower().endswith(EXT[o["kind"]]) else s + EXT[o["kind"]][0]
    return s[:-4] + ".psb" if psb and o["kind"] == "psd" else s


def depth(o: dict, src_bits: int) -> int:
    """Bits a channel of a full-size output: 8, 16 or 32 as it asks, or on "auto" those of the deepest scan."""
    b = o.get("bits", "auto")
    return src_bits if b == "auto" else int(b)


def psd_is_psb(preset: dict, W: int, H: int, src_bits: int) -> bool:
    return any(o.get("kind") == "psd" and psb_needed(W, H, depth(o, src_bits)) for o in preset.get("outputs", []))


def check_preset(p: dict) -> list[str]:
    """Problems with a preset, empty when it is fine."""
    bad, outs = [], p.get("outputs") or []
    if not outs:
        bad.append("no outputs")
    for k in FULL:
        if sum(o.get("kind") == k for o in outs) > 1:
            bad.append(f"at most one {k.upper()} per preset")
    if p.get("overwrite", "number") not in ("number", "skip", "replace"):
        bad.append("overwrite must be number, skip or replace")
    seen: dict[str, int] = {}
    for i, o in enumerate(outs, 1):
        if o.get("kind") not in EXT:
            bad.append(f"output {i}: kind must be one of {', '.join(EXT)}")
            continue
        if o["kind"] == "tiff" and o.get("compression", "zopfli") not in COMPRESSIONS:
            bad.append(f"output {i}: unknown compression")
        if o["kind"] in FULL and str(o.get("bits", "auto")) not in ("auto", "8", "16", "32"):
            bad.append(f"output {i}: bits must be auto, 8, 16 or 32")
        if o["kind"] not in FULL and o.get("size", "dpi") not in ("dpi", "long", "mb", "full"):
            bad.append(f"output {i}: size must be dpi, long, mb or full")
        if o["kind"] not in FULL and {"mb": o.get("mb", 0), "dpi": o.get("dpi", 300), "long": o.get("long", 2500)}.get(
                o.get("size", "dpi"), 1) <= 0:
            bad.append(f"output {i}: the size must be positive")
        if o["kind"] == "pdf" and o.get("image", "lossless") not in ("lossless", "jpeg"):
            bad.append(f"output {i}: a PDF holds the picture lossless or jpeg")
        if (o["kind"] == "jpg" or o.get("image") == "jpeg") and not 1 <= o.get("quality", 92) <= 100:
            bad.append(f"output {i}: JPEG quality is 1..100")
        n = file_name(o, {"name": "X", "n": 1, "date": "D", "preset": "P", "dpi": 0}).lower()
        if n in seen:
            bad.append(f"outputs {seen[n]} and {i} write the same file name")
        seen[n] = i
    return bad


def describe(preset: dict, name: str, n: int, W: int, H: int, src_dpi: float, src_bits: int = 8) -> list[dict]:
    """What each output of a preset will be, before anything is rendered."""
    ctx = {"name": name, "n": n, "date": date.today().isoformat(), "preset": preset.get("name", "")}
    psb, out = psd_is_psb(preset, W, H, src_bits), []
    for o in preset.get("outputs", []):
        k, s = plan_size(o, W, H, src_dpi)
        d = {"kind": o["kind"], "file": file_name(o, {**ctx, "dpi": round(src_dpi * s) if s else ""}, psb)}
        if o["kind"] in FULL:
            d["bits"] = depth(o, src_bits)
        if s is None:
            d["note"] = f"as large as fits in {o['mb']} MB"
        else:
            w, h = (min(m // k, max(1, int(m * s + 1e-9))) for m in (W, H))
            d.update(w=w, h=h, dpi=round(src_dpi * w / W, 1))
        out.append(d)
    return out


def load_presets(path: str) -> list[dict]:
    try:
        with open(path) as f:
            got = json.load(f)
        return got if isinstance(got, list) and got else [DEFAULT_PRESET]
    except (OSError, ValueError):
        return [DEFAULT_PRESET]


def save_presets(path: str, presets: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".part", "w") as f:
        json.dump(presets, f, indent=1)
    os.replace(path + ".part", path)


def target(path: str, policy: str, protect: set[str]) -> str | None:
    """Where to write: the path itself, a numbered sibling, or None to skip; never an input file."""
    real = lambda p: os.path.realpath(p)
    if real(path) in protect or (os.path.exists(path) and policy == "number"):
        stem, ext = os.path.splitext(path)
        n = 2
        while os.path.exists(f"{stem}_{n}{ext}") or real(f"{stem}_{n}{ext}") in protect:
            n += 1
        return f"{stem}_{n}{ext}"
    return None if os.path.exists(path) and policy == "skip" else path


# -- the whole export -----------------------------------------------------------------------------------------

PATCHES = "Patches"


@contextmanager
def _atomic(path: str):
    """Write to ``path``.part and put it in place only when complete."""
    part = path + ".part"
    try:
        yield part
        os.replace(part, path)
    finally:
        if os.path.exists(part):
            os.remove(part)


def export_set(mos, geom: dict, adj: dict | None, preset: dict, name: str, folder: str, icc: bytes | None,
               src_dpi: float = 1200.0, src_bits: int = 8, protect=(), progress=lambda p, m: None, cancel=None,
               patches=None) -> list[dict]:
    """Render the result and write every output of ``preset`` into ``folder``.  Returns one record per output.
    ``patches`` -- (rows, columns, values) of the patched pixels -- go into the picture before the color.  A PSD
    and a layered TIFF hold each scan as a layer with a mask, the patches on a layer above them and the color
    adjustments above that; at 32 bits, light and linear as Photoshop keeps it there, with the color in the layers."""
    bad = check_preset(preset)
    if bad:
        raise ValueError("; ".join(bad))
    W, H = size_of(geom)
    psb = psd_is_psb(preset, W, H, src_bits)
    protect = {os.path.realpath(p) for p in protect}
    ctx = {"name": name, "n": len(mos.paths), "date": date.today().isoformat(), "preset": preset.get("name", "")}
    policy = preset.get("overwrite", "number")
    os.makedirs(folder, exist_ok=True)
    jobs, results = [], []
    for o in preset["outputs"]:
        k, s = plan_size(o, W, H, src_dpi)
        want = os.path.join(folder, file_name(o, {**ctx, "dpi": round(src_dpi * s) if s else ""}, psb))
        path = target(want, policy, protect)
        if path is None:
            results.append({"kind": o["kind"], "path": want, "skipped": True})
        else:
            jobs.append({"o": o, "k": k, "s": s, "path": path})
    if not jobs:
        return results
    downs = {j["k"]: _Down(W, H, j["k"]) for j in jobs if j["o"]["kind"] not in FULL}
    tiff = next((j for j in jobs if j["o"]["kind"] == "tiff"), None)
    psd = next((j for j in jobs if j["o"]["kind"] == "psd"), None)
    tb, pb = (j and depth(j["o"], src_bits) for j in (tiff, psd))  # bits of the TIFF and the PSD, None when not written
    built = color.build(adj)
    ys = np.zeros(0, int)
    if patches is not None:
        ys, xs, vals = patches
        on = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        o = np.argsort(ys[on], kind="stable")
        ys, xs, vals = ys[on][o], xs[on][o], vals[on][o]
    T = color.tone(icc)

    def lin(x):                                                    # levels -> the light, linear, 32 bits hold
        i = np.clip(x * (color.N - 1) + 0.5, 0, color.N - 1).astype(np.uint16)
        return np.stack([T[c][i[..., c]] for c in range(3)], -1)

    # ponytail: at 32 bits the color is baked into the layers -- what Photoshop's levels and curves do to linear
    # light is not what they do here; adjustment layers there if someone needs to change it again in Photoshop
    bake = lambda px: lin(color.apply(px, built))
    ps = ps_adjustments(adj) if psd or (tiff and tiff["o"].get("layers")) else []
    ps_of = lambda b: [] if b == 32 else ps
    stacks = {}                                                    # bits -> (scan layers by scan, patch layer, its values)
    for b in {pb, tb if tiff and tiff["o"].get("layers") else None} - {None}:
        parts = {}
        for k, p in enumerate(mos.paths):
            box = mos.footprint(k, frame_of(geom), W, H, pad=2)    # wider than render's: holds all it draws of k
            if box:
                parts[k] = PixelLayer(os.path.splitext(os.path.basename(p))[0], (box[1], box[0], box[3], box[2]),
                                      (-1, 0, 1, 2, -2), folder, b)
        heal = PixelLayer(PATCHES, (ys[0], xs.min(), ys[-1] + 1, xs.max() + 1), (-1, 0, 1, 2), folder, b) if len(ys) else None
        stacks[b] = parts, heal, (bake(vals[:, None])[:, 0] if b == 32 else vals) if heal else None
    layers_of = lambda b: list(stacks[b][0].values()) + ([stacks[b][1]] if stacks[b][1] else [])
    comp = Planes(folder, "raw" if pb == 32 else "rle", psb) if psd else None    # the flattened picture, for every other program

    def piece(y0, k, xa, ya, px, valid, mask):                     # scan k in the strip at y0: onto its layers
        for b, (parts, _, _) in stacks.items():
            parts[k].put(y0 + ya, xa, np.dstack([valid, bake(px) if b == 32 else px, mask]))

    def feed():
        for y0, raw in strips(mos, geom, cancel=cancel, each=piece if stacks else None):
            i, j = np.searchsorted(ys, (y0, y0 + len(raw)))
            if i < j:
                raw[ys[i:j] - y0, xs[i:j]] = vals[i:j]
                for _, heal, hv in stacks.values():
                    if heal:
                        l = heal.rect[1]
                        a = np.zeros((ys[j - 1] + 1 - ys[i], heal.rect[3] - l, 4), np.float32)
                        a[ys[i:j] - ys[i], xs[i:j] - l] = np.c_[np.ones(j - i), hv[i:j]]
                        heal.put(ys[i], l, a)
            x = color.apply(raw, built)
            s8 = q8(x)
            for d in downs.values():
                d(y0, s8)
            full = {b: s8 if b == 8 else q16(x) if b == 16 else lin(x) for b in {tb, pb} - {None}}
            if comp:
                comp(full[pb])
            progress(0.02 + 0.8 * min(1.0, (y0 + ROWS) / H), f"rendering {min(H, y0 + ROWS)}/{H} rows")
            yield full.get(tb)

    full_record = lambda kind, path, t, n, b: {"kind": kind, "path": path, "bytes": os.path.getsize(path), "w": W, "h": H,
                                               "dpi": src_dpi, "bits": b, "layers": n, "seconds": round(time.time() - t, 1)}
    try:
        t0 = time.time()
        if tiff:
            n = len(layers_of(tb)) + len(ps_of(tb)) if tiff["o"].get("layers") else 0
            # ponytail: counts the color channels only -- transparency and masks are long runs, next to nothing
            # zipped (tjj: 885 MB of layers under 1.6 GB counted); a BigTIFF, which fewer programs open, is the cost
            more = n and (tb // 8) * 3 * sum((L.rect[2] - L.rect[0]) * (L.rect[3] - L.rect[1]) for L in layers_of(tb)) + (1 << 20)
            with _atomic(tiff["path"]) as part:
                write_tiff(part, feed(), W, H, tb, tiff["o"].get("compression", "zopfli"), icc, src_dpi,
                           f"Stitched from {len(mos.paths)} scans with ArtStitch", layered=more)
                if n:
                    progress(0.83, "adding the Photoshop layers")
                    add_tiff_layers(part, layer_writer(layers_of(tb), ps_of(tb)), tb)
            results.append(full_record("tiff", tiff["path"], t0, n, tb))
        else:
            for _ in feed():
                pass
        if psd:
            t1 = time.time()
            progress(0.84, f"saving {os.path.basename(psd['path'])}")
            with _atomic(psd["path"]) as part:
                write_psd(part, W, H, pb, icc, src_dpi, comp, layer_writer(layers_of(pb), ps_of(pb), psb), psb)
            results.append(full_record("psd", psd["path"], t1, len(layers_of(pb)) + len(ps_of(pb)), pb))
    finally:
        for p in [comp] + [L.planes for b in stacks for L in layers_of(b)]:
            if p:
                p.discard()
    small = [j for j in jobs if j["o"]["kind"] not in FULL]
    for i, j in enumerate(small):
        if cancel and cancel():
            raise Cancelled()
        progress(0.85 + 0.15 * i / len(small), f"saving {os.path.basename(j['path'])}")
        o, mid, t1 = j["o"], downs[j["k"]].img, time.time()
        dpi = lambda im: src_dpi * im.shape[1] / W
        if o["kind"] == "png":
            enc = lambda im, lv=6: _png(im, icc, dpi(im), lv)
            fin = lambda im: _png(im, icc, dpi(im), 9)
        elif o["kind"] == "pdf":
            q = o.get("quality", 92) if o.get("image") == "jpeg" else None
            enc = lambda im: _pdf(im, icc, dpi(im), q, 6)
            fin = lambda im: _pdf(im, icc, dpi(im), q, 9)
        else:
            q = o.get("quality", 92)
            enc = fin = lambda im: _jpg(im, icc, dpi(im), q)
        if j["s"] is None:
            img, data = _capped(mid, W, H, int(o["mb"] * 1e6), enc, fin)
        else:
            img = _scaled(mid, W, H, j["s"])
            data = fin(img)
        with _atomic(j["path"]) as part, open(part, "wb") as f:
            f.write(data)
        results.append({"kind": o["kind"], "path": j["path"], "bytes": len(data), "w": img.shape[1], "h": img.shape[0],
                        "dpi": round(dpi(img), 2), "seconds": round(time.time() - t1, 1)})
    progress(1.0, "done")
    return results
