"""Color adjustments: one function of float RGB, shared by the preview and the export.

Per-channel levels and curves are folded into three 16-bit lookup tables; hue,
saturation and lightness are one affine 3x3 matrix.  Identity settings cost nothing.

Settings (all optional)::

    {"levels": {"rgb": [black, gamma, white, out_black, out_white], "r": [...], "g": [...], "b": [...]},
     "curves": {"rgb": [[0, 0], [0.5, 0.6], [1, 1]], "r": [...], ...},
     "hue": 0, "sat": 0, "light": 0, "contrast": 0, "temp": 0, "tint": 0}

levels are 0..1 (gamma > 1 brightens the midtones), curves are control points on
a monotone cubic; hue is degrees, the rest -100..100 (temp: + warmer, tint: + magenta,
contrast: -100..100 is a factor 0.5..2 about mid gray).
"""

from __future__ import annotations

import math
import struct

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

N = 65536
CH = ("r", "g", "b")
LEVELS_ID = [0.0, 1.0, 1.0, 0.0, 1.0]
MATRIX_KEYS = ("hue", "sat", "light", "contrast", "temp", "tint")
LUMA = np.array([0.213, 0.715, 0.072])


def _levels(x: np.ndarray, p) -> np.ndarray:
    if not p or list(p) == LEVELS_ID:
        return x
    ib, g, iw, ob, ow = (float(v) for v in p)
    y = np.clip((x - ib) / max(iw - ib, 1e-6), 0.0, 1.0) ** (1.0 / max(g, 1e-3))
    return ob + y * (ow - ob)


def _curve(x: np.ndarray, pts) -> np.ndarray:
    if not pts:
        return x
    p = sorted((float(a), float(b)) for a, b in pts)
    xs, ys = [p[0][0]], [p[0][1]]
    for a, b in p[1:]:
        if a > xs[-1] + 1e-6:
            xs.append(a)
            ys.append(b)
    if len(xs) < 2 or (len(xs) == 2 and xs == [0.0, 1.0] and ys == [0.0, 1.0]):
        return x
    return np.clip(PchipInterpolator(xs, ys)(np.clip(x, xs[0], xs[-1])), 0.0, 1.0)    # flat past the ends, as Photoshop


def curve_samples(pts, n: int = 64) -> list[float]:
    """The curve through ``pts`` at ``n`` evenly spaced inputs, for drawing it exactly as it is applied."""
    return _curve(np.linspace(0.0, 1.0, n), pts).tolist()


def matrix(hue: float, sat: float, light: float, contrast: float = 0.0, temp: float = 0.0,
           tint: float = 0.0) -> np.ndarray | None:
    """3x4 affine: white balance, saturation, hue rotation about the gray axis, lightness, contrast."""
    if not (hue or sat or light or contrast or temp or tint):
        return None
    s, (c, n) = 1.0 + sat / 100.0, (math.cos(math.radians(hue)), math.sin(math.radians(hue)))
    S = np.outer(np.ones(3), LUMA) * (1 - s) + np.eye(3) * s
    H = np.array([[.213 + c * .787 - n * .213, .715 - c * .715 - n * .715, .072 - c * .072 + n * .928],
                  [.213 - c * .213 + n * .143, .715 + c * .285 + n * .140, .072 - c * .072 - n * .283],
                  [.213 - c * .213 - n * .787, .715 - c * .715 + n * .715, .072 + c * .928 + n * .072]])
    G = np.diag([2 ** (temp / 250), 2 ** (-tint / 250), 2 ** (-temp / 250)])
    l, k = light / 100.0, 2.0 ** (contrast / 100.0)
    a, b = (1 + l, 0.0) if l < 0 else (1 - l, l)
    return np.hstack([k * a * H @ S @ G, np.full((3, 1), k * b + 0.5 * (1 - k))]).astype(np.float32)


def build(adj: dict | None):
    """(luts (3, N) float32 or None, matrix or None) for a settings dict."""
    adj = adj or {}
    lv, cv = adj.get("levels") or {}, adj.get("curves") or {}
    x = np.linspace(0.0, 1.0, N)
    luts, used = [], False
    for c in CH:
        y = _curve(_levels(_curve(_levels(x, lv.get(c)), cv.get(c)), lv.get("rgb")), cv.get("rgb"))
        used |= not np.array_equal(y, x)
        luts.append(y.astype(np.float32))
    return (np.stack(luts) if used else None), matrix(*(float(adj.get(k, 0)) for k in MATRIX_KEYS))


def apply(x: np.ndarray, built) -> np.ndarray:
    """float32 (h, w, 3) in 0..1 -> adjusted float32 in 0..1 (the input may be returned as is)."""
    luts, M = built
    if luts is not None:
        idx = np.clip(x * (N - 1) + 0.5, 0, N - 1).astype(np.uint16)
        y = np.empty_like(x)
        for c in range(3):
            y[..., c] = luts[c][idx[..., c]]
        x = y
    if M is not None:
        x = np.clip(cv2.transform(np.ascontiguousarray(x), M), 0.0, 1.0)
    return x


def _trc(t: bytes, x: np.ndarray) -> np.ndarray:
    """An ICC tone curve (``curv`` or ``para``) at ``x``: the light, linear, that those levels stand for."""
    if t[:4] == b"curv":
        n, = struct.unpack_from(">I", t, 8)
        if n < 2:
            return x ** (struct.unpack_from(">H", t, 12)[0] / 256.0) if n else x
        return np.interp(x, np.linspace(0.0, 1.0, n), np.frombuffer(t, ">u2", n, 12) / 65535.0)
    if t[:4] != b"para":
        raise ValueError("not a tone curve")
    f, = struct.unpack_from(">H", t, 8)
    p = np.frombuffer(t, ">i4", (1, 3, 4, 5, 7)[f], 12) / 65536.0      # Y = (aX + b)^g + e from X = d, cX + off below
    g, a, b = (p[0], 1.0, 0.0) if f == 0 else p[:3]
    c, d = (0.0, -b / a) if f < 3 else p[3:5]
    e, off = (p[3], p[3]) if f == 2 else (p[5], p[6]) if f == 4 else (0.0, 0.0)
    return np.where(x >= d, np.maximum(a * x + b, 0.0) ** g + e, c * x + off)


def tone(icc: bytes | None) -> np.ndarray:
    """(3, N) float32: the light, linear, that each of N levels 0..1 of red, green and blue stands for under the
    profile's tone curves (``kTRC`` for gray), as Photoshop reads a 32-bit file; sRGB's without a profile, or one
    made of tables only."""
    x, tags = np.linspace(0.0, 1.0, N), {}
    try:
        for i in range(struct.unpack_from(">I", icc, 128)[0] if icc else 0):
            sig, at, n = struct.unpack_from(">4sII", icc, 132 + 12 * i)
            tags[sig] = icc[at:at + n]
    except struct.error:
        tags = {}
    out = []
    for s in (b"rTRC", b"gTRC", b"bTRC"):
        t = tags.get(s, tags.get(b"kTRC"))
        try:
            out.append(_trc(t, x))
        except (TypeError, struct.error, ValueError, IndexError):
            out.append(np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4))
    return np.stack(out).astype(np.float32)


def histogram(img: np.ndarray) -> list[list[int]]:
    """256-bin histograms of R, G, B and luma of a float image."""
    q = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
    luma = np.clip((img @ LUMA.astype(np.float32)) * 255 + 0.5, 0, 255).astype(np.uint8)
    return [np.bincount(a.ravel(), minlength=256).tolist() for a in (q[..., 0], q[..., 1], q[..., 2], luma)]


def auto_levels(hist: list[list[int]], clip: float = 0.001, per_channel: bool = True) -> dict:
    """Black and white points that clip ``clip`` of the pixels at each end (Auto Tone / Auto Contrast)."""
    pts = []
    for h in hist[:3]:
        c = np.cumsum(h) / max(sum(h), 1)
        pts.append((int(np.searchsorted(c, clip)), int(np.searchsorted(c, 1 - clip))))
    if per_channel:
        return {c: [b / 255, 1.0, max(w, b + 1) / 255, 0.0, 1.0] for c, (b, w) in zip(CH, pts)}
    b, w = min(p[0] for p in pts), max(p[1] for p in pts)
    return {"rgb": [b / 255, 1.0, max(w, b + 1) / 255, 0.0, 1.0]}
