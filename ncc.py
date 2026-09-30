"""Masked normalized cross-correlation by FFT -- the registration primitive.

Adapted from ASFQ/stitch_core.py (Padfield 2012, "Masked object registration in
the Fourier domain").  Only the generic part is carried over: ASFQ's optical
density / tissue mask belongs to histology and does not travel.  Here the
signal is a high-passed gray level and the mask is "this pixel is inside the
tile", optionally minus the scanner lid.

Convention (verified in test_artstitch.py): ``ncc[ky, kx]`` scores placing B at
translation ``(tx, ty)`` relative to A, i.e. ``A(x) ~ B(x - t)``, with the usual
FFT wrap-around for negative shifts.
"""

from __future__ import annotations

import numpy as np
from scipy import fft as sfft
from scipy import ndimage as ndi


def fast_shape(h: int, w: int) -> tuple[int, int]:
    return (sfft.next_fast_len(h), sfft.next_fast_len(w))


def pack_fft(signal: np.ndarray, mask: np.ndarray, fshape: tuple[int, int],
             workers: int = -1) -> dict[str, np.ndarray]:
    """The three transforms a tile needs, whichever side of a pair it is on."""
    m = mask.astype(np.float32)
    f = np.where(mask, signal, 0).astype(np.float32)
    rf = lambda x: sfft.rfft2(x, fshape, workers=workers).astype(np.complex64)
    return {"F1": rf(f), "F2": rf(f * f), "F3": rf(m)}


def masked_ncc_map(pa: dict, pb: dict, fshape: tuple[int, int], min_overlap: int,
                   workers: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Masked NCC for every integer shift, plus the overlap count at each shift.

    Every moment the Pearson correlation over the *shared* masked region needs
    is itself a correlation, so six inverse transforms give the whole map.
    """
    ir = lambda X: sfft.irfft2(X, fshape, workers=workers)
    A1, A2, A3 = pa["F1"], pa["F2"], pa["F3"]
    B1, B2, B3 = pb["F1"], pb["F2"], pb["F3"]

    n = np.maximum(ir(A3 * np.conj(B3)), 0.0)          # overlap area
    fg = ir(A1 * np.conj(B1))                          # sum f*g
    f1 = ir(A1 * np.conj(B3))                          # sum f over overlap
    g1 = ir(A3 * np.conj(B1))                          # sum g over overlap
    f2 = ir(A2 * np.conj(B3))
    g2 = ir(A3 * np.conj(B2))

    valid = n >= float(min_overlap)
    sn = np.where(valid, n, 1.0)
    num = fg - f1 * g1 / sn
    den = np.sqrt(np.maximum(f2 - f1 * f1 / sn, 0.0) * np.maximum(g2 - g1 * g1 / sn, 0.0))
    ncc = np.where(valid & (den > 1e-8), num / np.maximum(den, 1e-8), 0.0)
    return np.clip(ncc, -1.0, 1.0).astype(np.float32), n.astype(np.float32)


def shift_from_index(ky: int, kx: int, fshape: tuple[int, int]) -> tuple[int, int]:
    fy, fx = fshape
    return int(kx if kx < fx // 2 else kx - fx), int(ky if ky < fy // 2 else ky - fy)


def _subpixel(ncc: np.ndarray, ky: int, kx: int) -> tuple[float, float]:
    """Parabolic fit through the three samples around the peak, per axis."""
    fy, fx = ncc.shape

    def par(m, c, p):
        d = m - 2 * c + p
        return 0.0 if abs(d) < 1e-9 else float(np.clip(0.5 * (m - p) / d, -0.75, 0.75))

    return (par(ncc[ky, (kx - 1) % fx], ncc[ky, kx], ncc[ky, (kx + 1) % fx]),
            par(ncc[(ky - 1) % fy, kx], ncc[ky, kx], ncc[(ky + 1) % fy, kx]))


def peak(ncc: np.ndarray, counts: np.ndarray, fshape: tuple[int, int],
         limit: tuple[float, float], saturation: float, exclusion: int = 6,
         centre: tuple[float, float] = (0.0, 0.0)) -> dict | None:
    """Best shift and how much to believe it.

    Shifts are considered within ``limit`` (x, y) of ``centre`` -- the origin unless
    the caller knows roughly where to look.  Scores are weighted by
    sqrt(n / saturation) while the peak is *found* -- a shift that overlaps
    forty pixels can reach NCC 1.0 by accident -- and the plain NCC is reported.
    ``peak_ratio`` is the winner over the best score outside a small box around
    it.
    """
    fy, fx = fshape
    ty = np.where(np.arange(fy) < fy // 2, np.arange(fy), np.arange(fy) - fy)
    tx = np.where(np.arange(fx) < fx // 2, np.arange(fx), np.arange(fx) - fx)
    allowed = (np.abs(ty - centre[1])[:, None] <= limit[1]) & (np.abs(tx - centre[0])[None, :] <= limit[0])
    w = np.sqrt(np.clip(counts / float(max(saturation, 1.0)), 0.0, 1.0))
    weighted = ncc * w
    scores = np.where(allowed, weighted, -2.0)
    ky, kx = np.unravel_index(int(np.argmax(scores)), scores.shape)
    if scores[ky, kx] <= -1.5:
        return None
    yy = (np.arange(fy)[:, None] - ky + fy // 2) % fy - fy // 2
    xx = (np.arange(fx)[None, :] - kx + fx // 2) % fx - fx // 2
    far = (np.abs(yy) > exclusion) | (np.abs(xx) > exclusion)
    second = max(float(np.max(np.where(allowed & far, weighted, -2.0))), 1e-3)
    dx, dy = _subpixel(ncc, int(ky), int(kx))
    sx, sy = shift_from_index(int(ky), int(kx), fshape)
    return {"tx": sx + dx, "ty": sy + dy, "ncc": float(ncc[ky, kx]),
            "ratio": float(weighted[ky, kx] / second), "n": float(counts[ky, kx])}


def high_pass(img: np.ndarray, sigma: float) -> np.ndarray:
    """Remove smooth illumination differences, keep texture."""
    a = img.astype(np.float32)
    return a - ndi.gaussian_filter(a, sigma, mode="nearest")
