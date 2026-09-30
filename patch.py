"""Patches: dust, specks and hairs painted out of the result.

A patch is a brush stroke -- points and a radius -- over something that should not be there.  Its pixels are
replaced by a copy of the picture a little way off, moved by whole pixels so the grain of the paper and the paint
come through exactly, and the copy is then corrected by the smoothest surface that makes it meet the pixels around
the stroke (a harmonic membrane: Photoshop's healing brush does the same).  The copy keeps its texture and takes the
color and brightness of the place it goes; nothing is blurred.  The copy comes from the nearby place whose own
surroundings look most like the stroke's surroundings (least squared difference over a ring around the stroke, never
overlapping it), or from where the person dragged it.  A long stroke is healed in stretches, one after another along
it, each copied from a place of its own: one copy can carry on only the lines that run the way it was moved, and a
hair crosses lines at every angle.  The stretches still to heal count as no picture at all, so nothing of the hair
is matched, copied or met; the stretches healed already count as picture, so each copy tends to carry on the last.

Patches are kept in mosaic pixels and healed in the straightened plane at full size, where the result is cut from;
a heal is redone only when its stroke, the straightening or an earlier patch it overlaps changes.  At full size (the
export, 100 % on screen) the healed pixels are put in as they are; zoomed out, what they change is spread over the
screen pixels.  Adding the change at full size would not do: ``cv2.remap`` puts source positions on a 1/32 px
grid, and two windows of the same plane can fall either side of a step of it, which at the hard rim of a speck is a
level or two -- a faint ghost of the speck left in the patch.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

import export
from geometry import Frame, rot

MAX_R = 150.0                                 # px of the full-size picture
MAX_AREA = 400_000                            # px under one stroke: up to ~3 s to heal; patches are for dust, not areas
EMPTY = (np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32))


def ring_of(r: float) -> int:
    """Width of the ring around a stroke that is matched when looking for a place to copy from."""
    return max(3, int(round(r / 2)))


def reach_of(r: float) -> int:
    """How far from the stroke the copy may come from."""
    return int(max(48, 6 * r))


def stretch_of(r: float) -> float:
    """About how long each stretch of a long stroke is: short enough to cross one thing, long enough to match."""
    return max(24.0, 4.0 * r)


def length(pts) -> float:
    p = np.asarray(pts, float).reshape(-1, 2)
    return float(np.hypot(*np.diff(p, axis=0).T).sum())


def area(pts, r: float) -> float:
    """About how many pixels a stroke covers."""
    return 2 * r * length(pts) + math.pi * r * r


def count(pts, r: float) -> int:
    """How many stretches a stroke is healed in: one for a speck.  Lengths are the same in every frame of the plane."""
    return max(1, int(round(length(pts) / stretch_of(r))))


def pieces(Q: np.ndarray, n: int) -> list[np.ndarray]:
    """The polyline ``Q`` cut into ``n`` stretches of equal length."""
    if n == 1:
        return [Q]
    S = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(Q, axis=0).T))])
    cut = np.linspace(0.0, S[-1], n + 1)
    at = lambda c: np.array([[np.interp(c, S, Q[:, 0]), np.interp(c, S, Q[:, 1])]])
    return [np.vstack([at(a), Q[(S > a) & (S < b)], at(b)]) for a, b in zip(cut[:-1], cut[1:])]


def stamp(Q: np.ndarray, r: float, x0: int, y0: int, w: int, h: int, along: bool = False):
    """The pixels of a (h, w) window at (x0, y0) whose centers lie within ``r`` of the polyline ``Q``; with ``along``,
    also how far along the polyline the point of it nearest to each pixel is."""
    near, s = np.full((h, w), np.inf), np.zeros((h, w))
    S = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(Q, axis=0).T))])
    segs = list(zip(Q[:-1], Q[1:], S[:-1])) or [(Q[0], Q[0], 0.0)]
    for a, b, s0 in segs:
        lo = np.floor(np.minimum(a, b) - r).astype(int) - (x0, y0)
        hi = np.ceil(np.maximum(a, b) + r).astype(int) + 1 - (x0, y0)
        xa, ya, xb, yb = max(lo[0], 0), max(lo[1], 0), min(hi[0], w), min(hi[1], h)
        if xa >= xb or ya >= yb:
            continue
        yy, xx = np.mgrid[ya + y0:yb + y0, xa + x0:xb + x0]
        P = np.stack([xx, yy], -1).astype(float)
        ab = b - a
        t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        d2 = ((P - a - t[..., None] * ab) ** 2).sum(-1)
        win, sw = near[ya:yb, xa:xb], s[ya:yb, xa:xb]
        closer = d2 < win
        win[closer] = d2[closer]
        sw[closer] = s0 + t[closer] * math.sqrt(float(ab @ ab))
    m = near <= r * r
    return (m, s) if along else m


def find_source(img: np.ndarray, cov: np.ndarray, mask: np.ndarray, ring: int, reach: int):
    """(dx, dy): where in the window to copy ``mask`` from, or None when nowhere will do."""
    k = 2 * ring + 1
    ys, xs = np.nonzero(mask)                                             # dilated round the stroke only, not the
    y0, y1, x0, x1 = max(ys.min() - ring, 0), ys.max() + ring + 1, max(xs.min() - ring, 0), xs.max() + ring + 1
    F = np.zeros_like(mask)                                               # whole reach: a 300 px brush took 1 s
    F[y0:y1, x0:x1] = cv2.dilate(mask[y0:y1, x0:x1].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    ys, xs = np.nonzero(F)
    fy0, fy1, fx0, fx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    around = (F & ~mask & cov)[fy0:fy1, fx0:fx1].astype(np.float32)
    n = float(around.sum())
    if n < 8:
        return None
    I = np.ascontiguousarray(img - img[F & cov].mean(0), np.float32)          # centered: the sums below cancel less
    ssd = cv2.matchTemplate(I, np.ascontiguousarray(I[fy0:fy1, fx0:fx1]), cv2.TM_SQDIFF, mask=around)
    # a copy that would take in the stroke itself, or pixels outside the picture, is no copy
    bad = cv2.matchTemplate((mask | ~cov).astype(np.float32), F[fy0:fy1, fx0:fx1].astype(np.float32), cv2.TM_CCORR) > 0.5
    ty, tx = np.mgrid[0:ssd.shape[0], 0:ssd.shape[1]]
    dx, dy = tx - fx0, ty - fy0
    near = (dx * dx + dy * dy) / float(reach * reach)
    cost = np.where(bad | (near > 1), np.inf, ssd / (3 * n) * (1 + 0.2 * near))  # of equals, the nearer
    i = int(np.argmin(cost))
    return None if not np.isfinite(cost.flat[i]) else (int(dx.flat[i]), int(dy.flat[i]))


def membrane(mask: np.ndarray, bound: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """(n, 3) on the ``mask`` pixels (in ``np.nonzero`` order): harmonic inside, equal to ``bound`` on the
    neighbors just outside where ``ok`` (elsewhere the edge is free).  The mask must not touch the window edge."""
    ys, xs = np.nonzero(mask)
    n = len(ys)
    idx = np.full(mask.shape, -1, np.int64)
    idx[ys, xs] = np.arange(n)
    diag = np.full(n, 1e-9)                                               # a part with no fixed edge settles at 0
    b = np.zeros((n, bound.shape[2]))
    rows, cols = [np.arange(n)], [np.arange(n)]
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        ny, nx = ys + dy, xs + dx
        j = idx[ny, nx]
        inner = j >= 0
        edge = ~inner & ok[ny, nx]
        diag += inner | edge
        rows.append(np.nonzero(inner)[0])
        cols.append(j[inner])
        b[edge] += bound[ny[edge], nx[edge]]
    vals = np.concatenate([diag, -np.ones(sum(len(r) for r in rows[1:]))])
    A = sparse.csc_matrix((vals, (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return np.asarray(spsolve(A, b)).reshape(n, -1)


def heal(img: np.ndarray, cov: np.ndarray, mask: np.ndarray, d=None, ring: int = 3, reach: int = 48):
    """New values for the ``mask`` pixels of a window (float RGB 0..1, coverage), and the (dx, dy) they were
    copied from: (None, None) when there is nowhere to copy from."""
    if d is None:
        d = find_source(img, cov, mask, ring, reach)
        if d is None:
            return None, None
    dx, dy = d
    nb = np.zeros_like(mask)                                              # the neighbors just outside the stroke
    nb[:-1] |= mask[1:]
    nb[1:] |= mask[:-1]
    nb[:, :-1] |= mask[:, 1:]
    nb[:, 1:] |= mask[:, :-1]
    by, bx = np.nonzero(nb & ~mask)
    bound = np.zeros_like(img)
    bound[by, bx] = img[by, bx] - img[by + dy, bx + dx]
    ok = np.zeros_like(mask)
    ok[by, bx] = cov[by, bx] & cov[by + dy, bx + dx]
    ys, xs = np.nonzero(mask)
    vals = img[ys + dy, xs + dx] + membrane(mask, bound, ok)
    return np.clip(vals, 0.0, 1.0).astype(np.float32), (int(dx), int(dy))


def heal_stroke(img: np.ndarray, cov: np.ndarray, Q: np.ndarray, r: float, ds: list, x0: int = 0, y0: int = 0):
    """Heal the stroke ``Q`` (plane px; the window starts at (x0, y0)) into ``img``, stretch by stretch along it.
    ``ds``: per stretch, where to copy it from, or None to find a place.  Returns the pixels healed and, per stretch,
    where its copy came from (None: nowhere).  The stroke must keep clear of the window's edge."""
    h, w = cov.shape
    mask, s = stamp(Q, r, x0, y0, w, h, along=True)
    mask &= cov
    Y, X = np.nonzero(mask)
    n = len(ds)
    K = np.minimum((s[Y, X] * (n / length(Q))).astype(int), n - 1) if n > 1 else np.zeros(len(Y), int)
    ring, reach = ring_of(r), reach_of(r)
    todo, got = mask.copy(), []
    for k, d in enumerate(ds):
        ys, xs = Y[K == k], X[K == k]
        if not len(ys):
            got.append(None)
            continue
        # healed in a window of its own, nothing done to the whole window: a long scratch has hundreds of stretches
        m = ring + 2 + (reach if d is None else max(abs(d[0]), abs(d[1])))
        wy, wx = max(ys.min() - m, 0), max(xs.min() - m, 0)
        win = np.s_[wy:ys.max() + 1 + m, wx:xs.max() + 1 + m]
        part = np.zeros(todo[win].shape, bool)
        part[ys - wy, xs - wx] = True
        vals, d = heal(img[win], cov[win] & ~(todo[win] & ~part), part, d, ring, reach)
        got.append(d)
        if vals is not None:
            img[ys, xs] = vals
            todo[ys, xs] = False
    return mask & ~todo, got


@dataclass
class Heal:
    x0: int = 0                               # plane px of mask[0, 0]
    y0: int = 0
    mask: np.ndarray | None = None            # bool; None: this patch changes nothing
    vals: np.ndarray | None = None            # (n, 3) float32: the healed pixels (``np.nonzero(mask)`` order)
    delta: np.ndarray | None = None           # (n, 3) float32: what they add to the unpatched picture
    ds: list | None = None                    # per stretch, plane px its copy came from, relative to it (None: nowhere)

    def meets(self, x0: int, y0: int, x1: int, y1: int) -> bool:
        h, w = self.mask.shape
        return self.x0 < x1 and self.x0 + w > x0 and self.y0 < y1 and self.y0 + h > y0


def _paste(img: np.ndarray, x0: int, y0: int, Y, X, V) -> None:
    """Healed pixels (plane y, x, values) into a full-size window at plane (x0, y0)."""
    ys, xs = Y - y0, X - x0
    ok = (ys >= 0) & (ys < img.shape[0]) & (xs >= 0) & (xs < img.shape[1])
    img[ys[ok], xs[ok]] = V[ok]


def _heal_one(mos, a: float, p, Q, r: float, ds: list, box, before: list[Heal]) -> Heal:
    x0, y0, x1, y1 = box
    img, cov = mos.render(Frame(1.0, (x0, y0), a, p), 0, 0, x1 - x0, y1 - y0)
    base = img.copy()
    for h in before:                                                      # in order: a later one covers an earlier one
        ys, xs = np.nonzero(h.mask)
        _paste(img, x0, y0, ys + h.y0, xs + h.x0, h.vals)
    healed, ds = heal_stroke(img, cov, Q, r, ds, x0, y0)
    if not healed.any():
        return Heal(ds=ds)
    ys, xs = np.nonzero(healed)
    ya, xa = ys.min(), xs.min()
    vals = img[ys, xs].astype(np.float32)
    return Heal(x0 + xa, y0 + ya, healed[ya:ys.max() + 1, xa:xs.max() + 1], vals, vals - base[ys, xs], ds)


def _exclusive(heals: list[Heal]):
    """All the heals as (plane y, plane x, values, delta) arrays, each pixel once: where patches overlap, the later."""
    parts = []
    for j, h in enumerate(heals):
        if h.mask is None:
            continue
        ys, xs = np.nonzero(h.mask)
        keep = np.ones(len(ys), bool)
        for g in heals[j + 1:]:
            if g.mask is None or not g.meets(h.x0, h.y0, h.x0 + h.mask.shape[1], h.y0 + h.mask.shape[0]):
                continue
            yy, xx = ys + h.y0 - g.y0, xs + h.x0 - g.x0
            inb = (yy >= 0) & (yy < g.mask.shape[0]) & (xx >= 0) & (xx < g.mask.shape[1])
            keep[inb] &= ~g.mask[yy[inb], xx[inb]]
        parts.append((ys[keep] + h.y0, xs[keep] + h.x0, h.vals[keep], h.delta[keep]))
    return tuple(np.concatenate(v) for v in zip(*parts)) if parts else EMPTY


class Retouch:
    """The heals of one set's patches, remembered between calls."""

    def __init__(self):
        self._one: dict = {}                  # patch id -> (key, Heal)
        self._all = (None, [], EMPTY)         # (key of everything, heals, their _exclusive arrays)
        self._lock = threading.Lock()

    def __call__(self, mos, geom: dict, specs: list[dict]):
        """([Heal] in the order of ``specs``, the arrays ``apply`` takes) for the straightened plane of ``geom``."""
        a, p = export.turn_of(geom)
        geo = [round(a, 12), round(float(p[0]), 6), round(float(p[1]), 6)]
        whole = json.dumps([geo, specs], sort_keys=True)
        with self._lock:
            if self._all[0] == whole:
                return self._all[1], self._all[2]
            plane = Frame(1.0, (0.0, 0.0), a, p)
            heals, keys, one = [], [], {}
            for spec in specs:
                Q = plane.to_output(np.asarray(spec["pts"], float).reshape(-1, 2))
                r, src = float(spec["r"]), spec.get("src") or []           # src: per stretch, mosaic px or None
                ds = [None if k >= len(src) or src[k] is None else tuple(int(v) for v in np.round(np.asarray(src[k], float) @ rot(a)))
                      for k in range(count(spec["pts"], r))]
                pad = ring_of(r) + 2 + max(reach_of(r) if d is None else max(abs(d[0]), abs(d[1])) for d in ds)
                lo, hi = np.floor(Q.min(0) - r).astype(int) - pad, np.ceil(Q.max(0) + r).astype(int) + 1 + pad
                box = (int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1]))
                deps = [i for i, h in enumerate(heals) if h.mask is not None and h.meets(*box)]
                key = hashlib.sha1(json.dumps([geo, spec, [keys[i] for i in deps]], sort_keys=True).encode()).hexdigest()
                hit = self._one.get(spec["id"])
                h = hit[1] if hit and hit[0] == key else _heal_one(mos, a, p, Q, r, ds, box, [heals[i] for i in deps])
                one[spec["id"]] = (key, h)
                heals.append(h)
                keys.append(key)
            self._one, self._all = one, (whole, heals, _exclusive(heals))
            return heals, self._all[2]


def apply(img: np.ndarray, origin, s: float, flat) -> None:
    """Put the patches (``flat`` from ``Retouch``) into a window of the plane drawn at scale ``s`` whose pixel
    (0, 0) is plane point ``origin`` (whole pixels).  Zoomed out, each full-size pixel's change is shared among the
    window pixels its footprint falls in."""
    Y, X, V, D = flat
    if not len(Y):
        return
    if abs(s - 1.0) < 1e-9:
        return _paste(img, int(round(origin[0])), int(round(origin[1])), Y, X, V)
    H, W = img.shape[:2]

    def axis(v, o):                           # footprint of each plane pixel, in window pixels: [u, u + s)
        u = (v - 0.5 - o) * s + 0.5
        i = np.floor(u).astype(np.int64)
        w0 = np.minimum(u + s, i + 1) - u
        return i, w0, s - w0

    iy, wy0, wy1 = axis(Y, origin[1])
    ix, wx0, wx1 = axis(X, origin[0])
    for oy, wy in ((0, wy0), (1, wy1)):
        for ox, wx in ((0, wx0), (1, wx1)):
            yy, xx, w = iy + oy, ix + ox, wy * wx
            ok = (w > 1e-9) & (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            if ok.any():
                np.add.at(img, (yy[ok], xx[ok]), D[ok] * w[ok, None])
    np.clip(img, 0.0, 1.0, out=img)
