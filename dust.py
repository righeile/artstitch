"""Dust: specks, lint and hairs that lay on the scanner glass, found in the result so they can be patched.

Dust is small and stands out from what is around it: brighter than anything a disk of ``D`` px fits into (a
white top-hat), or darker (a black one).  So do the grain of the paper and every dab of pastel; what tells them
apart is by how much.  A speck is scored by its top-hat over what the top-hat typically is in the picture around
it (the median over 5 x 5 cells of 32 px of each cell's mean), so the same speck scores high on smooth paint and
low in coarse grain, as it looks.  Specks scoring from ``KEEP`` are kept, strongest first; which of them to patch
is the person's to choose, a picture of each in hand.

A tiny hair is too faint for that: no pixel of it stands out further than the grain, and a watercolor's own
specks of paint outscore it.  Along its length it does stand out, so hairs are found apart, by a line filter, and
listed first.

Everything is found in the result as it is patched so far, at full size, strip by strip: the picture is never
held whole.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

import export
import patch

D = 15                                        # px: dust narrower than this is found (a third of a mm at 1200 dpi)
KEEP = 10.0                                   # score a speck needs to be kept at all (below, thousands of grains)
LEVELS = 10                                   # and levels of 255 it must stand out by
REACH = 5.0, 6                                # score and levels down to which a speck's pixels count as its own:
                                              # at 3, 4 a speck on pastel spreads into the grain, twice as wide
CELL = 32                                     # px: the texture of the picture is measured in cells this size
ROWS = 32 * CELL                              # strip height
MARGIN = 3 * CELL                             # rows drawn above and below a strip: the texture median sees them
THUMB = 48                                    # px of picture round each speck, for the page to show
COLS = 32                                     # of them in a row of the sheet
MOST = 2000                                   # specks kept at most, the strongest
BIG = 4000                                    # px: larger than this is not dust but a mark on the picture
LARGE = 10.0                                  # px: a dot (not a line) of a larger radius starts left out: of 21
                                              # from score 12 on a pastel and a watercolor, 18 were paint
HAIR = 7                                      # px: hairs are found narrower than this (1 to 3 px at 1200 dpi)
ALONG = 3.0                                   # a hair's score: its line filter over the grain's, times this (the
                                              # grain, averaged along a line, stands out a third as far as a pixel)
HAIR_REACH = 2.5                              # line filter over the grain's down to which the line is the hair's
HAIR_KEEP = 8.0                               # score a hair needs to be kept at all: on a watercolor, many from 8 to
                                              # 10 were hairs; under 8 too little of one reaches HAIR_REACH to trace
                                              # (at a reach of 1.7: 2000 more, mostly grain, and strong ones merged)
THIN = 3.5                                    # px: a hair is no wider than this at half its depth; painted lines
                                              # are (on a watercolor, of those 4 px wide nearly all were paint)
SIDES = 0.5                                   # its two sides differ by no more than this of its depth, in color: a
                                              # line along the edge of a wash divides two paints, a hair lies on one
LINE = 0.4                                    # the line filter's peak over the top-hat's: a hair fills this much of
                                              # the filter's 26 px, a dot only its diameter (flakes on a pastel 0.2-0.35)
SHORT = 12.0                                  # px: a hair shorter than this is a speck
ANGLES = 12
ACROSS = np.arange(-8.0, 8.01, 0.5)           # px either side of a hair where its profile is read
LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def typical(th: np.ndarray, cell: int = CELL, floor: float = 1.0) -> np.ndarray:
    """What the top-hat ``th`` typically is around each pixel, in levels, plus ``floor``."""
    h, w = th.shape
    c = cv2.resize(th, (max(1, round(w / cell)), max(1, round(h / cell))), interpolation=cv2.INTER_AREA)
    c = cv2.medianBlur(c.astype(np.float32), 5) if min(c.shape) >= 3 else c.astype(np.float32)
    return cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR) + floor


def _lines():
    """Line filters for a half-size picture at ANGLES angles: 7 px along a line, then that 3 px either side of the
    center averaged, so 13 px of line (26 of the picture) for two 7 x 7 filters; one of 9 x 9 or more costs 20 times."""
    along, ends = [], []
    for a in range(ANGLES):
        c, s = math.cos(math.pi * a / ANGLES), math.sin(math.pi * a / ANGLES)
        k = np.zeros((7, 7), np.float32)
        cv2.line(k, (round((3 - 3 * c) * 16), round((3 - 3 * s) * 16)), (round((3 + 3 * c) * 16), round((3 + 3 * s) * 16)),
                 1.0, 1, cv2.LINE_AA, 4)
        along.append(k / k.sum())
        k = np.zeros((7, 7), np.float32)
        for x, y in ((3 - 3 * c, 3 - 3 * s), (3 + 3 * c, 3 + 3 * s)):
            x0, y0 = min(int(x), 5), min(int(y), 5)
            fx, fy = x - x0, y - y0
            k[y0:y0 + 2, x0:x0 + 2] += np.array([[(1 - fx) * (1 - fy), fx * (1 - fy)], [(1 - fx) * fy, fx * fy]], np.float32)
        ends.append(k / k.sum())
    return list(zip(along, ends))


LINES = _lines()


def spine(mask: np.ndarray, cost: np.ndarray):
    """A path between the two ends of a connected ``mask`` (the pixels furthest apart through it), the cheapest when
    a step costs its length times ``cost`` (per pixel): (ys, xs) in order along it.  Where every step costs its
    length, a path along a wide band may hug one edge of it; a cost falling to the middle keeps it there.  The ends
    are found by length alone: by cost, the costly rim is furthest, and the path would hook off into it."""
    ys, xs = np.nonzero(mask)
    n = len(ys)
    idx = np.full(mask.shape, -1)
    idx[ys, xs] = np.arange(n)
    c = cost[ys, xs]
    a, b, d, e = [], [], [], []
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        y2, x2 = ys + dy, xs + dx
        ok = (y2 < mask.shape[0]) & (x2 >= 0) & (x2 < mask.shape[1])
        j = np.full(n, -1)
        j[ok] = idx[y2[ok], x2[ok]]
        k = np.nonzero(j >= 0)[0]
        a.append(k), b.append(j[k]), d.append(np.full(len(k), math.hypot(dy, dx))), e.append(d[-1] * (c[k] + c[j[k]]) / 2)
    a, b = np.concatenate(a), np.concatenate(b)
    G = coo_matrix((np.concatenate(d), (a, b)), shape=(n, n)).tocsr()
    s = int(np.argmax(dijkstra(G, directed=False, indices=0)))
    t = int(np.argmax(dijkstra(G, directed=False, indices=s)))
    _, pred = dijkstra(coo_matrix((np.concatenate(e), (a, b)), shape=(n, n)).tocsr(), directed=False, indices=s,
                       return_predecessors=True)
    path = [t]
    while path[-1] != s:
        path.append(int(pred[path[-1]]))
    return ys[path], xs[path]


def dense(P: np.ndarray) -> np.ndarray:
    """The polyline ``P`` with a point every pixel or less."""
    n = np.maximum(np.ceil(np.hypot(*np.diff(P, axis=0).T)), 1).astype(int)
    return np.vstack([a + (b - a) * np.arange(k)[:, None] / k for a, b, k in zip(P[:-1], P[1:], n)] + [P[-1:]])


def across(img: np.ndarray, Q: np.ndarray, sign: float = 1.0) -> tuple[float, float]:
    """(width at half depth, how differently its two sides look over its depth) of the line ``Q`` (points a pixel
    apart) in ``img`` (float RGB 0..1), a line lighter than round it (``sign`` -1: darker).  The sides are read 5 to
    8 px out, where no other part of the line is nearer: the other arm of a curl is not a side, and a tight curl's
    inside is none.  They are told apart in color: a wash's edge can be as light as the next wash."""
    d = np.gradient(Q, axis=0)
    d /= np.hypot(*d.T)[:, None] + 1e-9
    X, Y = Q[:, :1] - ACROSS * d[:, 1:], Q[:, 1:] + ACROSS * d[:, :1]
    c = cv2.remap(img, X.astype(np.float32), Y.astype(np.float32), cv2.INTER_LINEAR) * 255.0
    p = sign * (c @ LUMA)
    mid = len(ACROSS) // 2
    k = np.arange(len(ACROSS)) + np.argmax(p[:, mid - 4:mid + 5], 1)[:, None] - 4     # each centered on the line
    m = np.median(np.take_along_axis(p, np.clip(k, 0, mid * 2), 1), 0)
    near = cKDTree(Q).query(np.c_[X.ravel(), Y.ravel()])[0].reshape(X.shape)
    free = (np.abs(ACROSS) >= 5) & (near >= np.abs(ACROSS) - 1)
    left, right = free & (ACROSS < 0), free & (ACROSS > 0)
    if not (left.any() or right.any()):
        return math.inf, math.inf
    base = float(np.median(p[left | right]))
    depth = max(float(m[mid]) - base, 1.0)
    lo = hi = mid
    while lo > 0 and m[lo - 1] >= base + depth / 2:
        lo -= 1
    while hi < 2 * mid and m[hi + 1] >= base + depth / 2:
        hi += 1
    if not (left.any() and right.any()):
        return (hi - lo + 1) * 0.5, 0.0
    diff = np.median(c[left], 0) - np.median(c[right], 0)
    return (hi - lo + 1) * 0.5, float(np.sqrt(np.mean(diff ** 2))) / depth


def hairs(img: np.ndarray, g: np.ndarray, cov: np.ndarray, rows: slice | None = None, top: bool = True,
          bottom: bool = True):
    """Hairs in a strip (float RGB 0..1, ``g`` its luma, uint8): [(item, y, x)] as ``specks``, and the mask of their
    pixels.  The line filter runs on the top-hat at half size; a hair is followed along the middle of what it finds,
    cut to where the hair's own pixels are, and kept if long and thin with the same paint either side: a dot or
    shorter is a speck, wider or between two paints was painted."""
    h, w = g.shape
    h2, w2 = h // 2, w // 2
    c2 = cv2.resize(cov.astype(np.uint8), (w2, h2), interpolation=cv2.INTER_NEAREST)
    near = cv2.erode(c2, np.ones((3, 3), np.uint8)) == 0                   # the paper's edge inside the strip
    K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (HAIR, HAIR))
    out, taken = [], np.zeros((h, w), np.uint8)
    for kind, op in (("light", cv2.MORPH_TOPHAT), ("dark", cv2.MORPH_BLACKHAT)):
        th = cv2.morphologyEx(g, op, K)
        t2 = cv2.resize(th[:2 * h2, :2 * w2].astype(np.float32), (w2, h2), interpolation=cv2.INTER_AREA)
        R = np.max([cv2.filter2D(cv2.filter2D(t2, -1, a), -1, b) for a, b in LINES], axis=0)
        z = R / typical(R, CELL // 2, 0.5)
        lab, n = ndimage.label((z >= HAIR_REACH) & (c2 > 0), np.ones((3, 3), bool))
        if not n:
            continue
        on = lab > 0
        peak = np.zeros(n, np.float32)
        np.maximum.at(peak, lab[on] - 1, z[on])
        boxes, typ = ndimage.find_objects(lab), None
        for i in np.nonzero(peak * ALONG >= HAIR_KEEP)[0]:
            sl = boxes[i]
            if sl[1].start == 0 or sl[1].stop == w2 or (top and sl[0].start == 0) or (bottom and sl[0].stop == h2):
                continue
            comp = lab[sl] == i + 1
            if (comp & near[sl]).any() or R[sl][comp].max() < LINE * t2[sl][comp].max():
                continue                                                    # the paper's edge, or a dot
            if typ is None:
                typ = typical(th)
            win = slice(2 * sl[0].start, 2 * sl[0].stop), slice(2 * sl[1].start, 2 * sl[1].stop)
            own = cv2.dilate(np.repeat(np.repeat(comp, 2, 0), 2, 1).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            own &= th[win] >= HAIR_REACH * typ[win]                         # the hair's own pixels, near its line
            ys, xs = spine(comp, 1.0 / z[sl] ** 2)                         # along the ridge of the line filter
            P = ndimage.uniform_filter1d(np.c_[2 * xs + 0.5, 2 * ys + 0.5], 5, axis=0, mode="nearest")
            parts, _ = ndimage.label(own, np.ones((3, 3), bool))
            on = parts[P[:, 1].astype(int), P[:, 0].astype(int)]
            own = np.isin(parts, on[on > 0])                                # and not the grain's beside it
            qy, qx = np.nonzero(own)
            if len(qy) < SHORT:
                continue
            Q = np.c_[qx, qy].astype(float)
            k = np.argmin(((Q[:, None] - P[None]) ** 2).sum(2), 1)          # the spine, cut to the hair's ends
            P = P[k.min():k.max() + 1]
            span = float(np.hypot(*np.diff(P, axis=0).T).sum()) if len(P) > 1 else 0.0
            cy = float(qy.mean()) + win[0].start
            if span < SHORT or (rows is not None and not rows.start <= cy < rows.stop):
                continue
            pts = cv2.approxPolyDP(P.astype(np.float32)[:, None], 0.75, False)[:, 0].astype(float)
            r = float(polyline_distance(Q, pts).max()) + 2.0
            fw, sides = across(img, dense(P) + (win[1].start, win[0].start), 1.0 if kind == "light" else -1.0)
            if fw > THIN or sides > SIDES:
                continue
            taken[win] |= cv2.dilate(own.astype(np.uint8), np.ones((5, 5), np.uint8))
            pts += (win[1].start, win[0].start)
            out.append(({"kind": kind, "shape": "hair", "score": round(float(peak[i]) * ALONG, 1),
                         "levels": int(th[win][own].max()), "area": len(qy), "pts": np.round(pts, 2).tolist(),
                         "r": round(r, 2)}, cy, float(qx.mean()) + win[1].start))
    return out, taken > 0


def polyline_distance(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Distance of each point of ``P`` (n, 2) to the polyline ``Q`` (m, 2)."""
    if len(Q) == 1:
        return np.hypot(*(P - Q[0]).T)
    best = np.full(len(P), np.inf)
    for a, b in zip(Q[:-1], Q[1:]):
        ab = b - a
        t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        best = np.minimum(best, np.hypot(*(P - a - t[:, None] * ab).T))
    return best


def stroke(xs: np.ndarray, ys: np.ndarray) -> tuple[list, float]:
    """A patch stroke (points, radius) over a speck's pixels: a dot, or a line along a fiber or a hair."""
    P = np.c_[xs, ys].astype(float)
    c = P.mean(0)
    Q = c[None]
    if len(P) >= 6:
        vals, vecs = np.linalg.eigh(np.cov((P - c).T))
        t = (P - c) @ vecs[:, 1]                                          # along the longest axis
        span, width = t.max() - t.min(), 4 * math.sqrt(max(vals[0], 0.0)) + 1
        if span > 2.5 * max(width, 3.0):                                  # a fiber: follow it, bent or not
            n = int(np.clip(span / max(width, 4.0), 2, 24))
            k = np.minimum(((t - t.min()) / (span + 1e-9) * n).astype(int), n - 1)
            Q = np.array([P[k == j].mean(0) for j in range(n) if (k == j).any()])
            ends = [P[np.argmin(t)], P[np.argmax(t)]]
            Q = np.vstack([ends[0], Q, ends[1]])
    r = float(polyline_distance(P, Q).max()) + 2.0                        # the pixels, and the halo round them
    return np.round(Q, 2).tolist(), round(r, 2)


def found_in(img: np.ndarray, cov: np.ndarray, rows: slice | None = None, top: bool = True, bottom: bool = True):
    """Hairs, then the specks that are not part of one, in a strip (float RGB 0..1): see ``specks``."""
    g = np.clip(img @ LUMA * 255.0 + 0.5, 0, 255).astype(np.uint8)
    got, taken = hairs(img, g, cov, rows, top, bottom)
    return got + specks(g, cov, rows, top, bottom, taken)


def specks(g: np.ndarray, cov: np.ndarray, rows: slice | None = None, top: bool = True, bottom: bool = True,
           taken: np.ndarray | None = None):
    """Specks in a strip (``g`` its luma, uint8): [(item, y, x)] with the item's points in strip px, those whose
    center falls in ``rows`` only, none on the ``taken`` pixels.  Nothing touching the picture's edge (the strip's
    sides, and its ``top`` and ``bottom`` when those are the picture's) is dust: it is the paper's edge, or a speck
    cut in half that shows as a line."""
    h, w = cov.shape
    K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (D, D))
    out = []
    for kind, th in (("light", cv2.morphologyEx(g, cv2.MORPH_TOPHAT, K)), ("dark", cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, K))):
        z = th / typical(th)
        lab, n = ndimage.label((z >= REACH[0]) & (th >= REACH[1]) & cov, np.ones((3, 3), bool))
        if not n:
            continue
        on = lab > 0                                                      # ndimage.maximum sorts the whole strip
        L = lab[on] - 1
        peak, lev = np.zeros(n, np.float32), np.zeros(n, np.uint8)
        np.maximum.at(peak, L, z[on])
        np.maximum.at(lev, L, th[on])
        boxes = ndimage.find_objects(lab)
        for i in np.nonzero((peak >= KEEP) & (lev >= LEVELS))[0]:
            sl = boxes[i]
            if sl[1].start == 0 or sl[1].stop == w or (top and sl[0].start == 0) or (bottom and sl[0].stop == h):
                continue
            yy, xx = np.nonzero(lab[sl] == i + 1)
            if len(yy) > BIG:
                continue
            ys, xs = yy + sl[0].start, xx + sl[1].start
            cy = float(ys.mean())
            if (rows is not None and not rows.start <= cy < rows.stop) or (taken is not None and taken[ys, xs].any()):
                continue                                                  # found again in the strip it is centered in
            pts, r = stroke(xs, ys)
            out.append(({"kind": kind, "shape": "speck", "score": round(float(peak[i]), 1), "levels": int(lev[i]),
                         "area": len(ys), "pts": pts, "r": r}, cy, float(xs.mean())))
    return out


def find(mos, geom: dict, retouch=None, progress=None, cancel=None):
    """Dust in the result of ``geom``: (items, hairs first, each strongest first, with points in result px and an
    ``i`` that is their place in the sheet; the sheet: THUMB-px pictures of them in rows of COLS, uint8 RGB,
    unadjusted).  ``retouch(y0, img)`` puts the patches so far into a strip."""
    W, H = export.size_of(geom)
    fr = export.frame_of(geom)
    got, thumbs = [], []
    h2 = THUMB // 2
    for y0 in range(0, H, ROWS):
        if cancel and cancel():
            raise export.Cancelled()
        if progress:
            progress(y0 / H, "looking for dust")
        ya, yb = max(y0 - MARGIN, 0), min(y0 + ROWS + MARGIN, H)
        img, cov = mos.render(fr, 0, ya, W, yb - ya)
        if retouch:
            retouch(ya, img)
        for item, cy, cx in found_in(img, cov, slice(y0 - ya, y0 - ya + ROWS), ya == 0, yb == H):
            item["pts"] = [[x, y + ya] for x, y in item["pts"]]
            y, x = int(round(cy)), int(round(cx))
            t = np.zeros((THUMB, THUMB, 3), np.float32)
            a, b, c, d = max(y - h2, 0), min(y + h2, img.shape[0]), max(x - h2, 0), min(x + h2, W)
            t[a - (y - h2):b - (y - h2), c - (x - h2):d - (x - h2)] = img[a:b, c:d]
            got.append(item)
            thumbs.append(export.q8(t))
    order = sorted(range(len(got)), key=lambda k: (got[k]["shape"] != "hair", -got[k]["score"]))[:MOST]
    items = [{**got[k], "i": j} for j, k in enumerate(order)]
    sheet = np.zeros((max(1, math.ceil(len(items) / COLS)) * THUMB, COLS * THUMB, 3), np.uint8)
    for j, k in enumerate(order):
        r, c = divmod(j, COLS)
        sheet[r * THUMB:(r + 1) * THUMB, c * THUMB:(c + 1) * THUMB] = thumbs[k]
    if progress:
        progress(1.0, f"{len(items)} specks")
    return items, sheet
