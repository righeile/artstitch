"""The artwork's edges against the scanner lid: the angle that straightens it, and the crop inside it.

Flatbed tiles are usually scanned with the frame drawn just outside the paper, so
the lid shows only as a strip a few millimeters wide, and a color test on a
whole-mosaic preview cannot see it.  So the edges are measured where they are:
four bands along the sides of the mosaic are rendered at high resolution (1 px =
2 mosaic px, then 1 px = 8 for wide margins) and read row by row.

* the lid color of a side is the mode of its outermost covered pixels;
* going inwards, the paper starts where the color leaves the lid's for a few
  pixels in a row (sub-pixel interpolated); rows with no lid strip say nothing;
* a robust line per side; the four tilts give the angle that straightens the
  sheet, their disagreement the confidence (a hand-cut sheet is not a rectangle);
* no edges -> the picture's own straight lines, or nothing (then it is manual);
* the crop is the largest rectangle inside the covered area and inside the four
  edge lines (pulled in by ``inset``), then grown to touch them exactly.

Everything is in mosaic coordinates, so results drive ``Frame`` directly.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from geometry import Frame, rot

SIDES = ("left", "right", "top", "bottom")
PASSES = ((2, 200), (8, 250))                  # (mosaic px per band px, band depth in band px)
TAU = 0.07                                     # color distance that separates paper from lid


def tilt(direction: float) -> float:
    """Angle of a line's direction from its nearest axis, in (-pi/4, pi/4]: the rotation that levels it."""
    return ((direction + math.pi / 4) % (math.pi / 2)) - math.pi / 4


# -- bands ---------------------------------------------------------------------

def _orient(a, side):
    """Band array turned so that rows run along the side and columns inwards from it."""
    if side == "left":
        return a
    if side == "right":
        return a[:, ::-1]
    return a.swapaxes(0, 1) if side == "top" else a.swapaxes(0, 1)[:, ::-1]


def _band_xy(r, x, side, h, w):
    """Oriented (along, across) -> band (column, row)."""
    return {"left": (x, r), "right": (w - 1 - x, r), "top": (r, x), "bottom": (r, h - 1 - x)}[side]


def _mode(px: np.ndarray, q: float = 0.03) -> tuple[np.ndarray, float]:
    """Most common color (bins of ``q``) and its share."""
    k = np.round(px / q).astype(np.int64)
    u, inv, cnt = np.unique(k[:, 0] * 1_000_000 + k[:, 1] * 1_000 + k[:, 2], return_inverse=True, return_counts=True)
    j = int(cnt.argmax())
    return np.median(px[inv == j], axis=0), float(cnt[j] / len(px))


def _strip_ends(img: np.ndarray, cov: np.ndarray, tau: float = TAU, min_strip: int = 3, run: int = 4):
    """For an oriented band (paper to the right of a lid strip): per row, where the strip ends.

    Returns rows and sub-pixel columns of the paper edge, only for rows that do
    show a lid strip of at least ``min_strip`` pixels.
    """
    H, W = cov.shape
    c0 = cov.argmax(1)
    rows = np.nonzero(cov.any(1) & (c0 < W - 30))[0]
    if len(rows) < 50:
        return None
    ring = np.concatenate([img[rows, c0[rows] + 1], img[rows, c0[rows] + 2]])
    lid, _ = _mode(ring)
    d = np.sqrt(((img - lid) ** 2).mean(-1)).astype(np.float32)
    d[~cov] = 0
    d = cv2.GaussianBlur(d, (0, 0), 1.2)
    m = np.minimum.reduce([d[:, j:W - run + 1 + j] for j in range(run)])       # paper for `run` pixels in a row
    cand = ((m > tau) & (np.arange(m.shape[1])[None, :] >= c0[:, None] + 2))[rows]
    ok = cand.any(1)
    first = cand.argmax(1)[ok]
    rr = rows[ok]
    keep = first - c0[rr] >= min_strip
    rr, first = rr[keep], first[keep]
    d0, d1 = d[rr, first - 1], d[rr, first]
    return rr.astype(np.float32), first - 1 + np.clip((tau - d0) / np.maximum(d1 - d0, 1e-6), 0, 1)


def off_glass(img: np.ndarray, depth: int = 48, tau: float = TAU) -> set[str]:
    """The sides of one raw scan where the sheet runs off the glass, on a proxy of it.

    A side that shows the lid beyond the paper had the paper's own edge on the
    glass, flat and held down by the lid.  A side with no lid is a side the sheet
    carries on over: there the paper lifts off the glass and the picture bends
    with it, which is the strip worth cutting before anything is matched.
    """
    g = img.astype(np.float32) / (65535.0 if img.dtype == np.uint16 else 255.0)
    out = set()
    for s in SIDES:
        band = np.ascontiguousarray(_orient(g, s)[:, :depth])
        got = _strip_ends(band, np.ones(band.shape[:2], bool), tau)
        if got is not None and _fit_line(np.c_[got[1], got[0]].astype(np.float32), 2.0) is not None:
            continue                                        # the paper's own edge, with the lid beyond it: flat
        if _mode(band.reshape(-1, 3))[1] < 0.5:
            out.add(s)                                      # all picture to the edge: the sheet runs off here
        # else the band is one flat color through and through: lid deeper than the band, and the paper
        # further in still.  Nothing ran off this side, so nothing is cut.
    return out


def _fit_line(pts: np.ndarray, tol: float):
    """Robust line through points: (direction angle, point, inlier points, rms, nearby points) or None."""
    if len(pts) < 50:
        return None
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    for _ in range(3):
        d = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)
        inl = pts[d < tol]
        if len(inl) < 40:
            return None
        vx, vy, x0, y0 = cv2.fitLine(inl, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    d = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)
    ok = d < tol
    if ok.sum() < 40 or ok.sum() < 0.3 * len(pts):
        return None
    return math.atan2(vy, vx), (float(x0), float(y0)), pts[ok], float(np.sqrt(np.mean(d[ok] ** 2))), pts[d < 8 * tol]


def side_lines(render, ext, passes=PASSES, tau: float = TAU) -> dict:
    """Paper-edge line of every side that shows one: {side: {angle, p0, pts, rms, f, span}}."""
    ext0 = np.floor(ext[0])
    span = ext[1] - ext0
    found: dict = {}
    for f, depth in passes:
        if len(found) == 4:
            break
        W, H = int(math.ceil(span[0] / f)) + 1, int(math.ceil(span[1] / f)) + 1
        d = min(depth, W // 2, H // 2)
        fr = Frame(scale=1.0 / f, origin=ext0)
        win = {"left": (0, 0, d, H), "right": (W - d, 0, d, H), "top": (0, 0, W, d), "bottom": (0, H - d, W, d)}
        for s in SIDES:
            if s in found:
                continue
            x0, y0, w, h = win[s]
            img, cov = render(fr, x0, y0, w, h)
            got = _strip_ends(_orient(img, s), _orient(cov, s), tau)
            if got is None:
                continue
            bx, by = _band_xy(got[0], got[1], s, h, w)
            pts = np.c_[ext0[0] + f * (x0 + bx), ext0[1] + f * (y0 + by)].astype(np.float32)
            fit = _fit_line(pts, tol=2.0 * f)
            if fit is None:
                continue
            ang, p0, inl, rms, near = fit
            v = np.array([math.cos(ang), math.sin(ang)])
            found[s] = {"angle": ang, "p0": p0, "pts": inl, "near": near, "rms": rms, "f": f,
                        "span": float(np.ptp((inl - np.array(p0)) @ v))}
    return found


# -- no edges: the picture's own lines -----------------------------------------------

def lines_angle(img: np.ndarray, max_deg: float = 10.0) -> tuple[float, float]:
    """Dominant near-axis direction of the picture's straight lines: (radians, strength 0..1)."""
    g = (img @ np.array([0.299, 0.587, 0.114], np.float32) * 255).astype(np.uint8)
    g = cv2.GaussianBlur(g, (0, 0), 1.2)
    med = float(np.median(g))
    edges = cv2.Canny(g, max(10, 0.66 * med), max(30, 1.33 * med))
    n = min(g.shape)
    segs = cv2.HoughLinesP(edges, 1, math.pi / 1800, threshold=int(0.06 * n), minLineLength=int(0.12 * n),
                           maxLineGap=int(0.02 * n))
    if segs is None:
        return 0.0, 0.0
    segs = segs.reshape(-1, 4).astype(float)                     # OpenCV 4: (n, 1, 4); OpenCV 5: (n, 4)
    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    length = np.hypot(dx, dy)
    theta = np.degrees(np.arctan2(dy, dx))
    r = theta - 90.0 * np.round(theta / 90.0)                    # tilt from the nearest axis
    keep = np.abs(r) <= max_deg
    if keep.sum() < 3:
        return 0.0, 0.0
    bins = np.arange(-max_deg, max_deg + 0.05, 0.05)
    h, _ = np.histogram(r[keep], bins, weights=length[keep])
    k = int(np.argmax(np.convolve(h, np.hanning(9), mode="same")))
    near = keep & (np.abs(r - (bins[k] + 0.025)) < 0.15)
    ang = float(np.average(r[near], weights=length[near])) if near.any() else bins[k] + 0.025
    return math.radians(ang), float(length[near].sum() / max(length[keep].sum(), 1e-6))


# -- the crop ---------------------------------------------------------------------------

def largest_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Largest axis-aligned all-True rectangle: (x0, y0, x1, y1), exclusive ends."""
    h, w = mask.shape
    heights = np.zeros(w, np.int32)
    best = (0, 0, 0, 0, 0)
    for y in range(h):
        heights = np.where(mask[y], heights + 1, 0)
        stack: list[tuple[int, int]] = []                       # (start, height)
        for x in range(w + 1):
            cur = int(heights[x]) if x < w else 0
            start = x
            while stack and stack[-1][1] >= cur:
                s, hh = stack.pop()
                if hh * (x - s) > best[0]:
                    best = (hh * (x - s), s, y - hh + 1, x, y + 1)
                start = s
            stack.append((start, cur))
    return best[1:]


def crop_rect(cov: np.ndarray, f: int, ext0, sides: dict, alpha: float, pivot, inset: float = 12.0) -> list[float]:
    """Largest rectangle of the straightened plane that is covered and inside the edge lines.

    ``cov``: covered pixels of a 1 px = ``f`` mosaic px rendering whose pixel
    (0, 0) is at ``ext0``.  The straightened plane is the mosaic rotated by
    ``-alpha`` about ``pivot``.  Returns [x0, y0, x1, y1] there, in mosaic px.
    """
    ext0, pivot, R = np.asarray(ext0, float), np.asarray(pivot, float), rot(alpha)
    to_q = lambda P: pivot + (np.asarray(P, float) - pivot) @ R
    to_x = lambda Q: pivot + (np.asarray(Q, float) - pivot) @ R.T
    er = int(math.ceil(inset / f)) + 1
    ok = cv2.erode(cov.astype(np.uint8), np.ones((2 * er + 1, 2 * er + 1), np.uint8)) > 0
    H, W = ok.shape
    lim = {}                                                    # exact edge lines in the straightened plane: v = a + b u
    for s, e in sides.items():
        q = to_q(e["pts"])
        u, v = (q[:, 1], q[:, 0]) if s in ("left", "right") else (q[:, 0], q[:, 1])
        b, a = np.polyfit(u, v, 1)
        qn = to_q(e["near"])
        un, vn = (qn[:, 1], qn[:, 0]) if s in ("left", "right") else (qn[:, 0], qn[:, 1])
        inward = (vn - (a + b * un)) * (1 if s in ("left", "top") else -1)
        a += max(0.0, float(np.percentile(inward, 97))) * (1 if s in ("left", "top") else -1)
        lim[s] = (a, b)

    def inside(x0, y0, x1, y1) -> bool:
        """The rectangle keeps ``inset`` away from every edge line (linear, so its corners decide)."""
        for s, (a, b) in lim.items():
            lo, hi = (y0, y1) if s in ("left", "right") else (x0, x1)
            near = {"left": max, "top": max, "right": min, "bottom": min}[s]
            line = near(a + b * lo, a + b * hi)
            edge = {"left": x0, "right": x1, "top": y0, "bottom": y1}[s]
            if (edge - line) * (1 if s in ("left", "top") else -1) < inset:
                return False
        return True

    def covered(x0, y0, x1, y1) -> bool:
        t = np.arange(0.0, 1.0 + 1e-9, f / max(x1 - x0, y1 - y0, 1.0))
        z = np.zeros_like(t)
        b = np.concatenate([np.c_[x0 + t * (x1 - x0), z + y0], np.c_[x0 + t * (x1 - x0), z + y1],
                            np.c_[z + x0, y0 + t * (y1 - y0)], np.c_[z + x1, y0 + t * (y1 - y0)]])
        X = to_x(b)
        i = np.round((X[:, 0] - ext0[0]) / f).astype(int)
        j = np.round((X[:, 1] - ext0[1]) / f).astype(int)
        return bool(((i >= 0) & (i < W) & (j >= 0) & (j < H)).all() and ok[j, i].all())

    valid = lambda r: inside(*r) and covered(*r)
    corners = to_q(np.array([ext0, ext0 + [W * f, 0], ext0 + [0, H * f], ext0 + [W * f, H * f]]))
    q0, q1 = corners.min(0), corners.max(0)
    cs = max(float(f), math.ceil(float((q1 - q0).max()) / 800))
    gx, gy = int((q1[0] - q0[0]) / cs), int((q1[1] - q0[1]) / cs)
    Q = np.stack(np.meshgrid(q0[0] + (np.arange(gx) + 0.5) * cs, q0[1] + (np.arange(gy) + 0.5) * cs), -1)
    X = to_x(Q.reshape(-1, 2))
    i = np.round((X[:, 0] - ext0[0]) / f).astype(int)
    j = np.round((X[:, 1] - ext0[1]) / f).astype(int)
    inb = (i >= 0) & (i < W) & (j >= 0) & (j < H)
    cell = np.zeros(gx * gy, bool)
    cell[inb] = ok[j[inb], i[inb]]
    cell = cell.reshape(gy, gx)
    for s, (a, b) in lim.items():                                # half-planes, conservative by half a cell
        u, v = (Q[..., 1], Q[..., 0]) if s in ("left", "right") else (Q[..., 0], Q[..., 1])
        cell &= (v - (a + b * u)) * (1 if s in ("left", "top") else -1) >= inset + cs * (0.5 + abs(b) / 2)
    cx0, cy0, cx1, cy1 = largest_rectangle(cell)
    rect = [q0[0] + cx0 * cs, q0[1] + cy0 * cs, q0[0] + cx1 * cs, q0[1] + cy1 * cs]
    for _ in range(int(cs) + 2):                                 # the cells were only conservative at their centers
        if valid(rect):
            break
        rect = [rect[0] + 1, rect[1] + 1, rect[2] - 1, rect[3] - 1]
    for _ in range(int(2 * cs) + 4):                             # ... then grow each side until it touches something
        moved = False
        for k, step in ((0, -1), (1, -1), (2, 1), (3, 1)):
            r = list(rect)
            r[k] += step
            if valid(r):
                rect, moved = r, True
        if not moved:
            break
    return [float(v) for v in rect]


def detect(render, ext, inset: float = 12.0, angle_deg: float | None = None, preview_f: int = 8,
           sides: dict | None = None) -> dict:
    """Angle and crop of a stitched set.

    ``render(frame, x0, y0, w, h)`` -> (RGB float 0..1, covered) is ``Mosaic.render``;
    ``ext`` the layout extent.  ``angle_deg`` given: keep it and only find the crop.
    ``angle_deg`` in the result feeds ``Frame(angle=radians(..), pivot=..)``; ``crop`` is
    the window in the straightened plane (whole pixels), both in mosaic px.  ``sides`` reuses
    the edge lines of an earlier call (they come back under ``_sides``, which is numpy, not JSON).
    """
    ext0 = np.floor(ext[0])
    span = ext[1] - ext0
    sides = side_lines(render, ext) if sides is None else sides
    tilts = {s: tilt(e["angle"]) for s, e in sides.items()}
    W, H = int(math.ceil(span[0] / preview_f)) + 1, int(math.ceil(span[1] / preview_f)) + 1
    img, cov = render(Frame(scale=1.0 / preview_f, origin=ext0), 0, 0, W, H)
    strength, spread = 0.0, 0.0
    if tilts:
        method = "edges"
        alpha = float(np.average(list(tilts.values()), weights=[sides[s]["span"] for s in tilts]))
        spread = math.degrees(max(tilts.values()) - min(tilts.values()))
        conf = "high" if len(tilts) >= 3 and spread < 0.15 else "medium" if len(tilts) >= 2 and spread < 0.6 else "low"
    else:
        alpha, strength = lines_angle(img)
        method = "lines" if strength > 0.25 else "none"
        alpha = alpha if method == "lines" else 0.0
        conf = "medium" if strength > 0.5 else "low"
    if angle_deg is not None:
        alpha, method, conf = math.radians(angle_deg), "manual", "manual"
    pivot = (ext0 + span / 2).tolist()
    x0, y0, x1, y1 = crop_rect(cov, preview_f, ext0, sides, alpha, pivot, inset)
    rect = [math.ceil(x0), math.ceil(y0), math.floor(x1), math.floor(y1)]
    return {"angle_deg": math.degrees(alpha), "method": method, "confidence": conf, "spread_deg": spread,
            "line_strength": strength, "pivot": pivot, "crop": rect, "_sides": sides,
            "sides": {s: {"found": s in sides, **({"tilt_deg": math.degrees(tilts[s]), "rms_px": sides[s]["rms"],
                                                   "inliers": len(sides[s]["pts"]), "span_px": sides[s]["span"],
                                                   "f": sides[s]["f"]} if s in sides else {})} for s in SIDES}}
