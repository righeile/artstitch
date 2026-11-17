"""Photographs of a flat artwork, put back into one plane.

A scan is the plane already.  A photograph is the plane seen from somewhere, and any two
photographs of a flat thing differ by a homography -- exactly, whatever the camera did,
because the subject is flat.  So the one thing a camera set needs that a scan set does not
is to be flattened first; after that it is a set of scans, and the coarse search, the mesh,
the seams, the patches and the export all work on it unchanged.

The plane is one of the photographs' own -- the best connected one.  Nothing here knows the
camera, so nothing here can know which way the wall faced: if that photograph was taken at an
angle, the finished picture keeps that angle.  Hold the camera square for one shot and the
whole set comes out square.

Each flattened photograph is cut down to the largest upright rectangle inside the warped
frame, so that no tile carries any "nothing here" corners -- a mask only this stage would
know about, which nothing downstream reads.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image

MIN_INLIERS = 25          # matches that survive RANSAC before a pair is believed to be the same wall
WORK_PX = 1500            # long side the features are found on: more is slower and finds no more pairs
MAX_STRETCH = 3.0         # a homography that blows a photograph up by more than this is a bad fit, not a steep angle


def _quiet(p: float, m: str) -> None:
    pass


def _features(gray: np.ndarray):
    sift = cv2.SIFT_create(nfeatures=2000)
    return sift.detectAndCompute(np.clip(gray, 0, 255).astype(np.uint8), None)


def _pair(ka, da, kb, db) -> tuple[np.ndarray | None, int]:
    """Homography taking photo b's pixels to photo a's, and how many matches agreed."""
    if da is None or db is None or len(da) < 4 or len(db) < 4:
        return None, 0
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(db, da, k=2)
    good = [p[0] for p in m if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < MIN_INLIERS:
        return None, 0
    src = np.float32([kb[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
    dst = np.float32([ka[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 3.0, maxIters=20000, confidence=0.999)
    n = 0 if inl is None else int(inl.sum())
    return (H, n) if H is not None and n >= MIN_INLIERS else (None, 0)


def _tree(n: int, w: np.ndarray, root: int) -> list[int]:
    """Parent of every photo in the spanning tree that keeps the strongest pair at each step (-1 at the root)."""
    par = [-1] * n
    seen = {root}
    while len(seen) < n:
        best = max(((w[a, b], a, b) for a in seen for b in range(n) if b not in seen), default=(0, -1, -1))
        if best[0] <= 0:
            break
        par[best[2]] = best[1]
        seen.add(best[2])
    return par if len(seen) == n else [-2 if k not in seen else par[k] for k in range(n)]


def _inscribed(quad: np.ndarray) -> tuple[int, int, int, int]:
    """The largest upright rectangle inside a convex quadrilateral, as (x0, y0, x1, y1).

    A vertical line crosses a convex shape twice, so the quadrilateral is just a top and a bottom
    over a range of x, and the best rectangle between two of 257 columns is found by running those
    two in.  What it costs against the true optimum is a column's width; what it saves is every
    "nothing here" corner, which no later stage would know to leave alone.
    """
    lo, hi = quad.min(0), quad.max(0)
    xs = np.linspace(lo[0], hi[0], 257)
    cuts = np.full((4, len(xs)), np.nan)
    for i, ((ax, ay), (bx, by)) in enumerate(zip(quad, np.roll(quad, -1, 0))):
        if ax == bx:
            continue
        t = (xs - ax) / (bx - ax)
        cuts[i] = np.where((t >= 0) & (t <= 1), ay + t * (by - ay), np.nan)
    with np.errstate(invalid="ignore"):
        top = np.where(np.isnan(cuts).all(0), np.inf, np.nanmin(cuts, 0))
        bot = np.where(np.isnan(cuts).all(0), -np.inf, np.nanmax(cuts, 0))
    best = (0.0, lo[0], lo[1], lo[0] + 1, lo[1] + 1)
    for j in range(len(xs) - 1):
        y0 = np.maximum.accumulate(top[j:])                  # the lowest top, and the highest bottom,
        y1 = np.minimum.accumulate(bot[j:])                  # over every column the rectangle spans
        area = np.where(y1 > y0, (y1 - y0) * (xs[j:] - xs[j]), 0.0)
        i = int(area.argmax())
        if area[i] > best[0]:
            best = (area[i], xs[j], y0[i], xs[j + i], y1[i])
    _, x0, y0, x1, y1 = best
    return int(np.ceil(x0)), int(np.ceil(y0)), int(np.floor(x1)), int(np.floor(y1))


def focal_px(path, width: int) -> float | None:
    """The focal length of a photograph in its own pixels, from what the camera wrote in the file.

    A rectangle shot with the camera level and tilted only up or down -- a tripod squared to the wall,
    which is how anyone careful does it -- is a symmetric trapezoid, and a symmetric trapezoid is the
    one case the corners cannot be read alone: a tall rectangle tilted far looks exactly like a short
    one tilted little.  The focal length settles it, and the camera already knows it.
    """
    try:
        with Image.open(path) as im:
            ex = im.getexif()
        eq35 = ex.get(41989)                                     # FocalLengthIn35mmFilm, whole millimetres
        if eq35:
            return float(eq35) / 36.0 * width                    # 36 mm is the long side of the 35 mm frame
    except Exception:                                            # a file with no camera behind it, or none that said
        pass
    return None


def rectangle(quad: np.ndarray, centre, f: float | None = None, ratio: float | None = None) -> tuple[float, float, str]:
    """How wide and how tall the rectangle a camera saw as ``quad`` really is, in the same pixels.

    A rectangle photographed by a camera with square pixels and no skew tells you its own proportions
    once the focal length is known, and if the camera was turned about both axes it tells you the
    focal length too -- Zhang and He's whiteboard trick, where the two vanishing points of the sides
    are conjugate.  Turned about one axis only there are no two vanishing points to use, and without
    ``f`` from the file there is nothing left but to average the opposite sides, which is right for a
    parallelogram and about three per cent out at a tilt of fifteen degrees.

    ``quad`` is the corners, top-left, top-right, bottom-left, bottom-right.  ``ratio`` is the
    artwork measured instead of guessed, which settles it outright.  Returns (w, h, how): whatever
    the proportions, the rectangle is the size that keeps the area the quadrilateral had.

    How well each route does, measured on eight real photographs of a chessboard (OpenCV's ``left*.jpg``,
    640 x 480, a lens bent enough that k1 = -0.25) whose 8 x 5 grid of inner corners is a true 1.6:1:
    with ``f`` from the file, within 3 % and usually within 1 %, the lens bend left in;  from the corners
    alone, typically 3 % but once 25 %;  from the sides, 2 % to 25 %, worse than the corners more often
    than not.  So: the size typed in beats the file, the file beats the corners, and the last two lines
    of defence are both only worth a glance -- which is why the caller is told which one answered.
    """
    m = np.c_[np.asarray(quad, float) - np.asarray(centre, float), np.ones(4)]
    m1, m2, m3, m4 = m
    cross2 = lambda a, b: float(a[0] * b[1] - a[1] * b[0])
    side = lambda a, b: float(np.hypot(*(a - b)[:2]))
    area = 0.5 * abs(cross2(m3 - m2, m4 - m1))                            # the quadrilateral's own area
    flat = (0.5 * (side(m1, m2) + side(m3, m4)), 0.5 * (side(m1, m3) + side(m2, m4)), "sides")
    if ratio:
        return math.sqrt(area * ratio), math.sqrt(area / ratio), "measured"
    d2, d3 = np.cross(m2, m4) @ m3, np.cross(m3, m4) @ m2
    if abs(d2) < 1e-9 or abs(d3) < 1e-9:
        return flat
    n2 = (np.cross(m1, m4) @ m3) / d2 * m2 - m1
    n3 = (np.cross(m1, m4) @ m2) / d3 * m3 - m1
    how = "camera"
    if f is not None and f > 0:
        f2 = float(f) ** 2
    elif abs(n2[2] * n3[2]) < 1e-9:                      # one vanishing point at infinity: f is not in the corners
        return flat
    else:
        f2, how = -(n2[0] * n3[0] + n2[1] * n3[1]) / (n2[2] * n3[2]), "corners"
        if not (1.0 < f2 < 1e10):                        # shot square on, or four corners that were no rectangle
            return flat
    r2 = ((n2[0] ** 2 + n2[1] ** 2) / f2 + n2[2] ** 2) / ((n3[0] ** 2 + n3[1] ** 2) / f2 + n3[2] ** 2)
    if not (4e-4 < r2 < 2.5e3):
        return flat
    ratio = math.sqrt(r2)
    return math.sqrt(area * ratio), math.sqrt(area / ratio), how


def homographies(store, paths, progress=_quiet) -> tuple[list[np.ndarray], dict]:
    """One homography per photograph, taking its own pixels into the best-connected one's plane."""
    n = len(paths)
    if n == 1:
        return [np.eye(3)], {"reference": 0, "inliers": {}}
    shape = store.get(paths[0]).shape
    f = max(1, int(round(max(shape) / WORK_PX)))
    feats = []
    for k, p in enumerate(paths):
        progress(0.1 + 0.3 * k / n, f"looking at photo {k + 1}")
        feats.append(_features(store.proxy(p, f, gray=True)))
    w = np.zeros((n, n))
    H = {}
    for a in range(n):
        for b in range(a + 1, n):
            progress(0.4 + 0.4 * (a * n + b) / (n * n), f"matching {a + 1} and {b + 1}")
            Hab, inl = _pair(*feats[a], *feats[b])
            if Hab is not None:
                w[a, b] = w[b, a] = inl
                H[a, b], H[b, a] = Hab, np.linalg.inv(Hab)
    root = int(w.sum(1).argmax())
    par = _tree(n, w, root)
    if -2 in par:
        bad = [str(k + 1) for k, v in enumerate(par) if v == -2]
        raise ValueError("no overlap found with photo " + ", ".join(bad) + " -- the photos must overlap each other")
    # proxy pixel i covers source [f i, f i + f), so u_src = f u_proxy + (f - 1) / 2
    A = np.array([[1.0 / f, 0, -(f - 1) / (2.0 * f)], [0, 1.0 / f, -(f - 1) / (2.0 * f)], [0, 0, 1.0]])
    Ai = np.linalg.inv(A)
    out = [None] * n
    out[root] = np.eye(3)
    for k in sorted(range(n), key=lambda k: _depth(par, k)):
        if out[k] is None:
            out[k] = out[par[k]] @ H[par[k], k]
    return [Ai @ (M / M[2, 2]) @ A for M in out], {"reference": root, "inliers": {f"{a + 1}-{b + 1}": int(w[a, b])
                                                                                 for a in range(n) for b in range(a + 1, n) if w[a, b]}}


def _depth(par: list[int], k: int) -> int:
    d = 0
    while par[k] != -1:
        k, d = par[k], d + 1
    return d


def flatten(store, paths, out_dir, progress=_quiet, Hs=None) -> tuple[list[str], dict]:
    """Write a flattened copy of every photograph and return the new paths.

    ``Hs`` are homographies worked out before -- how the set is put back onto a plane chosen by hand
    rather than onto a photograph's own (see ``rectangle``); the report carries them either way.
    """
    report: dict = {"reference": 0, "inliers": {}}
    if Hs is None:
        Hs, report = homographies(store, paths, progress)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    new, kept, origin = [], [], []
    for k, (p, H) in enumerate(zip(paths, Hs)):
        progress(0.8 + 0.2 * k / len(paths), f"flattening photo {k + 1}")
        img = store.get(p).rgb
        h, w = img.shape[:2]
        box = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
        quad = cv2.perspectiveTransform(box, H).reshape(4, 2)
        side = np.hypot(*(np.roll(quad, -1, 0) - quad).T)
        if side.max() > MAX_STRETCH * max(w, h):
            raise ValueError(f"photo {k + 1} does not look like the same flat thing as the others")
        x0, y0, x1, y1 = _inscribed(quad)
        x0, y0, x1, y1 = x0 + 2, y0 + 2, x1 - 2, y1 - 2        # cubic reads two pixels out: stay that far inside the frame
        T = np.array([[1.0, 0, -x0], [0, 1.0, -y0], [0, 0, 1.0]])
        flat = cv2.warpPerspective(img, T @ H, (x1 - x0 + 1, y1 - y0 + 1), flags=cv2.INTER_CUBIC)
        kept.append(round((x1 - x0 + 1) * (y1 - y0 + 1) / float(w * h), 3))
        origin.append([x0, y0])
        q = out / (Path(p).stem + ".tif")
        tifffile.imwrite(q, flat, compression="zlib", photometric="rgb")
        store.drop(q)                                   # flattened again onto another plane: forget the old pixels
        new.append(str(q))
    report["kept"] = kept
    report["h"] = [H.tolist() for H in Hs]          # photograph pixel -> the plane, before the cut
    report["origin"] = origin                       # where the cut starts in it: flattened pixel + this = plane
    return new, report
