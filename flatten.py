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

from pathlib import Path

import cv2
import numpy as np
import tifffile

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


def flatten(store, paths, out_dir, progress=_quiet) -> tuple[list[str], dict]:
    """Write a flattened copy of every photograph and return the new paths."""
    Hs, report = homographies(store, paths, progress)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    new, kept = [], []
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
        q = out / (Path(p).stem + ".tif")
        tifffile.imwrite(q, flat, compression="zlib", photometric="rgb")
        new.append(str(q))
    report["kept"] = kept
    return new, report
