"""Photographs of a flat artwork, put back into one plane.

A scan is the plane already.  A photograph is the plane seen from somewhere, and any two
photographs of a flat thing differ by a homography -- exactly, whatever the camera did,
because the subject is flat.  So the one thing a camera set needs that a scan set does not
is to be flattened first; after that it is a set of scans, and the coarse search, the mesh,
the seams, the patches and the export all work on it unchanged.

The plane is one of the photographs' own: the one the others lose the least detail onto, which
is the one taken nearest and most square-on.  Put a frontal photograph onto a steep one's plane
and its far side is squeezed into a few pixels; the other way round only the steep one's far side
is stretched, and that had little detail to lose.  Nothing here knows the camera, so nothing here
can know which way the wall faced: if even that photograph was taken at an angle, the finished
picture keeps that angle.  Hold the camera square for one shot, from no farther off than the
rest, and the whole set comes out square.

A real lens is not a pinhole: it bends straight lines, most at the corners, and then two
photographs differ by a homography only after that bend is taken out.  So the bend is read off
the same matches (``lens``) and taken out in the same warp.

Not every wall can be shot from in front of it.  Past about 50 degrees between two photographs
SIFT finds nothing to match, because a patch seen that far round is squashed along one direction
and SIFT forgives only rotation and scale.  So photographs that no chain of sure matches joins are
looked at again through simulated tilts (ASIFT, Morel and Yu 2009): each one squashed every way,
at up to five times, and its features looked up among the other photograph's own.  That is a few
seconds a photograph, so it is spent only where plain SIFT left the set in pieces.  Seen that far
round, the far end of the wall shrinks toward the horizon, and put back into the plane it would be
all stretch and no detail, more pixels than the rest of the photograph together: it is cut off
where it would be stretched MAX_STRETCH times as much as the photograph's best part.

Each flattened photograph is cut down to the largest upright rectangle inside the warped
frame, so that no tile carries any "nothing here" corners -- a mask only this stage would
know about, which nothing downstream reads.  A photograph from far round lands as a long wedge,
and the rectangle inside a wedge holds less of it: about a half at 40 to 50 degrees, under a third at 60.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image
from scipy.optimize import minimize_scalar
from scipy.sparse.csgraph import connected_components

import edges

MIN_INLIERS = 25          # matches that survive RANSAC before a pair is believed to be the same wall
SURE_INLIERS = 100        # fewer, and the pair was probably seen too far round for SIFT: worth the squashed views
WORK_PX = 1500            # long side the features are found on: more is slower and finds no more pairs
MAX_STRETCH = 3.0         # lengths: a photograph blown up this much where it is stretched least is a bad fit; a part of one
                          # stretched this much more than that is seen too far round to keep; the plane is the photograph's
                          # that shrinks none of the others this much, or the least
LENS_GAIN = 0.9           # a lens bend is believed only if it takes a tenth off how much the matches disagree


def _quiet(p: float, m: str) -> None:
    pass


def _features(gray: np.ndarray, tilted: bool = False):
    """SIFT; ``tilted``: on every squashed view of the photograph too, at its own pixels (ASIFT)."""
    sift = cv2.SIFT_create(nfeatures=2000)
    return (cv2.AffineFeature_create(sift, maxTilt=5) if tilted else sift).detectAndCompute(np.clip(gray, 0, 255).astype(np.uint8), None)


def _matches(ka, da, kb, db):
    """Photo b's features found again in photo a: (b's points, a's points), or None if too few."""
    if da is None or db is None or len(da) < 4 or len(db) < 4:
        return None
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(db, da, k=2)
    good = [p[0] for p in m if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < MIN_INLIERS:
        return None
    return np.float64([kb[g.queryIdx].pt for g in good]), np.float64([ka[g.trainIdx].pt for g in good])


def _pair(src, dst) -> tuple[np.ndarray | None, np.ndarray]:
    """Homography taking b's points to a's, and which of them agreed (all False if too few did).

    Fitted in whichever photograph sees the overlap with more pixels: there a pixel of error is less of the wall.
    The other way round, a photograph taken far round would set the fit, and its far end is a few pixels for a lot.
    """
    def fit(s, d):
        H, inl = cv2.findHomography(s, d, cv2.USAC_MAGSAC, 3.0, maxIters=20000, confidence=0.999)
        inl = np.zeros(len(s), bool) if inl is None else inl.ravel() > 0
        return (H, inl) if H is not None and inl.sum() >= MIN_INLIERS else (None, np.zeros(len(s), bool))
    H, inl = fit(src, dst)
    if H is not None and np.median(np.linalg.det(H) / (np.c_[src[inl], np.ones(inl.sum())] @ H[2]) ** 3) < 1:
        Hi, inl = fit(dst, src)                         # a's pixels are the coarser: measure in b's
        H = None if Hi is None else np.linalg.inv(Hi)
    return H, inl


def _rim(w: int, h: int) -> np.ndarray:
    """A photograph's frame, 32 points a side: closely enough to follow the curve a lens's bend gives it."""
    t = np.linspace(0, 1, 33)[:-1, None]
    return np.concatenate([t * [w - 1, 0], [w - 1, 0] + t * [0, h - 1], [w - 1, h - 1] - t * [w - 1, 0], [0, h - 1] - t * [0, h - 1]])


def _w(M, pts) -> np.ndarray:
    """``M``'s denominator at ``pts``, signed positive where it is largest: ``M`` magnifies areas det(M) / w^3 there,
    so w is how little it stretches each one, and where it changes sign the plane's horizon runs through."""
    w = np.c_[pts, np.ones(len(pts))] @ M[2]
    return w * np.sign(w[np.abs(w).argmax()])


def _least(M, pts) -> float:
    """How much ``M`` magnifies a photograph's areas where it magnifies them least, ``pts`` its frame (pinhole pixels):
    below 1 the plane has fewer pixels for that part than the photograph had."""
    return abs(float(np.linalg.det(M))) / float(_w(M, pts).max()) ** 3


def _clip(pts, M) -> np.ndarray:
    """The part of a photograph's frame ``pts`` (pinhole pixels) that ``M`` stretches at most MAX_STRETCH times as much as
    its least-stretched part.  w is linear in the photograph, so the cut is a straight line across it."""
    w = _w(M, pts)
    lim = w.max() / MAX_STRETCH ** (2 / 3)             # areas go as w^-3, lengths as w^-3/2
    out = []
    for p, q, a, b in zip(pts, np.roll(pts, -1, 0), w, np.roll(w, -1)):
        if a >= lim:
            out.append(p)
        if (a >= lim) != (b >= lim):
            out.append(p + (lim - a) / (b - a) * (q - p))
    return np.array(out)


def _K(shape, A=None) -> np.ndarray:
    """Photo pixel -> where the lens is described: centred, in units of the long side (``A``: at a proxy's scale)."""
    h, w = shape[:2]
    s = float(max(w, h))
    K = np.array([[s, 0, (w - 1) / 2.0], [0, s, (h - 1) / 2.0], [0, 0, 1.0]])
    return K if A is None else A @ K


def undistort(p, K, k1: float):
    """Photo pixels where a pinhole would have put them: the division model, x / (1 + k1 |x|^2).  Barrel is k1 < 0."""
    if not k1:
        return p
    x = (p - K[:2, 2]) / K[0, 0]
    return K[:2, 2] + K[0, 0] * x / (1 + k1 * (x * x).sum(-1, keepdims=True))


def distort(p, K, k1: float):
    """The other way: where the lens put what a pinhole would have put at ``p``."""
    if not k1:
        return p
    x = (p - K[:2, 2]) / K[0, 0]
    return K[:2, 2] + K[0, 0] * x * 2 / (1 + np.sqrt(np.maximum(1 - 4 * k1 * (x * x).sum(-1, keepdims=True), 0)))


def lens(pairs) -> float:
    """How much the lens bends straight lines (k1, see ``undistort``), from points matched between photographs.

    Through a pinhole, two photographs of a flat thing differ by a homography exactly; through a real lens
    they do not, by the bend.  So the bend is whatever makes the matches a homography again, judged where
    they were seen.  0.0 unless it takes a tenth off their disagreement: a camera that straightened its own
    pictures (any phone does) gets nothing.  ``pairs``: (points in a, the same points in b, a's K, b's K).

    Measured on the corners of OpenCV's eight chessboard photographs (``left*.jpg``) matched across all 28
    pairs: k1 = -0.40, and their disagreement goes from 1.13 px to 0.78.  (The photographs' own features say
    nothing there: the camera stood still and the board moved, so what matches is the room.)  Synthetic
    photographs through k1 = -0.12 give it back within 5 %.  Handheld photographs from cameras that
    straighten their own (OpenCV's newspaper set, the Oxford graffiti) gain 2 and 3 % and get 0.
    """
    def rms(k1: float) -> float:
        sse = n = 0
        for a, b, Ka, Kb in pairs:
            ua, ub = undistort(a, Ka, k1), undistort(b, Kb, k1)
            H, _ = cv2.findHomography(ub, ua, 0)
            if H is None:
                return math.inf
            e = distort(cv2.perspectiveTransform(ub.reshape(-1, 1, 2), H).reshape(-1, 2), Ka, k1) - a
            sse, n = sse + float((e * e).sum()), n + len(e)
        return math.sqrt(sse / n)
    if not pairs:
        return 0.0
    best = minimize_scalar(rms, bounds=(-0.8, 0.5), method="bounded", options={"xatol": 1e-4})
    return round(float(best.x), 5) if best.fun < LENS_GAIN * rms(0.0) else 0.0


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


def _inscribed(poly: np.ndarray, cells: int = 512) -> tuple[int, int, int, int]:
    """The largest upright rectangle inside a photograph's warped frame, any simple polygon, as (x0, y0, x1, y1).

    Found on a grid ``cells`` across and kept a cell in from the frame, so it holds the photograph's own
    pixels only -- a lens's bend makes the frame's sides curves.  A cell is what it costs against the true
    optimum; what it saves is every "nothing here" corner, which no later stage would know to leave alone.
    """
    lo = poly.min(0)
    q = float((poly.max(0) - lo).max()) / cells
    gw, gh = np.ceil((poly.max(0) - lo) / q).astype(int) + 1
    m = np.zeros((gh, gw), np.uint8)
    cv2.fillPoly(m, [np.round((poly - lo) / q * 16).astype(np.int32).reshape(-1, 1, 2)], 1, cv2.LINE_8, 4)
    x0, y0, x1, y1 = edges.largest_rectangle(cv2.erode(m, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0)
    return (int(np.ceil(lo[0] + x0 * q)), int(np.ceil(lo[1] + y0 * q)),
            int(np.floor(lo[0] + (x1 - 1) * q)), int(np.floor(lo[1] + (y1 - 1) * q)))


def focal_px(path) -> float | None:
    """The focal length of a photograph in its own pixels, from what the camera wrote in the file.

    A rectangle shot with the camera level and tilted only up or down -- a tripod squared to the wall,
    which is how anyone careful does it -- is a symmetric trapezoid, and a symmetric trapezoid is the
    one case the corners cannot be read alone: a tall rectangle tilted far looks exactly like a short
    one tilted little.  The focal length settles it, and the camera already knows it.
    """
    try:
        with Image.open(path) as im:
            eq35 = im.getexif().get_ifd(0x8769).get(41989)       # FocalLengthIn35mmFilm, whole millimetres, in the Exif block
            if eq35:                                             # reckoned on the 35 mm frame's diagonal, as cameras do
                return float(eq35) / 43.27 * math.hypot(*im.size)
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
    """One homography per photograph, taking its own pixels -- with the lens's bend taken out, the report's
    ``k1`` -- into the plane of the one that loses the others the least detail, the report's ``reference``.
    They are chained from the best-connected one, where the chains are shortest."""
    n = len(paths)
    if n == 1:
        return [np.eye(3)], {"reference": 0, "inliers": {}, "k1": 0.0}
    shape = store.get(paths[0]).shape
    f = max(1, int(round(max(shape) / WORK_PX)))
    # proxy pixel i covers source [f i, f i + f), so u_src = f u_proxy + (f - 1) / 2
    A = np.array([[1.0 / f, 0, -(f - 1) / (2.0 * f)], [0, 1.0 / f, -(f - 1) / (2.0 * f)], [0, 0, 1.0]])
    Ai = np.linalg.inv(A)
    feats, Ks = [], [_K(store.get(p).shape, A) for p in paths]
    for k, p in enumerate(paths):
        progress(0.1 + 0.3 * k / n, f"looking at photo {k + 1}")
        feats.append(_features(store.proxy(p, f, gray=True)))
    found = {}
    for a in range(n):
        for b in range(a + 1, n):
            progress(0.4 + 0.3 * (a * n + b) / (n * n), f"matching {a + 1} and {b + 1}")
            m = _matches(*feats[a], *feats[b])
            if m is not None:
                found[a, b] = m
    fits = {ab: _pair(*m) for ab, m in found.items()}
    tied = np.zeros((n, n), bool)
    for (a, b), (_, inl) in fits.items():
        tied[a, b] = inl.sum() >= SURE_INLIERS
    _, part = connected_components(tied, directed=False)
    cross = [(a, b) for a in range(n) for b in range(a + 1, n) if part[a] != part[b]]
    tilted = {}
    for i, (a, b) in enumerate(cross):                  # pieces, or held by a thread: try squashed views
        progress(0.7 + 0.1 * i / len(cross), f"matching {a + 1} and {b + 1} from other angles")
        for k in (a, b):
            if k not in tilted:
                tilted[k] = _features(store.proxy(paths[k], f, gray=True), tilted=True)
        mb, ma = _matches(*feats[a], *tilted[b]), _matches(*feats[b], *tilted[a])     # each way round: (query's, train's)
        ms = [m for m in (mb, ma and ma[::-1]) if m is not None]
        if ms:
            m = tuple(np.concatenate(z) for z in zip(*ms))
            fit = _pair(*m)
            if fit[1].sum() >= SURE_INLIERS:            # forty views of each make chance agreements likelier: only a sure pair counts
                found[a, b], fits[a, b] = m, fit
    k1 = lens([(dst[i], src[i], Ks[a], Ks[b]) for (a, b), (src, dst) in found.items()
               for i in [fits[a, b][1]] if i.any()])
    if k1:                                              # again on the straightened points: the corners agree now too
        fits = {(a, b): _pair(undistort(src, Ks[b], k1), undistort(dst, Ks[a], k1)) for (a, b), (src, dst) in found.items()}
    w = np.zeros((n, n))
    H = {}
    for (a, b), (Hab, inl) in fits.items():
        if Hab is not None:
            w[a, b] = w[b, a] = inl.sum()
            H[a, b], H[b, a] = Hab, np.linalg.inv(Hab)
    root = int(w.sum(1).argmax())
    par = _tree(n, w, root)
    if -2 in par:
        bad = [str(k + 1) for k, v in enumerate(par) if v == -2]
        raise ValueError("no overlap found with photo " + ", ".join(bad) + " -- the photos must overlap each other")
    out = [None] * n
    out[root] = np.eye(3)
    for k in sorted(range(n), key=lambda k: _depth(par, k)):
        if out[k] is None:
            out[k] = out[par[k]] @ H[par[k], k]
    out = [Ai @ (M / M[2, 2]) @ A for M in out]
    rims = [undistort(_rim(*store.get(p).shape[1::-1]), _K(store.get(p).shape), k1) for p in paths]
    def finest(r):                                      # the least photo r's plane leaves any photograph; 0: it blows one up
        m = [_least(np.linalg.inv(out[r]) @ M, q) for M, q in zip(out, rims)]
        return min(m) if max(m) <= MAX_STRETCH ** 2 else 0.0
    ref = max(range(n), key=lambda r: (finest(r), r == root))
    out = [np.linalg.inv(out[ref]) @ M for M in out]
    return [M / M[2, 2] for M in out], {"reference": ref, "k1": k1, "inliers": {
        f"{a + 1}-{b + 1}": int(w[a, b]) for a in range(n) for b in range(a + 1, n) if w[a, b]}}


def _depth(par: list[int], k: int) -> int:
    d = 0
    while par[k] != -1:
        k, d = par[k], d + 1
    return d


def _warp(img, M, K, k1: float, size):
    """``img`` put where ``M`` takes its pinhole pixels (see ``undistort``), ``size`` = (w, h)."""
    if not k1:
        return cv2.warpPerspective(img, M, size, flags=cv2.INTER_CUBIC)
    w, h = size
    out = np.empty((h, w) + img.shape[2:], img.dtype)
    Mi = np.linalg.inv(M)
    for y0 in range(0, h, 256):                         # in strips: a map of the whole is 8 bytes a pixel
        y1 = min(y0 + 256, h)
        v, u = np.mgrid[y0:y1, 0:w]
        q = cv2.perspectiveTransform(np.c_[u.ravel(), v.ravel()].astype(np.float64).reshape(-1, 1, 2), Mi).reshape(-1, 2)
        out[y0:y1] = cv2.remap(img, distort(q, K, k1).astype(np.float32).reshape(y1 - y0, w, 2), None, cv2.INTER_CUBIC)
    return out


def flatten(store, paths, out_dir, progress=_quiet, Hs=None, k1: float = 0.0) -> tuple[list[str], dict]:
    """Write a flattened copy of every photograph and return the new paths.

    ``Hs`` (with the lens's ``k1``) are homographies worked out before -- how the set is put back onto a
    plane chosen by hand rather than onto a photograph's own (see ``rectangle``); the report carries
    them either way.
    """
    report: dict = {"reference": 0, "inliers": {}}
    if Hs is None:
        Hs, report = homographies(store, paths, progress)
        k1 = report["k1"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    new, kept, origin = [], [], []
    for k, (p, H) in enumerate(zip(paths, Hs)):
        progress(0.8 + 0.2 * k / len(paths), f"flattening photo {k + 1}")
        img = store.get(p).rgb
        h, w = img.shape[:2]
        K = _K(img.shape)
        rim = undistort(_rim(w, h), K, k1)
        x0, y0, x1, y1 = _inscribed(cv2.perspectiveTransform(_clip(rim, H).reshape(-1, 1, 2), H).reshape(-1, 2))
        x0, y0, x1, y1 = x0 + 2, y0 + 2, x1 - 2, y1 - 2        # cubic reads two pixels out: stay that far inside the frame
        if _least(H, rim) > MAX_STRETCH ** 2 or x1 <= x0 or y1 <= y0:      # blown up as a whole, or nothing left of it
            raise ValueError(f"photo {k + 1} does not look like the same flat thing as the others")
        T = np.array([[1.0, 0, -x0], [0, 1.0, -y0], [0, 0, 1.0]])
        flat = _warp(img, T @ H, K, k1, (x1 - x0 + 1, y1 - y0 + 1))
        kept.append(round((x1 - x0 + 1) * (y1 - y0 + 1) / float(w * h), 3))
        origin.append([x0, y0])
        q = out / (Path(p).stem + ".tif")
        tifffile.imwrite(q, flat, compression="zlib", photometric="rgb")
        store.drop(q)                                   # flattened again onto another plane: forget the old pixels
        new.append(str(q))
    report["kept"], report["k1"] = kept, k1
    report["h"] = [H.tolist() for H in Hs]          # photograph pixel -> the plane, before the cut
    report["origin"] = origin                       # where the cut starts in it: flattened pixel + this = plane
    return new, report
