"""Where every tile sits, and the maps between a tile and the mosaic.

A tile's pose is a rigid motion about its own center plus a smooth displacement
mesh on the mosaic plane:

    P = R(theta) (u - c) + o          rigid placement of source pixel u
    X = P + D(P)                      mesh D: bilinear over nodes ``g`` px apart

The mesh is what a flatbed needs.  Between two scans of the same sheet the scan
direction is nearly affine, but along the sensor line the paper's height
(cockling, lid pressure) moves the image by tens of pixels, smoothly and
differently on every scan; a rigid or affine model leaves that as ghosting.

Everything that draws pixels goes through ``tile_maps``: one map from output
pixel to source pixel, composed from the tile's pose, the mesh and the output
frame (zoom, straighten, crop).  A tile is therefore resampled once, however
many of those it passes through.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def rot(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


class Frame:
    """Output pixel (p, q) <-> mosaic coordinates: X = A [p, q] + b.

    ``Frame(scale=1/8, origin=(x0, y0))`` is a plain zoomed window;  a rotation
    ``angle`` about ``pivot`` is how straightening enters (mosaic X of output
    pixel = pivot + R(angle)(...)).
    """

    def __init__(self, scale: float = 1.0, origin=(0.0, 0.0), angle: float = 0.0,
                 pivot=(0.0, 0.0)):
        self.A = rot(angle) / scale
        # output (0, 0) sits at ``origin`` in the *straightened* plane; that plane
        # is the mosaic rotated by ``angle`` about ``pivot``.
        self.b = np.asarray(pivot, float) + rot(angle) @ (np.asarray(origin, float) - np.asarray(pivot, float))

    def sub(self, x0: float, y0: float) -> "Frame":
        """The same frame with output pixel (x0, y0) renamed (0, 0)."""
        fr = Frame.__new__(Frame)
        fr.A = self.A
        fr.b = self.A @ np.array([x0, y0], float) + self.b
        return fr

    def to_mosaic(self, pq: np.ndarray) -> np.ndarray:
        return pq @ self.A.T + self.b

    def to_output(self, X: np.ndarray) -> np.ndarray:
        return (X - self.b) @ np.linalg.inv(self.A).T


class Layout:
    """Poses and meshes of ``n`` tiles.  ``sizes`` are (w, h) in source pixels."""

    def __init__(self, sizes, theta=None, o=None, g: float = 512.0, anchor: int = 0, trim=None):
        self.sizes = [(int(w), int(h)) for w, h in sizes]
        n = len(self.sizes)
        #: Source pixels cut off every side of each scan before anything is matched, cut or painted: the
        #: strip where the paper lifts off the glass at the scanner's edge and the picture bends with it.
        self.trim = np.zeros(n) if trim is None else np.asarray(trim, float).reshape(n).copy()
        self.theta = np.zeros(n) if theta is None else np.asarray(theta, float).copy()
        self.o = (np.array([self.centre(k) for k in range(n)]) if o is None
                  else np.asarray(o, float).reshape(n, 2).copy())
        self.g = float(g)
        self.anchor = int(anchor)
        self.org: list[np.ndarray] = [None] * n
        self.D: list[np.ndarray] = [None] * n
        for k in range(n):
            self.reset_mesh(k)

    # -- construction --------------------------------------------------------

    def centre(self, k: int) -> np.ndarray:
        w, h = self.sizes[k]
        return np.array([(w - 1) / 2.0, (h - 1) / 2.0])

    def corners(self, k: int) -> np.ndarray:
        w, h = self.sizes[k]
        return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)

    def reset_mesh(self, k: int) -> None:
        P = self.rigid(k, self.corners(k))
        lo, hi = P.min(0), P.max(0)
        org = np.floor(lo / self.g) * self.g - self.g
        cnt = np.ceil((hi - org) / self.g).astype(int) + 2
        self.org[k] = org
        self.D[k] = np.zeros((cnt[1], cnt[0], 2))

    # -- maps -----------------------------------------------------------------

    def rigid(self, k: int, u: np.ndarray) -> np.ndarray:
        return (np.asarray(u, float) - self.centre(k)) @ rot(self.theta[k]).T + self.o[k]

    def rigid_inv(self, k: int, P: np.ndarray) -> np.ndarray:
        return (np.asarray(P, float) - self.o[k]) @ rot(self.theta[k]) + self.centre(k)

    def weights(self, k: int, P: np.ndarray):
        """Bilinear node indices and weights of mosaic points P in tile k's mesh."""
        D = self.D[k]
        gy, gx = D.shape[:2]
        f = (np.asarray(P, float) - self.org[k]) / self.g
        x = np.clip(f[:, 0], 0, gx - 1.000001)
        y = np.clip(f[:, 1], 0, gy - 1.000001)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = x - x0, y - y0
        idx = np.stack([y0 * gx + x0, y0 * gx + x0 + 1, (y0 + 1) * gx + x0, (y0 + 1) * gx + x0 + 1], 1)
        w = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], 1)
        return idx, w

    def disp(self, k: int, P: np.ndarray) -> np.ndarray:
        idx, w = self.weights(k, P)
        flat = self.D[k].reshape(-1, 2)
        return np.einsum("nc,ncd->nd", w, flat[idx])

    def fwd(self, k: int, u: np.ndarray) -> np.ndarray:
        """source pixel -> mosaic"""
        P = self.rigid(k, u)
        return P + self.disp(k, P)

    def inv(self, k: int, X: np.ndarray, iters: int = 8) -> np.ndarray:
        """mosaic -> source pixel, by fixed point on P = X - D(P)."""
        X = np.asarray(X, float)
        P = X.copy()
        for _ in range(iters):
            P = X - self.disp(k, P)
        return self.rigid_inv(k, P)

    def bbox(self, k: int, n: int = 24) -> np.ndarray:
        """[[x0, y0], [x1, y1]] of the tile in the mosaic (walks its border)."""
        w, h = self.sizes[k]
        t = np.linspace(0, 1, n)
        edge = np.concatenate([np.c_[t * (w - 1), 0 * t], np.c_[t * (w - 1), 0 * t + h - 1],
                               np.c_[0 * t, t * (h - 1)], np.c_[0 * t + w - 1, t * (h - 1)]])
        X = self.fwd(k, edge)
        return np.array([X.min(0), X.max(0)])

    def extent(self) -> np.ndarray:
        b = np.array([self.bbox(k) for k in range(len(self.sizes))])
        return np.array([b[:, 0].min(0), b[:, 1].max(0)])

    # -- rendering maps ---------------------------------------------------------

    def tile_maps(self, k: int, frame: Frame, hw: tuple[int, int], cs: int = 16):
        """Source coordinates (ux, uy) of every output pixel, float32 (h, w).

        Computed on a grid ``cs`` pixels apart and interpolated bilinearly; the
        map is smooth (a rigid motion plus a coarse mesh), so this is exact to
        a small fraction of a pixel and costs almost nothing.  The nodes sit at
        block centers so that ``cv2.resize`` -- which interpolates float data
        with float weights, unlike ``warpAffine`` whose 1/32 weight steps would
        put a +-0.25 px sawtooth into the map -- lands each output pixel exactly.
        """
        h, w = hw
        gh = int(math.ceil(h / cs)) + 2
        gw = int(math.ceil(w / cs)) + 2
        ys = (np.arange(gh) - 1 + 0.5) * cs - 0.5
        xs = (np.arange(gw) - 1 + 0.5) * cs - 0.5
        pq = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
        u = self.inv(k, frame.to_mosaic(pq)).reshape(gh, gw, 2).astype(np.float32)
        up = cv2.resize(u, (gw * cs, gh * cs), interpolation=cv2.INTER_LINEAR)[cs:cs + h, cs:cs + w]
        return np.ascontiguousarray(up[..., 0]), np.ascontiguousarray(up[..., 1])

    def valid_from_maps(self, k: int, ux: np.ndarray, uy: np.ndarray, margin: float = 0.0):
        w, h = self.sizes[k]
        m = max(float(margin), float(self.trim[k]))
        return (ux >= m) & (ux <= w - 1 - m) & (uy >= m) & (uy <= h - 1 - m)

    # -- persistence -----------------------------------------------------------

    def to_dict(self) -> dict:
        return {"sizes": self.sizes, "theta": self.theta.tolist(), "o": self.o.tolist(),
                "g": self.g, "anchor": self.anchor, "trim": self.trim.tolist(),
                "org": [o.tolist() for o in self.org],
                "D": [{"shape": list(d.shape), "data": np.round(d, 3).ravel().tolist()} for d in self.D]}

    @classmethod
    def from_dict(cls, d: dict) -> "Layout":
        lay = cls(d["sizes"], d["theta"], d["o"], d["g"], d["anchor"], d.get("trim"))
        lay.org = [np.array(o, float) for o in d["org"]]
        lay.D = [np.array(m["data"], float).reshape(m["shape"]) for m in d["D"]]
        return lay
