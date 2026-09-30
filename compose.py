"""Seams and rendering: from a ``Layout`` to pixels.

Seams.  OpenCV's graph-cut seam finder is the right tool but only tolerable at
about a megapixel (its cost is superlinear; 4 MP did not finish in ten
minutes).  So the cut is made at ~1 MP, then refined level by level: at each
finer scale only a narrow band around the current boundary of each neighboring
pair is re-cut, the rest of both tiles being pinned by the masks.  The result is
a label map (which tile owns which pixel) at 1/4 scale.

Rendering.  Every output pixel is one interpolation of one or two source
pixels: ``Layout.tile_maps`` composes pose, mesh and output frame into a single
source coordinate per output pixel, and the label map -- bilinearly upsampled and
sharpened to a ~3 px ramp -- says how much each tile contributes.  No pyramid
blend: a wide blend is what softens detail, and the seams sit where the tiles
already agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np
from scipy import ndimage as ndi

from align import MARGIN_PX
from geometry import Frame, Layout

cv2.ocl.setUseOpenCL(False)                  # the seam finder must not wander onto a GPU

Progress = Callable[[float, str], None]
_quiet: Progress = lambda p, m: None

SEAM_MIN_F = 4                               # finest scale of the label map
SEAM_TARGET_PX = 1.2e6                       # size of the first, full graph cut


@dataclass
class Plan:
    """How the tiles are cut and matched in brightness."""

    f: int                                   # label map scale: 1 px = f mosaic px
    ext0: np.ndarray                         # mosaic coordinate of label pixel (0, 0)
    labels: np.ndarray                       # uint8, tile index or 255 = no data
    gains: np.ndarray                        # (n, 3) per-tile per-channel multipliers

    def to_dict(self) -> dict:
        ok, png = cv2.imencode(".png", self.labels)
        return {"f": self.f, "ext0": self.ext0.tolist(), "gains": self.gains.tolist(),
                "shape": list(self.labels.shape), "labels_png": png.tobytes().hex()}

    @classmethod
    def from_dict(cls, d: dict) -> "Plan":
        lab = cv2.imdecode(np.frombuffer(bytes.fromhex(d["labels_png"]), np.uint8), cv2.IMREAD_UNCHANGED)
        return cls(int(d["f"]), np.array(d["ext0"], float), lab, np.array(d["gains"], float))


def _to255(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float32) * (255.0 / 65535.0 if a.dtype == np.uint16 else 1.0)


def render_low(lay: Layout, store, paths, k: int, f: int, frame: Frame, hw, margin: float = MARGIN_PX):
    """Tile k drawn into ``frame`` from its 1/f block-mean proxy: (float32 RGB 0..255, valid)."""
    ux, uy = lay.tile_maps(k, frame, hw, cs=8)
    src = store.proxy(paths[k], f)
    im = cv2.remap(src, ((ux - (f - 1) / 2) / f).astype(np.float32), ((uy - (f - 1) / 2) / f).astype(np.float32),
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return _to255(im), lay.valid_from_maps(k, ux, uy, margin=max(margin, f))


def level_shape(ext: np.ndarray, f: int) -> tuple[int, int]:
    span = ext[1] - ext[0]
    return int(math.ceil(span[1] / f)) + 1, int(math.ceil(span[0] / f)) + 1


def fit_gains(lay: Layout, imgs, valids) -> np.ndarray:
    """Per-tile, per-channel gain that makes the overlaps agree (anchor tile = 1).

    Off unless a tile is more than 0.4 % from its neighbors: a gain that is
    only noise would soften nothing but would also fix nothing.
    """
    n = len(imgs)
    rows, rhs, wts = [], [], []
    for a in range(n):
        for b in range(a + 1, n):
            m = valids[a] & valids[b]
            if m.sum() < 5000:
                continue
            A_, B_ = imgs[a][m], imgs[b][m]
            ok = (A_.min(1) > 8) & (A_.max(1) < 247) & (B_.min(1) > 8) & (B_.max(1) < 247)
            if ok.sum() < 2000:
                continue
            r = np.log(B_[ok].mean(0) / np.maximum(A_[ok].mean(0), 1e-3))          # (3,)
            row = np.zeros(n)
            row[a], row[b] = 1.0, -1.0
            rows.append(row)
            rhs.append(r)
            wts.append(math.sqrt(ok.sum()))
    g = np.ones((n, 3))
    if not rows:
        return g
    M = np.array(rows) * np.array(wts)[:, None]
    R = np.array(rhs) * np.array(wts)[:, None]
    keep = [k for k in range(n) if k != lay.anchor]
    x, *_ = np.linalg.lstsq(M[:, keep], R, rcond=None)
    if np.abs(x).max() < 0.004:
        return g
    g[keep] = np.exp(np.clip(x, -0.7, 0.7))
    return g


def _cut(imgs, corners, masks, cost: str = "COST_COLOR") -> list[np.ndarray]:
    """OpenCV graph-cut seams: returns the masks after the cut (mutually exclusive)."""
    finder = cv2.detail_GraphCutSeamFinder(cost)
    ms = [m.copy() for m in masks]
    out = finder.find([np.ascontiguousarray(i, np.float32) for i in imgs], list(corners), ms)
    res = out if out is not None else ms
    return [np.asarray(m.get() if hasattr(m, "get") else m) for m in res]


def _fill_gaps(lab: np.ndarray, valid_any: np.ndarray) -> np.ndarray:
    """The cut leaves a thin line of pixels owned by nobody; give each to its nearest tile."""
    hole = valid_any & (lab == 255)
    if not hole.any() or hole.all():
        return lab
    idx = ndi.distance_transform_edt(lab == 255, return_distances=False, return_indices=True)
    return np.where(hole, lab[idx[0], idx[1]], lab)


def _adjacent(lab: np.ndarray) -> set[tuple[int, int]]:
    out = set()
    for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1], lab[1:])):
        m = (a != b) & (a != 255) & (b != 255)
        out |= {(int(min(x, y)), int(max(x, y))) for x, y in set(zip(a[m].tolist(), b[m].tolist()))}
    return out


def _refine(lay: Layout, store, paths, lab: np.ndarray, f: int, ext: np.ndarray, gains: np.ndarray,
            radius: int = 3) -> np.ndarray:
    """Re-cut a band around every boundary of the label map, at scale f."""
    frame = Frame(scale=1.0 / f, origin=ext[0])
    H, W = lab.shape
    k = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    for i, j in sorted(_adjacent(lab)):
        band = (cv2.dilate((lab == i).astype(np.uint8), k) & cv2.dilate((lab == j).astype(np.uint8), k)) > 0
        ys, xs = np.nonzero(band)
        if len(ys) == 0:
            continue
        y0, y1 = max(ys.min() - 2, 0), min(ys.max() + 3, H)
        x0, x1 = max(xs.min() - 2, 0), min(xs.max() + 3, W)
        fr = frame.sub(x0, y0)
        hw = (y1 - y0, x1 - x0)
        sub = lab[y0:y1, x0:x1]
        bnd = band[y0:y1, x0:x1]
        imgs, masks = [], []
        for t in (i, j):
            im, v = render_low(lay, store, paths, t, f, fr, hw)
            imgs.append(im * gains[t])
            masks.append((((sub == t) | bnd) & v).astype(np.uint8) * 255)
        try:
            ri, rj = _cut(imgs, [(0, 0), (0, 0)], masks, "COST_COLOR_GRAD")
        except cv2.error:
            continue
        new = sub.copy()
        new[bnd & (ri > 0) & (rj == 0)] = i
        new[bnd & (rj > 0) & (ri == 0)] = j
        lab[y0:y1, x0:x1] = new
    return lab


def plan_seams(lay: Layout, store, paths, progress: Progress = _quiet) -> Plan:
    """Cut the tiles apart along low-difference paths and match their brightness."""
    n = len(paths)
    ext = lay.extent()
    if n == 1:
        return Plan(SEAM_MIN_F, ext[0], np.zeros(level_shape(ext, SEAM_MIN_F), np.uint8), np.ones((1, 3)))
    span = ext[1] - ext[0]
    f = SEAM_MIN_F
    while span.prod() / f ** 2 > SEAM_TARGET_PX:
        f *= 2
    frame = Frame(scale=1.0 / f, origin=ext[0])
    hw = level_shape(ext, f)
    progress(0.05, f"cutting at 1/{f}")
    rendered = [render_low(lay, store, paths, k, f, frame, hw) for k in range(n)]
    imgs = [r[0] for r in rendered]
    valids = [r[1] for r in rendered]
    gains = fit_gains(lay, imgs, valids)
    imgs = [im * gains[k] for k, im in enumerate(imgs)]

    res = _cut(imgs, [(0, 0)] * n, [(v * 255).astype(np.uint8) for v in valids])
    lab = np.full(hw, 255, np.uint8)
    for k, m in enumerate(res):
        lab[m > 0] = k
    lab = _fill_gaps(lab, np.any(valids, axis=0))
    del imgs, rendered

    while f > SEAM_MIN_F:
        f //= 2
        hw = level_shape(ext, f)
        lab = np.repeat(np.repeat(lab, 2, 0), 2, 1)[:hw[0], :hw[1]]
        if lab.shape != hw:
            lab = np.pad(lab, ((0, hw[0] - lab.shape[0]), (0, hw[1] - lab.shape[1])), mode="edge")
        progress(0.3 + 0.6 * (1 - math.log2(f / SEAM_MIN_F) / 4), f"refining seams at 1/{f}")
        lab = _refine(lay, store, paths, lab, f, ext, gains)
    return Plan(f, ext[0].copy(), lab, gains)


class Mosaic:
    """Layout + seams + tiles: ``render`` any region of any frame."""

    def __init__(self, lay: Layout, store, paths, plan: Plan, feather: float = 3.0):
        self.lay, self.store, self.paths, self.plan, self.feather = lay, store, paths, plan, feather
        self._mask: dict[int, np.ndarray] = {}

    def _label_mask(self, k: int) -> np.ndarray:
        if k not in self._mask:
            self._mask[k] = (self.plan.labels == k).astype(np.float32)
        return self._mask[k]

    def weights(self, k: int, frame: Frame, w: int, h: int) -> np.ndarray:
        """Contribution of tile k to a (h, w) window of ``frame``: a sharpened label ramp."""
        p = self.plan
        M = np.hstack([frame.A / p.f, ((frame.b - p.ext0) / p.f)[:, None]]).astype(np.float64)
        m = cv2.warpAffine(self._label_mask(k), M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)      # <=1/8 px position steps: fine for a mask
        return np.clip((m - 0.5) * (p.f / max(self.feather, 0.5)) + 0.5, 0.0, 1.0)

    def footprint(self, k: int, frame: Frame, w: int, h: int, pad: int = 1):
        """(x0, y0, x1, y1): the part of a (h, w) window of ``frame`` that tile k can reach, ``pad`` px wider, or None."""
        (a, b), (c, d) = self.lay.bbox(k)
        q = frame.to_output(np.array([[a, b], [c, b], [a, d], [c, d]]))
        lo, hi = np.floor(q.min(0)).astype(int) - pad, np.ceil(q.max(0)).astype(int) + 1 + pad
        x0, y0, x1, y1 = max(lo[0], 0), max(lo[1], 0), min(hi[0], w), min(hi[1], h)
        return (int(x0), int(y0), int(x1), int(y1)) if x0 < x1 and y0 < y1 else None

    def render(self, frame: Frame, x0: int, y0: int, w: int, h: int, interp: int | None = None, each=None):
        """Window [x0, x0+w) x [y0, y0+h) of ``frame``: float32 RGB in 0..1, and coverage.

        Source: the full-resolution tile when the frame is near 1:1 (Lanczos),
        otherwise the block-mean proxy that is closest without being finer than
        the output -- averaging is what downscaling must do, and it is cheaper.

        ``each(k, xa, ya, pixels, valid, mask)``, when given, gets each tile over all of it that the window holds
        (its own layer): its pixels, where it has any, and the mask that stacks it on the tiles before it as the
        blend does -- w_k / (w_0 + ... + w_k) -- so that the stack of layers is the picture.
        """
        fr = frame.sub(x0, y0)
        scale = 1.0 / math.sqrt(abs(np.linalg.det(fr.A)))
        fp = 1
        while fp * 2 <= 1.0 / scale * 1.0001:
            fp *= 2
        if interp is None:
            interp = cv2.INTER_LANCZOS4 if fp == 1 else cv2.INTER_LINEAR
        acc = np.zeros((h, w, 3), np.float32)
        wsum = np.zeros((h, w), np.float32)
        for k in range(len(self.paths)):
            wk = self.weights(k, fr, w, h)
            if each:
                box = self.footprint(k, fr, w, h)
                if box is None:
                    continue
                xa, ya, xb, yb = box
            else:
                nz = np.nonzero(wk.max(0) > 0)[0]
                if len(nz) == 0:
                    continue
                rows = np.nonzero(wk.max(1) > 0)[0]
                xa, xb, ya, yb = int(nz[0]), int(nz[-1]) + 1, int(rows[0]), int(rows[-1]) + 1
            sub = fr.sub(xa, ya)
            ux, uy = self.lay.tile_maps(k, sub, (yb - ya, xb - xa), cs=16)
            src = self.store.get(self.paths[k]).rgb if fp == 1 else self.store.proxy(self.paths[k], fp)
            im = cv2.remap(src, ((ux - (fp - 1) / 2) / fp).astype(np.float32),
                           ((uy - (fp - 1) / 2) / fp).astype(np.float32),
                           interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            top = 65535.0 if src.dtype == np.uint16 else 255.0
            valid = self.lay.valid_from_maps(k, ux, uy, margin=0.0)
            wc = wk[ya:yb, xa:xb] * valid
            px = im.astype(np.float32) * (self.plan.gains[k] / top).astype(np.float32)
            acc[ya:yb, xa:xb] += px * wc[..., None]
            wsum[ya:yb, xa:xb] += wc
            if each:
                s = wsum[ya:yb, xa:xb]
                each(k, xa, ya, px, valid, np.divide(wc, s, out=np.zeros_like(wc), where=s > 0))
        out = acc / np.maximum(wsum, 1e-6)[..., None]
        return np.clip(out, 0.0, 1.0), wsum > 1e-3
