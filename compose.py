"""Seams and rendering: from a ``Layout`` to pixels.

Seams.  OpenCV's graph-cut seam finder is the right tool but only tolerable at
about a megapixel (its cost is superlinear; 4 MP did not finish in ten
minutes).  So the cut is made at ~1 MP, then refined level by level: at each
finer scale only a narrow band around the current boundary of each neighboring
pair is re-cut, the rest of both tiles being pinned by the masks.  The result is
a label map (which tile owns which pixel) at 1/4 scale.  Paint one photograph
shows glared and another shows clean is given to the clean one (``_glare``).

Rendering.  Every output pixel is one interpolation of one or two source
pixels: ``Layout.tile_maps`` composes pose, mesh and output frame into a single
source coordinate per output pixel, and the label map -- bilinearly upsampled and
sharpened to a ~3 px ramp -- says how much each tile contributes.  Detail is
never blended: a wide blend is what softens it, and the cut goes where the
tiles already agree.

What is blended is the low frequencies, through ``fit_field``: a smooth
multiplicative correction per tile, fitted so that the tiles' blurred versions
agree.  That is the whole of a lamp that falls off across the bed, a
photograph's vignette, a slide's drift from field to field -- and the slow half
of what Reposition leaves at a seam.  On flat paper on a flatbed it fits 1.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import lsq_linear

from align import MARGIN_PX
from geometry import Frame, Layout

cv2.ocl.setUseOpenCL(False)                  # the seam finder must not wander onto a GPU

Progress = Callable[[float, str], None]
_quiet: Progress = lambda p, m: None

SEAM_MIN_F = 4                               # finest scale of the label map
SEAM_TARGET_PX = 1.2e6                       # size of the first, full graph cut
SEAM_FIELD_F = 16                            # scale the correction is fitted at
FIELD_PX = 1024                              # mosaic px between correction nodes: how sharply it may bend
VIG_MAX = 0.47                               # most a corner may be lifted: exp(0.47) = 1.6x
VIG_RIDGE = 0.2                              # how much overlap it takes before a falloff is believed
NODE_PX = max(round(FIELD_PX / SEAM_FIELD_F), 1) * SEAM_FIELD_F         # the spacing actually used, a whole 1/f px
GLARE_LEVELS = 12.0                          # whiter than the other tile by this much in every channel (of 255): maybe glare


@dataclass
class Plan:
    """How the tiles are cut and matched in brightness."""

    f: int                                   # label map scale: 1 px = f mosaic px
    ext0: np.ndarray                         # mosaic coordinate of label pixel (0, 0)
    labels: np.ndarray                       # uint8, tile index or 255 = no data
    gains: np.ndarray                        # (n, 3) per-tile per-channel multipliers
    field: np.ndarray | None = None          # (n, gy, gx, 3) low-band correction to ADD, a node every NODE_PX
    vig: np.ndarray | None = None            # (3, 2) one falloff c1 r^2 + c2 r^4 shared by every scan

    def to_dict(self) -> dict:
        ok, png = cv2.imencode(".png", self.labels)
        d = {"f": self.f, "ext0": self.ext0.tolist(), "gains": self.gains.tolist(),
             "shape": list(self.labels.shape), "labels_png": png.tobytes().hex()}
        if self.field is not None:
            d["field"] = self.field.tolist()              # already rounded by ``fit_field``
        if self.vig is not None:
            d["vig"] = self.vig.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Plan":
        lab = cv2.imdecode(np.frombuffer(bytes.fromhex(d["labels_png"]), np.uint8), cv2.IMREAD_UNCHANGED)
        fld = d.get("field")                 # sets planned before the field existed were gain-only
        vg = d.get("vig")
        return cls(int(d["f"]), np.array(d["ext0"], float), lab, np.array(d["gains"], float),
                   None if fld is None else np.array(fld, float),
                   None if vg is None else np.array(vg, float))


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


def tile_radius(lay: Layout, k: int, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Distance of each source pixel from the middle of its own scan, 1.0 at the corner."""
    cx, cy = lay.centre(k)
    w, h = lay.sizes[k]
    return np.hypot(ux - cx, uy - cy) / (math.hypot(w - 1, h - 1) / 2.0)


def vig_factor(coef: np.ndarray, r: np.ndarray) -> np.ndarray:
    """The multiplier that takes ``coef``'s falloff back out: (h, w, 3) from (h, w)."""
    r2 = (r * r)[..., None]
    return np.exp(-(r2 * coef[:, 0] + r2 * r2 * coef[:, 1])).astype(np.float32)


def shade(lay: Layout, k: int, im: np.ndarray, gain, vig, frame: Frame, hw) -> np.ndarray:
    """Tile k's pixels with its gain and the shared falloff taken out -- what the cut and the blend see."""
    out = im * gain
    if vig is not None:
        out = out * vig_factor(vig, tile_radius(lay, k, *lay.tile_maps(k, frame, hw, cs=8)))
    return out


def fit_vignette(lay: Layout, maps, imgs, valids, anchor: int, sample: int = 30000) -> np.ndarray | None:
    """One falloff ``c1 r^2 + c2 r^4``, in log light, shared by every scan -- or None if there is none.

    A flatbed lamp is even and a sheet lies on the glass, so for a scan this fits zero and is dropped.
    A photograph and a microscope field both darken towards the corner, every frame the same way, and
    that is what makes it findable at all: where two frames overlap, one of them is near its middle and
    the other near its edge, so the overlap says how much darker the edge is.  A single curve shared by
    all of them is also the only version that is well determined -- a free field per frame would mostly
    fit the picture.

    Two frames are not enough.  Split one picture down the middle and the two halves meet mirror-image,
    each at the same radius as the other, so the overlap says nothing about radius at all and the fit
    takes whatever brightness difference it finds and blames it on an enormous falloff (measured: 400 %
    at the corner).  From three it has somewhere to stand.  The cap is the second guard: a real lens or
    lamp does not lose more than about a third of its light at the corner.

    ``maps[k]`` are tile k's source coordinates on the same low-resolution grid as ``imgs[k]``.
    """
    n = len(imgs)
    if n < 3:
        return None
    rows, rhs = [], []
    rng = np.random.default_rng(0)
    for a in range(n):
        for b in range(a + 1, n):
            m = valids[a] & valids[b]
            A_, B_ = imgs[a], imgs[b]
            m &= (A_.min(2) > 10) & (A_.max(2) < 245) & (B_.min(2) > 10) & (B_.max(2) < 245)
            idx = np.flatnonzero(m.ravel())
            if len(idx) < 500:
                continue
            if len(idx) > sample:
                idx = rng.choice(idx, sample, replace=False)
            ra = tile_radius(lay, a, maps[a][0].ravel()[idx], maps[a][1].ravel()[idx])
            rb = tile_radius(lay, b, maps[b][0].ravel()[idx], maps[b][1].ravel()[idx])
            row = np.zeros((len(idx), 2 + n))
            row[:, 0], row[:, 1] = ra ** 2 - rb ** 2, ra ** 4 - rb ** 4
            row[:, 2 + a], row[:, 2 + b] = 1.0, -1.0
            rows.append(row)
            rhs.append(np.log(A_.reshape(-1, 3)[idx]) - np.log(B_.reshape(-1, 3)[idx]))
    if not rows:
        return None
    M, R = np.vstack(rows), np.vstack(rhs)
    keep = [0, 1] + [2 + k for k in range(n) if k != anchor]          # the anchor's gain is the gauge
    M = M[:, keep]
    reg = np.zeros((2, M.shape[1]))             # a firm pull to no falloff at all: see VIG_RIDGE
    reg[0, 0] = reg[1, 1] = math.sqrt(len(M)) * VIG_RIDGE
    M = np.vstack([M, reg])
    R = np.vstack([R, np.zeros((2, 3))])
    lo = np.full(M.shape[1], -np.inf)
    hi = np.full(M.shape[1], np.inf)
    lo[:2], hi[:2] = -1.5, 0.0        # light falls off towards a corner, it does not rise: the free fit bends
    coef = np.array([lsq_linear(M, R[:, c], bounds=(lo, hi)).x[:2] for c in range(3)])    # (3, 2)
    if np.abs(coef.sum(1)).max() < 0.02:        # corner within 2 % of the middle: noise, and correcting it costs
        return None
    over = np.abs(coef.sum(1)).max() / VIG_MAX
    return coef / over if over > 1 else coef    # capped as a whole, so the channels keep their proportions


def fit_field(imgs, valids, f: int, hw) -> np.ndarray:
    """The low half of a two-band blend: what to ADD to each tile so the tiles' blurred versions agree.

    Blend detail across a seam and you soften it, which is why the cut is hard.  But a hard cut shows
    every slow difference between the two tiles as a step, and with Reposition there is one: the paper's
    bow is a few pixels, and a few pixels of shift in a gradient is a brightness difference.

    So the two bands are drawn differently.  Detail comes from whichever tile owns the pixel, as before;
    the slow part is shared out until the tiles agree.  Each PAIR is blurred over its own overlap and
    only there -- blur a tile over its own edge instead and the one-sided average of a gradient comes
    out biased, which on a real set was worth twice the difference it was meant to measure.  Then the
    pairs are settled against each other (damped Jacobi: a plain one swaps the two tiles round forever).

    The result is smooth, so it survives being kept at one value every ``NODE_PX``, and it is zero
    wherever the tiles already agree -- which is why flat paper on a flatbed does not notice it.
    ``imgs`` are 0..255 and the result is in 0..1, the units the renderer works in.
    """
    n = len(imgs)
    sig = max(FIELD_PX / f / 2.0, 1.0)
    step = max(int(round(FIELD_PX / f)), 1)                  # = NODE_PX mosaic px
    gy, gx = int(math.ceil(hw[0] / step)) + 1, int(math.ceil(hw[1] / step)) + 1
    iy = np.minimum(np.arange(gy) * step, hw[0] - 1)         # node (i, j) IS low-res pixel (i*step, j*step),
    ix = np.minimum(np.arange(gx) * step, hw[1] - 1)         # so ``Mosaic`` can find it again at NODE_PX
    blur = lambda a: cv2.GaussianBlur(a, (0, 0), sig, borderType=cv2.BORDER_REPLICATE)
    at = lambda a: a[np.ix_(iy, ix)]

    pairs = []
    for a in range(n):
        for b in range(a + 1, n):
            m = (valids[a] & valids[b]).astype(np.float32)
            if m.sum() < 50:
                continue
            sup = blur(m)
            d = blur((imgs[a] - imgs[b]) * m[..., None]) / (255.0 * np.maximum(sup, 1e-6)[..., None])
            pairs.append((a, b, at(np.clip(sup, 0.0, 1.0))[..., None], at(np.clip(d, -0.25, 0.25))))
    if not pairs:
        return np.zeros((n, gy, gx, 3))

    D = np.zeros((n, gy, gx, 3))
    den = np.zeros((n, gy, gx, 1))
    for a, b, w, _ in pairs:
        den[a] += w
        den[b] += w
    ok = den > 1e-6
    for _ in range(40):
        num = np.zeros_like(D)
        for a, b, w, d in pairs:
            num[a] += w * (D[b] - d)
            num[b] += w * (D[a] + d)
        D = 0.5 * D + 0.5 * np.where(ok, num / np.maximum(den, 1e-6), 0.0)
    D = np.clip(D * np.clip(den / 0.05, 0.0, 1.0), -0.25, 0.25)          # fades out where no pair reaches
    return np.round(D, 5)                    # the precision it is saved at, so an opened set renders identically


def _glare(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Where tile ``a`` shows glare that ``b`` does not, inside ``m`` (both valid; gains already matched).

    A reflection of the lamp in varnish adds white, the same amount in every channel, and it moves with
    the camera, so the next photograph has that paint clean.  A shadow multiplies instead.  So each patch
    where ``a`` is brighter in every channel is judged whole, by which of the two explains it better --
    ``b`` plus white, or ``b`` scaled -- and only glare is kept.  A misregistered edge differs too, but as a
    line a pixel or two wide, so a patch must be five pixels across to be judged at all.  On grey paint the two cannot be told
    apart and the patch is left alone: the low band (``fit_field``) is what takes out a broad shadow.
    """
    d = a - b
    cand = (m & (d.min(-1) > GLARE_LEVELS)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    if not cand.any():
        return cand > 0
    n, lab = cv2.connectedComponents(cand, connectivity=8)
    add = ((d - d.mean(-1, keepdims=True)) ** 2).sum(-1) + 1.0          # what is left after white, and a level of noise
    mul = (a * a).sum(-1) - (a * b).sum(-1) ** 2 / np.maximum((b * b).sum(-1), 1e-6)      # ... after scaling b
    glare = np.bincount(lab.ravel(), add.ravel(), n) < 0.5 * np.bincount(lab.ravel(), mul.ravel(), n)
    glare[0] = False
    return (cv2.dilate(glare[lab].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & m     # and its rim


def _unglared(imgs, valids) -> list[np.ndarray]:
    """Each tile's valid mask without the glare that another tile shows that paint free of.  The darkest
    tile at a pixel is never brighter than another, so every pixel keeps at least one tile."""
    out = [v.copy() for v in valids]
    for a in range(len(imgs)):
        for b in range(len(imgs)):
            m = valids[a] & valids[b]
            rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
            if a == b or len(rows) == 0:
                continue
            sl = np.s_[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
            out[a][sl] &= ~_glare(imgs[a][sl], imgs[b][sl], m[sl])
    return out


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
            vig=None, radius: int = 3) -> np.ndarray:
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
        imgs, vs = [], []
        for t in (i, j):
            im, v = render_low(lay, store, paths, t, f, fr, hw)
            imgs.append(shade(lay, t, im, gains[t], vig, fr, hw))
            vs.append(v)
        masks = [(((sub == t) | bnd) & u).astype(np.uint8) * 255 for t, u in zip((i, j), _unglared(imgs, vs))]   # glare judged whole, not in the band
        try:
            ri, rj = _cut(imgs, [(0, 0), (0, 0)], masks, "COST_COLOR_GRAD")
        except cv2.error:
            continue
        new = sub.copy()
        new[bnd & (ri > 0) & (rj == 0)] = i
        new[bnd & (rj > 0) & (ri == 0)] = j
        lab[y0:y1, x0:x1] = new
    return lab


def plan_seams(lay: Layout, store, paths, progress: Progress = _quiet, even: bool = True) -> Plan:
    """Cut the tiles apart along low-difference paths and match their brightness.

    ``even`` is the photometric half: the shared falloff and the low-band blend.  Both are nothing
    on scans that already agree, so it is on; off leaves every tile exactly as the scanner gave it
    but for one gain apiece, which is what the app did before either existed."""
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
    maps = [lay.tile_maps(k, frame, hw, cs=8) for k in range(n)]
    vig = fit_vignette(lay, maps, imgs, valids, lay.anchor) if even else None
    if vig is not None:
        imgs = [im * vig_factor(vig, tile_radius(lay, k, *maps[k])) for k, im in enumerate(imgs)]
    gains = fit_gains(lay, imgs, valids)
    imgs = [im * gains[k] for k, im in enumerate(imgs)]

    res = _cut(imgs, [(0, 0)] * n, [(v * 255).astype(np.uint8) for v in _unglared(imgs, valids)])
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
        lab = _refine(lay, store, paths, lab, f, ext, gains, vig)

    if not even:
        return Plan(f, ext[0].copy(), lab, gains)
    progress(0.95, "evening out the lighting")
    hw0 = level_shape(ext, SEAM_FIELD_F)
    fr0 = Frame(scale=1.0 / SEAM_FIELD_F, origin=ext[0])
    R = [render_low(lay, store, paths, k, SEAM_FIELD_F, fr0, hw0) for k in range(n)]
    low = [shade(lay, k, r[0], gains[k], vig, fr0, hw0) for k, r in enumerate(R)]
    fld = fit_field(low, _unglared(low, [r[1] for r in R]), SEAM_FIELD_F, hw0)      # glare is no slow difference
    return Plan(f, ext[0].copy(), lab, gains, fld, vig)


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

    def correction(self, k: int, frame: Frame, w: int, h: int) -> np.ndarray:
        """``fit_field``'s smooth low-band correction for tile k over a (h, w) window of ``frame``."""
        M = np.hstack([frame.A / NODE_PX, ((frame.b - self.plan.ext0) / NODE_PX)[:, None]]).astype(np.float64)
        return cv2.warpAffine(self.plan.field[k].astype(np.float32), M, (w, h),
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)

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
            if self.plan.vig is not None:
                px *= vig_factor(self.plan.vig, tile_radius(self.lay, k, ux, uy))
            if self.plan.field is not None:
                px = px + self.correction(k, sub, xb - xa, yb - ya)
            acc[ya:yb, xa:xb] += px * wc[..., None]
            wsum[ya:yb, xa:xb] += wc
            if each:
                s = wsum[ya:yb, xa:xb]
                each(k, xa, ya, px, valid, np.divide(wc, s, out=np.zeros_like(wc), where=s > 0))
        out = acc / np.maximum(wsum, 1e-6)[..., None]
        return np.clip(out, 0.0, 1.0), wsum > 1e-3
