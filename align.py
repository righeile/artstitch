"""Alignment: from N loose scans to poses + smooth meshes that make them agree.

Stages (each one narrows what the next has to search):

1. ``coarse_edges``  every pair, 1/32 scale: masked NCC over rotation and
   translation (the rotation search is what lets a crooked scan connect), over
   the whole range or, given a rough hand placement, around what it implies.
2. ``solve_rigid``   poses from those pairwise results (max spanning tree, then
   a robust least squares over every edge); the best-connected tile is the
   anchor, and the mosaic stands the way it does.  Its pixels are still bent
   with everyone else's: a scan is not flat, and the one that happens to be the
   anchor has no better claim to the artwork's shape than the rest.
3. ``refine``        1/16 -> 1/2 scale: dense patch matching between the
   renditions of every overlapping pair, then one sparse least squares for the
   per-tile displacement meshes.  Repeated, because each solve is linear but the
   renditions it measured moved when it was applied.

The mesh update is applied on the mosaic plane in place of the tile plane it
really lives on; the difference is (gradient of the update) x (size of the mesh
already there), second order, and the next iteration measures it away.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import cv2
import numpy as np
import scipy.sparse as sp
from scipy.optimize import least_squares
from scipy.sparse.linalg import splu

from geometry import Frame, Layout, rot
from ncc import fast_shape, high_pass, masked_ncc_map, pack_fft, peak

Progress = Callable[[float, str], None]
_quiet: Progress = lambda p, m: None

MARGIN_PX = 12          # scanner edges carry shadow lines; nothing is matched or blended there


def _hp(store, path, f: int, sigma: float) -> np.ndarray:
    t = store.get(path)
    key = (f, "hp", sigma)
    if key not in t.proxies:
        t.proxies[key] = high_pass(store.proxy(path, f, gray=True), sigma)
    return t.proxies[key]


# -----------------------------------------------------------------------------
# 1. coarse pairwise search
# -----------------------------------------------------------------------------

def _rotated(img: np.ndarray, phi: float) -> np.ndarray:
    h, w = img.shape
    c = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    R = rot(phi)
    M = np.hstack([R, (c - R @ c)[:, None]]).astype(np.float32)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


def _quality(ncc: float, ratio: float, overlap: float) -> float:
    a = np.clip((ncc - 0.25) / 0.5, 0, 1)
    b = np.clip((ratio - 1.0) / 0.6, 0, 1)
    c = np.clip(overlap / 0.30, 0, 1)
    return float(a * (0.4 + 0.6 * b) * (0.5 + 0.5 * c))


def coarse_pair(sa: np.ndarray, sb: np.ndarray, angles: np.ndarray, min_frac: float = 0.05,
                near: tuple | None = None, ma: np.ndarray | None = None, mb: np.ndarray | None = None,
                unit: float | None = None) -> dict | None:
    """Best (rotation, shift) of B against A at one scale; B rotates about its center.

    ``near`` = ((tx, ty), r) confines the shift to within r of (tx, ty), the way a hand
    placement implies it.  Such a hint may ask for a big turn, so B then sits on a canvas
    that holds it at every angle; the shift returned is still that of B's own center.
    ``ma``/``mb`` say where A and B have content (all of them if not given); ``unit`` is the
    area overlaps are counted in (the smaller picture if not given).
    """
    ha, wa = sa.shape
    hb, wb = sb.shape
    ox = oy = 0
    if near:
        c, s = np.abs(np.cos(angles)), np.abs(np.sin(angles))
        ox = max(0, math.ceil(((wb * c + hb * s).max() - wb) / 2)) + 1
        oy = max(0, math.ceil(((wb * s + hb * c).max() - hb) / 2)) + 1
    sb0 = np.pad(sb, ((oy, oy), (ox, ox)))
    mb0 = np.pad(np.ones(sb.shape, np.float32) if mb is None else mb.astype(np.float32), ((oy, oy), (ox, ox)))
    # every shift with an overlap must fall in the half of the transform its sign says it does
    fshape = fast_shape(2 * max(ha, sb0.shape[0]), 2 * max(wa, sb0.shape[1]))
    pa = pack_fft(sa, np.ones(sa.shape, bool) if ma is None else ma, fshape, workers=1)
    min_px = unit or min(ha * wa, hb * wb)
    if near:
        limit, centre = (near[1], near[1]), (near[0][0] - ox, near[0][1] - oy)
    else:
        limit, centre = (0.97 * max(wa, wb), 0.97 * max(ha, hb)), (0.0, 0.0)

    def at(phi: float):
        turned = abs(phi) > 1e-9
        sb_r = _rotated(sb0, phi) if turned else sb0
        mb = _rotated(mb0, phi) > 0.99 if turned else mb0 > 0.5
        ncc, cnt = masked_ncc_map(pa, pack_fft(sb_r, mb, fshape, workers=1), fshape,
                                  min_overlap=int(min_frac * min_px), workers=1)
        pk = peak(ncc, cnt, fshape, limit, saturation=0.25 * min_px, centre=centre)
        if pk:
            pk["tx"] += ox
            pk["ty"] += oy
            pk["overlap"] = pk["n"] / min_px
            pk["q"] = _quality(pk["ncc"], pk["ratio"], pk["overlap"])
            pk["phi"] = float(phi)
        return pk

    res = [at(float(a)) for a in angles]
    ok = [r for r in res if r]
    if not ok:
        return None
    best = max(ok, key=lambda r: r["q"])
    k = int(np.argmin([abs(r["phi"] - best["phi"]) if r else 1e9 for r in res]))
    if 0 < k < len(res) - 1 and res[k - 1] and res[k + 1]:      # parabolic refinement of the angle
        m, c, p = res[k - 1]["q"], res[k]["q"], res[k + 1]["q"]
        d = m - 2 * c + p
        if d < -1e-9:
            phi = best["phi"] + float(np.clip(0.5 * (m - p) / d, -0.5, 0.5)) * (angles[1] - angles[0])
            if abs(phi - best["phi"]) > 1e-6:
                r2 = at(phi)
                if r2 and r2["q"] >= best["q"] * 0.98:
                    best = r2
    return best


def _good(e: dict) -> bool:
    """A match worth building poses from (a coarse pair result or its edge).  Unrelated sheets still find a peak
    (real scans: ncc .37, ratio 1.13, q .13: a smooth lawn against a smooth lawn); true overlaps score q of .6 and up."""
    return e["q"] >= 0.25 and e["ratio"] >= 1.10 and e["overlap"] >= 0.05


#: A match no longer worth doubting.  True overlaps score q of .6 and up and false ones .43 and down on
#: every real set measured; the bar sits between them, well clear of ``_good``'s deliberately low one.
SURE = 0.5


def _best_per_pair(edges: list[dict]) -> list[dict]:
    """The good edges, one per pair: a pair searched again at the other quarter turns has two."""
    best: dict[tuple[int, int], dict] = {}
    for e in edges:
        k = (e["i"], e["j"])
        if _good(e) and e["q"] > best.get(k, {"q": -1.0})["q"]:
            best[k] = e
    return list(best.values())


def _turn(lay: Layout, i: int, j: int) -> float:
    """How far tile j is turned against tile i in a placement, in (-pi, pi]."""
    return (lay.theta[j] - lay.theta[i] + math.pi) % (2 * math.pi) - math.pi


def coarse_edges(store, paths, f: int = 32, max_deg: float = 3.0, step_deg: float = 0.5,
                 workers: int = 4, progress: Progress = _quiet, hint: Layout | None = None,
                 pairs: list[tuple[int, int]] | None = None, anywhere: bool = False,
                 turns: tuple[int, ...] = (0,), tilts=None) -> list[dict]:
    """Every pair (or just ``pairs``), rotation and translation.  Edge maps tile j into tile i's pixels:
    u_i = R(phi) (u_j - C_j) + C_j + f * t   with C_j the center of j's proxy in source px.

    ``hint`` is a rough hand placement.  Its turns say which angles to try for a pair; where to look
    is what the placement implies (a pair it puts apart is skipped), or, with ``anywhere``, the whole sheet.
    ``turns`` are the quarter turns tried on top (a scan that went into the scanner sideways or upside
    down); the best of them is kept.  Any turn but the plain one searches the whole sheet.

    ``tilts`` (radians, one per scan) are how the paper lay on the glass, as the scanner itself knows it.
    They say nothing about where a pair overlaps -- the sheet is moved between scans -- only how far one
    scan is turned against another, so the search runs about ``tilts[j] - tilts[i]`` instead of about 0.
    A hint is a whole hand placement and says this itself, so it wins.
    """
    n = len(paths)
    sig = [_hp(store, p, f, 2.0) for p in paths]
    angles = np.radians(np.arange(-max_deg, max_deg + 1e-9, step_deg))
    if pairs is None:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if hint is not None and not anywhere:
        together = set(overlapping_pairs(hint, 0.0))
        pairs = [p for p in pairs if p in together]
    done = [0]

    def one(ij):
        i, j = ij
        hp_, wp_ = sig[j].shape
        Cj = f * np.array([(wp_ - 1) / 2.0, (hp_ - 1) / 2.0]) + (f - 1) / 2.0
        phi0 = (_turn(hint, i, j) if hint is not None else
                0.0 if tilts is None else float(tilts[j] - tilts[i]))
        if hint is None and turns == (0,):
            r = coarse_pair(sig[i], sig[j], phi0 + angles)
        else:
            if anywhere or turns != (0,) or hint is None:
                near = ((0.0, 0.0), 0.97 * max(*sig[i].shape, *sig[j].shape))
            else:       # u_i = R(phi0)(u_j - c_j) + R(-theta_i)(o_j - o_i) + c_i, written about C_j
                t0 = (rot(-hint.theta[i]) @ (hint.o[j] - hint.o[i]) + hint.centre(i)
                      - rot(phi0) @ (hint.centre(j) - Cj) - Cj)
                near = (t0 / f, 0.2 * min(*sig[i].shape, *sig[j].shape))
            rs = [coarse_pair(sig[i], sig[j], phi0 + q * math.pi / 2 + angles, near=near) for q in turns]
            r = max((x for x in rs if x), key=lambda x: x["q"], default=None)
        done[0] += 1
        progress(0.02 + 0.18 * done[0] / len(pairs), f"pair {i + 1}-{j + 1}")
        if r is None:
            return None
        return {"i": i, "j": j, "phi": r["phi"], "t": f * np.array([r["tx"], r["ty"]]), "C": Cj,
                "ncc": r["ncc"], "ratio": r["ratio"], "overlap": r["overlap"], "q": r["q"]}

    with ThreadPoolExecutor(workers) as ex:
        return [e for e in ex.map(one, pairs) if e]


# -----------------------------------------------------------------------------
# 2. poses
# -----------------------------------------------------------------------------

def _edge_samples(lay: Layout, e: dict, nx: int = 7, ny: int = 9):
    """Correspondences (u_i, u_j) an edge implies, on a grid of tile j that lands inside tile i."""
    wj, hj = lay.sizes[e["j"]]
    wi, hi = lay.sizes[e["i"]]
    xs, ys = np.meshgrid(np.linspace(0, wj - 1, nx), np.linspace(0, hj - 1, ny))
    Uj = np.c_[xs.ravel(), ys.ravel()]
    Ui = (Uj - e["C"]) @ rot(e["phi"]).T + e["C"] + e["t"]
    ok = (Ui[:, 0] >= 0) & (Ui[:, 0] <= wi - 1) & (Ui[:, 1] >= 0) & (Ui[:, 1] <= hi - 1)
    return Ui[ok], Uj[ok]


def solve_rigid(lay: Layout, corrs: list[tuple], edges_for_init: list[dict] | None = None) -> None:
    """Fit theta, o of every tile but the anchor to correspondences (i, j, U_i, U_j, w).

    Robust (soft-L1); initial values come from the max spanning tree of
    ``edges_for_init`` when given, otherwise from ``lay`` as it is.  Scans nothing ties to the anchor make
    groups of their own, each fit about its first scan, which keeps its pose: the group is right within itself
    and cannot drift against the rest.  A tile the tree does not reach keeps the pose ``lay`` had.
    """
    n = len(lay.sizes)
    a = lay.anchor
    comp = connected(n, [{"i": i, "j": j} for i, j, *_ in corrs])
    root: dict = {}
    for k in range(n):
        root.setdefault(comp[k], k)
    root[comp[a]] = a
    fixed = set(root.values())
    if edges_for_init:
        seen = set(fixed)
        todo = sorted(edges_for_init, key=lambda e: -e["q"])
        while len(seen) < n:
            for e in todo:
                i, j = e["i"], e["j"]
                if (i in seen) != (j in seen):
                    if i in seen:                       # place j from i
                        R = rot(lay.theta[i])
                        lay.theta[j] = lay.theta[i] + e["phi"]
                        lay.o[j] = R @ (rot(e["phi"]) @ (lay.centre(j) - e["C"]) + e["C"] + e["t"] - lay.centre(i)) + lay.o[i]
                        seen.add(j)
                    else:                               # place i from j: invert the edge
                        phi = -e["phi"]
                        lay.theta[i] = lay.theta[j] + phi
                        # u_j = R(-phi)(u_i - C - t) + C   ->  center of i in j's pixels
                        cj = rot(-e["phi"]) @ (lay.centre(i) - e["C"] - e["t"]) + e["C"]
                        lay.o[i] = rot(lay.theta[j]) @ (cj - lay.centre(j)) + lay.o[j]
                        seen.add(i)
                    break
            else:
                break                                   # disconnected: the rest keep their poses
    free = [k for k in range(n) if k not in fixed]
    if not free or not corrs:
        return
    col = {k: 3 * m for m, k in enumerate(free)}

    def unpack(x):
        th, o = lay.theta.copy(), lay.o.copy()
        for k in free:
            th[k], o[k] = x[col[k]], x[col[k] + 1:col[k] + 3]
        return th, o

    def resid(x):
        th, o = unpack(x)
        out = []
        for i, j, Ui, Uj, w in corrs:
            Xi = (Ui - lay.centre(i)) @ rot(th[i]).T + o[i]
            Xj = (Uj - lay.centre(j)) @ rot(th[j]).T + o[j]
            out.append(((Xi - Xj) * math.sqrt(w)).ravel())
        return np.concatenate(out)

    x0 = np.concatenate([[lay.theta[k], *lay.o[k]] for k in free])
    scale = np.tile([1e-2, 30.0, 30.0], len(free))
    sol = least_squares(resid, x0, loss="soft_l1", f_scale=4.0, x_scale=scale)
    th, o = unpack(sol.x)
    lay.theta[:], lay.o[:] = th, o


def _apart(lay: Layout, groups: list[list[int]], gap: float = 600.0) -> None:
    """Groups of scans that nothing matched to each other start piled at the origin; put them side by side, the
    anchor's first, so they read as separate pieces.  ponytail: one row in scan order; "Place scans by hand" does better."""
    right = None
    for ks in sorted(groups, key=lambda g: (lay.anchor not in g, g[0])):
        box = np.array([lay.bbox(k) for k in ks])
        lo, hi = box[:, 0, 0].min(), box[:, 1, 0].max()
        if right is not None:
            lay.o[ks, 0] += right + gap - lo
            hi += right + gap - lo
        right = hi


def _group_picture(lay: Layout, store, paths, ks: list[int], f: int):
    """Scans ``ks`` drawn together as posed, high-passed, at 1/f: (signal, where it has content, mosaic origin)."""
    box = np.array([lay.bbox(k) for k in ks])
    lo, hi = box[:, 0].min(0), box[:, 1].max(0)
    hw = (int(math.ceil((hi[1] - lo[1]) / f)) + 1, int(math.ceil((hi[0] - lo[0]) / f)) + 1)
    fr = Frame(scale=1.0 / f, origin=lo)
    acc, cnt = np.zeros(hw, np.float32), np.zeros(hw, np.float32)
    for k in ks:
        img, v = render_hp(lay, store, paths, k, f, fr, hw, 2.0)
        acc += img * v
        cnt += v
    return acc / np.maximum(cnt, 1.0), cnt > 0, lo


def join_groups(lay: Layout, store, paths, groups: list[list[int]], max_deg: float = 3.0, f: int = 32,
                progress: Progress = _quiet) -> list[list[int]]:
    """Groups that no single pair of scans ties together, matched as wholes: each group drawn as one picture, every
    quarter turn, anywhere on the sheet (stitch the pairs, then the pairs together, as it helps by hand).  The best
    match moves one group onto the other, and again until nothing more matches; the anchor's group stays put.
    Returns the groups left."""
    angles = np.radians(np.arange(-max_deg, max_deg + 1e-9, 0.5))
    unit = min(float(np.prod(_hp(store, p, f, 2.0).shape)) for p in paths)    # overlap counts in scans, as for a pair
    groups = sorted((sorted(g) for g in groups), key=lambda g: (lay.anchor not in g, g[0]))
    pics: dict = {}
    while len(groups) > 1:
        best = None
        for x in range(len(groups)):
            for y in range(x + 1, len(groups)):
                for g in (groups[x], groups[y]):
                    if tuple(g) not in pics:
                        pics[tuple(g)] = _group_picture(lay, store, paths, g, f)
                (sa, ma, loa), (sb, mb, lob) = pics[tuple(groups[x])], pics[tuple(groups[y])]
                near = ((0.0, 0.0), 0.97 * max(*sa.shape, *sb.shape))
                progress(0.22, "scans " + ", ".join(str(k + 1) for k in groups[x]) + " against "
                         + ", ".join(str(k + 1) for k in groups[y]))
                with ThreadPoolExecutor(4) as ex:
                    rs = list(ex.map(lambda q: coarse_pair(sa, sb, q * math.pi / 2 + angles, near=near,
                                                           ma=ma, mb=mb, unit=unit), range(4)))
                for r in rs:
                    if r and _good(r) and (best is None or r["q"] > best[0]["q"]):
                        best = (r, x, y, loa, lob, sb.shape)
        if best is None:
            break
        r, x, y, loa, lob, (hb, wb) = best
        cb = np.array([(wb - 1) / 2.0, (hb - 1) / 2.0])
        P, Q, R = lob + f * cb, loa + f * (cb + [r["tx"], r["ty"]]), rot(r["phi"])   # X_x = R (X_y - P) + Q
        for k in groups[y]:
            lay.theta[k] += r["phi"]
            lay.o[k] = R @ (lay.o[k] - P) + Q
            lay.reset_mesh(k)
        groups[x] = sorted(groups[x] + groups.pop(y))
    return groups


def pick_anchor(n: int, edges: list[dict], among=None) -> int:
    """The best-connected scan (of ``among``, if given)."""
    score = np.zeros(n)
    for e in edges:
        score[e["i"]] += e["q"]
        score[e["j"]] += e["q"]
    if among is not None:
        score[[k for k in range(n) if k not in among]] = -np.inf
    return int(np.argmax(score))


def quarter_turns(n: int, edges: list[dict], start: int = 0) -> list:
    """Quarter turns (0-3) of every scan against ``start``, along the matches; None where none lead."""
    q: list = [None] * n
    q[start] = 0
    todo = [start]
    while todo:
        i = todo.pop()
        for e in edges:
            for a, b, s in ((e["i"], e["j"], 1), (e["j"], e["i"], -1)):
                if a == i and q[b] is None:
                    q[b] = (q[a] + s * round(e["phi"] / (math.pi / 2))) % 4
                    todo.append(b)
    return q


def connected(n: int, edges: list[dict]) -> list[int]:
    """Component label per tile."""
    lab = list(range(n))

    def find(x):
        while lab[x] != x:
            lab[x] = lab[lab[x]]
            x = lab[x]
        return x

    for e in edges:
        lab[find(e["i"])] = find(e["j"])
    return [find(k) for k in range(n)]


# -----------------------------------------------------------------------------
# 3. dense matching and the mesh solve
# -----------------------------------------------------------------------------

def render_hp(lay: Layout, store, paths, k: int, f: int, frame: Frame, hw, sigma: float):
    """Tile k's high-passed gray proxy drawn into ``frame`` (scale 1/f), and its validity."""
    ux, uy = lay.tile_maps(k, frame, hw, cs=8)
    src = _hp(store, paths[k], f, sigma)
    img = cv2.remap(src, ((ux - (f - 1) / 2) / f).astype(np.float32),
                    ((uy - (f - 1) / 2) / f).astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, lay.valid_from_maps(k, ux, uy, margin=max(MARGIN_PX, f))


def _par(m, c, p):
    d = m - 2 * c + p
    return 0.0 if abs(d) < 1e-9 else float(np.clip(0.5 * (m - p) / d, -0.75, 0.75))


def dense_pair(lay: Layout, store, paths, a: int, b: int, f: int, T: int, S: int,
               sigma: float, min_std: float = 0.8, min_ncc: float = 0.3):
    """Patch displacement of b's rendition against a's.  Returns (X, delta, ncc) or None.

    X: patch centers in mosaic px.  delta: where a's content at X sits in b, relative
    to X, in full-resolution px.
    """
    ba, bb = lay.bbox(a), lay.bbox(b)
    lo, hi = np.maximum(ba[0], bb[0]), np.minimum(ba[1], bb[1])
    span = (hi - lo) / f
    W_, H_ = int(span[0]), int(span[1])
    if W_ < T + 2 * S + 4 or H_ < T + 2 * S + 4:
        return None
    fr = Frame(scale=1.0 / f, origin=lo)
    A, va = render_hp(lay, store, paths, a, f, fr, (H_, W_), sigma)
    B, vb = render_hp(lay, store, paths, b, f, fr, (H_, W_), sigma)
    tpl_ok = cv2.boxFilter(va.astype(np.float32), -1, (T, T), normalize=True, borderType=cv2.BORDER_CONSTANT)
    win_ok = cv2.boxFilter(vb.astype(np.float32), -1, (T + 2 * S, T + 2 * S), normalize=True,
                           borderType=cv2.BORDER_CONSTANT)
    step = max(T // 2, 8)
    out = []
    for y0 in range(S, H_ - T - S, step):
        for x0 in range(S, W_ - T - S, step):
            cy, cx = y0 + T // 2, x0 + T // 2
            if tpl_ok[cy, cx] < 0.999 or win_ok[cy, cx] < 0.999:
                continue
            tpl = A[y0:y0 + T, x0:x0 + T]
            if float(tpl.std()) < min_std:
                continue
            r = cv2.matchTemplate(B[y0 - S:y0 + T + S, x0 - S:x0 + T + S], tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, (px, py) = cv2.minMaxLoc(r)
            if mx < min_ncc or px in (0, r.shape[1] - 1) or py in (0, r.shape[0] - 1):
                continue                                    # on the search border: the true shift may lie outside
            sx = _par(r[py, px - 1], r[py, px], r[py, px + 1])
            sy = _par(r[py - 1, px], r[py, px], r[py + 1, px])
            out.append((x0 + (T - 1) / 2.0, y0 + (T - 1) / 2.0, px + sx - S, py + sy - S, mx))
    if not out:
        return None
    o = np.array(out)
    X = fr.to_mosaic(o[:, :2])
    return X, o[:, 2:4] * f, o[:, 4]


#: How dearly the mesh holds its scan's shape.  The paper is flat and stiff: shifting, turning and
#: stretching a scan is free, and bending it is not.  Curvature alone prices only how sharply it bends,
#: not how far, and over a scan a gentle bow reaches millimetres for almost nothing -- so what a bend
#: costs is its own size, away from the affine part.  Past 3e-2 the seams themselves start to open.
STIFF = 1e-2

def _bending(gy: int, gx: int):
    """Thin plate: d2/dx2, d2/dy2 and the twist d2/dxdy, which weighs as much as the two of them.

    Leave the twist out and a tile may be wrung into a saddle -- nothing along its middle, growing to
    every corner -- for free, which is just what a mesh does where no match holds it.  What stays free
    is what should: the affine part, so the scan may still be shifted, turned and stretched.
    """
    d1 = lambda m: sp.diags([-1.0, 1.0], [0, 1], shape=(m - 1, m))
    d2 = lambda m: sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(m - 2, m))
    return sp.vstack([sp.kron(sp.identity(gy), d2(gx)), sp.kron(d2(gy), sp.identity(gx)),
                      math.sqrt(2.0) * sp.kron(d1(gy), d1(gx))])


def _unaffine(gy: int, gx: int):
    """How far a mesh's nodes stand from the affine field that fits them best, as a matrix.

    A scan may be shifted, turned and stretched as freely as this says nothing: those are the affine
    fields, and the scanner is not exact.  Everything else is the paper bending, and this is its size.
    """
    y, x = np.mgrid[0:gy, 0:gx]
    M = np.c_[np.ones(gy * gx), x.ravel() - (gx - 1) / 2.0, y.ravel() - (gy - 1) / 2.0]
    return sp.csr_matrix(np.eye(gy * gx) - M @ np.linalg.pinv(M))


#: What holds the meshes' one free direction.  With no scan pinned, adding the same field to every
#: mesh leaves every pair exactly where it was, and this decides which of those the solve picks: the
#: smallest, so the bend they share goes to nothing.  It is a numerical floor as much as a choice --
#: under 1e-5 the normal equations are singular to the factorisation and a weakly tied pair wanders
#: tens of pixels; over 1e-2 it starts damping the bend the matches did ask for.
GAUGE = 3e-4


def solve_mesh(lay: Layout, sets: list[tuple], f: int, sigma_c: float = 3.0, robust: int = 3) -> None:
    """One sparse least squares for the meshes' update, added to ``lay.D``.

    ``sets`` are (a, b, X, delta, ncc).  Row: dD_a(X) - dD_b(X) = delta.
    """
    # Every scan's mesh moves and none is held at zero.  A sheet bows a little differently every time,
    # so pinning one would hand the artwork that scan's own bow -- which the seams, agreeing with each
    # other to a fraction of a pixel, would never show.  Left this way, what the meshes have in common
    # is what ``STIFF`` prices, so the bend they share goes to nothing and the artwork comes out the
    # average of the scans rather than a copy of whichever one happened to be the anchor.
    free = list(range(len(lay.sizes)))
    off, N = {}, 0
    for k in free:
        off[k] = N
        N += lay.D[k].shape[0] * lay.D[k].shape[1]
    if not sets or N == 0:
        return
    rows, cols, vals, rhs, wts, src = [], [], [], [], [], []
    r0 = 0
    for si, (a, b, X, delta, ncc) in enumerate(sets):
        m = len(X)
        w = ncc ** 2 / (0.15 * f) ** 2
        for k, sign in ((a, 1.0), (b, -1.0)):
            idx, wgt = lay.weights(k, X)
            rows.append(np.repeat(np.arange(r0, r0 + m), 4))
            cols.append((off[k] + idx).ravel())
            vals.append((sign * wgt).ravel())
        rhs.append(delta)
        wts.append(w)
        src.append(np.full(m, si))
        r0 += m
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r0, N))
    rhs = np.vstack(rhs)
    w = np.concatenate(wts)

    L = sp.block_diag([_bending(*lay.D[k].shape[:2]) for k in free], format="csr") / sigma_c
    R = sp.block_diag([_unaffine(*lay.D[k].shape[:2]) for k in free], format="csr")
    reg = L.T @ L + (R.T @ R) * STIFF + sp.identity(N) * GAUGE

    rob = np.ones(r0)
    for _ in range(robust):
        Wd = sp.diags(w * rob)
        x = splu((A.T @ Wd @ A + reg).tocsc()).solve(A.T @ Wd @ rhs)
        res = np.hypot(*(A @ x - rhs).T)
        c = max(1.5 * f * 0.15 * 3, 1.0)                        # ~3 sigma of the matcher at this scale
        rob = 1.0 / (1.0 + (res / c) ** 2)
    for k in free:
        gy, gx = lay.D[k].shape[:2]
        lay.D[k] += x[off[k]:off[k] + gy * gx].reshape(gy, gx, 2)


def pair_residuals(sets: list[tuple], lay: Layout) -> dict:
    """RMS full-resolution residual of each pair's matches (what is left to fix)."""
    out = {}
    for a, b, X, delta, ncc in sets:
        r = np.hypot(delta[:, 0], delta[:, 1])
        out[(a, b)] = {"rms": float(np.sqrt(np.average(r ** 2, weights=ncc ** 2))),
                       "median": float(np.median(r)), "n": int(len(r))}
    return out


LEVELS = [(16, 32, 16, 3), (8, 32, 8, 2), (4, 32, 6, 1), (2, 48, 4, 1)]      # f, patch, search, iterations
SETTLED_PX = 1.0        # the last level looks again (twice at most) while a pair is still this far off


def overlapping_pairs(lay: Layout, min_frac: float = 0.04) -> list[tuple[int, int]]:
    n = len(lay.sizes)
    bb = [lay.bbox(k) for k in range(n)]
    out = []
    for a in range(n):
        for b in range(a + 1, n):
            lo, hi = np.maximum(bb[a][0], bb[b][0]), np.minimum(bb[a][1], bb[b][1])
            if (hi - lo).min() <= 0:
                continue
            area = float(np.prod(hi - lo))
            small = min(np.prod(bb[a][1] - bb[a][0]), np.prod(bb[b][1] - bb[b][0]))
            if area >= min_frac * small:
                out.append((a, b))
    return out


def refine(lay: Layout, store, paths, levels=LEVELS, progress: Progress = _quiet, workers: int = 6) -> dict:
    """Coarse to fine dense matching + mesh solves.  Returns the final pair residuals.

    A pair that first overlaps enough to be matched at a fine level (a narrow strip) can still be far off when the
    last level looks, and what it reports is that look: then it looks again after the solve."""
    total = sum(it for *_, it in levels)
    step_i = 0
    last: dict = {}
    for li, (f, T, S, iters) in enumerate(levels):
        sigma = 4.0 if f >= 4 else 6.0
        it = 0
        while it < iters or (li == len(levels) - 1 and it < iters + 2
                             and max((v["rms"] for v in last.values()), default=0.0) > SETTLED_PX):
            pairs = overlapping_pairs(lay)
            with ThreadPoolExecutor(workers) as ex:
                res = list(ex.map(lambda ab: dense_pair(lay, store, paths, ab[0], ab[1], f, T, S, sigma), pairs))
            sets = [(a, b, *r) for (a, b), r in zip(pairs, res) if r is not None]
            last = pair_residuals(sets, lay)
            step_i += 1
            it += 1
            progress(0.25 + 0.7 * min(step_i, total) / total, f"1/{f}: " + ", ".join(
                f"{a + 1}-{b + 1} {v['rms']:.1f}px" for (a, b), v in last.items()))
            if f == levels[0][0] and it == 1:                    # the first look also refits the rigid poses
                # rigid refit on the dense matches: X (as seen) + delta is where b's content really sits
                corrs = []
                for a, b, X, delta, ncc in sets:
                    Ua = lay.inv(a, X)
                    Ub = lay.inv(b, X + delta)
                    corrs.append((a, b, Ua, Ub, float(np.mean(ncc ** 2)) / 25.0))
                solve_rigid(lay, corrs)
                for k in range(len(lay.sizes)):
                    lay.reset_mesh(k)
                continue
            solve_mesh(lay, sets, f)
    return last


def hand_layout(sizes, poses) -> Layout:
    """Rough hand placements [{x, y, theta, anchor}] as a Layout.  The anchor keeps its pixels (a tile
    turned more than 45 degrees cannot), so the whole placement is turned and moved to put it upright at home."""
    th = (np.radians([p["theta"] for p in poses]) + np.pi) % (2 * np.pi) - np.pi
    lay = Layout(sizes, th, [[p["x"], p["y"]] for p in poses])
    flag = next((k for k, p in enumerate(poses) if p.get("anchor")), 0)
    a = flag if abs(th[flag]) < np.pi / 4 else int(np.argmin(np.abs(th)))
    lay.anchor = a
    lay.o[:] = (lay.o - lay.o[a]) @ rot(th[a]) + lay.centre(a)
    lay.theta[:] = th - th[a]
    for k in range(len(sizes)):
        lay.reset_mesh(k)
    return lay


def trim_sides(lay: Layout, px: np.ndarray) -> None:
    """Cut ``px`` off each scan on the sides where the sheet ran off the glass.

    A sheet larger than the glass lifts where it carries on over the edge, and the picture bends
    with it.  Those are the sides another scan covers -- that is what makes them a seam -- so the
    strip is cut and the neighbour supplies those pixels instead.  A side where the artwork's own
    edge lay on the glass has no scan standing behind it: it lay flat, and keeps every pixel.

    Never more than a third of how far a neighbour on that side reaches in, and the shallowest
    neighbour sets it: the strip two scans have in common is what they are matched on, and where
    that is a sliver (two pairs joined corner to corner, say) cutting it costs more than the bend.
    """
    for a, b in overlapping_pairs(lay):
        for k, j in ((a, b), (b, a)):
            d = (lay.o[j] - lay.o[k]) @ rot(lay.theta[k])            # the neighbour, in k's own frame
            s = (1 if d[0] > 0 else 0) if abs(d[0]) > abs(d[1]) else (3 if d[1] > 0 else 2)
            w, h = lay.sizes[k]
            P = lay.rigid_inv(k, lay.rigid(j, lay.corners(j)))        # the neighbour's own corners, in k's frame
            lo, hi = np.maximum(P.min(0), 0), np.minimum(P.max(0), [w - 1, h - 1])
            deep = max(hi[s // 2] - lo[s // 2], 0.0)                  # how far this neighbour reaches in, px
            cut = min(px[k], deep / 3.0)                              # left, right, top, bottom
            lay.trim[k, s] = cut if lay.trim[k, s] == 0 else min(lay.trim[k, s], cut)


def align_set(store, paths, progress: Progress = _quiet, poses=None, max_deg: float = 3.0,
              tilts=None, trim=None) -> tuple[Layout, dict]:
    """Paths of one artwork's scans -> (Layout, report).

    ``poses`` are rough hand placements: they say where to look (``coarse_edges``) and are what
    a tile keeps if nothing can be matched to it; without them the whole range is searched.
    ``tilts`` (degrees, one per scan) are how the paper lay on the glass: the rotation search starts
    there instead of at 0, so what it still has to find is only what the measurement got wrong.
    ``trim`` is the strip (source px, one per scan) cut off the sides where the sheet ran off the
    glass and the picture bends with it -- the sides another scan covers, so nothing is lost.  It
    is decided once the poses are known, and holds for the mesh, the seams and the render.
    """
    n = len(paths)
    for k, p in enumerate(paths):
        store.get(p)
    sizes = [(store.get(p).shape[1], store.get(p).shape[0]) for p in paths]
    lay = Layout(sizes)
    report: dict = {"edges": [], "tiles": n}
    rads = None if tilts is None else np.radians(np.asarray(tilts, float).reshape(n))
    if n == 1:
        return lay, report
    hint = hand_layout(sizes, poses) if poses is not None else None
    progress(0.01, "coarse search")
    edges = coarse_edges(store, paths, max_deg=max_deg, progress=progress, hint=hint, tilts=rads)
    good = _best_per_pair(edges)
    # Whether anything is still missing is asked of the matches that are plainly real, not of every one
    # that cleared the bar: a marginal false pair is enough to tie two halves of a set together, and then
    # a scan that went on the glass upside down is never looked for.  Both are cheap to ask again about.
    sure = connected(n, [e for e in good if e["q"] >= SURE])
    if len(set(sure)) > 1:          # not all tied together: look everywhere, and at the other quarter turns
        rest = [(i, j) for i in range(n) for j in range(i + 1, n) if sure[i] != sure[j]]
        more = coarse_edges(store, paths, max_deg=max_deg, progress=lambda p, m: progress(0.2, m), hint=hint,
                            pairs=rest, anywhere=True, turns=(1, 2, 3) if hint is None else (0, 1, 2, 3),
                            tilts=rads)
        edges, good = edges + more, _best_per_pair(edges + more)
    comp = connected(n, good)
    report["edges"] = [{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in e.items()} for e in edges]
    if hint is None:                # upright the way scan 1 is: the anchor is the best-connected scan turned like it
        turned = quarter_turns(n, good)
        lay.anchor = pick_anchor(n, good, among=[k for k in range(n) if turned[k] == 0])
    else:                                                        # ponytail: a poor match never outvotes the hand; QC names tiles nothing matched
        lay.anchor, lay.theta[:], lay.o[:] = hint.anchor, hint.theta, hint.o
    for k in range(n):
        lay.reset_mesh(k)
    corrs = []
    for e in good:
        Ui, Uj = _edge_samples(lay, e)
        if len(Ui):
            corrs.append((e["i"], e["j"], Ui, Uj, e["q"]))
    solve_rigid(lay, corrs, good)
    groups = [[k for k in range(n) if comp[k] == c] for c in dict.fromkeys(comp)]
    if len(groups) > 1:
        groups = join_groups(lay, store, paths, groups, max_deg, progress=progress)
    if hint is None:
        _apart(lay, groups)
    for k in range(n):
        lay.reset_mesh(k)
    report["disconnected"] = groups if len(groups) > 1 else []
    if trim is not None:
        trim_sides(lay, np.asarray(trim, float).reshape(n))
    report["residual"] = {f"{a + 1}-{b + 1}": v for (a, b), v in refine(lay, store, paths, progress=progress).items()}
    a = lay.anchor
    turn = {k: round(_turn(lay, a, k) / (math.pi / 2)) % 4 for k in next(g for g in groups if a in g)}
    report["turned"] = {str(k + 1): 90 * t for k, t in turn.items() if t} if hint is None else {}
    return lay, report
