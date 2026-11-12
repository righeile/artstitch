"""Checks for ArtStitch.  ``python test_artstitch.py`` runs them all (about a minute), or use pytest.

The alignment checks cut a synthetic painting into scans with known poses and a smooth flatbed-like
warp, then ask where the recovered layout disagrees with the truth *between scans that show the same
paint* -- which is what a visible seam is.  The rest are the places a wrong sign or an off-by-one
would hide: shift direction, size cap, lossless TIFF, color order, straighten direction.
"""

from __future__ import annotations

import math
import os
import struct
import sys
import tempfile
import time
import zlib
from functools import lru_cache

import cv2
import imagecodecs
import numpy as np
import tifffile
from scipy import fft as sfft
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import color
import edges
import export
import tiles
from align import _good, align_set, coarse_pair, hand_layout, overlapping_pairs
from geometry import Frame, Layout, rot
from ncc import fast_shape, masked_ncc_map, pack_fft, peak
from session import Session

TMP = tempfile.mkdtemp(prefix="artstitch_test_")


# --- registration primitives ---------------------------------------------------------------------

def test_ncc_shift_direction_and_window():
    rng = np.random.default_rng(0)
    A = ndi.gaussian_filter(rng.standard_normal((160, 200)).astype(np.float32), 2)
    B = A[12:132, 30:190]                                  # B(x) = A(x + t), t = (30, 12), so A(x) = B(x - t)
    fs = fast_shape(160 + 120, 200 + 160)
    on = lambda a: pack_fft(a, np.ones(a.shape, bool), fs)
    n, c = masked_ncc_map(on(A), on(B), fs, 200)
    pk = peak(n, c, fs, (190, 150), saturation=1000.0)
    assert abs(pk["tx"] - 30) < 0.1 and abs(pk["ty"] - 12) < 0.1 and pk["ncc"] > 0.95, pk
    near = peak(n, c, fs, (6, 6), saturation=1000.0, centre=(27, 9))     # a window around a hint finds it ...
    assert abs(near["tx"] - 30) < 0.1 and abs(near["ty"] - 12) < 0.1, near
    away = peak(n, c, fs, (6, 6), saturation=1000.0, centre=(-60, 40))   # ... and only what is inside it
    assert away["ncc"] < 0.6, away


def test_coarse_pair_unequal_sizes():
    """Tiles of different shape: a shift that is more than half the transform used to read as a negative one."""
    rng = np.random.default_rng(1)
    F = ndi.gaussian_filter(rng.standard_normal((200, 260)).astype(np.float32), 2.5)
    sa, sb = F[20:110, 0:160], F[60:190, 140:220]          # B sits at t = (140, 40) against A
    r = coarse_pair(sa, sb, np.array([0.0]), min_frac=0.02)
    assert r and abs(r["tx"] - 140) < 0.3 and abs(r["ty"] - 40) < 0.3, r


def test_hand_layout_keeps_relative_poses():
    sizes = [(5400, 4200)] * 3 + [(4200, 5400)]
    poses = [dict(x=3000, y=2000, theta=10.0, anchor=True), dict(x=8000, y=2100, theta=11.5),
             dict(x=3100, y=7000, theta=9.0), dict(x=8100, y=7000, theta=100.0)]
    lay = hand_layout(sizes, poses)
    assert lay.anchor == 0 and lay.theta[0] == 0 and np.allclose(lay.o[0], lay.centre(0))
    th = np.radians([p["theta"] for p in poses])
    o = np.array([[p["x"], p["y"]] for p in poses], float)
    for i, j in ((0, 1), (1, 2), (2, 3), (0, 3)):           # what one tile looks like from the other is untouched
        assert abs((lay.theta[j] - lay.theta[i]) - (th[j] - th[i])) < 1e-12
        assert np.allclose(rot(-lay.theta[i]) @ (lay.o[j] - lay.o[i]), rot(-th[i]) @ (o[j] - o[i]))
    poses[0]["theta"] = 90.0                                # an anchor turned on its side cannot keep its pixels
    turned = hand_layout(sizes, poses)
    assert turned.anchor != 0 and turned.theta[turned.anchor] == 0


def test_group_sets_by_name():
    for n in ("a_1", "a_2", "b", "c 1", "c 2", "c 3"):
        tifffile.imwrite(os.path.join(TMP, n + ".tif"), np.zeros((4, 6, 3), np.uint8))
    got = tiles.group_sets([os.path.join(TMP, f) for f in sorted(os.listdir(TMP)) if f.endswith(".tif")])
    assert {s["name"]: len(s["files"]) for s in got if s["name"] in "abc"} == {"a": 2, "b": 1, "c": 3}
    assert tiles.set_key("Irchel_redsign 2.tiff") == ("Irchel_redsign", 2) and tiles.set_key("x.tif") == ("x", 0)


# --- color ---------------------------------------------------------------------------------------

def test_color_adjustments():
    ramp = np.linspace(0, 1, 101, dtype=np.float32)[None, :, None].repeat(3, 2)
    assert color.build({}) == (None, None) and color.apply(ramp, color.build({})) is ramp
    x = np.full((1, 1, 3), 0.25, np.float32)
    lv = lambda **a: color.apply(x, color.build(a))[0, 0]
    assert np.allclose(lv(levels={"rgb": [0.0, 2.0, 1.0, 0.0, 1.0]}), 0.5, atol=1e-4)               # gamma > 1 brightens
    assert np.allclose(lv(levels={"rgb": [0.25, 1.0, 0.75, 0.0, 1.0]}), 0.0, atol=1e-4)
    only_r = lv(levels={"r": [0.0, 1.0, 0.5, 0.0, 1.0]})
    assert np.allclose(only_r, [0.5, 0.25, 0.25], atol=1e-4)                                         # channels are independent
    order = lv(levels={"r": [0.0, 2.0, 1.0, 0.0, 1.0], "rgb": [0.2, 1.0, 1.0, 0.0, 1.0]})
    assert abs(order[0] - (0.5 - 0.2) / 0.8) < 1e-3, order                                          # the channel's own first
    assert abs(lv(curves={"rgb": [[0, 0], [0.5, 0.6], [1, 1]]})[0] - color.curve_samples([[0, 0], [0.5, 0.6], [1, 1]], 5)[1]) < 1e-3
    y = color.apply(np.full((1, 1, 3), 0.5, np.float32), color.build({"curves": {"rgb": [[0, 0], [0.5, 0.6], [1, 1]]}}))
    assert abs(float(y[0, 0, 0]) - 0.6) < 1e-3                                                       # through its control points
    px = np.array([[[0.8, 0.3, 0.1]]], np.float32)
    gray = color.apply(px, color.build({"sat": -100}))[0, 0]
    assert np.allclose(gray, gray[0], atol=1e-5)
    assert np.allclose(color.apply(px, color.build({"hue": 180})), color.apply(px, color.build({"hue": -180})), atol=1e-5)
    h = color.histogram((np.random.default_rng(2).random((100, 100, 3)) * 0.6 + 0.2).astype(np.float32))
    b = color.auto_levels(h, 0.001)
    assert all(0.19 < b[c][0] < 0.25 and 0.75 < b[c][2] < 0.81 for c in "rgb"), b


# --- saving --------------------------------------------------------------------------------------

def _flat(img: np.ndarray):
    """A 'mosaic' that is just a picture, so the exporter can be run without stitching anything."""
    class Flat:
        paths = ["a", "b"]

        def footprint(self, k, fr, w, h, pad=1):                  # scan a is all of it, b none
            return (0, 0, w, h) if k == 0 else None

        def render(self, fr, x0, y0, w, h, interp=None, each=None):
            ox, oy = (int(round(v)) for v in fr.b)
            crop = img[oy + y0:oy + y0 + h, ox + x0:ox + x0 + w].astype(np.float32) / 255.0
            one = np.ones(crop.shape[:2], np.float32)
            if each:
                each(0, 0, 0, crop, one > 0, one)
            return crop, one > 0
    return Flat()


def _picture(h: int, w: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = ndi.gaussian_filter(rng.standard_normal((h, w, 3)).astype(np.float32), (6, 6, 0)) * 40
    return np.clip(128 + base + rng.standard_normal((h, w, 3)) * 10, 0, 255).astype(np.uint8)


def test_tiff_is_lossless_whatever_the_compression():
    for bits, dtype in ((8, np.uint8), (16, np.uint16)):
        H, W = 600, 517                                            # not a multiple of the strip height
        img = (_picture(H, W) if bits == 8 else _picture(H, W).astype(np.uint16) * 257 + 3)
        sizes = {}
        for comp in export.COMPRESSIONS if bits == 8 else ("deflate", "none"):
            p = os.path.join(TMP, f"t{bits}_{comp}.tif")
            export.write_tiff(p, (img[y:y + export.ROWS] for y in range(0, H, export.ROWS)), W, H, bits, comp,
                              b"ICCDATA" * 20, 1200.0)
            back = tifffile.imread(p)
            assert back.dtype == dtype and np.array_equal(back, img), (bits, comp)
            with tifffile.TiffFile(p) as tf:
                pg = tf.pages[0]
                assert pg.tags["InterColorProfile"].value in (b"ICCDATA" * 20, np.frombuffer(b"ICCDATA" * 20, np.uint8).tolist())
                assert abs(pg.tags["XResolution"].value[0] / pg.tags["XResolution"].value[1] - 1200) < 1e-6
            sizes[comp] = os.path.getsize(p)
        assert sizes["deflate"] < 0.9 * sizes["none"], sizes
        if bits == 8:
            assert sizes["zopfli"] <= sizes["deflate"] and sizes["lzma"] < sizes["deflate"], sizes


def test_export_presets_sizes_and_names():
    ok = export.DEFAULT_PRESET
    assert export.check_preset(ok) == []
    two = {**ok, "outputs": [{"kind": "tiff", "name": "a"}, {"kind": "tiff", "name": "b"}]}
    assert any("one TIFF" in m for m in export.check_preset(two))
    dup = {**ok, "outputs": [{"kind": "png", "name": "x"}, {"kind": "png", "name": "X.png"}]}
    assert any("same file name" in m for m in export.check_preset(dup))
    assert any("positive" in m for m in export.check_preset({**ok, "outputs": [{"kind": "png", "size": "mb", "mb": 0}]}))
    d = export.describe(ok, "art", 4, 12000, 9000, 1200.0)
    assert [(o["kind"], o.get("w"), o.get("h")) for o in d] == [("tiff", 12000, 9000), ("png", 3000, 2250),
                                                                ("png", None, None), ("jpg", 3000, 2250)]
    assert d[1]["file"] == "art_300dpi.png" and d[1]["dpi"] == 300.0 and "10 MB" in d[2]["note"]
    assert export.file_name({"kind": "png", "name": "{name}_{dpi}dpi"}, {"name": "a/b", "dpi": 300}) == "a_b_300dpi.png"
    assert export.file_name({"kind": "png", "name": "x.PNG"}, {"name": "n"}) == "x.PNG"
    base = os.path.join(TMP, "exists.png")
    open(base, "w").close()
    assert export.target(base, "number", set()) == os.path.join(TMP, "exists_2.png")
    assert export.target(base, "skip", set()) is None and export.target(base, "replace", set()) == base
    assert export.target(base, "replace", {os.path.realpath(base)}) == os.path.join(TMP, "exists_2.png")   # never an input


def test_export_set_files_cap_and_overwrite():
    img = _picture(1600, 2400)
    geom = {"angle_deg": 0.0, "pivot": [0.0, 0.0], "crop": [0, 0, 2400, 1600]}
    preset = {"name": "t", "overwrite": "number", "outputs": [
        {"kind": "tiff", "name": "{name}", "compression": "deflate"},
        {"kind": "png", "name": "{name}_{dpi}", "size": "dpi", "dpi": 300},
        {"kind": "png", "name": "{name}_small", "size": "mb", "mb": 1},
        {"kind": "jpg", "name": "{name}", "size": "long", "long": 900, "quality": 90}]}
    folder = os.path.join(TMP, "out")
    res = export.export_set(_flat(img), geom, {}, preset, "art", folder, b"ICCDATA" * 20, 1200.0, 8)
    got = {r["path"].split(os.sep)[-1]: r for r in res}
    assert set(got) == {"art.tiff", "art_300.png", "art_small.png", "art.jpg"}, set(got)
    assert np.array_equal(tifffile.imread(os.path.join(folder, "art.tiff")), img)                # lossless, end to end
    small = got["art_small.png"]["bytes"]
    assert 850_000 < small <= 1_000_000 == int(1 * 1e6), small                                    # decimal MB, and close to it
    assert os.path.getsize(os.path.join(folder, "art_small.png")) == small
    assert (got["art_300.png"]["w"], got["art_300.png"]["h"]) == (600, 400)
    from PIL import Image
    with Image.open(os.path.join(folder, "art_300.png")) as im:
        assert im.size == (600, 400) and abs(im.info["dpi"][0] - 300) < 0.5 and im.info.get("icc_profile")
    with Image.open(os.path.join(folder, "art.jpg")) as im:
        assert im.size == (900, 600)
        ref = cv2.resize(img, (900, 600), interpolation=cv2.INTER_AREA).astype(float)
        mse = ((np.asarray(im).astype(float) - ref) ** 2).mean()
        assert 10 * math.log10(255 ** 2 / mse) > 28
    again = export.export_set(_flat(img), geom, {}, preset, "art", folder, None, 1200.0, 8)       # numbered, not overwritten
    assert {os.path.basename(r["path"]) for r in again} == {"art_2.tiff", "art_300_2.png", "art_small_2.png", "art_2.jpg"}
    skip = export.export_set(_flat(img), geom, {}, {**preset, "overwrite": "skip"}, "art", folder, None, 1200.0, 8)
    assert all(r.get("skipped") for r in skip)


def _ps_layers(b: bytes, at: int, psb: bool, bits: int):
    """Photoshop layer information at b[at:]: ([(name, {key: data}, {channel: pixels or None}, rect)], end).  Walks
    every length, so a wrong one fails here rather than in Photoshop."""
    dt = np.dtype(">u2" if bits == 16 else np.uint8)
    n, = struct.unpack_from(">h", b, at)
    at, recs, out = at + 2, [], []
    for _ in range(n):
        rect, nc = struct.unpack_from(">4i", b, at), struct.unpack_from(">H", b, at + 16)[0]
        at += 18
        ch = [struct.unpack_from(">hQ" if psb else ">hI", b, at + i * (10 if psb else 6)) for i in range(nc)]
        at += nc * (10 if psb else 6)
        sig, mode, opacity, extra = *struct.unpack_from(">4s4sB", b, at), struct.unpack_from(">I", b, at + 12)[0]
        assert (sig, mode, opacity) == (b"8BIM", b"norm", 255)
        at += 16
        end = at + extra
        ml, = struct.unpack_from(">I", b, at)
        mask = struct.unpack_from(">4i", b, at + 4) if ml else None
        assert ml in (0, 20) and (ml == 0 or b[at + 20:at + 24] == bytes(4))   # black outside, no flags
        at += 4 + ml
        at += 4 + struct.unpack_from(">I", b, at)[0]                          # blending ranges
        at += (1 + b[at] + 3) // 4 * 4                                         # the ASCII name
        blocks = {}
        while at < end:
            sig, key, ln = struct.unpack_from(">4s4sI", b, at)
            blocks[key], at = b[at + 12:at + 12 + ln], at + 12 + ln
        assert at == end
        k, = struct.unpack_from(">I", blocks[b"luni"])
        recs.append((blocks[b"luni"][4:4 + 2 * k].decode("utf-16-be"), blocks, rect, ch, mask))
    for name, blocks, rect, ch, mask in recs:
        px = {}
        for cid, ln in ch:
            t, l, bo, r = mask if cid == -2 else rect
            comp, = struct.unpack_from(">H", b, at)
            if (bo - t) * (r - l) == 0:
                assert comp == 0 and ln == 2
                px[cid] = None
            else:
                assert comp == 3                                                # ZIP with prediction
                z, h, w = zlib.decompress(b[at + 2:at + ln]), bo - t, r - l
                if bits == 32:                  # each row's bytes differenced, the most significant of every value first
                    u = np.cumsum(np.frombuffer(z, np.uint8).reshape(h, 4 * w), axis=1, dtype=np.uint8)
                    px[cid] = u.reshape(h, 4, w).transpose(0, 2, 1).copy().view(">f4")[..., 0].astype(np.float32)
                else:
                    a = np.frombuffer(z, dt).reshape(h, w).astype(dt.newbyteorder("="))
                    px[cid] = np.cumsum(a, axis=1, dtype=a.dtype)              # the prediction undone, modulo 2^bits
            at += ln
        out.append((name, blocks, px, rect))
    return out, at


def _psd_read(path: str) -> dict:
    b = open(path, "rb").read()
    sig, ver, nch, H, W, bits, mode = struct.unpack_from(">4sH6xHIIHH", b)
    assert (sig, nch, mode) == (b"8BPS", 3, 3)
    psb, L = ver == 2, ">Q" if ver == 2 else ">I"
    l_ = struct.calcsize(L)
    at = 30 + struct.unpack_from(">I", b, 26)[0]
    rend, res, at = at + 4 + struct.unpack_from(">I", b, at)[0], {}, at + 4
    while at < rend:
        _, rid, _, ln = struct.unpack_from(">4sHHI", b, at)
        res[rid], at = b[at + 12:at + 12 + ln], at + 12 + ln + (ln & 1)
    lend, li = at + l_ + struct.unpack_from(L, b, at)[0], struct.unpack_from(L, b, at + l_)[0]
    at, layers = at + 2 * l_, []
    if li:
        layers, e = _ps_layers(b, at, psb, bits)
        assert 0 <= at + li - e < 4
        at += li
    at += 4 + struct.unpack_from(">I", b, at)[0]                              # global layer mask
    if at < lend:
        assert b[at:at + 8] == b"8BIM" + export.LAYERS_KEY[bits] and bits != 8
        ln = struct.unpack_from(L, b, at + 8)[0]
        layers, e = _ps_layers(b, at + 8 + l_, psb, bits)
        at += 8 + l_ + ln
        assert 0 <= at - e < 4
    assert at == lend
    if bits == 32:                                                              # then the composite: raw
        assert struct.unpack_from(">H", b, at)[0] == 0 and len(b) == at + 2 + 12 * H * W
        flat = np.frombuffer(b, ">f4", 3 * H * W, at + 2).reshape(3, H, W).astype(np.float32)
        return {"psb": psb, "bits": bits, "res": res, "layers": layers, "flat": np.moveaxis(flat, 0, -1)}
    assert struct.unpack_from(">H", b, at)[0] == 1                              # or RLE
    counts = np.frombuffer(b, ">u4" if psb else ">u2", 3 * H, at + 2)
    at += 2 + counts.nbytes
    counts = counts.astype(int)
    rows = []
    for n in counts:
        rows.append(np.frombuffer(imagecodecs.packbits_decode(b[at:at + n]), np.uint8))
        at += n
    assert at == len(b)
    flat = np.stack(rows).reshape(3, H, W, bits // 8)
    flat = flat[..., 0] if bits == 8 else flat[..., 0].astype(np.uint16) * 256 + flat[..., 1]
    return {"psb": psb, "bits": bits, "res": res, "layers": layers, "flat": np.moveaxis(flat, 0, -1)}


def test_photoshop_layers_psd_and_pdf():
    img = _picture(700, 900)
    geom = {"angle_deg": 0.0, "pivot": [0.0, 0.0], "crop": [0, 0, 900, 700]}
    adj = {"levels": {"r": [0.05, 1.2, 0.95, 0.0, 1.0], "rgb": [0.03, 1.1, 0.97, 0.0, 1.0]},
           "curves": {"g": [[0.1, 0.05], [0.4, 0.45], [1, 1]], "rgb": [[0, 0], [0.25, 0.22], [0.75, 0.8], [1, 1]]},
           "hue": 8, "sat": 15, "temp": 12}
    preset = {"name": "t", "overwrite": "replace", "outputs": [
        {"kind": "tiff", "name": "{name}", "compression": "deflate", "layers": True}, {"kind": "psd", "name": "{name}"},
        {"kind": "png", "name": "{name}", "size": "full"}, {"kind": "pdf", "name": "{name}", "size": "dpi", "dpi": 600},
        {"kind": "pdf", "name": "{name}_small", "size": "mb", "mb": 0.03, "image": "jpeg", "quality": 85}]}
    from PIL import Image, ImageCms
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    folder = os.path.join(TMP, "ps")
    res = {os.path.basename(r["path"]): r for r in export.export_set(_flat(img), geom, adj, preset, "art", folder, icc)}
    assert res["art.psd"]["layers"] == res["art.tiff"]["layers"] == 6
    flat = np.asarray(Image.open(os.path.join(folder, "art.png")))
    p = _psd_read(os.path.join(folder, "art.psd"))
    names = ["a", "Levels: red, green, blue", "Curves: red, green, blue", "Levels", "Curves", export.MIXER]
    assert [x[0] for x in p["layers"]] == names and not p["psb"]                # bottom first, in the order they apply
    assert np.array_equal(np.stack([p["layers"][0][2][c] for c in range(3)], -1), img)    # the scan: unadjusted
    assert (p["layers"][0][2][-1] == 255).all() and (p["layers"][0][2][-2] == 255).all()   # opaque, all shown
    assert np.array_equal(p["flat"], flat)                                                # the composite: adjusted
    assert p["res"][0x040F] == icc and p["res"][0x03ED][:4] == struct.pack(">I", 1200 << 16)
    lv = struct.unpack_from(">H5h5h", p["layers"][1][1][b"levl"])
    assert lv == (2, 0, 255, 0, 255, 100, 13, 242, 0, 255, 120), lv                     # composite untouched, red set
    cv = p["layers"][2][1][b"curv"]
    assert struct.unpack_from(">BHHHH", cv) == (0, 1, 0, 0b100, 3)                     # green only, its three points
    assert struct.unpack_from(">6H", cv, 9) == (13, 26, 115, 102, 255, 255)            # (output, input): flat past them
    assert color.curve_samples([[0.2, 0.1], [0.5, 0.8], [0.9, 1]], 11)[:3] == [0.1] * 3   # ... here as there
    M = color.matrix(8, 15, 0, 0, 12, 0)
    assert struct.unpack_from(">15h", p["layers"][5][1][b"mixr"], 4) == tuple(
        v for m in np.round(M * 100).astype(int) for v in (m[0], m[1], m[2], 0, m[3]))
    with tifffile.TiffFile(os.path.join(folder, "art.tiff")) as tf:                     # the TIFF: flattened, and
        assert np.array_equal(tf.pages[0].asarray(), flat)                             # the same layers in a tag
        v = bytes(tf.pages[0].tags[37724].value)
    assert v.startswith(export.PS_MAGIC + b"8BIMLayr") and struct.unpack_from(">I", v, len(export.PS_MAGIC) + 8)[0] == len(v) - len(export.PS_MAGIC) - 12
    lay, end = _ps_layers(v, len(export.PS_MAGIC) + 12, False, 8)
    assert [x[0] for x in lay] == names and 0 <= len(v) - end < 4
    assert np.array_equal(np.stack([lay[0][2][c] for c in range(3)], -1), img)
    try:
        import pymupdf
    except ImportError:
        pymupdf = None
    if pymupdf:
        with pymupdf.open(os.path.join(folder, "art.pdf")) as doc:
            page = doc[0]
            pix = pymupdf.Pixmap(doc, page.get_images()[0][0])
            assert (round(page.rect.width, 3), round(page.rect.height, 3)) == (54.0, 42.0)   # the artwork's own size
            got = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, 3)
            assert np.array_equal(got, cv2.resize(flat, (450, 350), interpolation=cv2.INTER_AREA))   # lossless
        with pymupdf.open(os.path.join(folder, "art_small.pdf")) as doc:
            assert doc[0].get_images()[0][2] > 100 and res["art_small.pdf"]["bytes"] <= 30_000
    # 16 bits, and a PSB: the layers move into an Lr16 block, lengths and row counts widen
    img16 = img.astype(np.uint16) * 257 + 3
    comp, base = export.Planes(folder, "rle", psb=True), export.PixelLayer("x", (0, 0, 700, 900), (0, 1, 2), folder, 16)
    for y in range(0, 700, export.ROWS):
        comp(img16[y:y + export.ROWS])
        base.put(y, 0, (img16[y:y + export.ROWS] ^ 0x5A5A) / 65535.0)
    path = os.path.join(TMP, "deep.psb")
    export.write_psd(path, 900, 700, 16, None, 300.0, comp, export.layer_writer([base], export.ps_adjustments({"sat": 9}), True), True)
    q = _psd_read(path)
    assert q["psb"] and q["bits"] == 16 and np.array_equal(q["flat"], img16) and [x[0] for x in q["layers"]] == ["x", export.MIXER]
    assert np.array_equal(np.stack([q["layers"][0][2][c] for c in range(3)], -1), img16 ^ 0x5A5A)
    assert export.describe({"outputs": [{"kind": "psd", "name": "a"}]}, "a", 1, 31000, 900, 1200.0)[0]["file"] == "a.psb"
    assert export.describe({"outputs": [{"kind": "psd", "name": "a"}]}, "a", 1, 12000, 18000, 1200.0)[0]["file"] == "a.psd"
    assert export.describe({"outputs": [{"kind": "psd", "name": "a"}]}, "a", 1, 14889, 21602, 1200.0)[0]["file"] == "a.psb"   # tjj
    assert any("one PSD" in m for m in export.check_preset({"outputs": [{"kind": "psd", "name": "a"}, {"kind": "psd", "name": "b"}]}))


def test_16_and_32_bits_in_and_out():
    """32 bits a channel are Photoshop's: light, linear, under the profile's tone curves.  A deep scan is read at 16
    bits, and each full-size file is written at the depth it asks for -- at 32 with the color in its layers."""
    from PIL import ImageCms
    x = np.linspace(0.0, 1.0, color.N)
    srgb = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    assert np.abs(color.tone(None) - srgb).max() < 1e-6 and np.abs(color.tone(icc) - srgb).max() < 1e-3
    curv = b"curv" + bytes(4) + struct.pack(">IH2x", 1, 563)                              # gamma 563/256, about 2.2
    gray = bytes(128) + struct.pack(">I4sII", 1, b"kTRC", 144, len(curv)) + curv         # one curve for all three
    assert np.abs(color.tone(gray) - x ** (563 / 256)).max() < 1e-6
    # in: a floating point TIFF and a 16-bit PNG come back as the same 16-bit levels
    T = color.tone(icc)
    img16 = _picture(300, 400).astype(np.uint16) * 256 + np.random.default_rng(1).integers(0, 256, (300, 400, 3), np.uint16)
    p = os.path.join(TMP, "float.tif")
    tifffile.imwrite(p, np.stack([T[c][img16[..., c]] for c in range(3)], -1), photometric="rgb",
                     extratags=[(34675, 7, len(icc), icc, True)])
    assert tiles.bits(p) == 32 and np.abs(tiles.decode(p).rgb.astype(int) - img16).max() <= 1
    p = os.path.join(TMP, "deep.png")
    cv2.imwrite(p, img16[..., ::-1])
    assert tiles.bits(p) == 16 and np.array_equal(tiles.decode(p).rgb, img16)
    # out: the picture, its scan and its patches, all light and linear; no adjustment layers
    img = _picture(300, 400)
    geom = {"angle_deg": 0.0, "pivot": [0.0, 0.0], "crop": [0, 0, 400, 300]}
    adj = {"levels": {"rgb": [0.03, 1.1, 0.97, 0.0, 1.0]}, "sat": 15}
    ys, xs, vals = np.array([100, 100, 101]), np.array([50, 51, 50]), np.full((3, 3), 0.5, np.float32)
    preset = {"name": "t", "overwrite": "replace", "outputs": [
        {"kind": "tiff", "name": "{name}", "compression": "deflate", "layers": True, "bits": 32},
        {"kind": "psd", "name": "{name}", "bits": 32}, {"kind": "png", "name": "{name}", "size": "full"}]}
    folder = os.path.join(TMP, "deep")
    res = {r["kind"]: r for r in export.export_set(_flat(img), geom, adj, preset, "art", folder, icc,
                                                   patches=(ys, xs, vals))}
    built = color.build(adj)
    lin = lambda v: np.stack([T[c][np.clip(v[..., c] * 65535 + 0.5, 0, 65535).astype(np.uint16)] for c in range(3)], -1)
    raw = img.astype(np.float32) / 255.0
    raw[ys, xs] = vals
    adjusted = color.apply(raw, built)
    with tifffile.TiffFile(res["tiff"]["path"]) as tf:
        flat = tf.pages[0].asarray()
        assert flat.dtype == np.float32 and tf.pages[0].predictor == 3
        v = bytes(tf.pages[0].tags[37724].value)
    assert np.abs(flat - lin(adjusted)).max() < 1e-4
    assert np.abs(tiles.decode(res["tiff"]["path"]).rgb.astype(int) - export.q16(adjusted)).max() <= 1   # reads back
    q = _psd_read(res["psd"]["path"])
    assert q["bits"] == 32 and np.array_equal(q["flat"], flat) and res["psd"]["bits"] == res["tiff"]["bits"] == 32
    assert v.startswith(export.PS_MAGIC + b"8BIMLr32")
    for lay in (q["layers"], _ps_layers(v, len(export.PS_MAGIC) + 12, False, 32)[0]):
        assert [L[0] for L in lay] == ["a", export.PATCHES]
        scan, heal = lay[0][2], lay[1][2]
        assert (scan[-1] == 1).all() and (scan[-2] == 1).all()
        got = np.stack([scan[c] for c in range(3)], -1)
        assert np.abs(got - lin(color.apply(img.astype(np.float32) / 255.0, built))).max() < 1e-4
        on = heal[-1] > 0                                                  # row by row, as ys and xs are
        assert on.sum() == 3 and np.abs(np.stack([heal[c] for c in range(3)], -1)[on] - lin(adjusted)[ys, xs]).max() < 1e-4
    # one file at 16 and one at 32: each its own layers
    mixed = {**preset, "outputs": [{**preset["outputs"][0], "bits": 16}, {"kind": "psd", "name": "{name}", "bits": "auto"}]}
    res = {r["kind"]: r for r in export.export_set(_flat(img), geom, adj, mixed, "mix", folder, icc, src_bits=32)}
    with tifffile.TiffFile(res["tiff"]["path"]) as tf:
        assert tf.pages[0].dtype == np.uint16
        v = bytes(tf.pages[0].tags[37724].value)
    assert v.startswith(export.PS_MAGIC + b"8BIMLr16")
    assert len(_ps_layers(v, len(export.PS_MAGIC) + 12, False, 16)[0]) == 1 + len(export.ps_adjustments(adj)) == res["tiff"]["layers"]
    assert _psd_read(res["psd"]["path"])["bits"] == 32 and res["psd"]["layers"] == 1
    f32, path = lin(adjusted), os.path.join(TMP, "deep32.psb")                  # a PSB: 8-byte lengths, a raw composite
    comp, base = export.Planes(folder, "raw", psb=True), export.PixelLayer("x", (0, 0, 300, 400), (0, 1, 2), folder, 32)
    for y in range(0, 300, export.ROWS):
        comp(f32[y:y + export.ROWS])
        base.put(y, 0, f32[y:y + export.ROWS])
    export.write_psd(path, 400, 300, 32, None, 300.0, comp, export.layer_writer([base], [], True), True)
    q = _psd_read(path)
    assert q["psb"] and np.array_equal(q["flat"], f32) and np.array_equal(np.stack([q["layers"][0][2][c] for c in range(3)], -1), f32)
    one = lambda o, sb=8: export.describe({"outputs": [{"kind": "psd", "name": "a", **o}]}, "a", 1, 12000, 18000, 1200.0, sb)[0]
    assert (one({})["file"], one({"bits": 32})["file"], one({}, 32)["bits"], one({"bits": 8}, 32)["bits"]) == ("a.psd", "a.psb", 32, 8)
    assert any("bits" in m for m in export.check_preset({"outputs": [{"kind": "tiff", "bits": 24}]}))


# --- edges: the sheet against the scanner lid ------------------------------------------------------

def _sheet(alpha_deg: float, size=(5400, 4000), paper=(5000, 3600)):
    c, hw, R = np.array(size) / 2.0, np.array(paper) / 2.0, rot(math.radians(alpha_deg))

    def render(fr, x0, y0, w, h, interp=None):
        pq = np.stack(np.meshgrid(np.arange(x0, x0 + w), np.arange(y0, y0 + h)), -1).reshape(-1, 2).astype(float)
        X = fr.to_mosaic(pq)
        inside = (np.abs((X - c) @ R) <= hw).all(1)                  # the sheet's own axes: R(-alpha)(X - c)
        tex = 0.04 * np.sin(X[:, 0] / 37.0) * np.sin(X[:, 1] / 53.0) + 0.03 * np.sin((X[:, 0] + 2 * X[:, 1]) / 211.0)
        img = np.where(inside[:, None], np.array([0.72, 0.66, 0.55]) + tex[:, None], np.array([0.93, 0.95, 0.98]))
        return img.reshape(h, w, 3).astype(np.float32), np.ones((h, w), bool)
    return render, c, hw, R


def test_edges_angle_and_crop_inside_the_sheet():
    for alpha in (1.7, -2.4, 0.0):
        render, c, hw, R = _sheet(alpha)
        r = edges.detect(render, np.array([[0.0, 0.0], [5400.0, 4000.0]]))
        assert r["method"] == "edges" and abs(r["angle_deg"] - alpha) < 0.05, (alpha, r["angle_deg"], r["method"])
        x0, y0, x1, y1 = r["crop"]
        fr = Frame(1.0, (x0, y0), math.radians(r["angle_deg"]), r["pivot"])
        corners = fr.to_mosaic(np.array([[0, 0], [x1 - x0, 0], [x1 - x0, y1 - y0], [0, y1 - y0]], float))
        assert (np.abs((corners - c) @ R) <= hw).all(), (alpha, "crop leaves the sheet")
        assert (x1 - x0) > 0.97 * 5000 * math.cos(math.radians(abs(alpha))) - 60 and (y1 - y0) > 3400, r["crop"]
    assert abs(edges.tilt(math.radians(91.0)) - math.radians(1.0)) < 1e-12 and abs(edges.tilt(math.radians(-1.0)) + math.radians(1.0)) < 1e-12


# --- alignment against the truth -------------------------------------------------------------------

def _art(h: int, w: int, seed: int = 1) -> np.ndarray:
    """A paint-like 1/f texture (every scale has content) with hard-edged blocks."""
    rng = np.random.default_rng(seed)
    hh, ww = h // 2, w // 2
    filt = 1.0 / np.maximum(np.hypot(sfft.fftfreq(hh)[:, None], sfft.rfftfreq(ww)[None, :]), 1.0 / 3000) ** 1.25
    ch = []
    for _ in range(3):
        f = sfft.irfft2(sfft.rfft2(rng.standard_normal((hh, ww)).astype(np.float32)) * filt, (hh, ww))
        f = cv2.resize(f.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        ch.append(f + rng.standard_normal((h, w)).astype(np.float32) * (0.15 * f.std()))
    out = np.empty((h, w, 3), np.uint8)
    for c in range(3):
        m = 0.75 * ch[0] + 0.45 * ch[c]
        lo, hi = np.percentile(m[::7, ::7], [1, 99])
        out[..., c] = np.clip((m - lo) / (hi - lo) * 235 + 10, 0, 255).astype(np.uint8)
    for _ in range(40):
        x0, y0 = int(rng.integers(0, w - 400)), int(rng.integers(0, h - 400))
        out[y0:y0 + int(rng.integers(80, 900)), x0:x0 + int(rng.integers(80, 900))] = rng.integers(20, 235, 3, dtype=np.uint8)
    return out


def _warp(u, k):
    """A smooth displacement per scan (cockling), analytic so it continues past the scan's edge."""
    x, y, ph = u[..., 0], u[..., 1], 1.3 * k
    return np.stack([5.0 * np.sin(y / 2300 + ph) + 3.0 * np.sin(x / 3100 + 2 * ph),
                     4.0 * np.sin(x / 2700 + 3 * ph) + 3.0 * np.cos(y / 3600 + ph)], -1)


def _pos(g, u):
    return (u - g["c"]) @ rot(g["theta"]).T + g["o"] + _warp(u, g["k"])


def _inv(g, X):
    u = (X - g["o"]) @ rot(g["theta"]) + g["c"]
    for _ in range(6):
        u = (X - g["o"] - _warp(u, g["k"])) @ rot(g["theta"]) + g["c"]
    return u


def _cut(art, g, cs=32):
    w, h = g["size"]
    gy, gx = math.ceil(h / cs) + 2, math.ceil(w / cs) + 2
    us = np.stack(np.meshgrid((np.arange(gx) - 1 + 0.5) * cs - 0.5, (np.arange(gy) - 1 + 0.5) * cs - 0.5), -1)
    X = cv2.resize(_pos(g, us).astype(np.float32), (gx * cs, gy * cs), interpolation=cv2.INTER_LINEAR)[cs:cs + h, cs:cs + w]
    return cv2.remap(art, np.ascontiguousarray(X[..., 0]), np.ascontiguousarray(X[..., 1]), cv2.INTER_LINEAR)


def _gt(size, theta_deg, o, k):
    w, h = size
    return {"size": size, "theta": math.radians(theta_deg), "o": np.array(o, float), "k": k,
            "c": np.array([(w - 1) / 2, (h - 1) / 2])}


@lru_cache(maxsize=None)
def scans(kind: str):
    """Scans of a synthetic painting, files + truth: four (2 x 2, a third overlapping); ``turn``: one of them lies
    on its side; ``quarters``: one upside down and one on its side; ``eight``: two rows of four, each a little
    crooked; ``columns``: two columns of three, the columns sharing a strip too narrow for any one pair."""
    rng = np.random.default_rng(5)
    if kind in ("eight", "columns"):
        art, gt = (_art(7400, 16800), []) if kind == "eight" else (_art(10400, 10800), [])
        rows, cols = ((2200, 5000), (2800, 6400, 10000, 13600)) if kind == "eight" else ((2100, 5100, 8100), (2700, 7850))
        for y, x in ([(y, x) for y in rows for x in cols] if kind == "eight" else [(y, x) for x in cols for y in rows]):
            th = rng.uniform(-0.6, 0.6)
            gt.append(_gt((5400, 4200), th, (x + rng.uniform(-40, 40), y + rng.uniform(-40, 40)), len(gt)))
    else:
        art = _art(7400, 9600)
        gt = [_gt((5400, 4200), 0.0, (2800, 2200), 0), _gt((5400, 4200), 0.6, (6700, 2250), 1),
              _gt((5400, 4200), -0.5, (2750, 5200), 2), _gt((5400, 4200), 0.3, (6650, 5150), 3)]
        if kind == "turn":
            gt[3] = _gt((4200, 5400), 90.4, (6650, 5150), 3)
        if kind == "tilted":                 # further apart than the +-3 deg search: only the scanner's tilts find these
            gt = [_gt((5400, 4200), 0.0, (2800, 2200), 0), _gt((5400, 4200), 11.0, (6700, 2250), 1),
                  _gt((5400, 4200), -8.0, (2750, 5200), 2), _gt((5400, 4200), 14.0, (6650, 5150), 3)]
        if kind == "quarters":
            gt[2] = _gt((5400, 4200), 179.5, (2750, 5200), 2)
            gt[3] = _gt((4200, 5400), -89.7, (6650, 5150), 3)
    paths = []
    for k, g in enumerate(gt):
        p = os.path.join(TMP, f"{kind}_{k + 1}.tif")
        tifffile.imwrite(p, _cut(art, g))
        paths.append(p)
    return paths, gt


def _pair_error(lay, gt, margin=200):
    """Per pair of scans (rms, max px) between where the layout puts the same paint, from each scan."""
    out = {}
    for i in range(len(gt)):
        for j in range(i + 1, len(gt)):
            w, h = gt[i]["size"]
            xs, ys = np.meshgrid(np.linspace(margin, w - margin, 60), np.linspace(margin, h - margin, 60))
            ui = np.c_[xs.ravel(), ys.ravel()]
            uj = _inv(gt[j], _pos(gt[i], ui))
            wj, hj = gt[j]["size"]
            ok = (uj[:, 0] > margin) & (uj[:, 0] < wj - margin) & (uj[:, 1] > margin) & (uj[:, 1] < hj - margin)
            if ok.sum() >= 50:
                e = np.hypot(*(lay.fwd(i, ui[ok]) - lay.fwd(j, uj[ok])).T)
                out[(i + 1, j + 1)] = (float(np.sqrt((e ** 2).mean())), float(e.max()))
    return out


def _hand(gt, dx=(450, -300, 380, -500), dy=(-250, 400, 320, -350), dth=(0.8, -1.1, 1.0, -0.7)):
    return [dict(x=float(g["o"][0] + dx[k]), y=float(g["o"][1] + dy[k]), theta=math.degrees(g["theta"]) + dth[k],
                 anchor=k == 0) for k, g in enumerate(gt)]


def _check_alignment(kind: str, poses_of=None, pairs: int = 6):
    paths, gt = scans(kind)
    lay, rep = align_set(tiles.TileStore(), paths, poses=poses_of(gt) if poses_of else None)
    a = lay.anchor
    assert lay.theta[a] == 0 and np.array_equal(lay.o[a], lay.centre(a)) and not lay.D[a].any(), "the anchor moved"
    assert not rep["disconnected"], rep["disconnected"]
    err = _pair_error(lay, gt)
    assert len(err) == pairs, err
    assert all(rms < 0.25 and mx < 1.0 for rms, mx in err.values()), (kind, err)
    return lay, rep


def test_align_finds_the_scans_alone():
    _check_alignment("grid")


def test_align_from_a_rough_hand_placement():
    """Half a tile-width of the overlap off, a degree crooked: roughly is enough."""
    _check_alignment("grid", _hand)


def test_align_from_a_gross_hand_placement():
    """More than the local window allows: the whole range is searched for those pairs instead."""
    _check_alignment("grid", lambda gt: _hand(gt, (1100, -700, 900, -1300), (-600, 1000, 800, -900), (0.8, -1.9, 1.0, -0.7)))


def test_align_with_a_scan_on_its_side():
    _check_alignment("turn", _hand)


def test_align_with_a_gap_left_between_overlapping_scans():
    def gap(gt):
        p = _hand(gt)
        p[3]["x"] += 1500
        p[3]["y"] += 1500
        return p
    _check_alignment("grid", gap)


def _line_up(gt):
    """What the Arrange window starts from: upright (a sideways scan turned a quarter), side by side, nothing overlapping."""
    x, out = 0.0, []
    for k, g in enumerate(gt):
        w, h = g["size"]
        out.append(dict(x=x + w / 2, y=h / 2, theta=90.0 if abs(g["theta"]) > 1.0 else 0.0, anchor=k == 0))
        x += w + 600
    return out


def test_align_from_scans_lined_up_side_by_side():
    """Nothing overlaps, so the whole sheet is searched -- around the turn the hand gave the sideways scan."""
    for kind in ("grid", "turn"):
        _check_alignment(kind, _line_up)


def test_align_eight_scans_lined_up():
    """The hardest set so far: eight scans, sixteen overlaps, nothing to start from."""
    _check_alignment("eight", _line_up, pairs=16)


def test_align_with_scans_upside_down_and_on_their_side():
    """Nothing to start from, one scan turned 180 degrees and one 90 (the way a sheet fits an A4 scanner): the other
    quarter turns are searched, and the result stands the way scan 1 does."""
    lay, rep = _check_alignment("quarters")
    assert lay.anchor in (0, 1) and rep["turned"] == {"3": 180, "4": 270}, (lay.anchor, rep["turned"])


def test_align_starts_from_the_tilts_the_scanner_measured():
    """Scans lying further apart than the rotation search reaches: on their own they never tie together, and
    told how the paper lay on the glass -- by eye, a few tenths of a degree out -- they line up like any set."""
    paths, gt = scans("tilted")
    store = tiles.TileStore()
    _, rep = align_set(store, paths)
    assert rep["disconnected"], "turns this far apart are outside the +-3 deg search; they should not be found"
    tilts = [math.degrees(g["theta"]) + d for g, d in zip(gt, (0.0, 0.3, -0.25, 0.2))]
    lay, rep = align_set(store, paths, tilts=tilts)
    assert not rep["disconnected"], rep["disconnected"]
    err = _pair_error(lay, gt)
    assert len(err) == 6 and all(rms < 0.25 and mx < 1.0 for rms, mx in err.values()), err


def test_the_side_that_shows_the_lid_is_the_side_that_lay_flat():
    """Where the sheet ran off the glass there is no lid beyond the paper, and that is the side to cut."""
    rng = np.random.default_rng(0)
    img = rng.integers(20, 235, (600, 800, 3)).astype(np.uint8)
    assert edges.off_glass(img) == {"left", "right", "top", "bottom"}      # all picture: it ran off every side
    lid = img.copy()
    lid[:, :40] = 250                                                     # the lid shows along the left
    assert edges.off_glass(lid) == {"right", "top", "bottom"}
    lid[-30:] = 18                                                        # and a dark lid along the bottom
    assert edges.off_glass(lid) == {"right", "top"}
    assert edges.off_glass(np.full((600, 800, 3), 250, np.uint8)) == set()  # lid everywhere: nothing ran off


def test_the_bend_at_the_glass_edge_is_cut_off_before_anything_uses_it():
    """One gate decides which of a scan's pixels may be matched on, cut through or painted, side by side."""
    lay = Layout([(1000, 800)], trim=[[10, 20, 30, 40]])             # left, right, top, bottom
    ux, uy = np.meshgrid(np.arange(1000.0), np.arange(800.0))
    v = lay.valid_from_maps(0, ux, uy)
    xs, ys = np.nonzero(v.any(0))[0], np.nonzero(v.any(1))[0]
    assert (xs[0], 999 - xs[-1], ys[0], 799 - ys[-1]) == (10, 20, 30, 40)
    assert int(lay.valid_from_maps(0, ux, uy, margin=50).any(0).argmax()) == 50   # never less than the margin
    assert Layout([(1000, 800)]).valid_from_maps(0, ux, uy).all()                 # no trim: all of it


def test_the_trim_is_mm_of_the_sides_that_ran_off_the_glass():
    """What the user types becomes pixels of each scan, on the sides where the sheet left the glass."""
    paths, _ = scans("grid")
    found = edges.off_glass
    try:
        edges.off_glass = lambda img, **kw: {"left", "bottom"}       # which sides is tested above
        s = Session(paths, tiles.TileStore())
        s.trim_mm = 2.0                                              # 94 px at the 1200 dpi assumed of these files
        assert s.trim_px() == [[2.0 * 1200 / 25.4, 0.0, 0.0, 2.0 * 1200 / 25.4]] * 4
        assert s.trimmed == {k: ["bottom", "left"] for k in range(4)}
        s.trim_mm = 300.0                                            # a typo cannot eat the overlap: a tenth, at most
        assert s.trim_px() == [[420.0, 0.0, 0.0, 420.0]] * 4
        s.trim_mm = 0.0
        assert s.trim_px() == [[0.0] * 4] * 4
        assert Session(paths[:1], tiles.TileStore()).trim_px() == [[0.0] * 4]      # one scan is never cut
    finally:
        edges.off_glass = found


def test_align_joins_groups_that_no_single_pair_ties():
    """Two columns whose shared strip is too narrow for any one pair (4.6 % of a scan): matched as wholes, the way
    stitching pairs first and then the pairs helps by hand."""
    paths, gt = scans("columns")
    lay, rep = align_set(tiles.TileStore(), paths)
    assert not rep["disconnected"], rep["disconnected"]
    assert not [e for e in rep["edges"] if _good(e) and (e["i"] < 3) != (e["j"] < 3)], "a single pair did it"
    err = _pair_error(lay, gt, margin=20)
    assert len(err) >= 7 and all(rms < 0.3 and mx < 1.0 for rms, mx in err.values()), err
    assert max(v["rms"] for v in rep["residual"].values()) < 1.0, rep["residual"]    # what Checks reads (6 px: a look before the last solve)


@lru_cache(maxsize=None)
def two_sheets():
    """Two different paintings, two scans of each (a folder where two sheets share one name): 1800 px shared inside a pair."""
    paths = []
    for sheet, seed in enumerate((1, 2)):
        art = _art(4200, 7400, seed)
        for k, x in enumerate((2300, 5100)):
            p = os.path.join(TMP, f"sheets_{2 * sheet + k + 1}.tif")
            tifffile.imwrite(p, _cut(art, _gt((4600, 4200), 0.0, (x, 2100), 2 * sheet + k)))
            paths.append(p)
    return paths


def test_a_smooth_overlap_is_no_match():
    """Numbers from real scans: half-overlapped lawn of two different sheets (ncc .37, ratio 1.13) against a true overlap."""
    assert not _good(dict(ncc=0.374, ratio=1.133, overlap=0.548, q=0.133))
    assert _good(dict(ncc=0.861, ratio=6.326, overlap=0.376, q=1.0))


def test_two_sheets_under_one_name_are_not_joined():
    """Nothing is matched between different paintings, so they stay apart and say so -- no invented overlap.  A batch
    does not export pieces side by side: the set fails, with its session kept to be opened and placed by hand."""
    import batch
    paths = two_sheets()
    lay, rep = align_set(tiles.TileStore(), paths)
    assert rep["disconnected"] == [[0, 1], [2, 3]], rep["disconnected"]
    assert sorted(overlapping_pairs(lay)) == [(0, 1), (2, 3)], overlapping_pairs(lay)
    assert max(v["rms"] for v in rep["residual"].values()) < 1.0, rep["residual"]     # the sheet without the anchor too (4 px when it was left piled up)
    folder = os.path.join(TMP, "sheets")
    os.makedirs(folder, exist_ok=True)
    for k, p in enumerate(paths, 1):
        os.symlink(p, os.path.join(folder, f"x_{k}.tif"))
    rows = batch.rows_from(batch.scan(folder))
    preset = {"name": "t", "folder": "out", "overwrite": "replace", "outputs": [{"kind": "jpg", "name": "{name}", "size": "long", "long": 600}]}
    batch.run(rows, preset, tiles.TileStore(), {})
    r = rows[0]
    assert r["state"] == "failed" and "could not be matched" in r["error"] and not r["outputs"], r
    assert os.path.exists(r["session"]) and not [f for f in os.listdir(os.path.join(folder, "out")) if f.endswith(".jpg")]


# --- the whole pipeline on the same scans ----------------------------------------------------------

_session: dict = {}


def _stitched() -> Session:
    if "s" not in _session:
        paths, gt = scans("grid")
        s = Session(paths, tiles.TileStore())
        s.auto()
        _session["s"] = s
    return _session["s"]


def test_session_stitches_without_errors_and_keeps_the_anchor_bit_exact():
    s = _stitched()
    assert not [q for q in s.qc() if q["level"] == "error"], s.qc()
    lay = s.layout
    a = lay.anchor
    s.geom = s.default_geom()                                      # no straightening: the canvas as it is
    folder = os.path.join(TMP, "sess")
    res = s.export({"name": "p", "overwrite": "replace", "outputs": [{"kind": "tiff", "name": "whole", "compression": "none"}]}, folder)
    out = tifffile.imread(res[0]["path"])
    x0, y0 = s.geom["crop"][:2]
    cx, cy = (int(v) for v in lay.o[a])
    tile = s.store.get(s.paths[a]).rgb
    got = out[cy - 100 - y0:cy + 100 - y0, cx - 100 - x0:cx + 100 - x0]
    assert np.array_equal(got, tile[cy - 100:cy + 100, cx - 100:cx + 100]), "the anchor was resampled"


def test_layers_are_the_scans_with_masks_and_the_patches_above():
    """In a PSD and a layered TIFF each scan is a layer holding all of itself that the picture shows, with a mask
    that cuts it to its part of the stitch, and the patches are a layer above them: stacked as Photoshop stacks
    them, they are the flattened picture."""
    import patch
    s = _stitched()
    lab, n = s.plan.labels, len(s.paths)
    near = sum(ndi.maximum_filter(lab == k, 60) for k in range(n))       # scans cut within 120 px (4 px a label)
    cx, cy = s.plan.ext0 + s.plan.f * np.array(np.unravel_index(np.argmax(near), lab.shape)[::-1])
    geom0 = s.geom
    try:
        s.geom = s.default_geom()
        s.set_crop([cx - 300, cy - 250, cx + 300, cy + 250])
        s.edit_patches("add", "result", pts=[[200, 150], [260, 170]], r=6)
        folder = os.path.join(TMP, "parts")
        res = {r["kind"]: r for r in s.export({"name": "p", "overwrite": "replace", "outputs": [
            {"kind": "psd", "name": "p"}, {"kind": "tiff", "name": "p", "compression": "deflate", "layers": True}]}, folder)}
        p = _psd_read(res["psd"]["path"])
        names = [os.path.splitext(os.path.basename(q))[0] for q in s.paths] + [export.PATCHES]
        assert [x[0] for x in p["layers"]] == names and res["psd"]["layers"] == res["tiff"]["layers"] == 5
        flat = p["flat"].astype(float)
        acc, scans = np.zeros_like(flat), []
        for name, _, px, (t, l, b, r) in p["layers"]:
            a = px[-1] / 255.0 * (px[-2] / 255.0 if -2 in px else 1.0)
            c = np.stack([px[i] for i in range(3)], -1).astype(float)
            acc[t:b, l:r] += (c - acc[t:b, l:r]) * a[..., None]
            if -2 in px:
                scans.append(((px[-2] > 0).mean(), (px[-1] > 0).mean()))
                assert scans[-1][0] < scans[-1][1], name                                 # all of it, a part shown
        off = np.abs(acc - flat)
        print(f"      stack vs flattened: max {off.max():.2f}, mean {off.mean():.3f}; shown / held {np.round(scans, 2).tolist()}")
        assert sum(0 < m for m, _ in scans) >= 3
        assert off.max() <= 1 and off.mean() < 0.05, (off.max(), off.mean())             # the masks' 8 bits
        c0, c1 = (np.stack([p["layers"][k][2][i] for i in range(3)], -1).astype(float) for k in (0, 1))
        both = (p["layers"][0][2][-1] > 0) & (p["layers"][1][2][-1] > 0)
        print(f"      two scans where both are: {np.abs(c0 - c1)[both].mean():.2f} apart")
        assert both.mean() > 0.5 and np.abs(c0 - c1)[both].mean() < 3                  # each its own pixels, in place
        heal = p["layers"][-1][2][-1] > 0
        assert 50 < heal.sum() < 2000, heal.sum()                                      # the stroke, not its box
        with tifffile.TiffFile(res["tiff"]["path"]) as tf:
            assert np.array_equal(tf.pages[0].asarray(), p["flat"])
            v = bytes(tf.pages[0].tags[37724].value)
        assert [x[0] for x in _ps_layers(v, len(export.PS_MAGIC) + 12, False, 8)[0]] == names
    finally:
        s.patches, s._undo, s._retouch, s.geom = [], [], patch.Retouch(), geom0
        s._touch()


def test_batch_stop_finishes_the_set_in_progress_and_skips_the_rest():
    import batch
    paths, _ = scans("grid")
    folder = os.path.join(TMP, "batch")
    os.makedirs(folder, exist_ok=True)
    for name in ("a", "b"):
        for k, p in enumerate(paths, 1):
            os.symlink(p, os.path.join(folder, f"{name}_{k}.tif"))
    rows = batch.rows_from(batch.scan(folder))
    assert [(r["name"], r["include"]) for r in rows] == [("a", True), ("b", True)], rows
    preset = {"name": "t", "folder": "out", "overwrite": "replace", "outputs": [{"kind": "jpg", "name": "{name}", "size": "long", "long": 600}]}
    stop: list = []                                                 # "stop" is pressed while the first set is being made
    batch.run(rows, preset, tiles.TileStore(), {}, progress=lambda p, m="": stop.append(1) if m.startswith("a (") else None,
              cancel=lambda: bool(stop))
    assert rows[0]["state"] in ("ok", "review") and os.path.exists(rows[0]["outputs"][0]["path"]), rows[0]
    assert rows[1]["state"] == "canceled" and not rows[1]["outputs"], rows[1]


def test_straighten_keep_and_level_line_direction():
    s = _stitched()
    g = s.straighten(angle_deg=1.0)
    assert g["method"] == "manual" and abs(g["angle_deg"] - 1.0) < 1e-9
    g = s.straighten(keep=True)
    assert g["method"] == "manual" and abs(g["angle_deg"] - 1.0) < 1e-9        # only the crop was looked for again
    g = s.level_line((0, 0), (1000, 1000 * math.tan(math.radians(1.0))))         # a line running 1 deg clockwise ...
    assert abs(g["angle_deg"] - 2.0) < 0.01, g["angle_deg"]                      # ... turns the sheet one more degree clockwise
    g = s.level_line((0, 0), (1000, -1000 * math.tan(math.radians(0.5))))
    assert abs(g["angle_deg"] - 1.5) < 0.01, g["angle_deg"]


def test_turning_the_result_moves_whole_pixels():
    """A quarter turn is a permutation of the same pixels (the anchor's still bit-exact), clockwise for +1, with the
    crop turned along; straightening again keeps the turn; four turns are none."""
    s = _stitched()
    s.geom = s.default_geom()
    g0, a = s.geom, s.layout.anchor
    cx, cy = (int(v) for v in s.layout.o[a])
    win = [cx - 150, cy - 100, cx + 150, cy + 100]                           # plane px, the middle of the anchor
    grab = lambda r: export.q8(s.render("result", r[0] - s.geom["crop"][0], r[1] - s.geom["crop"][1],
                                        r[2] - r[0], r[3] - r[1], 1.0)[0])
    before = grab(win)
    W, H = s.result_size()
    s.turn(1)
    assert s.geom["quarter"] == 1 and s.result_size() == (H, W)
    assert np.array_equal(grab(export.turn_rect(win, g0, 1)), np.rot90(before, -1)), "not a clockwise turn of the same pixels"
    for _ in range(3):
        s.turn(1)
    assert s.geom["quarter"] == 0 and s.geom["crop"] == g0["crop"]
    s.straighten(angle_deg=1.0)
    w1, h1 = s.result_size()
    s.turn(-1)
    g = s.straighten(angle_deg=1.0)
    assert g["quarter"] == 3 and s.result_size() == (h1, w1), (g["quarter"], s.result_size(), (w1, h1))


def test_patch_paints_out_a_speck_without_blurring():
    """A black speck on the anchor, painted out on a straightened, turned result: nothing outside the stroke
    changes, inside it another place's texture comes through exactly (copy + a membrane with no curvature), the
    speck is gone at full size, zoomed out, in the saved TIFF, after another turn and after saving and opening."""
    import patch
    s = _stitched()
    lay, a = s.layout, s.layout.anchor
    tile = s.store.get(s.paths[a])
    rgb0, proxies0 = tile.rgb, tile.proxies
    cx, cy = (int(v) for v in lay.o[a])
    lum = tile.rgb[cy - 600:cy + 600, cx - 600:cx + 600].astype(np.float32).mean(2)   # well inside what the anchor shows
    grain = np.abs(lum - cv2.GaussianBlur(lum, (0, 0), 2))
    spots = [(lum[y:y + 48, x:x + 48].mean(), x + 24, y + 24) for y in range(0, 1152, 24) for x in range(0, 1152, 24)
             if grain[y:y + 48, x:x + 48].mean() > 2 and grain[y:y + 48, x:x + 48].max() < 25 and lum[y:y + 48, x:x + 48].min() > 40]
    _, bx, by = max(spots)                                                 # textured, no hard edge, bright for a black speck
    sx, sy = cx - 600 + bx, cy - 600 + by                                  # anchor px = mosaic px
    r, R = 10.0, 80
    try:
        s.straighten(angle_deg=0.7)
        s.turn(1)

        def at(win_s=1.0):
            ang, piv = export.turn_of(s.geom)
            q = Frame(1.0, (0, 0), ang, piv).to_output(np.array([[sx, sy]], float))[0] - s.geom["crop"][:2]
            return q, (int(q[0]) - R, int(q[1]) - R, 2 * R, 2 * R)

        (vx, vy), win = at()
        clean = s.render("result", *win, 1.0)[0]
        clean_small = s.render("result", *win, 0.25)[0]
        tile.rgb, tile.proxies = rgb0.copy(), {}                          # the speck, in the proxies too
        cv2.circle(tile.rgb, (sx, sy), 6, (0, 0, 0), -1)
        s._touch()
        before = s.render("result", *win, 1.0)[0]
        yy, xx = np.mgrid[0:2 * R, 0:2 * R]
        dist = np.hypot(xx + win[0] - vx, yy + win[1] - vy)
        dark = lambda img, rad: int((img.mean(2)[dist <= rad] < 0.1).sum())
        assert dark(before, 5) > 50, "no speck to patch"
        t0 = time.time()
        s.edit_patches("add", "result", pts=[[vx, vy]], r=r)
        after = s.render("result", *win, 1.0)[0]
        took = time.time() - t0
        (p,) = s.patch_view()
        part0 = lambda: s.patch_view()[0]["parts"][0]                         # a speck is healed in one piece
        assert len(p["parts"]) == 1
        dx, dy = part0()["d"]
        assert part0()["auto"] and np.hypot(dx, dy) > 2 * r, p
        assert np.array_equal(after[dist > r + 1], before[dist > r + 1]), "pixels outside the stroke changed"
        assert dark(after, r) == 0, "the speck is still there"
        gone = np.abs(after - clean)[dist <= r - 1].mean() / np.abs(before - clean)[dist <= r - 1].mean()
        assert gone < 0.4, gone
        # inside, the copy's own texture: what the membrane added has no curvature, the texture has plenty
        src = np.roll(before, (-dy, -dx), (0, 1))
        inner = ndi.binary_erosion(dist <= r - 0.5, iterations=1)
        lap = lambda f: f[1:-1, 1:-1] * 4 - f[:-2, 1:-1] - f[2:, 1:-1] - f[1:-1, :-2] - f[1:-1, 2:]
        bend, grain = np.abs(lap(after - src))[inner[1:-1, 1:-1]], np.abs(lap(src))[inner[1:-1, 1:-1]]
        # 6/255: the render is 8-bit, and this stencil has a center coefficient of 4, so one
        # quantization level in one healed pixel already reads as 4/255.  0.01 was under that.
        assert bend.max() < 6 / 255 and grain.mean() > 10 * bend.mean(), (bend.max(), bend.mean(), grain.mean())
        small = s.render("result", *win, 0.25)[0]
        s.edit_patches("undo")
        small_before = s.render("result", *win, 0.25)[0]
        s.edit_patches("add", "result", pts=[[vx, vy]], r=r)
        k = np.abs(small_before - clean_small).max(2) > 0.05                    # where the speck shows, zoomed out
        assert k.any() and np.abs(small - clean_small)[k].mean() < 0.4 * np.abs(small_before - clean_small)[k].mean()
        # a place chosen by hand, then back to the automatic one
        s.edit_patches("source", "result", pid=p["id"], d=[dx + 7, dy - 5])
        assert part0()["d"] == [dx + 7, dy - 5] and not part0()["auto"]
        s.edit_patches("source", "result", pid=p["id"], d=None)
        assert part0()["d"] == [dx, dy] and s.patches[0]["src"] is None
        # undo walks back through the changes, not just the newest patch
        s.edit_patches("remove", pid=p["id"])
        s.edit_patches("undo")
        assert part0()["d"] == [dx, dy]
        s.edit_patches("undo")
        assert part0()["d"] == [dx + 7, dy - 5]
        s.edit_patches("source", "result", pid=p["id"], d=None)
        assert np.array_equal(s.render("result", *win, 1.0)[0], after)
        # saved: the result cut down around the speck, to keep the file small
        x0, y0 = s.geom["crop"][:2]
        s.set_crop([x0 + win[0] - 100, y0 + win[1] - 100, x0 + win[0] + 2 * R + 100, y0 + win[1] + 2 * R + 100])
        res = s.export({"name": "p", "overwrite": "replace", "outputs": [{"kind": "tiff", "name": "patched", "compression": "none"}]},
                       os.path.join(TMP, "patch"))
        off = np.abs(tifffile.imread(res[0]["path"])[100:100 + 2 * R, 100:100 + 2 * R].astype(int) - export.q8(after))
        assert off[dist <= r].max() == 0 and off.max() <= 1, "the saved picture is not the one shown"   # 1: remap's 1/32 px steps
        # opened again: the same picture
        path = s.save(os.path.join(TMP, "patch", "p.artstitch.json"))
        s2 = Session.load(path, s.store)
        (_, win2) = at()
        assert s2.patches == s.patches and np.array_equal(s2.render("result", *win2, 1.0)[0], s.render("result", *win2, 1.0)[0])
        # turned: healed again in the new frame, still over the speck
        s.turn(-1)
        (vx, vy), win = at()
        dist = np.hypot(xx + win[0] - vx, yy + win[1] - vy)
        assert dark(s.render("result", *win, 1.0)[0], r) == 0, "the patch did not follow the turn"
        print(f"      one patch healed in {took:.2f} s, copied from ({dx}, {dy}), {gone:.2f} of the speck left")
    finally:
        tile.rgb, tile.proxies = rgb0, proxies0
        s.patches, s._retouch = [], patch.Retouch()
        s._touch()


def test_patch_heals_a_long_hair_in_stretches():
    """A hair across an upright edge and a 45° one: one copy for the whole hair carries on only the edge that runs the
    way it was moved, a copy per stretch carries on both."""
    import patch
    rng = np.random.default_rng(3)
    H, W = 220, 460
    yy, xx = np.mgrid[0:H, 0:W]
    grain = ndi.gaussian_filter(rng.normal(0, 1, (H, W, 3)), (1.2, 1.2, 0))
    clean = (0.35 + 0.25 * (xx > 150) + 0.2 * (xx - yy > 230))[..., None] + 0.015 * grain / grain.std()
    clean = np.clip(clean, 0, 1).astype(np.float32)
    Q, r, cov = np.array([[60.0, 110.0], [400.0, 110.0]]), 3.0, np.ones((H, W), bool)   # crosses at x = 150 and 340
    hair = patch.stamp(Q, r, 0, 0, W, H)
    img = clean.copy()
    img[hair] = 0.05
    n = patch.count(Q, r)
    assert n > 10 and patch.count(Q[:1], r) == 1 and len(patch.pieces(Q, n)) == n
    err = {}
    for m in (1, n):
        im = img.copy()
        done, ds = patch.heal_stroke(im, cov, Q, r, [None] * m)
        assert np.array_equal(done, hair) and np.array_equal(im[~hair], img[~hair]) and None not in ds
        e = np.abs(im - clean).max(2)
        err[m] = [float(e[hair & (np.abs(xx - x) < 10)].mean()) for x in (100, 150, 340)]   # grain, upright, 45°
        if m == n:                                    # the stretches at the edges were copied along them
            (ux, uy), (sx, sy) = (ds[int((x - 60) / (340 / n))] for x in (150, 340))
            assert ux == 0 and sx == sy, ds
    assert max(err[n][1:]) < 0.035, err               # carried on as well as the grain allows
    assert err[1][2] > 2 * max(err[n][1:]), err       # one copy cannot carry on both


def test_dust_finds_specks_and_hairs_not_the_paint():
    """Specks and hairs, straight and curled, on grainy paint with soft edges: each found once and as what it is,
    a speck across the border between strips too, each covered by its stroke and a hair's stroke thin all along;
    neither the paint's edges nor the grain are found, nor a speck on the edge."""
    import dust
    rng = np.random.default_rng(5)
    H, W = 2300, 900
    yy, xx = np.mgrid[0:H, 0:W]
    base = 0.5 + 0.25 * ((xx - 450) ** 2 + (yy - 700) ** 2 < 250 ** 2) - 0.2 * (xx > 720)   # a disk, an edge
    grain = ndi.gaussian_filter(rng.normal(0, 1, (H, W)), 1.0)
    lum = ndi.gaussian_filter(base, 1.5) + 0.02 * grain / grain.std()
    marks = np.zeros((H, W), np.uint8)
    specks = [(-1, 200, 300), (1, 800, 400), (-1, 300, 1024), (-1, 450, 700), (1, 800, 1900), (-1, 2, 500)]
    for k, (sign, x, y) in enumerate(specks):
        cv2.circle(marks, (x, y), 3, k + 1, -1)
    cv2.line(marks, (150, 1700), (330, 1760), len(specks) + 1, 2)            # the hairs
    cv2.ellipse(marks, (550, 1400), (40, 40), 0, 0, 270, len(specks) + 2, 2)
    sign = np.r_[0, [s for s, _, _ in specks], -1, -1][marks]
    img = ndi.gaussian_filter(lum + 0.3 * sign, 0.8)                          # the scanner's own blur
    img = np.repeat(np.clip(img, 0, 1)[..., None], 3, 2).astype(np.float32)

    class Mos:
        def render(self, fr, x0, y0, w, h):
            return img[y0:y0 + h, x0:x0 + w].copy(), np.ones((h, w), bool)

    items, sheet = dust.find(Mos(), {"angle_deg": 0.0, "pivot": [W / 2, H / 2], "crop": [0, 0, W, H]})
    found = items                                                              # all kept, weakest too
    assert [it["i"] for it in items] == list(range(len(items))) and sheet.shape == (48 * -(-len(items) // 32), 48 * 32, 3)
    order = [(it["shape"] != "hair", -it["score"]) for it in items]
    assert order == sorted(order)                                              # hairs first, each strongest first
    matched = []
    for k in range(1, len(specks) + 3):
        Y, X = np.nonzero(marks == k)
        on = [it for it in found if (dust.polyline_distance(np.c_[X, Y], np.array(it["pts"])) <= it["r"]).all()]
        near = [it for it in items if dust.polyline_distance(np.c_[X, Y], np.array(it["pts"])).min() < 12]
        if k == 6:
            assert not near, ("a speck on the picture's edge", near)
        else:
            assert len(on) == 1 == len(near), (k, on, near)
            assert on[0]["shape"] == ("hair" if k > len(specks) else "speck"), (k, on)
            matched += on
    hair, curl = matched[-2:]
    assert np.hypot(*np.subtract(hair["pts"][-1], hair["pts"][0])) > 0.9 * np.hypot(180, 60), hair
    assert len(curl["pts"]) > 8 and hair["r"] < 7 and curl["r"] < 7, (hair, curl)     # followed round, not a disk
    stray = [it for it in found if not any(it is m for m in matched)]
    assert not stray, stray


def test_dust_found_is_patched_and_not_found_again():
    """In the session: specks on a scan are found in the result, patched from what was found (one step to undo),
    and not found again; straightening forgets what was found."""
    import patch
    s = _stitched()
    lay, a = s.layout, s.layout.anchor
    tile = s.store.get(s.paths[a])
    rgb0, proxies0 = tile.rgb, tile.proxies
    cx, cy = (int(v) for v in lay.o[a])
    top = np.iinfo(tile.rgb.dtype).max
    try:
        s.geom = s.default_geom()
        s.set_crop([cx - 300, cy - 200, cx + 300, cy + 200])
        tile.rgb, tile.proxies = rgb0.copy(), {}
        for x, y, v in ((cx - 150, cy - 60, 0), (cx + 120, cy + 70, top)):
            cv2.circle(tile.rgb, (x, y), 3, (v, v, v), -1)
        s._touch()
        s.find_dust()
        hit = lambda: [it for it in s.dust["items"] if it["score"] >= 12 and
                       min(np.hypot(*(np.mean(it["pts"], 0) - (x - cx + 300, y - cy + 200))) for x, y in ((cx - 150, cy - 60), (cx + 120, cy + 70))) < 6]
        got = hit()
        assert len(got) == 2, got
        undo = len(s._undo)
        s.edit_patches("add", "result", strokes=[{"pts": it["pts"], "r": it["r"]} for it in got])
        assert len(s.patches) == 2 and len(s._undo) == undo + 1
        s.find_dust()
        assert not hit(), hit()
        s.straighten(angle_deg=0.3)
        assert s.dust is None
    finally:
        tile.rgb, tile.proxies = rgb0, proxies0
        s.patches, s._undo, s._retouch = [], [], patch.Retouch()
        s._touch()


def test_lines_angle_reads_the_picture_lines_whatever_the_opencv():
    img = np.full((600, 800, 3), 230, np.uint8)
    t = math.tan(math.radians(1.5))
    for y in range(60, 560, 40):                                   # ruled lines running 1.5 deg clockwise
        cv2.line(img, (20, y), (780, int(round(y + 760 * t))), (40, 40, 40), 2, cv2.LINE_AA)
    ang, strength = edges.lines_angle(img.astype(np.float32) / 255)
    assert abs(math.degrees(ang) - 1.5) < 0.1 and strength > 0.5, (math.degrees(ang), strength)
    assert edges.lines_angle(np.full((300, 400, 3), 0.9, np.float32)) == (0.0, 0.0)   # a blank sheet: no lines


# --- add-ons -------------------------------------------------------------------------------------


def test_addons_reach_the_page_and_a_broken_one_is_left_out():
    import flask
    import app

    def good(host):
        bp = flask.Blueprint("addon_test", __name__)
        bp.add_url_rule("/api/addon-test", view_func=lambda: {"ok": True})
        host.app.register_blueprint(bp)
        host.add_script("/addons/test/t.js")
        host.add_style("/addons/test/t.css")

    class Entry:
        def __init__(self, name, fn):
            self.name, self.fn = name, fn

        def load(self):
            return self.fn

    saved, found = {k: list(v) for k, v in app.ADDONS.items()}, app.entry_points
    app.entry_points = lambda group: [Entry("broken", lambda host: 1 / 0), Entry("good", good)]
    try:
        assert app.load_addons() == ["good"]
        client = app.app.test_client()
        page = client.get("/").get_data(as_text=True)
        assert page.index("/static/app.js") < page.index("/addons/test/t.js") < page.index("</body>")
        assert "/addons/test/t.css" in page.split("</head>")[0]
        assert client.get("/api/addon-test").get_json() == {"ok": True}
        assert client.get("/api/health").get_json()["addons"] == ["good"]
    finally:
        app.entry_points = found
        app.ADDONS.update(saved)
    page = app.app.test_client().get("/").get_data(as_text=True)
    assert 'id="addon-scans"' in page and "/addons/" not in page          # no add-on: the slot stays empty


def test_the_page_can_hand_over_tilts_and_a_trim():
    """What the scanner measured and what the user typed reach the session through the routes."""
    import app

    paths, _ = scans("grid")
    c = app.app.test_client()
    tilts = [0.0, 11.0, -8.0, 14.0]
    r = c.post("/api/open", json={"paths": paths, "name": "tilted", "tilts": tilts}).get_json()
    assert r["tilts"] == tilts and r["trim_mm"] == app.TRIM_MM      # a flatbed set is trimmed unless told not to
    assert c.post("/api/open", json={"paths": paths, "tilts": [1.0]}).status_code >= 400   # one per scan
    assert c.post("/api/session/settings", json={"trim_mm": 99}).get_json()["trim_mm"] == 25.0
    assert c.post("/api/session/settings", json={"trim_mm": 3}).get_json()["trim_mm"] == 3.0
    assert abs(np.max(app.STATE["session"].trim_px()) - 3 * 1200 / 25.4) < 1e-6   # no dpi in the file


if __name__ == "__main__":
    failed = 0
    for name, fn in [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]:
        t0 = time.time()
        try:
            fn()
            print(f"ok    {name}  ({time.time() - t0:.1f}s)")
        except Exception as e:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print("all passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)
