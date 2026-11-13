# ArtStitch

## Installation

ArtStitch runs on your own computer and opens in your browser. It needs Python
3.12 or newer (`python3 --version` tells which you have; python.org has the
installers).

```bash
git clone https://github.com/righeile/artstitch.git
cd artstitch
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then start it: double-click **`run_mac.command`** (it uses the `.venv` by itself,
offers to install anything missing, and opens the page), or

```bash
.venv/bin/python app.py      # then open http://127.0.0.1:8800
```

On Windows, install into the Python that `python` runs
(`python -m pip install -r requirements.txt`), then double-click
**`run_windows.bat`** or run `python app.py`. The Windows launcher has not been
tried on Windows yet.

`--port` changes the port; `--state` changes where presets are kept (default
`~/.artstitch`). Starting a launcher stops an older copy still running.
`.venv/bin/python test_artstitch.py` (see Checks) confirms the install.

## What it does

Flatbed scans of one artwork, in two to eight parts (more works too), into one
picture. The parts are found by name, matched to each other, cut apart along the
path where they already agree (nothing is blended, so nothing is softened),
straightened, cropped, color-corrected and saved in as many versions as you
like. A folder with many artworks in it is done the same way, unattended.

The matching is ASFQ's stitching engine (masked normalized cross-correlation by
FFT, dense displacement mesh, graph-cut seams), with a rotation search added and
the histology parts taken out.

## Pages

1. **Scans**: point at the folder the scanner wrote. Scans of one artwork are
   found by name (`name`, `name 1`, `name 2` …); nothing is moved or modified.
   A file far larger than the others of its name is taken for an earlier result
   and left unchecked. Files can also be chosen by hand, and a saved session
   (`.artstitch.json`) opened again. Scans of 8 or 16 bits a channel are used as
   they are; 32-bit ones (floating point, light kept linear as Photoshop keeps it)
   are brought to 16 bits through the scan's color profile.
2. **Stitch**: every pair is searched for its overlap, rotation included
   (±3° by default; the box is on the page). Scans that came from ArtStitch Scan
   bring the tilt the scanner measured for each part, and the search starts from
   it, so the ±3° is spent on what the scanner could not know rather than on the
   tilt it already did. **Trim where it runs off the glass … mm**
   (12 by default) cuts that much off a scan before the overlaps are matched in
   detail, the seams cut or a pixel painted, on the sides where the sheet ran off
   the glass and lifted off it, the picture bending with it. Those are the sides
   another scan stands behind, which is what makes them a seam, so the neighbour
   supplies what comes off. The artwork's own outer edges lay flat under the lid
   and keep every pixel: nothing could give them back. A set of one scan is never
   cut, and never more than a tenth of a scan. Checks says what came off which. A scan that lay on its side or upside
   down in the scanner is found turned; the result stands the way the **first
   scan** does, and Checks names the scans that lay turned against it. Scans that
   no single pair ties together are joined group against group, the way halves
   or pairs are stitched first by hand. Checks lists what deserves a look:
   scans that still disagree after the alignment, scans that share nothing with
   the others (they are left standing apart, not joined by a guess). **Place scans by
   hand** opens a canvas to drag a scan to where it belongs and turn it; roughly
   is enough, the exact position is still found. It also starts from nothing
   ("Line them up"), and a scan on its side only needs turning by 90°.
3. **Straighten & crop**: the paper edges are found from the scanner lid showing
   around the sheet, at full resolution, and give the angle and the largest
   rectangle inside the sheet. Without edges: type an angle, drag the slider, or
   **draw a level line** along something that should be horizontal. The crop
   rectangle can be dragged or typed. **↺ ↻** turn the result by 90°; a turn
   moves whole pixels and resamples nothing.
4. **Patch**: dust, specks and hairs. Click on a speck, drag along a hair; the
   brush is in pixels of the full-size picture (`[` `]` change it). Each stroke is
   filled with a copy of the picture from close by, where the ring around the
   stroke matches best, shifted in color and brightness to meet its edge, so the
   grain carries on and nothing is blurred. A long stroke is filled in stretches
   (about two brush widths each), each copied from a place of its own, so a hair
   across several edges carries each of them on. Hover over a patch to see where
   its copies came from (yellow) and drag one somewhere better if the patch
   shows; the stretches after it then choose again to meet it. Option-click takes
   a patch out; ⌘Z undoes. Patches stay on their dust
   whatever the straightening, crop or turn, are saved with the session and go
   into every export. Stitching again removes them: the pixels under them may
   have moved.
   **Find dust** looks through the whole result (about 45 s for 300 megapixels)
   for hairs, specks and lint that stand out from the grain round them, by
   default 12 times as much; two sliders go lower, hairs to 8 times and specks
   to 10. A hair is a line thinner than a brush paints
   (3.5 px across at half its depth) with the same paint on both sides, so the
   edge of a watercolor wash is not one. Each is circled and shown as a small
   picture, hairs first, strongest first. Paint can stand out that much too:
   click a picture to leave it out, double-click to see it in place; **All**,
   **None**, **Hairs** and **Specks** choose many at once. **Patch N** patches
   the rest in one step, which ⌘Z takes back whole. A hair's side branch or faint
   tail may be left for the brush; straightening or cropping again forgets what
   was found.
5. **Color**: levels and curve per channel (or all three), then temperature,
   tint, lightness, contrast, saturation and hue. A curve's end points move
   sideways as well as up and down; past them the curve is flat, as in
   Photoshop. The preview and the export run the same code. Settings can be
   saved by name.
6. **Export**: a *preset* is a list of files to make from one result, each with a
   file name of its own. **Batch** is the same steps for a whole folder, all but Patch.

## Export presets

The result is rendered once, strip by strip, and every file of the preset is made
from that pass, so a 300-megapixel result needs a few hundred megabytes of memory.

| File | Sizes |
|---|---|
| TIFF (one per preset) | Always full size and lossless; 8, 16 or 32 bits a channel (**bits as the scans** by default). Compression: `zopfli` (default; Deflate squeezed a little harder, opens anywhere), `deflate`, `lzma`, `zstd` (smaller still, but not for Photoshop), `lzw`, `none`. BigTIFF when needed; the scan's dpi and ICC profile are kept. **Photoshop layers**: see below |
| Photoshop PSD (one per preset) | Always full size and lossless, 8, 16 or 32 bits a channel, with the layers below. Written as a PSB (`.psb`) above 30 000 px a side, or when it could pass 2 GB (from about 220 megapixels at 8 bits, 110 at 16, 55 at 32) |
| PNG, JPEG, PDF | 8 bits a channel. A dpi (300), a long side in pixels, full size, or **under N MB** (the largest picture that fits; MB is 1 000 000 bytes). A PDF is one page the size of the artwork, the picture in it lossless (as in a PNG) or JPEG |

**32 bits** are Photoshop's: floating point, holding light linearly, through the
file's color profile. Such a file is large, since floating point hardly compresses
(about six times the size at 8 bits), and it holds no adjustment layers: the
color is in the scan and patch layers instead, because Photoshop's levels and
curves act differently on linear light.

**Photoshop layers.** A PSD, and a TIFF with the box checked, hold each scan as a
layer of its own, named after it: the whole scan as far as the picture reaches,
placed and matched in brightness, before the color adjustments. A layer mask cuts
each scan to its part of the stitch; together the masks blend the scans exactly as
the stitch does, so painting on a mask moves a seam. The patches are a layer above
the scans that holds only the patched pixels. Each color adjustment is a Photoshop
adjustment layer above that, in the order they are applied: Levels
(red, green, blue), Curves (red, green, blue), Levels, Curves, and a Channel Mixer
for hue, saturation, light, contrast, temperature and tint (Photoshop's own Hue/
Saturation works differently; the mixer is the same arithmetic). Adjustments left
at zero get no layer. The finished picture is stored as well, so every other
program shows the result; the file is two to three times as large as without
layers (a 322-megapixel watercolor of five scans: a 1.4 GB TIFF instead of 0.5 GB).
Photoshop recomputes the picture from the layers. The scans and patches give it
back to within a level. The adjustments can move it by a few levels: level points
are whole steps of 255, the mixer is in whole percent, and Photoshop draws its own
curve through the same points.

File names take placeholders: `{name} {n} {date} {preset} {dpi}`. *If a file
already exists*: add a number, skip it, or replace it. A scan is never overwritten.
The default preset is `{name}.tiff`, `{name}_300dpi.png`, `{name}_small.png`
(under 10 MB) and `{name}.jpg`, saved in `stitched/` beside the scans (or any
full path). Presets are kept in `~/.artstitch`.

## Batch

**Find sets** groups the folder by name; sets of at least two scans are checked.
**Run the batch** stitches, straightens (by the paper edges; without them by the
picture's own straight lines if it has strong ones, otherwise left as scanned),
crops, colors with the chosen saved settings and saves with the chosen preset.
A set that fails is reported and the others carry on. That includes a set whose
scans cannot all be matched, in any quarter turn, alone or as groups (a scan of
another sheet under the same name, say): nothing is exported, its session is
saved, and opening it shows the groups side by side, ready to be placed by hand.
A set that finished with something worth a look (scans that still disagree, no
paper edges, paper edges that disagree, a small crop) is marked **review**; its
`.artstitch.json` sits next to its outputs, and opening it gives the editor
exactly as the batch left it. A
`artstitch_batch_<time>.csv` lists every set. With *if a file already exists* on
**skip**, running the same folder again only does what is missing.

## Add-ons

ArtStitch can be extended by packages of their own, installed into the same Python
(`.venv/bin/pip install …`). They are found when ArtStitch starts (the terminal
says `add-ons: …`), and taken out again by `pip uninstall`. Without one, nothing
changes. **ArtStitch Scan** ([righeile/artstitch-scan](https://github.com/righeile/artstitch-scan))
scans the parts of an artwork straight from a flatbed scanner on the Scans page.

An add-on declares a function under the entry-point group `artstitch.addons`;
it is called once with a `Host` (see `app.py`) that gives it the Flask app to add
routes to, the job runner, and a place for its own script and stylesheet on the
page. In the page, `window.ArtStitch` holds what its script may use (opening a
set -- with the tilt each scan lay at on the glass, if it knows it --
straightening by an angle, following a job), `ArtStitch.slot('scans')` is
where it puts itself on the Scans page, and every change to the open set is
announced as the document event `artstitch:session`. One that fails to load is
reported in the terminal and left out.

## What to expect

- The scan the others are matched to (the **anchor**) is never resampled: its
  pixels reach the result unchanged. Every other scan is resampled once, and the
  result is quantized once at the end.
- Flatbed scans of flat work: each scan gets a rotation, a shift and a smooth
  warp (a few pixels, from the lid pressing unevenly). Curled or folded paper,
  where the scans disagree about the shape and not just the position, will show
  in the seam.
- The coarse search accepts overlaps down to 5 % of the smaller scan; a third or
  more is comfortable. A match must also stand out from chance: two sheets that
  only look alike (a lawn beside another lawn) are not taken for an overlap.
- On synthetic scans with a known truth (`test_artstitch.py`) two scans agree to
  about 0.1 px rms after stitching, from a blind search and from a hand
  placement half a scan off and a degree crooked.

## Checks

```bash
python3 test_artstitch.py       # about 90 s: alignment (up to eight scans, turned scans, groups), edges, patches (a hair in stretches), dust, color, export (TIFF, PSD, PDF, layers, 8/16/32 bits), batch, session
```

The alignment checks cut a synthetic painting into scans of known pose and
compare where the result puts the same paint.
