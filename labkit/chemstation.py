"""Read an Agilent ChemStation run folder: the traces, and what made them.

A ``.D`` directory is one injection.  Inside it each detector channel is a
``.ch`` file -- ``MWD1A.ch`` is one wavelength of the diode array, ``FLD1A.ch``
the fluorescence channel -- and the numbers in it are a delta stream, not a
table.  Nothing here writes: a run folder is the instrument's, read only.

Only format version **130** is decoded.  A file that says anything else is
reported by its version rather than guessed at, because a wrong offset reads
as a plausible, wrong chromatogram.

The layout, measured on a 2026-10-16 S200 run off an Asterix ChemStation
(B.06.52), all big-endian:

    0x000  byte length, then that many UTF-16 characters: the version
    0x11A  int32  start of the run, in milliseconds
    0x11E  int32  end of the run
    0x35A  sample name          0x758  operator       0x957  date and time
    0xA0E  method               0xC11  instrument     0xFD7  vial position
    0x104C unit                 0x1075 signal description
    0x1274 double intercept     0x127C double slope
    0x1800 the data, in blocks

Each block is a uint16 whose high byte is 0x10 and whose low byte is the
number of int16 deltas that follow; a delta of -32768 means the next four
bytes are an absolute int32 instead.  The deltas run on across blocks, so a
block is a length and nothing more.
"""

from __future__ import annotations

import re
import statistics
import struct
from pathlib import Path

DATA = 0x1800
#: offset -> what the Pascal-style UTF-16 string there is called
TEXT = {0x35A: "sample", 0x758: "operator", 0x957: "when", 0xA0E: "method",
        0xC11: "instrument", 0xFD7: "vial", 0x104C: "units",
        0x1075: "description"}


def _text(d: bytes, off: int) -> str:
    n = d[off]
    return d[off + 1:off + 1 + 2 * n].decode("utf-16le", "replace").strip()


def _values(d: bytes) -> list[int]:
    """The delta stream, as the counts the detector recorded."""
    out, val, i = [], 0, DATA
    while i + 2 <= len(d):
        head = struct.unpack_from(">H", d, i)[0]
        i += 2
        if head >> 8 != 0x10 or head & 0xFF == 0:
            break                       # the tail of the file, not a block
        for _ in range(head & 0xFF):
            if i + 2 > len(d):
                break
            (step,) = struct.unpack_from(">h", d, i)
            i += 2
            if step == -0x8000:         # an absolute value, not a step
                (val,) = struct.unpack_from(">i", d, i)
                i += 4
            else:
                val += step
            out.append(val)
    return out


def signal(path: Path | str) -> dict:
    """One ``.ch`` file: its trace in detector units, and its own labels."""
    path = Path(path)
    d = path.read_bytes()
    version = d[1:1 + d[0] * 1].decode("latin-1").strip("\x00")
    if version != "130":
        raise ValueError("%s is ChemStation format %r, and this reads 130"
                         % (path.name, version))
    out = {k: _text(d, off) for off, k in TEXT.items()}
    start, end = struct.unpack_from(">ii", d, 0x11A)
    intercept, slope = struct.unpack_from(">dd", d, 0x1274)
    counts = _values(d)
    n = len(counts)
    step = (end - start) / (n - 1) if n > 1 else 0
    out.update(
        name=path.stem,
        file=path.name,
        start_min=start / 60000.0,
        end_min=end / 60000.0,
        n=n,
        x=[(start + i * step) / 60000.0 for i in range(n)],
        y=[intercept + c * slope for c in counts],
    )
    return out


# ------------------------------------------------------------------- peaks

def noise(y: list[float], win: int | None = None) -> float:
    """The baseline's band: the median peak-to-peak of short windows.

    Not the median step between two points -- that is the detector's own
    jitter and a slow wander of ten times its size sails past it.  A window
    of a twentieth of the run is wider than the jitter and narrower than any
    peak, so most windows hold baseline only and the median is the band the
    baseline wanders in, peaks or no peaks.

    Each window is read against the line between its own ends, because a
    baseline that falls 25 mAU across a run -- this run's first injection at
    230 nm does -- otherwise measures as a band of 1 mAU and hides every
    peak on it.
    """
    n = len(y)
    if n < 10:
        return 0.0
    win = win or max(5, n // 20)
    bands = []
    for i in range(0, n - win + 1, win):
        seg = y[i:i + win]
        slope = (seg[-1] - seg[0]) / (len(seg) - 1)
        flat = [v - slope * k for k, v in enumerate(seg)]
        bands.append(max(flat) - min(flat))
    return statistics.median(bands)


def peaks(x: list[float], y: list[float], floor: float | None = None,
          min_width: float = 0.05) -> list[dict]:
    """Maxima that stand out and have a width, as a peak must.

    A local maximum is not a peak: a trace turns over at every second point.
    What counts is how far a maximum stands above the higher of the two
    valleys flanking it -- its prominence -- and that has to beat three
    times the baseline's band.  And a peak is wide: the two artifacts in
    this run's A4 fluorescence trace stand 20 bands above the baseline and
    are one point across, so prominence alone calls a speck of dust a peak.
    A peak has to be at least *min_width* minutes wide at half its
    prominence, which no single point ever is.

    ponytail: the walk out of each maximum is O(n) per candidate and so
    O(n*k) in all; at 3000 points and a handful of candidates that is
    nothing.  A prominence stack is the upgrade if a trace ever gets long.
    """
    n = len(y)
    if n < 5:
        return []
    if floor is None:
        floor = 3 * noise(y)
    if floor <= 0:
        return []

    tops = []                           # plateaux collapse to their middle
    i = 1
    while i < n - 1:
        if y[i] > y[i - 1]:
            j = i
            while j + 1 < n and y[j + 1] == y[i]:
                j += 1
            if j + 1 < n and y[j + 1] < y[i]:
                tops.append((i + j) // 2)
            i = j + 1
        else:
            i += 1

    found = []
    for t in tops:
        left = t
        while left > 0 and y[left - 1] <= y[t]:
            left -= 1
        right = t
        while right < n - 1 and y[right + 1] <= y[t]:
            right += 1
        prom = y[t] - max(min(y[left:t + 1]), min(y[t:right + 1]))
        if prom < floor:
            continue
        half = y[t] - prom / 2
        a, b = t, t
        while a > left and y[a - 1] >= half:
            a -= 1
        while b < right and y[b + 1] >= half:
            b += 1
        width = x[b] - x[a]
        if width < min_width:
            continue
        found.append({"i": t, "t": x[t], "v": y[t], "prominence": prom,
                      "width_min": width})

    # Two maxima of the same height each walk past the other to the baseline,
    # so both report the whole peak's prominence.  Where the dip between two
    # of them is shallower than the floor they are one peak; the taller wins.
    merged: list[dict] = []
    for p in found:
        if merged:
            a, b = merged[-1]["i"], p["i"]
            if min(y[a:b + 1]) > max(y[a], y[b]) - floor:
                if p["v"] > merged[-1]["v"]:
                    merged[-1] = p
                continue
        merged.append(p)
    for p in merged:
        p.pop("i")
    return merged


# -------------------------------------------------------------- the folder

def _thin(x: list[float], y: list[float], cap: int = 1200):
    """Fewer points to draw, keeping every apex.

    Plain decimation walks past the top of a peak.  Each bucket gives up its
    lowest and its highest point, in the order they were measured, so a peak
    one point wide survives the trip to the browser.
    """
    n = len(y)
    if n <= cap:
        return x, y
    size = -(-n // (cap // 2))
    xs, ys = [], []
    for s in range(0, n, size):
        lo = hi = s
        for i in range(s, min(s + size, n)):
            if y[i] < y[lo]:
                lo = i
            if y[i] > y[hi]:
                hi = i
        for i in sorted((lo, hi)):
            xs.append(x[i])
            ys.append(y[i])
    return xs, ys


def _flow(folder: Path) -> float | None:
    """mL/min, out of the acquisition method the instrument wrote down."""
    acq = folder / "acq.txt"
    if not acq.exists():
        return None
    text = acq.read_bytes().decode("utf-16", "ignore")
    m = re.search(r"^Flow:\s+([\d.]+)\s*mL/min", text, re.M)
    return float(m.group(1)) if m else None


def injection(folder: Path | str, cap: int = 1200) -> dict:
    """One ``.D`` folder: every channel in it, thinned for drawing."""
    folder = Path(folder)
    flow = _flow(folder)
    out = {"dir": folder.name, "path": str(folder), "flow_ml_min": flow,
           "signals": [], "unreadable": []}
    for ch in sorted(folder.glob("*.ch")):
        try:
            s = signal(ch)
        except Exception as exc:                      # a channel, not the run
            out["unreadable"].append("%s: %s" % (ch.name, exc))
            continue
        s["noise"] = noise(s["y"])
        s["peaks"] = peaks(s["x"], s["y"])
        for key in ("sample", "operator", "when", "method", "instrument", "vial"):
            out.setdefault(key, s.pop(key))
            s.pop(key, None)
        x, y = _thin(s["x"], s["y"], cap)
        s["x"] = [round(v, 4) for v in x]      # 6 ms, and half the JSON
        s["y"] = [round(v, 5) for v in y]
        out["signals"].append(s)
    return out


def run(folder: Path | str, cap: int = 1200) -> dict:
    """A whole sequence folder, oldest injection first.

    ``ACQUIRING.TXT`` names the one the instrument is still filling, which is
    why a trial run looks short rather than empty.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError("%s is not a folder" % folder)
    dirs = sorted(p for p in folder.iterdir() if p.suffix.upper() == ".D")
    if not dirs:
        raise ValueError("no .D injections in %s -- is this a run folder?"
                         % folder.name)
    busy = ""
    acquiring = folder / "ACQUIRING.TXT"
    if acquiring.exists():
        busy = acquiring.read_bytes().decode("utf-16", "ignore").strip()
    runs = [injection(p, cap) for p in dirs]
    for r in runs:
        r["acquiring"] = r["dir"].upper() in busy.upper()
    return {"folder": str(folder), "name": folder.name, "runs": runs}


# ------------------------------------------------------------ for purifig

def export_text(folder, dirs, channel: str, flow: float | None = None,
                names: dict | None = None) -> str:
    """The chosen channel of several injections, as purifig already reads it.

    A UNICORN ASCII export is *pairs* of columns -- one (x, value) pair per
    curve, each with its own x, because the curves are not sampled together.
    Ours are not either: the fluorescence channel of this run has 2778 points
    where the diode array has 3001.  purifig reads that layout, so nothing
    needs writing on its side; a short column simply runs out.

    The curve's name goes in both cells of its pair. A single curve is not a
    paired layout any more -- two columns are a plain x/y table -- and the
    reader takes the name from the second column there and from the first
    here, so writing it twice is one curve name either way.

    *flow* in mL/min turns the x axis into volume, which is what a size
    exclusion figure is drawn against; without it the export is in minutes
    and says so.
    """
    folder = Path(folder)
    names = names or {}
    cols = []
    for name in dirs:
        path = folder / name
        chs = [p for p in path.glob("*.ch") if p.stem == channel]
        if not chs:
            raise ValueError("%s has no %s" % (name, channel))
        s = signal(chs[0])
        cols.append((names.get(name) or s["vial"] or name,
                     [v * flow for v in s["x"]] if flow else s["x"],
                     s["y"], s["units"]))
    if not cols:
        raise ValueError("nothing to export")

    xunit = "mL" if flow else "min"
    head, unit = [], []
    for label, _x, _y, units in cols:
        head += [label, label]
        unit += [xunit, units]
    rows = ["\t".join(head), "\t".join(unit)]
    for i in range(max(len(x) for _l, x, _y, _u in cols)):
        line = []
        for _label, x, y, _units in cols:
            line += ["%.4f" % x[i], "%.5f" % y[i]] if i < len(x) else ["", ""]
        rows.append("\t".join(line))
    return "\n".join(rows) + "\n"


# -------------------------------------------------------------- self-check

def _fixture(want, description="test"):
    """A one-minute ``.ch`` written to the rules above -- counts x 2.

    purifig checks its own side of the import against this too, so it lives
    out here rather than inside the check below.
    """
    body = bytearray()
    prev, i = 0, 0
    while i < len(want):                       # blocks of three, one escape
        take = want[i:i + 3]
        body += struct.pack(">H", 0x1000 | len(take))
        for v in take:
            if v == 900:                       # an absolute value
                body += struct.pack(">hi", -0x8000, v)
            else:
                body += struct.pack(">h", v - prev)
            prev = v
        i += 3
    d = bytearray(DATA) + body
    d[0:4] = b"\x03" + "130".encode("latin-1")
    struct.pack_into(">ii", d, 0x11A, 0, 60000)             # a one-minute run
    struct.pack_into(">dd", d, 0x1274, 0.0, 2.0)            # counts x 2
    d[0x1075] = len(description)
    d[0x1076:0x1076 + 2 * len(description)] = description.encode("utf-16le")
    return bytes(d)


def _check():
    """The decoder, against a stream written to the same rules."""
    want = [0, 5, 5, 4, 9, 40, 9, 4, 4, 4, 4, 3, 3, 3, 900, 3, 3]
    d = _fixture(want)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "MWD1A.ch"
        path.write_bytes(d)
        s = signal(path)
    assert s["n"] == len(want), (s["n"], len(want))
    assert [round(v) for v in s["y"]] == [v * 2 for v in want], s["y"][:5]
    assert s["description"] == "test" and s["units"] == ""
    assert s["x"][0] == 0 and abs(s["x"][-1] - 1) < 1e-9, s["x"][-1]

    x, y = s["x"], s["y"]
    got = peaks(x, y, floor=10, min_width=0)   # seventeen points, no width
    assert [round(p["v"]) for p in got] == [80, 1800], got   # not the 5s
    assert len(peaks(x, y, min_width=0)) == 1  # the band of this one is wide
    assert noise([0, 1] * 30) == 1, noise([0, 1] * 30)
    # One point standing twenty bands high is a speck of dust, not a peak.
    spike = [0.0, 0.1] * 100
    spike[101] = 4.0
    assert peaks(list(range(200)), spike, min_width=2) == []
    # Two maxima of one height, one dip: one peak, not two.
    flat = [0, 0, 9, 9, 8.95, 9, 9, 0, 0]
    assert len(peaks(list(range(len(flat))), flat, floor=1, min_width=0)) == 1
    xs, ys = _thin(list(range(1000)), [0] * 499 + [7] + [0] * 500, cap=100)
    assert max(ys) == 7 and len(xs) <= 110, (max(ys), len(xs))

    # The export, read back the way purifig reads it: two header rows, then a
    # pair of columns per curve, each pair carrying its curve's name.
    with tempfile.TemporaryDirectory() as tmp:
        for well in ("A1", "A2"):
            (Path(tmp) / (well + ".D")).mkdir()
            (Path(tmp) / (well + ".D") / "MWD1A.ch").write_bytes(d)
        text = export_text(tmp, ["A1.D", "A2.D"], "MWD1A", flow=0.5,
                           names={"A1.D": "A1 target"})
        lines = [l.split("\t") for l in text.strip().split("\n")]
    assert lines[0] == ["A1 target", "A1 target", "A2.D", "A2.D"], lines[0]
    assert lines[1] == ["mL", "", "mL", ""], lines[1]
    assert len(lines) == 2 + len(want), len(lines)
    assert lines[-1][0] == "%.4f" % (1.0 * 0.5), lines[-1]   # a minute at 0.5
    print("ok  %d points decoded, %d peaks, the apex survives thinning"
          % (s["n"], len(got)))


if __name__ == "__main__":
    _check()
