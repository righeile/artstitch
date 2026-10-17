#!/usr/bin/env python3
"""A microplate: its well names, and a layout saying what is in each well.

    from labkit import plate

    plate.canon_well("r02c01")         -> "B01"      (also "b1", "B-01")
    plate.label_wells(layout, wells)   -> {"variant": ["WT", None, ...]}
    plate.field_files(path)            -> the channel files of one field

A plate reader names an image by where it was taken, not by what was in the
well: ``r02c01f03p01-ch2sk1fk1fl1.tiff`` is row 2, column 1, field 3, channel
2.  The design is therefore not in the filename at all.  It is in the layout
-- which wells got which construct, which got the drug -- and `names` can only
get as far as the well.  This turns the well into the columns everything else
groups by.

A layout is plain JSON, so it can sit next to the results it labeled::

    {"format": "96-well",
     "labels": {"variant":   {"B01": "WT", "B02": "WT", "E01": "Y715C"},
                "treatment": {"B01": "DMSO", "B02": "baf"}}}

One entry of ``labels`` per column it makes: a two-factor design is two
entries, not a tree.

Only the standard library.  `browse` imports this, and `browse` has to run in
the apps that have no pandas.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

__all__ = ["FORMATS", "well_id", "canon_well", "clean_layout", "label_wells",
           "harmony_parts", "field_files", "later_channel"]

#: (rows, columns) of each SBS microplate.
FORMATS: Dict[str, Tuple[int, int]] = {
    "6-well": (2, 3), "12-well": (3, 4), "24-well": (4, 6),
    "48-well": (6, 8), "96-well": (8, 12), "384-well": (16, 24),
}

_WELL = re.compile(r"^([A-Za-z])[\s_-]?(\d{1,2})$")
_ROW_COL = re.compile(r"^r(\d{1,2})c(\d{1,2})$", re.I)


def well_id(row: int, col: int) -> str:
    """0-based (row, column) -> ``"B01"``.  Always two digits, so that ids
    compare and sort as text."""
    return f"{chr(65 + row)}{col + 1:02d}"


def canon_well(text) -> Optional[str]:
    """Any spelling of a well -> ``"B01"``; None when it is not one.

    ``B1``, ``b01``, ``B-1`` and Harmony's ``r02c01`` are the same well, and a
    layout typed by hand has to meet filenames written by an instrument.  The
    range is the largest plate here, A01 to P24: beyond it a letter and a
    number is far more likely a clone name than a well.
    """
    s = str(text).strip()
    m = _ROW_COL.match(s)
    if m:
        row, col = int(m.group(1)) - 1, int(m.group(2)) - 1
    else:
        m = _WELL.match(s)
        if not m:
            return None
        row, col = ord(m.group(1).upper()) - 65, int(m.group(2)) - 1
    return well_id(row, col) if 0 <= row < 16 and 0 <= col < 24 else None


def clean_layout(layout) -> dict:
    """A layout as this module writes it, from whatever was handed in.

    Well ids are canonicalized and empty labels dropped, so ``b1`` typed into a
    JSON file and ``B01`` clicked in the editor are one well.  A layout saved
    by Scrambler (``plate_format`` and ``assignments``) reads as a single
    ``condition`` column, because that is what it holds.
    """
    layout = layout if isinstance(layout, dict) else {}
    labels = layout.get("labels")
    if not isinstance(labels, dict):
        labels = {"condition": layout.get("assignments") or {}}
    out: Dict[str, Dict[str, str]] = {}
    for column, by_well in labels.items():
        kept = {canon_well(w): str(v).strip() for w, v in dict(by_well or {}).items()
                if canon_well(w) and v is not None and str(v).strip()}
        if kept and str(column).strip():
            out[str(column).strip()] = dict(sorted(kept.items()))
    fmt = layout.get("format") or layout.get("plate_format")
    return {"format": fmt if fmt in FORMATS else "96-well", "labels": out}


def label_wells(layout, wells: Iterable) -> Dict[str, List[Optional[str]]]:
    """What the layout says about each of `wells`, one list per column.

    None where it says nothing -- a well the layout does not mention, or a
    value that is not a well at all -- so the caller can tell "unlabeled" from
    a label and leave whatever was there alone.
    """
    ids = [canon_well(w) for w in wells]
    return {column: [by_well.get(w) for w in ids]
            for column, by_well in clean_layout(layout)["labels"].items()}


# --------------------------------------------------------------------------
# Harmony (Opera Phenix, Operetta): one TIFF per channel of each field
# --------------------------------------------------------------------------

#: row, column, field, plane - channel, slow-kinetic step, fast-kinetic step,
#: FLIM bin.  The channel number is its own group because it is the one piece
#: that differs between the files of a single field.
_HARMONY = re.compile(r"^(r\d+c\d+)(f\d+)(p\d+)-(ch)(\d+)(sk\d+)(fk\d+)(fl\d+)$", re.I)


def harmony_parts(name) -> Optional[List[str]]:
    """``r02c01f03p01-ch2sk1fk1fl1.tiff`` ->
    ``["B01", "f03", "p01", "ch2", "sk1", "fk1", "fl1"]``; None for any other name.

    The well comes back under the name it has on the plate, because that is the
    name a layout and a lab book use; the rest are Harmony's own.
    """
    m = _HARMONY.match(Path(str(name)).stem)
    well = canon_well(m.group(1)) if m else None
    if not well:
        return None
    return [well, m.group(2), m.group(3), m.group(4) + m.group(5), *m.groups()[5:]]


def _around_channel(name: str) -> Optional[Tuple[str, int, str]]:
    """(text before the channel number, the number, text after it)."""
    stem, suffix = Path(name).stem, Path(name).suffix
    m = _HARMONY.match(stem)
    return (stem[:m.start(5)], int(m.group(5)), stem[m.end(5):] + suffix) if m else None


def field_files(path) -> List[Path]:
    """Every channel of the field that `path` is one channel of, in order.

    A two-color field is two files differing only in ``ch1``/``ch2``; a reader
    that takes one of them for the image has lost the other color.  Any other
    name comes back as itself, so a caller can use this without asking first.
    """
    path = Path(path)
    around = _around_channel(path.name)
    if not around:
        return [path]
    head, _, tail = around
    # ponytail: probes ch1..ch16 by name rather than globbing a folder that can
    # hold 100,000 files. Widen the range if an instrument numbers past 16.
    found = [q for k in range(1, 17) if (q := path.with_name(f"{head}{k}{tail}")).exists()]
    return found or [path]


def later_channel(name: str, names=None) -> bool:
    """True for a channel of a field whose earlier channel is also in `names`.

    These are the files a listing leaves out so that a field appears once:
    opening either file reads the whole field (`field_files`), and listing both
    would run every field twice.  Without `names`, every channel after the
    first counts -- the answer for a caller that cannot read the folder first.
    """
    around = _around_channel(name)
    if not around:
        return False
    if names is None:
        return around[1] > 1
    return any(f"{around[0]}{k}{around[2]}" in names for k in range(1, around[1]))


if __name__ == "__main__":                                   # python -m labkit.plate
    import tempfile

    assert [canon_well(s) for s in ("B1", "b01", "B-1", "r02c01", " H12 ")] == \
        ["B01", "B01", "B01", "B01", "H12"]
    assert canon_well("P24") == "P24" and canon_well("r16c24") == "P24"
    assert [canon_well(s) for s in ("Q1", "A25", "A0", "WT1", "Y715C", "1", "", None)] == \
        [None] * 8, "past a 384-well plate it is a name, not a well"

    lay = {"format": "96-well", "labels": {"variant": {"b1": "WT", "E01": " Y715C ", "H1": ""}}}
    assert clean_layout(lay) == {"format": "96-well",
                                 "labels": {"variant": {"B01": "WT", "E01": "Y715C"}}}
    assert label_wells(lay, ["r02c01", "E1", "H01", "dish3"]) == \
        {"variant": ["WT", "Y715C", None, None]}
    assert clean_layout(None) == {"format": "96-well", "labels": {}}
    assert clean_layout({"plate_format": "24-well", "assignments": {"A01": "ctrl"}}) == \
        {"format": "24-well", "labels": {"condition": {"A01": "ctrl"}}}, "a Scrambler layout"

    name = "r05c01f79p01-ch2sk1fk1fl1.tiff"
    assert harmony_parts(name) == ["E01", "f79", "p01", "ch2", "sk1", "fk1", "fl1"]
    assert harmony_parts("20260815_1_lysosensor_Y715C_0.czi") is None
    both = {name, name.replace("ch2", "ch1")}
    assert later_channel(name, both) and not later_channel(name.replace("ch2", "ch1"), both)
    assert not later_channel(name, {name}), "a channel copied on its own still lists"
    assert later_channel(name) and not later_channel(name.replace("ch2", "ch1"))
    assert not later_channel("B1_3.tif", {"B1_3.tif"})

    with tempfile.TemporaryDirectory() as tmp:
        for k in (2, 1):
            (Path(tmp) / name.replace("ch2", f"ch{k}")).touch()
        (Path(tmp) / "plain.tif").touch()
        assert [p.name for p in field_files(Path(tmp) / name)] == \
            [name.replace("ch2", "ch1"), name], "channel order, whichever file was named"
        assert field_files(Path(tmp) / "plain.tif") == [Path(tmp) / "plain.tif"]

    print("labkit.plate: all checks passed.")
