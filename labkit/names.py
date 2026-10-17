#!/usr/bin/env python3
"""Acquisition filenames, taken apart.

    from labkit import names

    names.delimiters("_ OR -")            -> ["_", "-"]
    names.auto_meta(files)                 -> a DataFrame of date/variant/dish/...
    names.stamp_groups(objects, per_image) -> writes the condition label

A microscope filename carries the experiment's design in it --
``20260815_1_lysosensor_Y715C_0`` is date, dish, experiment, variant, field --
and every tool in this folder has to get that design back out before it can
group anything, color anything, or match a control to its own replicate.

**Nothing here guesses on one name alone.** Which position carries the
condition is only visible across the whole batch: one file cannot tell you that
part 3 is the thing that varies. So the split proposes a role per position and
the caller overrides any of them, and files with a different number of parts are
a separate *scheme* rather than being force-fitted -- silently assigning `date`
to a field-of-view number corrupts every date-matched comparison downstream, and
does it quietly.

The delimiter is a spec, not a character: ``"_ OR -"``, ``"_|-"`` and ``"_,-"``
each name two, and longest is tried first so ``"__ OR _"`` works.

Parsed columns are what the apps' overlays and replicate matching are built on
-- which column panels a plot, which value is the control, which columns a
control is matched on -- so they live here rather than in one app.

Only `re` and `pandas`; no plotting, no app.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .plate import harmony_parts

__all__ = [
    "ROLES", "ROLE_HELP", "GROUPING_ROLES", "GROUP_PREFERENCE",
    "delimiters", "split_name", "suggest_roles", "name_schemes", "parse_names",
    "merge_meta", "auto_meta", "auto_group_map", "stamp_groups", "label_one",
    "pick_unit_column", "n_experiments",
]

#: Suggested labels for a filename part. This list is a convenience, not a
#: constraint: `parse_names` accepts ANY string as a label and makes a column of
#: that name, so a naming scheme carrying something not listed here (a drug, a
#: cell line, a concentration) just needs the label typed in. The only structural
#: requirement on a filename is that its parts are separated by underscores.
#: "ignore" is the one reserved value — it drops the part.
ROLES = ["ignore", "date", "replicate", "dish", "experiment", "variant",
         "treatment", "condition", "concentration", "cell_line", "marker",
         "timepoint", "field", "channel", "other"]

ROLE_HELP = {
    "ignore": "dropped — not carried into the table",
    "date": "acquisition date — can be used to match controls",
    "replicate": "biological replicate — can be used to match controls",
    "dish": "dish / well — the unit that bootstrapping resamples within, "
            "and where a plate layout reads the well from",
    "experiment": "experiment or stain name, e.g. 'lysosensor'",
    "variant": "genotype / mutant — one plot panel per value by default",
    "treatment": "drug or treatment — can define panels, or be matched on",
    "condition": "any other experimental condition",
    "concentration": "dose or concentration",
    "cell_line": "cell line or patient line",
    "marker": "stain or reporter",
    "timepoint": "time point",
    "field": "field of view within a dish",
    "channel": "channel index",
    "other": "kept in the table, not used for grouping",
}

#: Labels that make sense to group panels by, or to match controls on.
GROUPING_ROLES = ["variant", "treatment", "condition", "concentration",
                  "cell_line", "marker", "timepoint", "date", "replicate",
                  "dish", "experiment"]

#: Labels that identify a resampling unit, most specific first.
_UNIT_PREFERENCE = ["dish", "replicate", "date"]

_KNOWN_VARIANTS = re.compile(r"^(WT|wt|[A-Z]\d{2,4}[A-Z]|KO|ko|HET|het)$")

#: A part that spells out its own role and then numbers itself -- `dish1`,
#: `rep 2`, `FOV_3`. Without this, `dish1` guesses as a generic "condition",
#: which is not merely untidy: no `dish` column then exists, so the statistics
#: section cannot offer `date+dish` as the experimental unit and the user is
#: left wondering why an obviously present label is not selectable.
_ROLE_WORD = re.compile(r"^(?P<word>[A-Za-z]{2,12})[\s_-]*(?P<num>\d{1,3})$")
#: Only unambiguous words. `WT1` also matches the pattern above, so the lookup
#: has to be an explicit whitelist rather than "any word followed by a number".
_ROLE_WORDS = {
    "dish": "dish", "well": "dish", "plate": "dish", "slide": "dish",
    "coverslip": "dish", "cs": "dish",
    "rep": "replicate", "reps": "replicate", "replicate": "replicate",
    "fov": "field", "field": "field", "pos": "field", "position": "field",
    "img": "field", "image": "field",
    "day": "date",
    "exp": "experiment", "experiment": "experiment",
}


def _role_from_word(part: str) -> Optional[str]:
    """`dish1` -> "dish". None when the part is not a role word plus a number."""
    m = _ROLE_WORD.match(part)
    return _ROLE_WORDS.get(m.group("word").lower()) if m else None
_DATE_LIKE = re.compile(r"^(20\d{6}|\d{6})$")
_INT_LIKE = re.compile(r"^\d{1,3}$")
#: What `plate.harmony_parts` returns, recognized from the parts alone.
_HARMONY_PARTS = re.compile(r"^[A-P]\d\d f\d+ p\d+ ch\d+ sk\d+ fk\d+ fl\d+$", re.I)


#: How a `sep` spec naming more than one delimiter is written. All three forms
#: are accepted because none of them is guessable: `"_ OR -"` reads the way the
#: request is spoken, `"_|-"` the way a regex user would write it, `"_,-"` the
#: way a list is usually typed.
_SEP_ALT = re.compile(r"\s+OR\s+|\s*\|\s*|\s*,\s*", re.I)


def delimiters(sep: str = "_") -> List[str]:
    """The literal delimiters a `sep` spec names, longest first.

    One delimiter is the common case, and `"_"` behaves exactly as it always
    did. Several are written with OR, a pipe or a comma between them --
    ``"_ OR -"``, ``"_|-"``, ``"_,-"`` -- for a folder that mixes `a_b-c`.

    Tokens are stripped, so the spaces around OR are separators in the *spec*
    and never delimiters themselves; splitting on spaces is `also_whitespace`,
    which is a separate flag because it is a decision about the names rather
    than about this string. An empty spec yields no delimiters, which leaves
    the stem whole -- the existing way to say "do not split".

    Longest first matters: with ``"__ OR _"`` a two-underscore run has to be
    tried as one delimiter before it is tried as two, or the alternation matches
    the shorter one and the longer never fires. Equal-length delimiters keep the
    order they were typed in — the sort is stable — so the result is the same
    list every time rather than whatever a set happened to iterate.
    """
    seen: List[str] = []
    for t in (t.strip() for t in _SEP_ALT.split(str(sep or ""))):
        if t and t not in seen:
            seen.append(t)
    return sorted(seen, key=len, reverse=True)


def split_name(name: str, sep: str = "_", also_whitespace: bool = False,
               split_trailing_digits: bool = False) -> List[str]:
    """Split one filename into its metadata parts.

    The contract is just: **parts are separated by the chosen delimiter(s)**,
    and the caller labels each position. Nothing else about the layout is
    assumed — any number of parts, in any order, carrying anything. `sep` may
    name several delimiters; see `delimiters`.

    The extension is dropped first. The two flags are opt-in rescues for names
    that predate that convention and are OFF by default, because both invent
    structure that is not in the name: ``also_whitespace`` also splits on runs of
    spaces (`P716L 34.czi`), and ``split_trailing_digits`` separates a trailing
    number from a leading label (`Y715C10` becomes `["Y715C", "10"]`). Leave them
    off for underscore-separated names, where `dish1` should stay one part.

    The one name that is not split on a delimiter is a plate reader's own
    (`r02c01f03p01-ch1sk1fk1fl1`): it has none between its pieces, and a fixed
    grammar instead. See `plate.harmony_parts`.
    """
    plate_parts = harmony_parts(name)
    if plate_parts:
        return plate_parts
    stem = Path(str(name)).stem
    seps = delimiters(sep) + ([" "] if also_whitespace else [])
    parts = [stem]
    if seps:
        # One alternation rather than a pass per delimiter: a run that mixes
        # them (`a_-b`) is then a single boundary instead of a boundary and an
        # empty part, which is what the old per-separator loop had to drop.
        rx = re.compile("(?:" + "|".join(re.escape(s) for s in seps) + ")+")
        parts = [q for q in rx.split(stem) if q != ""]
    if split_trailing_digits:
        nxt = []
        for p in parts:
            m = re.match(r"^([A-Za-z][A-Za-z0-9]*?)(\d{1,3})$", p)
            if m and not _KNOWN_VARIANTS.match(p):
                nxt += [m.group(1), m.group(2)]
            else:
                nxt.append(p)
        parts = nxt
    return parts


def suggest_roles(parts: Sequence[str], all_parts: Optional[Sequence[Sequence[str]]] = None
                  ) -> List[str]:
    """Propose a label per position. **Advisory only** — always let the user fix it.

    These guesses exist to save typing, not to define the format. A part that is
    a treatment, a dose or a cell line will usually come back as "variant" or
    "other", because nothing in the filename distinguishes them; the caller is
    expected to correct that.

    Uses shape (8-digit dates, known genotype spellings, parts that name their own
    role such as `dish1` or `FOV_3`, small integers) and, when
    ``all_parts`` is given, how much a position varies across the whole file set:
    a position that never changes is an experiment label, one that changes within
    every group is a field index.
    """
    n = len(parts)
    out = ["other"] * n
    varied: List[int] = []
    if all_parts:
        for i in range(n):
            vals = {tuple(p)[i] for p in all_parts if len(p) == n}
            varied.append(len(vals))
    else:
        varied = [2] * n

    if _HARMONY_PARTS.match(" ".join(parts)):
        # A plate reader's name says what each piece is. The well is the unit
        # and the field is the field; a plane, a channel or a time step is only
        # worth a column when the export holds more than one of it.
        rest = ["other", "channel", "timepoint", "other", "other"]
        return ["dish", "field"] + [r if varied[i + 2] > 1 else "ignore"
                                    for i, r in enumerate(rest)]

    for i, p in enumerate(parts):
        if _DATE_LIKE.match(p):
            out[i] = "date"
        elif _KNOWN_VARIANTS.match(p):
            out[i] = "variant"
        elif _role_from_word(p):
            out[i] = _role_from_word(p)
        elif _INT_LIKE.match(p):
            out[i] = "field" if i == n - 1 else "replicate"
        elif varied[i] <= 1:
            out[i] = "experiment"
        else:
            out[i] = "variant" if "variant" not in out else "condition"

    # A single unvarying alphabetic part is the stain/experiment name.
    for i, p in enumerate(parts):
        if out[i] == "variant" and varied[i] <= 1 and not _KNOWN_VARIANTS.match(p):
            out[i] = "experiment"
    return out


def name_schemes(names: Sequence[str], **split_kw) -> Dict[int, Dict]:
    """Group filenames by how many parts they split into.

    Returns ``{n_parts: {"files": [...], "parts": [[...], ...], "suggested": [...]}}``.
    Mixed folders are normal — `Y715C10.czi` and `0_lysosensor_Y715C_0.czi` do not
    share a layout — and each scheme gets its own role assignment rather than one
    being mangled to fit the other.
    """
    by: Dict[int, Dict] = {}
    parsed = [(nm, split_name(nm, **split_kw)) for nm in names]
    for nm, parts in parsed:
        by.setdefault(len(parts), {"files": [], "parts": []})
        by[len(parts)]["files"].append(nm)
        by[len(parts)]["parts"].append(parts)
    for n, d in by.items():
        d["suggested"] = suggest_roles(d["parts"][0], d["parts"])
    return dict(sorted(by.items(), key=lambda kv: -len(kv[1]["files"])))


def parse_names(names: Sequence[str], roles_by_count: Dict[int, Sequence[str]],
                **split_kw) -> pd.DataFrame:
    """Turn filenames into a metadata table using per-scheme role assignments.

    Repeated roles within one scheme are suffixed (`variant`, `variant_2`) rather
    than overwriting each other. Unparsed files still get a row, so nothing
    disappears silently.
    """
    rows: List[Dict] = []
    for nm in names:
        parts = split_name(nm, **split_kw)
        roles = list(roles_by_count.get(len(parts), []))
        rec: Dict[str, object] = {"image": nm, "n_parts": len(parts),
                                  "parsed": bool(roles)}
        seen: Dict[str, int] = {}
        for i, p in enumerate(parts):
            role = roles[i] if i < len(roles) else "other"
            if role == "ignore":
                continue
            seen[role] = seen.get(role, 0) + 1
            key = role if seen[role] == 1 else f"{role}_{seen[role]}"
            rec[key] = p
            rec[f"part{i}"] = p
        rows.append(rec)
    return pd.DataFrame(rows)


def merge_meta(df_obj: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Join filename-derived metadata onto the objects table, one column per role.

    A ratiometric batch already writes its own `date` into the objects table --
    `analyze_ratio_paths` stamps every row with the session key it used to pick a
    per-session calibration, under the column named `calibration.key_col`, which
    is `"date"`. `meta` derives a `date` column independently, by parsing the
    same filenames' underscore-separated parts. A plain `merge` on `image`
    colliding on that name doesn't error, it just silently renames BOTH copies
    to `date_x`/`date_y` -- so neither is called `date` any more, and every
    downstream control that only checks for the literal column name (`match
    control on`, `resampling unit`, the experimental-unit picker in Statistics)
    loses it without a word. This is why: it happens exactly when a per-session
    calibration is in force, which is exactly when matching controls by date is
    the point.

    Resolved by coalescing rather than by renaming either source: wherever the
    objects table already has a value (it is what the pipeline actually used,
    e.g. to key the calibration), that value wins; the filename guess fills in
    only what the pipeline did not supply. A non-ratiometric batch never writes
    `date` onto its own objects table, so for it this is a plain left join,
    identical to before.
    """
    shared = [c for c in meta.columns if c != "image" and c in df_obj.columns]
    obj = df_obj.merge(meta, on="image", how="left", suffixes=("", "_meta"))
    for c in shared:
        mc = f"{c}_meta"
        if mc in obj.columns:
            obj[c] = obj[c].where(obj[c].notna(), obj[mc])
            obj = obj.drop(columns=mc)
    return obj


#: Roles the default `group` label is taken from, best first. A run labeled
#: only by treatment should still get a usable `group`, so this is a list rather
#: than just "variant".
GROUP_PREFERENCE = ["variant", "condition", "treatment", "cell_line",
                    "concentration", "timepoint"]


def auto_meta(names: Sequence[str], **split_kw) -> pd.DataFrame:
    """`parse_names` with the roles `suggest_roles` proposes, one assignment per
    scheme. The starting point the naming panel shows, computed without it."""
    schemes = name_schemes(names, **split_kw)
    return parse_names(names, {n: d["suggested"] for n, d in schemes.items()},
                       **split_kw)


def auto_group_map(names: Sequence[str], *,
                   preference: Sequence[str] = GROUP_PREFERENCE,
                   **split_kw) -> Dict[str, str]:
    """``{filename: condition label}`` from the filename parts alone.

    Per name rather than per column: a folder holding two naming schemes gets a
    role assignment each, so the part carrying the condition can be `variant` in
    one scheme and `condition` in the other, and a single column would leave
    half the files unlabeled.
    """
    if not len(names):
        return {}
    meta = auto_meta(names, **split_kw)
    out: Dict[str, str] = {}
    for _, r in meta.iterrows():
        for c in preference:
            v = r.get(c)
            if isinstance(v, str) and v:
                out[str(r["image"])] = v
                break
    return out


def stamp_groups(*dfs, names: Optional[Sequence[str]] = None,
                 name_col: str = "image", group_col: str = "group",
                 **split_kw) -> Dict[str, str]:
    """Overwrite `group` from the filename parts, in place. Returns the map used.

    One file at a time cannot do this, which is why it is not in the workers:
    which position carries the condition is only decidable by looking at how the
    positions vary across the whole batch — a part that never changes is the
    stain name, a part that changes in every single file is a field index. So
    the label is assigned here, in the parent, once every name is known, and
    replaces the per-file placeholder.

    Does nothing when no part parses as a condition, which leaves that
    placeholder standing rather than overwriting a readable stem with a blank.
    """
    frames = [d for d in dfs
              if d is not None and len(d) and name_col in d.columns]
    if not frames:
        return {}
    if names is None:
        names = sorted({n for d in frames for n in d[name_col].astype(str)})
    m = auto_group_map(names, **split_kw)
    if not m:
        return {}
    for d in frames:
        col = d[name_col].astype(str).map(m)
        if group_col in d.columns:
            d[group_col] = col.where(col.notna(), d[group_col])
        else:
            d.insert(min(1, len(d.columns)), group_col, col)
    return m


def pick_unit_column(meta: pd.DataFrame,
                     preference: Sequence[str] = _UNIT_PREFERENCE) -> Optional[str]:
    """Most specific available column identifying a resampling unit."""
    for c in preference:
        if c in meta.columns and meta[c].notna().any():
            return c
    return None


#: Columns that, together, uniquely identify an independent experiment. Both
#: are used when present, because dish numbering is typically only unique
#: within a date (dish "1" on two different dates is not the same dish).
_EXPERIMENT_COLS = ["date", "dish"]


def _n_experiments(d: pd.DataFrame) -> Optional[int]:
    """Number of independent experiments -- unique date/dish combinations --
    behind a group of vesicles.

    Distinct from `n_images` (fields of view within a dish are not
    independent) and `n_vesicles` (vesicles within one dish are not
    independent): this is the number a reviewer means by "how many
    independent experiments". None when the filename metadata carries
    neither date nor dish, so the caller can show "not known" rather than a
    count that overstates independence it never measured.
    """
    cols = [c for c in _EXPERIMENT_COLS if c in d.columns and d[c].notna().any()]
    if not cols:
        return None
    return int(d[cols].drop_duplicates().shape[0])

def label_one(name: str, sep: str = "_") -> str:
    """A condition label for ONE filename, with no batch to compare it against.

    Deliberately weak, and meant to be overwritten: a worker that holds a single
    file cannot see which position carries the condition, so it writes this and
    `stamp_groups` replaces it once every name is known. The rule is "the last
    part that is not an index" -- drop anything that looks like a date, a bare
    number, or a word whose meaning is already known (`field`, `dish`), and take
    what is left.

    A plate reader's name has no condition in it at all, so the label is the
    well: six wells are six groups, where the stem would be one group per field.
    """
    stem = Path(str(name)).stem
    parts = split_name(stem, sep=sep)
    if harmony_parts(stem):
        return parts[0]
    keep = [p for p in parts
            if not _DATE_LIKE.match(p) and not _INT_LIKE.match(p)
            and not _role_from_word(p)]
    return keep[-1] if keep else stem


#: Kept for the caller that wants the old name.
n_experiments = _n_experiments


if __name__ == "__main__":                                   # python -m labkit.names
    # The cases that broke something. Not a substitute for an app's own
    # validation -- it is what makes this module checkable in the seven apps
    # that have none.
    assert delimiters("_") == ["_"]
    assert delimiters("_ OR -") == ["_", "-"]
    assert delimiters("_|-") == ["_", "-"], "a pipe names delimiters too"
    assert delimiters("__ OR _") == ["__", "_"], "longest first, or __ splits twice"
    assert delimiters("_ OR _") == ["_"], "typed twice is still one"
    assert delimiters("") == [], "no delimiter leaves the name whole"
    assert split_name("a_-b", sep="_ OR -") == ["a", "b"], \
        "a mixed run is one boundary, not a boundary plus an empty part"
    assert split_name("20260815_1_lysosensor_Y715C_0") == \
        ["20260815", "1", "lysosensor", "Y715C", "0"]
    assert split_name("whole name", sep="") == ["whole name"]

    assert label_one("20251117_3_lysosensor_E110A-E314A_0.czi") == "E110A-E314A", \
        "the condition, not the whole stem and not the trailing index"
    assert label_one("20260101_WT_dish1_field2.tif") == "WT"

    # Two layouts in one folder: each gets its own role list, and every file
    # still comes back with a label.
    mixed = ["20260101_WT_dish1_field1.tif", "20260101_Y715C_dish1_field1.tif",
             "ctrl_rep1.tif", "treat_rep1.tif"]
    assert set(name_schemes(mixed)) == {4, 2}
    assert sorted(auto_group_map(mixed)) == sorted(mixed)

    # A scheme with ONE file has nothing varying, so no position can be the
    # condition and that file gets no label. Pinned deliberately: inventing one
    # from a single name is the guess this module exists to stop making.
    assert auto_group_map(["20260101_WT_dish1_field1.tif", "ctrl_rep1.tif"]) == \
        {"20260101_WT_dish1_field1.tif": "WT"}

    df = pd.DataFrame({"image": ["20260101_WT_dish1_field1.tif",
                                 "20260101_Y715C_dish1_field1.tif"],
                       "group": ["field1", "field1"]})      # the placeholder
    stamp_groups(df)
    assert list(df["group"]) == ["WT", "Y715C"], "the batch decides, not one name"
    assert list(df.columns)[:2] == ["image", "group"], "`group` keeps its position"

    nogroup = pd.DataFrame({"image": ["0.tif", "1.tif"], "group": ["0", "1"]})
    stamp_groups(nogroup)
    assert list(nogroup["group"]) == ["0", "1"], \
        "nothing parsable leaves the placeholder standing"

    # A plate reader's names: no delimiter, the well instead of a condition.
    plate_names = [f"r{r:02d}c01f{f:02d}p01-ch1sk1fk1fl1.tiff" for r in (2, 5) for f in (1, 2)]
    assert split_name(plate_names[0], sep="-") == \
        ["B01", "f01", "p01", "ch1", "sk1", "fk1", "fl1"], "its own grammar, whatever the delimiter"
    assert name_schemes(plate_names)[7]["suggested"] == \
        ["dish", "field"] + ["ignore"] * 5, "one plane, one channel: neither is a column"
    assert list(auto_meta(plate_names)["dish"]) == ["B01", "B01", "E01", "E01"]
    assert label_one(plate_names[2]) == "E01", "the well, not one group per field"
    assert auto_group_map(plate_names) == {}, "no condition in the name: the layout supplies it"

    print("labkit.names: all checks passed.")
