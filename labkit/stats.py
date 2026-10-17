"""Group comparisons, with the p-values a figure is allowed to print.

Mann-Whitney, Wilcoxon signed-rank, Welch and paired t, corrected across the
whole family a single call produces.  It lives here rather than in one app
because a p-value that appears on a figure and a p-value that appears in a
table have to be the same number, in every app.

What it insists on, and why:

* **Paired tests are actually paired.**  Masking a frame pairs row *i* with row
  *i*, which is whatever order they happened to be in.  A paired test here
  requires ``pair_on`` (normally the cell, mouse or animal id), aligns on it,
  and reports how many pairs it had to drop as incomplete.
* **n comes with the p.**  Both group sizes, and the number of complete pairs.
* **An effect size comes with every p-value** -- Cliff's delta for the rank
  tests, Hedges' g for the t-tests -- because a corrected p-value alone never
  said whether the difference mattered.
* **More than two levels works.**  Every test is between one pair of levels --
  all pairs, adjacent ones, each against a control, or pairs named outright --
  so a third group is more pairs, not a third argument to a two-sample test.
* **More than one factor is a different question.**  "Does the drug affect the
  mutants differently from WT" is about the interaction, and pairwise testing
  cannot answer it -- WT significant and the mutant not is not a difference
  between them.  :func:`factorial` answers it, reads which factors were
  measured repeatedly in the same cell out of the data rather than trusting
  anyone to declare it, and has a rank version so it can agree with the
  brackets printed beside it.

:func:`brackets` is the bridge to the figure: it turns the returned table into
the list of comparisons :func:`labkit.plots.annotate_p` draws, so the numbers
over the bars are the numbers in the table by construction rather than by
someone retyping them.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.stats as ss

__all__ = [
    "TESTS", "DESIGNS", "Comparison", "compare", "factorial", "stars",
    "describe_groups", "describe_design", "brackets", "format_p",
]

#: name -> (callable, paired?, human label)
TESTS = {
    "mann-whitney": ("Mann-Whitney U", False),
    "wilcoxon": ("Wilcoxon signed-rank", True),
    "welch": ("Welch t", False),
    "student": ("Student t", False),
    "paired-t": ("paired t", True),
}

#: Other spellings accepted for the same tests.
_ALIASES = {
    "m-w": "mann-whitney",
    "mw": "mann-whitney",
    "mannwhitney": "mann-whitney",
    "student_paired": "paired-t",
    "paired": "paired-t",
    "t": "welch",
}


def _resolve_test(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "-")
    key = _ALIASES.get(key.replace("-", "_"), _ALIASES.get(key, key))
    if key not in TESTS:
        raise ValueError(
            f"Unknown test {name!r}. Choose from: {', '.join(sorted(TESTS))}"
        )
    return key


def stars(p_value: float, show_ns: bool = True) -> str:
    """Significance marker. Always a string -- never ``None``.

    Returning ``None`` for a non-significant p-value would make every caller
    that concatenates the marker raise ``TypeError`` whenever a comparison
    came out non-significant -- an annotation routine crashing on the most
    ordinary possible result.
    """
    if p_value is None or not np.isfinite(p_value):
        return ""
    # Five, not four: the lab's own ladder in `voltage_clamp_processing.py`
    # runs ***** below 1e-5, and a figure that stops at **** reads as a
    # weaker result than the same data drawn in the notebook.
    if p_value < 0.00001:
        return "*****"
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns" if show_ns else ""


@dataclass
class Comparison:
    """One two-group comparison and everything needed to report it."""

    group_a: str
    group_b: str
    n_a: int
    n_b: int
    statistic: float
    p_value: float
    test: str
    effect: float = float("nan")
    effect_name: str = ""
    median_a: float = float("nan")
    median_b: float = float("nan")
    difference: float = float("nan")
    n_pairs: int | None = None
    within: str | None = None
    p_corrected: float = float("nan")
    significant: bool = False
    message: str = ""

    @property
    def label(self) -> str:
        base = f"{self.group_a} vs {self.group_b}"
        return f"{self.within} · {base}" if self.within else base

    def as_dict(self) -> dict:
        return {
            "comparison": self.label,
            "within": self.within,
            "group_a": self.group_a,
            "group_b": self.group_b,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "n_pairs": self.n_pairs,
            "median_a": self.median_a,
            "median_b": self.median_b,
            "difference": self.difference,
            "effect": self.effect,
            "effect_name": self.effect_name,
            "test": self.test,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "p_corrected": self.p_corrected,
            "significant": self.significant,
            "stars": stars(self.p_corrected),
            "message": self.message,
        }


# --------------------------------------------------------------------------
# Effect sizes
# --------------------------------------------------------------------------

def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Rank-based effect size in [-1, 1]; positive when ``a`` runs larger.

    Paired with the rank tests because it makes the same assumptions they do:
    no normality, and invariance to any monotonic transform of the data.
    """
    if a.size == 0 or b.size == 0:
        return float("nan")
    greater = float(np.sum(a[:, None] > b[None, :]))
    lesser = float(np.sum(a[:, None] < b[None, :]))
    return (greater - lesser) / (a.size * b.size)


def _hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference, corrected for small-sample bias."""
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return float("nan")
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if pooled == 0:
        return float("nan")
    d = (np.mean(a) - np.mean(b)) / pooled
    correction = 1.0 - 3.0 / (4.0 * (na + nb) - 9.0)
    return float(d * correction)


# --------------------------------------------------------------------------

def _clean(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _align_pairs(
    frame: pd.DataFrame, value: str, group: str, a: str, b: str, pair_on: str
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Two arrays aligned on ``pair_on``, plus how many pairs were dropped.

    One value per (unit, group): if a cell has several rows for a condition
    they are averaged first, so a cell recorded twice cannot count twice.
    """
    subset = frame[frame[group].isin([a, b])][[pair_on, group, value]].dropna()
    wide = subset.groupby([pair_on, group])[value].mean().unstack(group)
    for column in (a, b):
        if column not in wide.columns:
            wide[column] = np.nan
    complete = wide[[a, b]].dropna()
    dropped = len(wide) - len(complete)
    return (
        complete[a].to_numpy(dtype=float),
        complete[b].to_numpy(dtype=float),
        len(complete),
        dropped,
    )


def _run(
    a_values: np.ndarray, b_values: np.ndarray, test: str
) -> tuple[float, float, float, str, str]:
    """``(statistic, p, effect, effect_name, message)``."""
    if a_values.size < 2 or b_values.size < 2:
        return (
            float("nan"), float("nan"), float("nan"), "",
            f"too few values ({a_values.size} and {b_values.size}) to test",
        )

    if test == "mann-whitney":
        statistic, p = ss.mannwhitneyu(a_values, b_values, alternative="two-sided")
        return statistic, p, _cliffs_delta(a_values, b_values), "Cliff's delta", ""
    if test == "wilcoxon":
        if np.allclose(a_values, b_values):
            return float("nan"), float("nan"), 0.0, "Cliff's delta", "all pairs identical"
        statistic, p = ss.wilcoxon(a_values, b_values, zero_method="wilcox")
        return statistic, p, _cliffs_delta(a_values, b_values), "Cliff's delta", ""
    if test == "welch":
        statistic, p = ss.ttest_ind(a_values, b_values, equal_var=False)
        return statistic, p, _hedges_g(a_values, b_values), "Hedges' g", ""
    if test == "student":
        statistic, p = ss.ttest_ind(a_values, b_values, equal_var=True)
        return statistic, p, _hedges_g(a_values, b_values), "Hedges' g", ""
    if test == "paired-t":
        statistic, p = ss.ttest_rel(a_values, b_values)
        return statistic, p, _hedges_g(a_values, b_values), "Hedges' g", ""
    raise ValueError(test)


def _pairs_to_compare(levels: list[str], how) -> list[tuple[str, str]]:
    """Which pairs of levels to test, from the ``how`` specification."""
    if isinstance(how, (list, tuple)) and how and isinstance(how[0], (list, tuple)):
        return [(str(x), str(y)) for x, y in how]
    if isinstance(how, str) and how not in ("all", "adjacent"):
        # A single level name: everything against that control.
        if how not in levels:
            raise ValueError(f"Control level {how!r} is not present. Levels: {levels}")
        return [(level, how) for level in levels if level != how]
    if how == "adjacent":
        return list(zip(levels[:-1], levels[1:]))
    return list(combinations(levels, 2))


def compare(
    data: pd.DataFrame,
    value: str,
    group: str,
    how="all",
    test: str = "mann-whitney",
    alpha: float = 0.05,
    correction: str = "fdr_bh",
    pair_on: str | None = None,
    within: str | None = None,
    unit: str | None = None,
) -> pd.DataFrame:
    """Compare groups and correct across the whole family.

    ``how``
        ``'all'`` for every pair, ``'adjacent'`` for consecutive levels only, a
        level name to test everything against that control, or an explicit list
        of pairs such as ``[('WT', 'E110A'), ('WT', 'R362G')]``.
    ``within``
        Run the comparison separately inside each level of this column, e.g.
        ``within='condition'`` to compare constructs at each pH.
    ``pair_on``
        Required for ``wilcoxon`` and ``paired-t``. The column identifying what
        is paired -- normally ``'cell_id'``.
    ``unit``
        Average to one value per unit before testing, so a cell contributing
        many sweeps counts once. *n* is then a number of cells.

    Returns one row per comparison with both group sizes, an effect size, the
    raw and corrected p-value, and a significance marker.
    """
    from statsmodels.stats.multitest import multipletests

    test = _resolve_test(test)
    paired = TESTS[test][1]

    for column in (value, group):
        if column not in data.columns:
            raise KeyError(f"{column!r} is not a column of data")
    if paired and not pair_on:
        raise ValueError(
            f"{TESTS[test][0]} is a paired test, so pair_on must name the column "
            f"identifying what is paired (usually 'cell_id'). Without it the two "
            f"samples are matched by row order, which is not a pairing."
        )
    if pair_on and pair_on not in data.columns:
        raise KeyError(f"pair_on={pair_on!r} is not a column of data")

    frame = data[np.isfinite(pd.to_numeric(data[value], errors="coerce"))].copy()
    frame[value] = pd.to_numeric(frame[value], errors="coerce")

    if unit:
        keys = [k for k in (within, group, unit) if k]
        frame = frame.groupby(keys, dropna=False)[value].mean().reset_index()

    blocks = (
        [(None, frame)]
        if not within
        else [(str(key), chunk) for key, chunk in frame.groupby(within, dropna=False)]
    )

    results: list[Comparison] = []
    for block_name, chunk in blocks:
        levels = [str(v) for v in pd.unique(chunk[group].dropna())]
        if len(levels) < 2:
            continue
        chunk = chunk.assign(**{group: chunk[group].astype(str)})
        for a, b in _pairs_to_compare(levels, how):
            if a not in levels or b not in levels:
                continue
            if paired:
                a_values, b_values, n_pairs, dropped = _align_pairs(
                    chunk, value, group, a, b, pair_on
                )
                message = f"{dropped} incomplete pair(s) dropped" if dropped else ""
                n_a = n_b = n_pairs
            else:
                a_values = _clean(chunk.loc[chunk[group] == a, value])
                b_values = _clean(chunk.loc[chunk[group] == b, value])
                n_pairs, message = None, ""
                n_a, n_b = a_values.size, b_values.size

            statistic, p, effect, effect_name, note = _run(a_values, b_values, test)
            results.append(
                Comparison(
                    group_a=a, group_b=b, n_a=int(n_a), n_b=int(n_b),
                    statistic=float(statistic), p_value=float(p), test=TESTS[test][0],
                    effect=float(effect), effect_name=effect_name,
                    median_a=float(np.median(a_values)) if a_values.size else float("nan"),
                    median_b=float(np.median(b_values)) if b_values.size else float("nan"),
                    difference=(
                        float(np.median(a_values) - np.median(b_values))
                        if a_values.size and b_values.size else float("nan")
                    ),
                    n_pairs=n_pairs, within=block_name,
                    message="; ".join(m for m in (message, note) if m),
                )
            )

    if not results:
        return pd.DataFrame(columns=list(Comparison("", "", 0, 0, 0, 0, "").as_dict()))

    # Correct across every comparison this call produced -- the family is
    # exactly the returned table, which is why it is corrected here and not
    # per block.
    p_values = np.array([r.p_value for r in results], dtype=float)
    testable = np.isfinite(p_values)
    corrected = np.full(p_values.shape, np.nan)
    if testable.any():
        corrected[testable] = multipletests(
            p_values[testable], alpha=alpha, method=correction
        )[1]
    for result, value_corrected in zip(results, corrected):
        result.p_corrected = float(value_corrected)
        result.significant = bool(np.isfinite(value_corrected) and value_corrected < alpha)

    table = pd.DataFrame([r.as_dict() for r in results])
    table.attrs["alpha"] = alpha
    table.attrs["correction"] = correction
    table.attrs["n_comparisons"] = int(testable.sum())
    return table


def describe_groups(
    data: pd.DataFrame, value: str, group: str, within: str | None = None, unit: str | None = None
) -> pd.DataFrame:
    """n, mean, SD, SEM and median per group -- what a figure legend needs."""
    frame = data.copy()
    frame[value] = pd.to_numeric(frame[value], errors="coerce")
    keys = [k for k in (within, group) if k]
    if unit:
        frame = frame.groupby(keys + [unit], dropna=False)[value].mean().reset_index()
    summary = frame.groupby(keys, dropna=False)[value].agg(
        n="count", mean="mean", sd="std", median="median"
    ).reset_index()
    summary["sem"] = summary["sd"] / np.sqrt(summary["n"].clip(lower=1))
    return summary


# --------------------------------------------------------------------------
# From a table to a figure
# --------------------------------------------------------------------------

def format_p(
    p_value: float,
    style: str = "auto",
    show_ns: bool = True,
    floor: float = 1e-4,
) -> str:
    """The string that goes on the figure for one p-value.

    ``style``
        ``'stars'``  ``***``, and ``ns`` when it is not significant.
        ``'exact'``  ``p = 0.031``, always the number.
        ``'auto'``   the number down to *floor*, then ``p < 0.0001``.
        ``'both'``   ``p = 0.031 *``.

    ``'auto'`` is the default because a number is the honest form.  With four
    animals a side, p = 0.04 and p = 0.06 are the same result, and a star
    beside one and nothing beside the other tells the reader they are not.
    Stars remain available because some journals require them.
    """
    if p_value is None or not np.isfinite(p_value):
        return ""
    if style == "stars":
        return stars(p_value, show_ns=show_ns)
    if not np.isfinite(p_value):
        return ""

    if p_value < floor:
        number = f"p < {floor:g}"
    elif p_value >= 0.1:
        number = f"p = {p_value:.2f}"
    elif p_value >= 0.001:
        number = f"p = {p_value:.3f}"
    else:
        number = f"p = {p_value:.1e}".replace("e-0", "e-")

    if style == "exact" or style == "auto":
        if style == "exact" and not show_ns and p_value >= 0.05:
            return ""
        return number
    if style == "both":
        marker = stars(p_value, show_ns=False)
        return f"{number} {marker}".strip()
    raise ValueError(
        f"Unknown p-value style {style!r}. Choose from: auto, exact, stars, both"
    )


def brackets(
    table: pd.DataFrame,
    use_corrected: bool = True,
    style: str = "auto",
    only_significant: bool = False,
    alpha: float = 0.05,
    within: str | None = None,
) -> list[dict]:
    """Turn :func:`compare`'s table into what :func:`labkit.plots.annotate_p` draws.

    Each entry is ``{'a', 'b', 'p', 'text', 'significant'}`` -- the two group
    labels the bracket spans, the p-value it is reporting, and the string to
    print.  Nothing is reformatted downstream, so the figure cannot disagree
    with the table.

    ``use_corrected`` picks ``p_corrected`` over ``p_value``; leave it on
    unless you are annotating a single planned comparison, in which case
    there is no family to correct across and the raw p is the right one.
    ``within`` keeps only the rows from one block of a ``within=`` comparison,
    for a figure that shows that block on its own.
    """
    if table is None or not len(table):
        return []
    column = "p_corrected" if use_corrected and "p_corrected" in table.columns else "p_value"
    rows = table
    if within is not None and "within" in rows.columns:
        rows = rows[rows["within"].astype(str) == str(within)]

    out: list[dict] = []
    for _, row in rows.iterrows():
        p = float(row[column])
        if not np.isfinite(p):
            continue
        significant = p < alpha
        if only_significant and not significant:
            continue
        out.append({
            "a": str(row["group_a"]),
            "b": str(row["group_b"]),
            "p": p,
            "text": format_p(p, style=style, show_ns=not only_significant),
            "significant": significant,
            "within": row.get("within"),
            "n_a": int(row["n_a"]) if np.isfinite(row.get("n_a", np.nan)) else None,
            "n_b": int(row["n_b"]) if np.isfinite(row.get("n_b", np.nan)) else None,
            "test": row.get("test", ""),
        })
    # Short spans first, so the stacking in annotate_p puts the pair that is
    # side by side underneath the pair that reaches across the whole axis.
    return out


def compare_arrays(
    groups: dict[str, "np.ndarray | list[float]"],
    how="all",
    test: str = "mann-whitney",
    alpha: float = 0.05,
    correction: str = "fdr_bh",
    units: dict[str, list] | None = None,
) -> pd.DataFrame:
    """:func:`compare` for data you already have as ``{label: values}``.

    The apps that never build a tidy frame -- the ones holding a list of
    measurements per condition in memory -- get the same table, the same
    correction and therefore the same annotations as the ones that do.

    ``units`` gives, per label, what each value belongs to -- normally a cell
    id, in the same order as the values. Without it a paired test has nothing
    to pair on and refuses to run.
    """
    ids = {str(k): list(v) for k, v in (units or {}).items()}
    frame = pd.DataFrame(
        [{"__group": str(label), "__value": float(v),
          "__unit": (ids.get(str(label)) or [None] * (i + 1))[i]}
         for label, values in groups.items()
         for i, v in enumerate(np.asarray(values, dtype=float).ravel())
         if np.isfinite(v)]
    )
    if not len(frame):
        return pd.DataFrame(columns=list(Comparison("", "", 0, 0, 0, 0, "").as_dict()))
    # Preserve the caller's ordering: pandas would otherwise sort the levels
    # and a "control vs mutant" comparison would come back reversed.
    order = {str(k): i for i, k in enumerate(groups)}
    frame["__group"] = pd.Categorical(
        frame["__group"], categories=sorted(order, key=order.get), ordered=True
    )
    frame = frame.sort_values("__group")
    return compare(frame, value="__value", group="__group", how=how,
                   test=test, alpha=alpha, correction=correction,
                   pair_on="__unit" if units else None)


# --------------------------------------------------------------------------
# More than one factor at a time
# --------------------------------------------------------------------------

#: What the factorial test can be run on.
DESIGNS = {
    "parametric": "on the measured values",
    "rank": "on aligned ranks",
}


def _terms(names: list[str]) -> list[tuple[str, ...]]:
    """Main effects first, then every interaction, smallest first."""
    return [t for size in range(1, len(names) + 1) for t in combinations(names, size)]


def _align(frame: pd.DataFrame, names: list[str], term: tuple[str, ...]) -> pd.Series:
    """The aligned response for one term (Wobbrock et al. 2011).

    Strip every effect but this one: the residual from the full cell means,
    plus this term's own effect, which is the inclusion-exclusion of the
    marginal means over its factors.  Ranking what is left and running the
    ordinary model on the ranks gives a test of *this* term that no other
    term's size can inflate -- which is what plain ranking gets wrong.
    """
    aligned = frame["y"] - frame.groupby(names, dropna=False)["y"].transform("mean")
    for size in range(len(term) + 1):
        sign = (-1) ** (len(term) - size)
        for subset in combinations(term, size):
            aligned += sign * (
                frame["y"].mean() if not subset
                else frame.groupby(list(subset), dropna=False)["y"].transform("mean")
            )
    return aligned


def factorial(
    data: pd.DataFrame,
    value: str,
    factors,
    subject: str | None = None,
    kind: str = "parametric",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Main effects and interactions for a design with more than one factor.

    Two factors are not two comparisons.  "Does PI3P change the mutants
    differently from WT" is a question about the *interaction*, and no amount
    of pairwise testing answers it: WT significant and E110A not is not a
    difference between them.

    ``factors``
        The columns that were varied, e.g. ``['construct', 'condition']``.
    ``subject``
        What was measured repeatedly -- normally ``'cell_id'``.  A factor whose
        levels a single subject carries more than one of is treated as
        within-subject, and the model gets a random intercept per subject.
        Left out, every factor is between-subject, which inflates *n* whenever
        it isn't true.
    ``kind``
        ``'parametric'`` for the values as measured, ``'rank'`` for the aligned
        rank transform -- the non-parametric answer, and the one to use when
        the pairwise tests beside it are rank tests.

    Returns one row per term with its df, statistic, p-value and partial eta
    squared; :func:`describe_design` turns the same table into the sentence a
    methods section needs.
    """
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    factors = [str(f) for f in factors]
    if kind not in DESIGNS:
        raise ValueError(f"kind must be one of {', '.join(DESIGNS)}, not {kind!r}")
    if len(set(factors)) != len(factors):
        raise ValueError("a factor cannot appear twice in the same design")
    for column in [value] + factors + ([subject] if subject else []):
        if column not in data.columns:
            raise KeyError(f"{column!r} is not a column of data")

    # Safe column names: a factor called "date" or a value with a bracket in it
    # is data, and has no business being parsed as part of a formula.
    names = [f"f{i}" for i in range(len(factors))]
    frame = pd.DataFrame({n: data[f].fillna("").astype(str) for n, f in zip(names, factors)})
    frame["y"] = pd.to_numeric(data[value], errors="coerce")
    if subject:
        frame["s"] = data[subject].fillna("").astype(str)
        frame = frame[frame["s"] != ""]
    frame = frame[np.isfinite(frame["y"])].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"no finite {value} to test")

    notes: list[str] = []
    if subject:
        rows_in = len(frame)
        frame = frame.groupby(names + ["s"], dropna=False, as_index=False)["y"].mean()
        if len(frame) < rows_in:
            notes.append(f"{rows_in} rows averaged to {len(frame)}, one per {subject} per combination")

    label = {n: f for n, f in zip(names, factors) if frame[n].nunique() > 1}
    notes += [f"{f} has one level here and was left out"
              for n, f in zip(names, factors) if n not in label]
    names = [n for n in names if n in label]
    if not names:
        raise ValueError("a design needs at least one factor with two levels present")

    within = [n for n in names
              if subject and frame.groupby("s")[n].nunique().max() > 1]
    # bool() because pandas 3 returns np.int64 from .size().max(), which makes this
    # np.True_ -- and attrs["repeated"] is compared with `is True` and json.dumps'd.
    repeated = bool(within and frame.groupby("s").size().max() > 1) if subject else False

    cells = int(np.prod([frame[n].nunique() for n in names]))
    empty = cells - frame.groupby(names, dropna=False).ngroups
    if empty:
        notes.append(f"{empty} of {cells} cells of the design are empty")

    # Sum-to-zero contrasts, not the default treatment coding.  Under treatment
    # coding the Wald test of a lower-order term is that term's effect *at the
    # reference level of the others* -- a simple effect wearing a main effect's
    # name.  A clean crossover, where WT rises by exactly what the mutant
    # falls, then reports a main effect of condition at p = 1e-52 when there is
    # none at all.
    term_of = {n: f"C({n}, Sum)" for n in names}
    formula = "y ~ " + " * ".join(term_of[n] for n in names)

    def fit(response: pd.Series) -> tuple[dict, object]:
        """Every term of the full model, keyed by the factors it involves."""
        work = frame.assign(y=np.asarray(response, dtype=float))
        out: dict[frozenset, dict] = {}
        try:
            if repeated:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = smf.mixedlm(formula, data=work, groups=work["s"]).fit(reml=False)
                    terms = fitted.wald_test_terms().table
                df_residual = float(len(work) - len(fitted.fe_params))
                for term, row in terms.iterrows():
                    if term == "Intercept":
                        continue
                    statistic = float(np.ravel(row["statistic"])[0])
                    out[frozenset(str(term).split(":"))] = {
                        "df": float(row["df_constraint"]), "df_residual": df_residual,
                        "statistic": statistic, "statistic_name": "chi2",
                        "p_value": float(row["pvalue"]),
                        "effect": statistic / (statistic + df_residual),
                    }
            else:
                fitted = smf.ols(formula, data=work).fit()
                table = anova_lm(fitted, typ=2)
                ss_residual = float(table.loc["Residual", "sum_sq"])
                for term, row in table.iterrows():
                    if term == "Residual":
                        continue
                    ss = float(row["sum_sq"])
                    out[frozenset(str(term).split(":"))] = {
                        "df": float(row["df"]), "df_residual": float(table.loc["Residual", "df"]),
                        "statistic": float(row["F"]), "statistic_name": "F",
                        "p_value": float(row["PR(>F)"]),
                        "effect": ss / (ss + ss_residual) if ss + ss_residual > 0 else float("nan"),
                    }
        except Exception as exc:
            # statsmodels and patsy raise a zoo of types here and every one of
            # them means the same thing to whoever asked: this design could not
            # be fitted.  Say why it probably could not -- "Singular matrix" on
            # its own does not tell anyone that half the cells are empty.
            raise ValueError(f"the model could not be fitted: {exc}"
                             + (f" -- {'; '.join(notes)}" if notes else "")) from exc
        return out, fitted

    effect_name = "partial η²" + (" (Wald)" if repeated else "")
    if kind == "rank":
        effect_name += ", on ranks"

    found, fitted = ({}, None)
    if kind == "parametric":
        found, fitted = fit(frame["y"])
    rows = []
    for term in _terms(names):
        if kind == "rank":
            # One fit per term: the alignment differs for each one.
            found, fitted = fit(_align(frame, names, term).rank())
        result = found.get(frozenset(term_of[n] for n in term))
        if result is None:
            continue
        rows.append({
            "term": " × ".join(label[n] for n in term),
            "effect_kind": "main effect" if len(term) == 1 else "interaction",
            "repeated": any(n in within for n in term),
            **result, "effect_name": effect_name,
            "significant": bool(result["p_value"] < alpha),
            "stars": stars(result["p_value"]),
        })
    if fitted is not None and getattr(fitted, "converged", True) is False:
        notes.append("the mixed model did not converge -- treat these p-values as provisional")

    table = pd.DataFrame(rows)
    table.attrs = {
        "value": value, "factors": [label[n] for n in names], "subject": subject or "",
        "within": [label[n] for n in within], "repeated": repeated, "kind": kind,
        "model": ("linear mixed model with a random intercept per subject"
                  if repeated else "ordinary least squares ANOVA (type II)"),
        "n": int(len(frame)),
        "n_subjects": int(frame["s"].nunique()) if subject else 0,
        "notes": notes,
    }
    return table


def describe_design(table: pd.DataFrame) -> str:
    """The sentence a methods section needs, from the table that was run."""
    at = table.attrs
    if not at:
        return ""
    factors = at["factors"]
    design = " × ".join(factors) if len(factors) > 1 else f"{factors[0]} alone"
    how = at["model"]
    if at["kind"] == "rank":
        how = f"an aligned rank transform ({how})"
    else:
        how = f"a{'n' if how[0] in 'aeiou' else ''} {how}"
    where = ""
    if at["within"]:
        where = (f", with {' and '.join(at['within'])} measured repeatedly in the same "
                 f"{at['subject'] or 'subject'}")
    counts = f"{at['n']} measurements"
    if at["n_subjects"]:
        counts += f" from {at['n_subjects']} {at['subject']}s"
    return f"{design}{where}, tested by {how} on {counts}."
