"""Disjoint fit/report splits, so a post-temperature number means something.

Vendors report calibration twice: raw, and again after fitting a temperature.
The second number is only informative if the temperature was fitted on items
other than the ones it is reported on. Fitting and reporting on the same set
measures how well a one-parameter model can be bent to fit a sample, which is
not a property of the decision model at all.

This module makes that rule mechanical rather than aspirational:

* :func:`deterministic_split` assigns items to a fit or report side by hashing
  their identifiers, so the assignment is identical across processes and does
  not move when the item set grows.
* :func:`held_out_calibration` fits the temperature on one side and reports on
  the other, and reports the in-sample number alongside it so the difference is
  visible instead of hidden.
* :func:`check_disjoint` refuses overlapping sides outright.

The assignment is deliberately not a shuffle. A shuffle re-draws every
assignment when an item is added, which silently invalidates any number
reported against an earlier version of the dataset. Hashing an item's own
identifier means item ``q17`` lands on the same side however many other items
exist.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

from .calibration import CalibrationReport, apply_temperature, calibration_report, fit_temperature
from .types import ChoiceQuestion

_HASH_SPAN = 1 << 64


def _unit_hash(item_id: str, salt: str) -> float:
    """Map an identifier to [0, 1), identically in every process.

    ``hash()`` on strings is salted per interpreter run, so it cannot be used
    for anything whose output is reported. This mirrors the reasoning in
    ``deciders.mock._stable_seed``.
    """
    payload = f"{salt}\x1f{item_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / _HASH_SPAN


@dataclass(frozen=True)
class Split:
    """Positional indices of the fit side and the report side.

    Indices refer to the sequence the split was built from, so the same object
    can index probabilities, labels and questions without re-deriving anything.
    """

    fit: tuple[int, ...]
    report: tuple[int, ...]

    def __post_init__(self) -> None:
        overlap = set(self.fit) & set(self.report)
        if overlap:
            raise ValueError(f"fit and report sides overlap on indices {sorted(overlap)}")

    @property
    def n_fit(self) -> int:
        return len(self.fit)

    @property
    def n_report(self) -> int:
        return len(self.report)

    @property
    def fit_fraction(self) -> float:
        """Realised fit share. Hash assignment only approximates the target."""
        total = self.n_fit + self.n_report
        if total == 0:
            raise ValueError("split is empty")
        return self.n_fit / total

    def require_both_sides(self) -> None:
        """Raise unless both sides can carry their job.

        A temperature cannot be fitted on nothing, and a calibration report
        over nothing is not a report.
        """
        if not self.fit:
            raise ValueError("fit side is empty: no items to fit a temperature on")
        if not self.report:
            raise ValueError("report side is empty: no items to report on")


def check_disjoint(fit_ids: Sequence[str], report_ids: Sequence[str]) -> None:
    """Raise if any identifier appears on both sides.

    Called on identifiers rather than indices because leakage in practice comes
    from the same item entering a dataset twice, not from an index bug.
    """
    shared = set(fit_ids) & set(report_ids)
    if shared:
        raise ValueError(f"fit/report leakage on {len(shared)} item(s): {sorted(shared)[:5]}")


def deterministic_split(
    item_ids: Sequence[str], fit_fraction: float = 0.5, salt: str = ""
) -> Split:
    """Split items by hashing their identifiers.

    Args:
        item_ids: One identifier per item. Must be unique — duplicate or blank
            identifiers cannot be assigned a stable side, and silently
            collapsing them is how leakage gets in.
        fit_fraction: Target share of items used to fit the temperature. The
            realised share is approximate, because each item is assigned
            independently.
        salt: Changes the assignment. Use it to check that a result is not an
            artefact of one particular split, and record which salt produced
            any number you report.

    Raises:
        ValueError: on a non-unique or blank identifier, or a fraction outside
            [0, 1].
    """
    if not 0.0 <= fit_fraction <= 1.0:
        raise ValueError(f"fit_fraction must be in [0, 1], got {fit_fraction!r}")
    if not item_ids:
        raise ValueError("need at least one item to split")
    if any(not item_id for item_id in item_ids):
        raise ValueError("every item needs a non-empty identifier to be split stably")
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("item identifiers must be unique")

    fit: list[int] = []
    report: list[int] = []
    for index, item_id in enumerate(item_ids):
        (fit if _unit_hash(item_id, salt) < fit_fraction else report).append(index)
    return Split(tuple(fit), tuple(report))


def split_questions(
    questions: Sequence[ChoiceQuestion], fit_fraction: float = 0.5, salt: str = ""
) -> Split:
    """:func:`deterministic_split` keyed on each question's ``item_id``."""
    return deterministic_split([q.item_id for q in questions], fit_fraction, salt)


@dataclass(frozen=True)
class HeldOutCalibration:
    """Calibration on the report side, with the temperature's provenance kept.

    Three reports over the *same* report-side items:

    ``raw``
        No scaling at all.
    ``calibrated``
        Scaled by ``temperature``, which was fitted on the disjoint fit side.
        This is the number that is honest to publish.
    ``in_sample``
        Scaled by ``in_sample_temperature``, refitted on the report side
        itself. Kept only to quantify what the shortcut buys.
    """

    n_fit: int
    n_report: int
    temperature: float
    in_sample_temperature: float
    raw: CalibrationReport
    calibrated: CalibrationReport
    in_sample: CalibrationReport

    @property
    def nll_optimism(self) -> float:
        """How much NLL the in-sample shortcut appears to save.

        Non-negative by construction up to the line search's tolerance:
        ``in_sample_temperature`` minimises NLL over exactly these items, so no
        other temperature can score lower on them.
        """
        return self.calibrated.nll - self.in_sample.nll

    @property
    def ece_optimism(self) -> float:
        """The same comparison on ECE. Positive means the shortcut flattered.

        Unlike :attr:`nll_optimism` this is **not** guaranteed non-negative:
        the temperature is fitted against NLL, not against ECE, and ECE is a
        binned statistic that the fit does not optimise. It is measured and
        reported, not assumed.
        """
        return self.calibrated.ece - self.in_sample.ece

    @property
    def ece_reduction(self) -> float:
        """ECE removed by the held-out temperature. Positive means it helped."""
        return self.raw.ece - self.calibrated.ece


def held_out_calibration(
    probabilities: Sequence[Sequence[float]],
    correct_index: Sequence[int],
    split: Split,
    n_bins: int = 10,
) -> HeldOutCalibration:
    """Fit the temperature on ``split.fit`` and report it on ``split.report``.

    Args:
        probabilities: One probability vector per item.
        correct_index: Index of the correct option per item.
        split: Built from the same item sequence, by :func:`deterministic_split`.
        n_bins: Bin count for the reported ECE.

    Raises:
        ValueError: if lengths disagree, an index is out of range for the item
            sequence, or either side of the split is empty.
    """
    if len(probabilities) != len(correct_index):
        raise ValueError("probabilities and correct_index must have equal length")
    if not probabilities:
        raise ValueError("need at least one observation")
    split.require_both_sides()
    n = len(probabilities)
    for index in (*split.fit, *split.report):
        if not 0 <= index < n:
            raise ValueError(f"split index {index} out of range for {n} items")

    def take(indices: Sequence[int]) -> tuple[list[Sequence[float]], list[int]]:
        return ([probabilities[i] for i in indices], [correct_index[i] for i in indices])

    fit_probs, fit_correct = take(split.fit)
    report_probs, report_correct = take(split.report)

    temperature = fit_temperature(fit_probs, fit_correct)
    in_sample_temperature = fit_temperature(report_probs, report_correct)

    def scaled(temp: float) -> list[tuple[float, ...]]:
        return [apply_temperature(p, temp) for p in report_probs]

    return HeldOutCalibration(
        n_fit=split.n_fit,
        n_report=split.n_report,
        temperature=temperature,
        in_sample_temperature=in_sample_temperature,
        raw=calibration_report(report_probs, report_correct, n_bins),
        calibrated=calibration_report(scaled(temperature), report_correct, n_bins),
        in_sample=calibration_report(scaled(in_sample_temperature), report_correct, n_bins),
    )
