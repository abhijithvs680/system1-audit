"""Calibration metrics for probabilistic decisions.

Pure standard library on purpose: an audit harness that needs a GPU stack
installed before it can check arithmetic is an audit harness nobody runs.

Definitions follow the usual convention where the model's *confidence* is the
probability it assigned to its own top prediction, and *accuracy* within a bin
is the fraction of those predictions that were right.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

_EPS = 1e-12


@dataclass(frozen=True)
class ReliabilityBin:
    """One bucket of a reliability diagram."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        """Signed calibration gap: positive means overconfident."""
        return self.mean_confidence - self.accuracy


@dataclass(frozen=True)
class CalibrationReport:
    """Summary of how well confidence tracked correctness."""

    n: int
    accuracy: float
    mean_confidence: float
    ece: float
    adaptive_ece: float
    mce: float
    brier: float
    nll: float
    bins: tuple[ReliabilityBin, ...]

    @property
    def overconfidence(self) -> float:
        """Mean confidence minus accuracy. Positive means overconfident."""
        return self.mean_confidence - self.accuracy


def _validate(confidences: Sequence[float], correct: Sequence[bool]) -> None:
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have equal length")
    if not confidences:
        raise ValueError("need at least one observation")
    for c in confidences:
        if not 0.0 <= c <= 1.0:
            raise ValueError(f"confidence out of range: {c!r}")


def _bin_stats(
    confidences: Sequence[float],
    correct: Sequence[bool],
    edges: Sequence[tuple[float, float]],
    assignment: Sequence[int],
) -> tuple[ReliabilityBin, ...]:
    n_bins = len(edges)
    counts = [0] * n_bins
    conf_sums = [0.0] * n_bins
    hit_sums = [0] * n_bins
    for conf, hit, b in zip(confidences, correct, assignment):
        counts[b] += 1
        conf_sums[b] += conf
        hit_sums[b] += 1 if hit else 0
    bins = []
    for b, (lo, hi) in enumerate(edges):
        if counts[b] == 0:
            bins.append(ReliabilityBin(lo, hi, 0, 0.0, 0.0))
            continue
        bins.append(
            ReliabilityBin(
                lower=lo,
                upper=hi,
                count=counts[b],
                mean_confidence=conf_sums[b] / counts[b],
                accuracy=hit_sums[b] / counts[b],
            )
        )
    return tuple(bins)


def equal_width_bins(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> tuple[ReliabilityBin, ...]:
    """Reliability bins of equal width over [0, 1].

    This is the binning behind the commonly reported ECE number.
    """
    _validate(confidences, correct)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    edges = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]
    assignment = [min(int(c * n_bins), n_bins - 1) for c in confidences]
    return _bin_stats(confidences, correct, edges, assignment)


def equal_mass_bins(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> tuple[ReliabilityBin, ...]:
    """Reliability bins holding (near) equal numbers of observations.

    Equal-width ECE is dominated by whichever bin the model happens to crowd
    into. Equal-mass binning is the standard remedy, reported as adaptive ECE.
    """
    _validate(confidences, correct)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    order = sorted(range(len(confidences)), key=lambda i: confidences[i])
    n = len(order)
    n_bins = min(n_bins, n)
    groups: list[list[int]] = []
    for b in range(n_bins):
        start = (b * n) // n_bins
        stop = ((b + 1) * n) // n_bins
        groups.append(order[start:stop])
    edges = []
    assignment = [0] * n
    for b, group in enumerate(groups):
        lo = confidences[group[0]]
        hi = confidences[group[-1]]
        edges.append((lo, hi))
        for i in group:
            assignment[i] = b
    return _bin_stats(confidences, correct, edges, assignment)


def expected_calibration_error(bins: Sequence[ReliabilityBin]) -> float:
    """Count-weighted mean absolute gap between confidence and accuracy."""
    total = sum(b.count for b in bins)
    if total == 0:
        raise ValueError("bins are empty")
    return sum(b.count * abs(b.gap) for b in bins) / total


def maximum_calibration_error(bins: Sequence[ReliabilityBin]) -> float:
    """Largest absolute gap over non-empty bins."""
    populated = [b for b in bins if b.count > 0]
    if not populated:
        raise ValueError("bins are empty")
    return max(abs(b.gap) for b in populated)


def brier_score(probabilities: Sequence[Sequence[float]], correct_index: Sequence[int]) -> float:
    """Multiclass Brier score: mean squared error of the full probability vector.

    Range is [0, 2]. Lower is better. For a binary problem this is twice the
    familiar ``mean((p - y) ** 2)``, which is the standard multiclass form.
    """
    if len(probabilities) != len(correct_index):
        raise ValueError("probabilities and correct_index must have equal length")
    if not probabilities:
        raise ValueError("need at least one observation")
    total = 0.0
    for probs, k in zip(probabilities, correct_index):
        if not 0 <= k < len(probs):
            raise ValueError(f"correct_index {k} out of range for {len(probs)} options")
        for j, p in enumerate(probs):
            target = 1.0 if j == k else 0.0
            total += (p - target) ** 2
    return total / len(probabilities)


def negative_log_likelihood(
    probabilities: Sequence[Sequence[float]], correct_index: Sequence[int]
) -> float:
    """Mean negative log probability of the correct option."""
    if len(probabilities) != len(correct_index):
        raise ValueError("probabilities and correct_index must have equal length")
    if not probabilities:
        raise ValueError("need at least one observation")
    total = 0.0
    for probs, k in zip(probabilities, correct_index):
        total -= math.log(max(probs[k], _EPS))
    return total / len(probabilities)


def apply_temperature(probabilities: Sequence[float], temperature: float) -> tuple[float, ...]:
    """Re-sharpen or soften a probability vector by temperature ``T``.

    Vendors publish calibration numbers "post-temperature fitting". Only
    probabilities are exposed by most decision APIs, not logits, so ``log p``
    is used as the pseudo-logit. That is the standard substitution and it is
    exact up to the softmax's shift invariance.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    logits = [math.log(max(p, _EPS)) / temperature for p in probabilities]
    peak = max(logits)
    exps = [math.exp(z - peak) for z in logits]
    total = sum(exps)
    return tuple(e / total for e in exps)


def fit_temperature(
    probabilities: Sequence[Sequence[float]],
    correct_index: Sequence[int],
    bounds: tuple[float, float] = (0.05, 20.0),
    tolerance: float = 1e-4,
) -> float:
    """Find the temperature minimising NLL, by golden-section search.

    NLL as a function of ``T`` is unimodal for softmax temperature scaling, so
    a derivative-free line search is enough and keeps the dependency list empty.
    """
    lo, hi = bounds
    if not 0.0 < lo < hi:
        raise ValueError("bounds must satisfy 0 < lo < hi")

    def loss(t: float) -> float:
        scaled = [apply_temperature(p, t) for p in probabilities]
        return negative_log_likelihood(scaled, correct_index)

    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - inv_phi * (b - a), a + inv_phi * (b - a)
    fc, fd = loss(c), loss(d)
    while b - a > tolerance:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = loss(d)
    return (a + b) / 2.0


def calibration_report(
    probabilities: Sequence[Sequence[float]],
    correct_index: Sequence[int],
    n_bins: int = 10,
) -> CalibrationReport:
    """Full calibration summary for a set of decisions."""
    if len(probabilities) != len(correct_index):
        raise ValueError("probabilities and correct_index must have equal length")
    if not probabilities:
        raise ValueError("need at least one observation")
    confidences = [max(p) for p in probabilities]
    correct = [
        max(range(len(p)), key=lambda j, p=p: p[j]) == k
        for p, k in zip(probabilities, correct_index)
    ]
    width_bins = equal_width_bins(confidences, correct, n_bins)
    mass_bins = equal_mass_bins(confidences, correct, n_bins)
    return CalibrationReport(
        n=len(probabilities),
        accuracy=sum(correct) / len(correct),
        mean_confidence=sum(confidences) / len(confidences),
        ece=expected_calibration_error(width_bins),
        adaptive_ece=expected_calibration_error(mass_bins),
        mce=maximum_calibration_error(width_bins),
        brier=brier_score(probabilities, correct_index),
        nll=negative_log_likelihood(probabilities, correct_index),
        bins=width_bins,
    )
