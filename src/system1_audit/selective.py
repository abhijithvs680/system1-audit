"""Selective prediction: what you get if the model is allowed to abstain.

The practical question for a routing or guardrail layer is not "how accurate is
it?" but "if I only trust it above some confidence and escalate the rest, what
error rate do I carry and how much traffic did I actually save?". That is a
risk-coverage curve, and it is the number a cost argument should rest on.

One caveat governs this whole module. The curve walks one item at a time, so a
point on it can sit *inside* a group of equally confident items. No confidence
cutoff can do that: a cutoff answers every item at or above it, all or none of
a tie group. So a curve point is a bound on what a deployment gets, not a
promise. ``feasible_coverage_at_risk`` and ``operating_point`` report the
achievable number; ``coverage_at_risk`` reports the bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RiskCoveragePoint:
    """Error rate when only the most confident ``coverage`` share is answered."""

    coverage: float
    risk: float
    threshold: float
    n_answered: int


@dataclass(frozen=True)
class OperatingPoint:
    """A confidence cutoff and what it actually delivers when deployed.

    Every field is measured at ``threshold`` applied as a real cutoff: answer
    an item when its confidence is at or above it, abstain otherwise. So
    ``risk`` is the error rate the budget is spent on, not the error rate of a
    prefix that stopped mid-tie, and ``coverage`` is the traffic actually
    answered. The three are mutually consistent by construction.
    """

    threshold: float
    coverage: float
    risk: float
    n_answered: int


@dataclass(frozen=True)
class SelectiveReport:
    """Risk-coverage summary."""

    n: int
    full_coverage_risk: float
    aurc: float
    curve: tuple[RiskCoveragePoint, ...]

    def coverage_at_risk(self, target_risk: float) -> float:
        """Largest curve coverage whose risk stays at or below ``target_risk``.

        This is an upper bound, not an operating point. The maximum is taken
        over every point on the curve, including points that stop inside a
        group of equally confident items, which no cutoff can reproduce. Use
        ``feasible_coverage_at_risk`` for the number a deployment gets; the two
        agree exactly when no two confidences tie.

        Returns 0.0 when even the single most confident prediction exceeds the
        target.
        """
        best = 0.0
        for point in self.curve:
            if point.risk <= target_risk:
                best = max(best, point.coverage)
        return best

    def feasible_points(self) -> tuple[RiskCoveragePoint, ...]:
        """The curve points a real confidence cutoff can actually land on.

        The curve is sorted by descending confidence, so every item at or above
        a given confidence occupies an unbroken prefix, and the *last* point of
        each tie group is exactly the state of a cutoff set at that confidence.
        Those are the only reachable points. With distinct confidences every
        point is reachable and this returns the whole curve.
        """
        reachable: list[RiskCoveragePoint] = []
        for index, point in enumerate(self.curve):
            is_last = index + 1 == len(self.curve)
            if is_last or self.curve[index + 1].threshold < point.threshold:
                reachable.append(point)
        return tuple(reachable)

    def operating_point(self, target_risk: float) -> OperatingPoint | None:
        """Best deployable cutoff whose *realised* risk stays within budget.

        Returns ``None`` when no cutoff meets the budget. Unlike taking a
        coverage from ``coverage_at_risk`` and a threshold from the curve
        separately, the returned threshold, coverage and risk all describe the
        same cutoff, so applying the threshold reproduces the other two.
        """
        if not 0.0 <= target_risk <= 1.0:
            raise ValueError("target_risk must be in [0, 1]")
        best: RiskCoveragePoint | None = None
        for point in self.feasible_points():
            if point.risk <= target_risk and (best is None or point.coverage > best.coverage):
                best = point
        if best is None:
            return None
        return OperatingPoint(
            threshold=best.threshold,
            coverage=best.coverage,
            risk=best.risk,
            n_answered=best.n_answered,
        )

    def feasible_coverage_at_risk(self, target_risk: float) -> float:
        """Coverage an actual confidence cutoff delivers under ``target_risk``.

        Zero when no cutoff meets the budget. Never exceeds
        ``coverage_at_risk``, and equals it when confidences are distinct.
        """
        point = self.operating_point(target_risk)
        return 0.0 if point is None else point.coverage

    def threshold_at_risk(self, target_risk: float) -> float | None:
        """Confidence cutoff achieving the best feasible coverage under budget.

        The threshold is chosen so that deploying it carries a realised error
        rate at or below ``target_risk``. Picking it off the raw curve instead
        can return a threshold that breaches the budget, because the curve
        point it came from answered only part of a tie group while the cutoff
        answers all of it.
        """
        point = self.operating_point(target_risk)
        return None if point is None else point.threshold


def risk_coverage_curve(
    confidences: Sequence[float], correct: Sequence[bool]
) -> SelectiveReport:
    """Build the full risk-coverage curve, one point per answered prefix.

    Predictions are sorted by confidence, most confident first. Point ``i`` is
    the error rate over the first ``i`` predictions.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have equal length")
    if not confidences:
        raise ValueError("need at least one observation")

    order = sorted(range(len(confidences)), key=lambda i: -confidences[i])
    n = len(order)
    errors = 0
    curve: list[RiskCoveragePoint] = []
    for rank, idx in enumerate(order, start=1):
        if not correct[idx]:
            errors += 1
        curve.append(
            RiskCoveragePoint(
                coverage=rank / n,
                risk=errors / rank,
                threshold=confidences[idx],
                n_answered=rank,
            )
        )
    aurc = sum(p.risk for p in curve) / n
    return SelectiveReport(
        n=n,
        full_coverage_risk=curve[-1].risk,
        aurc=aurc,
        curve=tuple(curve),
    )


def threshold_feasible_coverage(
    confidences: Sequence[float], correct: Sequence[bool], target_risk: float
) -> float:
    """Largest coverage a real confidence cutoff delivers under the budget.

    Convenience wrapper over ``SelectiveReport.feasible_coverage_at_risk`` for
    callers holding raw vectors. With continuous confidences this agrees with
    ``coverage_at_risk``; with a vote share, which takes few distinct values,
    it does not, and only this one is implementable.
    """
    if not 0.0 <= target_risk <= 1.0:
        raise ValueError("target_risk must be in [0, 1]")
    return risk_coverage_curve(confidences, correct).feasible_coverage_at_risk(target_risk)
