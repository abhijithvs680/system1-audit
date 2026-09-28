"""Selective prediction: what you get if the model is allowed to abstain.

The practical question for a routing or guardrail layer is not "how accurate is
it?" but "if I only trust it above some confidence and escalate the rest, what
error rate do I carry and how much traffic did I actually save?". That is a
risk-coverage curve, and it is the number a cost argument should rest on.
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
class SelectiveReport:
    """Risk-coverage summary."""

    n: int
    full_coverage_risk: float
    aurc: float
    curve: tuple[RiskCoveragePoint, ...]

    def coverage_at_risk(self, target_risk: float) -> float:
        """Largest coverage whose risk stays at or below ``target_risk``.

        Returns 0.0 when even the single most confident prediction exceeds the
        target.
        """
        best = 0.0
        for point in self.curve:
            if point.risk <= target_risk:
                best = max(best, point.coverage)
        return best

    def threshold_at_risk(self, target_risk: float) -> float | None:
        """Confidence threshold achieving the best coverage under ``target_risk``."""
        best: RiskCoveragePoint | None = None
        for point in self.curve:
            if point.risk <= target_risk and (best is None or point.coverage > best.coverage):
                best = point
        return None if best is None else best.threshold


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
