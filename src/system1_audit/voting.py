"""Aggregating several display orders into one decision, priced per forward pass.

H3 in ``RESEARCH_NOTES.md`` says the usable quantity for a routing layer is not
accuracy but coverage at a fixed error budget, and that voting across option
permutations may buy more of it per unit latency than moving to a larger model.
``permutation.audit_dataset`` already reports ``accuracy_first_order`` beside
``accuracy_modal_vote``, which is the wrong comparison twice over:

1. It compares on accuracy, which H3 explicitly rejects as the operational
   number.
2. It compares a one-pass decision against a ``K``-pass decision without
   recording that the second cost ``K`` times as much. A gain that is not
   divided by the passes it took is not a result, it is a bigger budget.

This module aggregates the transcript an audit already collected -- no extra
model calls -- into one decision per item under several strategies, scores each
on coverage at a fixed error budget, and reports the difference between two
strategies with a *paired* interval over items. Paired, because the strategies
answer the same questions: two independent intervals on two overlapping
quantities cannot say whether the difference is real.

One caveat is load-bearing and is repeated in the dataclass docstrings: this
module can price voting against *itself*, in passes. It cannot price voting
against a larger model, because that needs the larger model. The second half of
H3 stays unanswered here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from .permutation import PermutationAudit, PermutationResult
from .selective import risk_coverage_curve, threshold_feasible_coverage
from .significance import Interval, bootstrap_interval, wilson_interval

# A strategy turns one item's permutation transcript into a single decision:
# a label and the confidence to gate it on.
Strategy = Callable[[PermutationResult], tuple[str, float]]


@dataclass(frozen=True)
class AggregatedDecision:
    """One item's decision after aggregating its display orders."""

    item_id: str
    label: str
    confidence: float
    correct: bool


@dataclass(frozen=True)
class StrategyReport:
    """What one aggregation strategy delivers, and what it cost.

    ``coverage_at_risk`` is the prefix-based number from
    ``SelectiveReport.coverage_at_risk``. ``threshold_coverage`` is the coverage
    an actual confidence cutoff can deliver, which is lower whenever
    confidences tie, because a cutoff cannot answer half of a tie group. For a
    strategy whose confidence is a vote share the two diverge sharply, and the
    second one is the number a deployment gets.

    ``passes_per_item`` is the honest denominator: it is what the strategy cost,
    not what the harness happened to collect.
    """

    name: str
    n_items: int
    passes_per_item: int
    accuracy: float
    accuracy_interval: Interval
    risk_budget: float
    coverage_at_risk: float
    coverage_interval: Interval
    threshold_coverage: float
    distinct_confidences: int
    decisions: tuple[AggregatedDecision, ...]

    @property
    def coverage_per_pass(self) -> float:
        """Threshold-feasible coverage divided by the passes it took.

        Read the ceiling before reading the number: coverage cannot exceed 1.0,
        so this quantity cannot exceed ``1 / passes_per_item``. A single-pass
        baseline whose coverage is above ``1 / K`` therefore beats any ``K``-pass
        strategy here *by construction*, however much coverage the votes bought.

        That makes it a fair comparison only between strategies costing the same
        number of passes, or against another model's coverage at that model's own
        pass cost. It is not a verdict on whether voting is worth it, and using
        it as one would be arithmetic dressed up as a result.
        """
        return self.threshold_coverage / self.passes_per_item


@dataclass(frozen=True)
class VotingComparison:
    """Candidate minus baseline on the same items, with paired intervals.

    ``verdict`` is one of:

    ``no_op``
        The candidate returned the baseline's label and the same confidence on
        every item, so there is nothing to measure. This is what an
        order-invariant model produces, since all ``K`` passes are then the same
        call. Confidences are compared to ``_SAME_CONFIDENCE`` rather than for
        bitwise equality -- see ``_is_no_op``.
    ``no_difference_measured``
        The candidate did decide differently somewhere, but the difference
        scored identically on every resample: a zero-width interval at zero.
        Distinct from ``unresolved``, which is a statement about power.
    ``coverage_gain_resolved``
        The paired coverage interval excludes zero on the upside.
    ``coverage_loss_resolved``
        It excludes zero on the downside -- voting made the selective number
        worse.
    ``unresolved``
        The interval straddles zero with non-zero width. At this sample size the
        comparison does not support either direction.
    """

    risk_budget: float
    baseline: StrategyReport
    candidate: StrategyReport
    accuracy_gain: Interval
    coverage_gain: Interval
    extra_passes_per_item: int
    verdict: str

    @property
    def coverage_gain_per_extra_pass(self) -> float:
        """Coverage bought per additional forward pass.

        Zero extra passes returns 0.0 rather than dividing by zero: a strategy
        that costs no more than the baseline has no per-pass price.
        """
        if self.extra_passes_per_item <= 0:
            return 0.0
        return self.coverage_gain.point / self.extra_passes_per_item


# Two confidences closer than this are treated as the same number. Averaging
# ``K`` bitwise-identical floats does not return that float -- the mean of six
# copies of 0.7 is 0.7000000000000001 -- so a no-op detector that asked for
# bitwise equality would miss a provable no-op and downgrade it to a statement
# about sample size. The tolerance is far below any confidence difference a
# model produces and far above the residue of averaging.
_SAME_CONFIDENCE = 1e-12


def _is_no_op(
    pairs: Sequence[tuple[AggregatedDecision, AggregatedDecision]],
) -> bool:
    """True when the candidate decided exactly as the baseline on every item."""
    return all(
        b.label == c.label
        and math.isclose(b.confidence, c.confidence, rel_tol=0.0, abs_tol=_SAME_CONFIDENCE)
        for b, c in pairs
    )


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------


def aligned_vectors(result: PermutationResult) -> tuple[tuple[float, ...], ...]:
    """Each display's probabilities, re-ordered to the canonical label order.

    ``PermutationResult.displays`` stores probabilities in the order the options
    were *shown*. Averaging those directly would average position slots rather
    than labels, which is the exact confusion the permutation audit exists to
    measure.
    """
    canonical = result.canonical_labels
    vectors: list[tuple[float, ...]] = []
    for shown, probabilities in result.displays:
        by_label = dict(zip(shown, probabilities))
        vectors.append(tuple(by_label[label] for label in canonical))
    return tuple(vectors)


def first_order(result: PermutationResult) -> tuple[str, float]:
    """The single-pass baseline: answer the first display order and stop."""
    vector = aligned_vectors(result)[0]
    best = max(range(len(vector)), key=lambda i: vector[i])
    return result.canonical_labels[best], vector[best]


def mean_probability(result: PermutationResult) -> tuple[str, float]:
    """Average the aligned probability vectors, then take the argmax.

    Keeps a continuous confidence, so it stays usable as a selective-prediction
    gate. Under a bias planted on display positions this is the strategy with a
    reason to work: averaging over orders averages the position term toward a
    constant, which cancels in the argmax.
    """
    vectors = aligned_vectors(result)
    width = len(result.canonical_labels)
    mean = [sum(vec[j] for vec in vectors) / len(vectors) for j in range(width)]
    best = max(range(width), key=lambda i: mean[i])
    return result.canonical_labels[best], mean[best]


def modal_vote(result: PermutationResult) -> tuple[str, float]:
    """Majority vote over per-display top-1 labels; confidence is the vote share.

    This is the strategy ``accuracy_modal_vote`` already reports, exposed here so
    it can be priced. Its confidence takes at most ``K + 1`` distinct values, and
    ``StrategyReport.threshold_coverage`` is where that shows up.
    """
    return result.modal_label, result.modal_share


# --------------------------------------------------------------------------
# Threshold-feasible coverage
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Scoring one strategy
# --------------------------------------------------------------------------


def aggregate(
    audit: PermutationAudit, strategy: Strategy
) -> tuple[AggregatedDecision, ...]:
    """Apply ``strategy`` to every item in a completed audit."""
    decisions: list[AggregatedDecision] = []
    for result in audit.results:
        label, confidence = strategy(result)
        decisions.append(
            AggregatedDecision(
                item_id=result.item_id,
                label=label,
                confidence=confidence,
                correct=label == result.correct_label,
            )
        )
    return tuple(decisions)


def _accuracy(decisions: Sequence[AggregatedDecision]) -> float:
    return sum(1 for d in decisions if d.correct) / len(decisions)


def _coverage(decisions: Sequence[AggregatedDecision], target_risk: float) -> float:
    return threshold_feasible_coverage(
        [d.confidence for d in decisions], [d.correct for d in decisions], target_risk
    )


def strategy_report(
    audit: PermutationAudit,
    strategy: Strategy,
    risk_budget: float,
    name: str | None = None,
    passes_per_item: int | None = None,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> StrategyReport:
    """Score one strategy on a completed audit.

    ``passes_per_item`` defaults to 1 for ``first_order`` and to the audit's
    permutation count for anything else. Pass it explicitly for a custom
    strategy, because the default would otherwise overcharge a strategy that
    reads only some of the displays.
    """
    decisions = aggregate(audit, strategy)
    if passes_per_item is None:
        passes_per_item = 1 if strategy is first_order else audit.n_permutations
    if passes_per_item < 1:
        raise ValueError("passes_per_item must be >= 1")
    confidences = [d.confidence for d in decisions]
    hits = [d.correct for d in decisions]
    n_correct = sum(1 for d in decisions if d.correct)
    return StrategyReport(
        name=name or getattr(strategy, "__name__", "strategy"),
        n_items=len(decisions),
        passes_per_item=passes_per_item,
        accuracy=_accuracy(decisions),
        accuracy_interval=wilson_interval(n_correct, len(decisions), confidence),
        risk_budget=risk_budget,
        coverage_at_risk=risk_coverage_curve(confidences, hits).coverage_at_risk(
            risk_budget
        ),
        coverage_interval=bootstrap_interval(
            list(decisions),
            lambda sample: _coverage(sample, risk_budget),
            confidence,
            n_resamples,
            seed,
        ),
        threshold_coverage=_coverage(decisions, risk_budget),
        distinct_confidences=len(set(confidences)),
        decisions=decisions,
    )


# --------------------------------------------------------------------------
# Comparing two strategies
# --------------------------------------------------------------------------


def compare_strategies(
    audit: PermutationAudit,
    risk_budget: float,
    baseline: Strategy = first_order,
    candidate: Strategy = mean_probability,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> VotingComparison:
    """Paired comparison of two aggregation strategies on the same items.

    The resampling unit is the item, and both strategies are recomputed on each
    resampled item set, so the interval is on the *difference*. Resampling the
    two strategies independently would break the pairing that makes the
    comparison informative.
    """
    base_report = strategy_report(
        audit, baseline, risk_budget, None, None, confidence, n_resamples, seed
    )
    cand_report = strategy_report(
        audit, candidate, risk_budget, None, None, confidence, n_resamples, seed
    )
    pairs = list(zip(base_report.decisions, cand_report.decisions))

    identical = _is_no_op(pairs)

    def accuracy_gain(sample: Sequence[tuple[AggregatedDecision, ...]]) -> float:
        return _accuracy([c for _b, c in sample]) - _accuracy([b for b, _c in sample])

    def coverage_gain(sample: Sequence[tuple[AggregatedDecision, ...]]) -> float:
        return _coverage([c for _b, c in sample], risk_budget) - _coverage(
            [b for b, _c in sample], risk_budget
        )

    if identical:
        zero = Interval(0.0, 0.0, 0.0, confidence, "identical decisions, no resampling")
        return VotingComparison(
            risk_budget=risk_budget,
            baseline=base_report,
            candidate=cand_report,
            accuracy_gain=zero,
            coverage_gain=zero,
            extra_passes_per_item=cand_report.passes_per_item
            - base_report.passes_per_item,
            verdict="no_op",
        )

    acc = bootstrap_interval(pairs, accuracy_gain, confidence, n_resamples, seed)
    cov = bootstrap_interval(pairs, coverage_gain, confidence, n_resamples, seed)
    if cov.width == 0.0 and cov.point == 0.0:
        # Measured identical on every resample. Reporting this as "unresolved"
        # would claim the sample was too small to answer a question it answered.
        verdict = "no_difference_measured"
    elif cov.excludes(0.0):
        verdict = "coverage_gain_resolved" if cov.point > 0 else "coverage_loss_resolved"
    else:
        verdict = "unresolved"
    return VotingComparison(
        risk_budget=risk_budget,
        baseline=base_report,
        candidate=cand_report,
        accuracy_gain=acc,
        coverage_gain=cov,
        extra_passes_per_item=cand_report.passes_per_item - base_report.passes_per_item,
        verdict=verdict,
    )
