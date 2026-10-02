"""Permutation-vote aggregation, priced per forward pass.

The load-bearing test here is the no-op one: on a decider that ignores display
order, every permutation is the same call, so voting cannot change any decision
and the measured gain must be *exactly* zero. A harness that reports a gain
there is measuring its own resampling noise, which would make every positive
result in this module unreadable.
"""

import unittest

from system1_audit.deciders import SyntheticDecider
from system1_audit.permutation import audit_dataset
from system1_audit.types import ChoiceDecision, ChoiceQuestion
from system1_audit.voting import (
    aggregate,
    aligned_vectors,
    compare_strategies,
    first_order,
    mean_probability,
    modal_vote,
    strategy_report,
    threshold_feasible_coverage,
)

LABELS = ("billing", "bug", "sales", "other")


def questions(n):
    """``n`` items cycling through the labels as the correct answer."""
    out = []
    for i in range(n):
        correct = LABELS[i % len(LABELS)]
        out.append(ChoiceQuestion(f"state {i}", LABELS, correct, f"q{i}"))
    return tuple(out)


class OrderInvariantDecider:
    """Answers from state alone. Identical under every display order."""

    def __init__(self, answer_key, confidence=0.7):
        self.answer_key = answer_key
        self.confidence = confidence

    def decide_choice(self, state, labels):
        labels = tuple(labels)
        target = self.answer_key[state]
        rest = (1.0 - self.confidence) / (len(labels) - 1)
        return ChoiceDecision(
            labels, tuple(self.confidence if lab == target else rest for lab in labels)
        )


class AlignmentTests(unittest.TestCase):
    def test_vectors_are_realigned_to_canonical_labels(self):
        items = questions(4)
        key = {q.state: q.correct for q in items}
        audit = audit_dataset(OrderInvariantDecider(key), items, n_permutations=6)
        for result in audit.results:
            correct_index = result.canonical_labels.index(result.correct_label)
            for vector in aligned_vectors(result):
                # Order-invariant decider: the mass sits on the correct label in
                # every display, once the vector is back in canonical order.
                self.assertAlmostEqual(vector[correct_index], 0.7, places=9)

    def test_realignment_is_not_the_identity_on_a_permuted_display(self):
        items = questions(4)
        key = {q.state: q.correct for q in items}
        audit = audit_dataset(OrderInvariantDecider(key), items, n_permutations=6)
        result = audit.results[0]
        raw = [probs for _shown, probs in result.displays]
        aligned = aligned_vectors(result)
        # If realignment were a no-op the two would agree on every display; the
        # whole point is that they do not.
        self.assertTrue(any(tuple(r) != a for r, a in zip(raw, aligned)))


class ThresholdFeasibleCoverageTests(unittest.TestCase):
    def test_agrees_with_prefix_coverage_when_confidences_are_distinct(self):
        confidences = [0.9, 0.8, 0.7, 0.6, 0.5]
        correct = [True, True, True, False, False]
        self.assertAlmostEqual(
            threshold_feasible_coverage(confidences, correct, 0.0), 0.6, places=9
        )

    def test_a_tie_group_is_answered_whole_or_not_at_all(self):
        # Three items tie at 0.5; one of them is wrong. A cutoff at 0.5 takes all
        # three, so a zero-error budget cannot reach beyond the first two items.
        confidences = [0.9, 0.8, 0.5, 0.5, 0.5]
        correct = [True, True, True, False, True]
        self.assertAlmostEqual(
            threshold_feasible_coverage(confidences, correct, 0.0), 0.4, places=9
        )
        # The prefix curve is happy to stop mid-tie and claims more.
        from system1_audit.selective import risk_coverage_curve

        self.assertGreater(
            risk_coverage_curve(confidences, correct).coverage_at_risk(0.0), 0.4
        )

    def test_rejects_mismatched_lengths_and_empty_input(self):
        with self.assertRaises(ValueError):
            threshold_feasible_coverage([0.5], [True, False], 0.1)
        with self.assertRaises(ValueError):
            threshold_feasible_coverage([], [], 0.1)
        with self.assertRaises(ValueError):
            threshold_feasible_coverage([0.5], [True], 1.5)

    def test_whole_sample_qualifies_under_a_loose_budget(self):
        self.assertAlmostEqual(
            threshold_feasible_coverage([0.5, 0.5], [True, False], 0.5), 1.0, places=9
        )


class NoOpTests(unittest.TestCase):
    """Voting over an order-invariant decider must change nothing, exactly."""

    def setUp(self):
        self.items = questions(40)
        key = {q.state: q.correct for q in self.items}
        self.audit = audit_dataset(OrderInvariantDecider(key), self.items, n_permutations=6)

    def test_mean_probability_is_a_no_op(self):
        comparison = compare_strategies(
            self.audit, risk_budget=0.1, candidate=mean_probability, n_resamples=200
        )
        self.assertEqual(comparison.verdict, "no_op")
        self.assertEqual(comparison.accuracy_gain.point, 0.0)
        self.assertEqual(comparison.coverage_gain.point, 0.0)
        self.assertEqual(comparison.coverage_gain.width, 0.0)

    def test_averaging_identical_floats_is_not_bitwise_identical(self):
        # The reason the no-op detector needs a tolerance, reproduced so it
        # survives: every display's aligned vector here is bitwise identical, yet
        # the mean of six copies of 0.7 is not 0.7.
        vectors = aligned_vectors(self.audit.results[0])
        self.assertTrue(all(v == vectors[0] for v in vectors))
        self.assertNotEqual(sum([0.7] * 6) / 6, 0.7)
        base = aggregate(self.audit, first_order)
        voted = aggregate(self.audit, mean_probability)
        self.assertEqual([d.label for d in base], [d.label for d in voted])
        deltas = [abs(b.confidence - c.confidence) for b, c in zip(base, voted)]
        self.assertTrue(any(d > 0.0 for d in deltas))
        self.assertLess(max(deltas), 1e-12)

    def test_modal_vote_is_not_a_no_op_because_its_confidence_differs(self):
        # The label is unchanged, but the vote share replaces a probability, so
        # the selective number can move even where accuracy cannot.
        comparison = compare_strategies(
            self.audit, risk_budget=0.1, candidate=modal_vote, n_resamples=200
        )
        self.assertNotEqual(comparison.verdict, "no_op")
        self.assertEqual(comparison.accuracy_gain.point, 0.0)
        self.assertEqual(comparison.candidate.distinct_confidences, 1)

    def test_a_zero_width_interval_at_zero_is_not_called_unresolved(self):
        # modal_vote decides differently (a vote share, not a probability) but
        # scores identically on every resample. That is an answer, not a power
        # problem, and must not be labelled as one.
        comparison = compare_strategies(
            self.audit, risk_budget=0.1, candidate=modal_vote, n_resamples=200
        )
        self.assertEqual(comparison.coverage_gain.width, 0.0)
        self.assertEqual(comparison.coverage_gain.point, 0.0)
        self.assertEqual(comparison.verdict, "no_difference_measured")

    def test_extra_passes_are_charged_even_for_a_no_op(self):
        comparison = compare_strategies(
            self.audit, risk_budget=0.1, candidate=mean_probability, n_resamples=200
        )
        self.assertEqual(comparison.baseline.passes_per_item, 1)
        self.assertEqual(comparison.candidate.passes_per_item, 6)
        self.assertEqual(comparison.extra_passes_per_item, 5)
        self.assertEqual(comparison.coverage_gain_per_extra_pass, 0.0)


class PlantedPositionBiasTests(unittest.TestCase):
    """Averaging over display orders should cancel a position term."""

    def setUp(self):
        self.items = questions(60)
        key = {q.state: q.correct for q in self.items}
        self.decider = SyntheticDecider(
            answer_key=key, skill=0.25, position_weight=0.5, noise=0.1, seed=7
        )
        self.audit = audit_dataset(self.decider, self.items, n_permutations=8, seed=3)

    def test_mean_probability_recovers_accuracy_the_position_bias_cost(self):
        base = strategy_report(self.audit, first_order, 0.1, n_resamples=200)
        voted = strategy_report(self.audit, mean_probability, 0.1, n_resamples=200)
        self.assertGreater(voted.accuracy, base.accuracy)

    def test_the_accuracy_gain_is_resolved_as_a_paired_interval(self):
        comparison = compare_strategies(
            self.audit, risk_budget=0.1, candidate=mean_probability, n_resamples=500
        )
        self.assertGreater(comparison.accuracy_gain.point, 0.0)
        self.assertTrue(comparison.accuracy_gain.excludes(0.0))

    def test_voting_is_charged_the_passes_it_used(self):
        comparison = compare_strategies(
            self.audit, risk_budget=0.1, candidate=mean_probability, n_resamples=200
        )
        self.assertEqual(comparison.extra_passes_per_item, 7)
        self.assertLess(
            comparison.coverage_gain_per_extra_pass, comparison.coverage_gain.point + 1e-12
        )


class CoarseConfidenceTests(unittest.TestCase):
    """A vote share is a coarse gate, and coarseness costs coverage."""

    def setUp(self):
        self.items = questions(60)
        key = {q.state: q.correct for q in self.items}
        decider = SyntheticDecider(
            answer_key=key, skill=0.25, position_weight=0.4, noise=0.5, seed=11
        )
        self.audit = audit_dataset(decider, self.items, n_permutations=8, seed=5)

    def test_modal_vote_confidence_takes_few_distinct_values(self):
        voted = strategy_report(self.audit, modal_vote, 0.1, n_resamples=200)
        continuous = strategy_report(self.audit, mean_probability, 0.1, n_resamples=200)
        self.assertLessEqual(voted.distinct_confidences, self.audit.n_permutations + 1)
        self.assertGreater(continuous.distinct_confidences, voted.distinct_confidences)

    def test_threshold_coverage_never_exceeds_the_prefix_number(self):
        for strategy in (first_order, mean_probability, modal_vote):
            report = strategy_report(self.audit, strategy, 0.1, n_resamples=200)
            self.assertLessEqual(
                report.threshold_coverage, report.coverage_at_risk + 1e-12, report.name
            )

    def test_coverage_per_pass_cannot_exceed_one_over_the_pass_count(self):
        # The ceiling that makes this metric unfair across pass counts, asserted
        # so the docstring's warning is not just prose.
        for strategy in (first_order, mean_probability, modal_vote):
            report = strategy_report(self.audit, strategy, 0.1, n_resamples=100)
            self.assertLessEqual(
                report.coverage_per_pass, 1.0 / report.passes_per_item + 1e-12, report.name
            )

    def test_coverage_per_pass_divides_by_the_cost(self):
        voted = strategy_report(self.audit, mean_probability, 0.1, n_resamples=200)
        self.assertAlmostEqual(
            voted.coverage_per_pass,
            voted.threshold_coverage / self.audit.n_permutations,
            places=12,
        )


class ReportShapeTests(unittest.TestCase):
    def setUp(self):
        self.items = questions(24)
        key = {q.state: q.correct for q in self.items}
        decider = SyntheticDecider(answer_key=key, skill=0.4, noise=0.3, seed=2)
        self.audit = audit_dataset(decider, self.items, n_permutations=6, seed=1)

    def test_aggregate_returns_one_decision_per_item_in_order(self):
        decisions = aggregate(self.audit, modal_vote)
        self.assertEqual(
            [d.item_id for d in decisions], [r.item_id for r in self.audit.results]
        )

    def test_modal_vote_matches_the_audits_own_vote_accuracy(self):
        # Same strategy the permutation audit already reports, so the two must
        # agree; if they diverge, one of them is aggregating differently.
        voted = strategy_report(self.audit, modal_vote, 0.1, n_resamples=100)
        self.assertAlmostEqual(voted.accuracy, self.audit.accuracy_modal_vote, places=12)

    def test_first_order_matches_the_audits_own_first_order_accuracy(self):
        base = strategy_report(self.audit, first_order, 0.1, n_resamples=100)
        self.assertAlmostEqual(
            base.accuracy, self.audit.accuracy_first_order, places=12
        )

    def test_every_reported_quantity_carries_an_interval(self):
        report = strategy_report(self.audit, mean_probability, 0.1, n_resamples=200)
        self.assertTrue(report.accuracy_interval.contains(report.accuracy))
        self.assertTrue(0.0 <= report.coverage_interval.lower <= 1.0)
        self.assertTrue(0.0 <= report.coverage_interval.upper <= 1.0)

    def test_results_are_reproducible_across_calls(self):
        a = compare_strategies(self.audit, 0.1, n_resamples=300, seed=4)
        b = compare_strategies(self.audit, 0.1, n_resamples=300, seed=4)
        self.assertEqual(a.coverage_gain.lower, b.coverage_gain.lower)
        self.assertEqual(a.coverage_gain.upper, b.coverage_gain.upper)

    def test_custom_passes_per_item_is_honoured(self):
        report = strategy_report(
            self.audit, mean_probability, 0.1, passes_per_item=3, n_resamples=100
        )
        self.assertEqual(report.passes_per_item, 3)
        with self.assertRaises(ValueError):
            strategy_report(
                self.audit, mean_probability, 0.1, passes_per_item=0, n_resamples=100
            )

    def test_verdict_is_one_of_the_documented_strings(self):
        comparison = compare_strategies(self.audit, 0.1, n_resamples=300)
        self.assertIn(
            comparison.verdict,
            {
                "no_op",
                "no_difference_measured",
                "coverage_gain_resolved",
                "coverage_loss_resolved",
                "unresolved",
            },
        )


if __name__ == "__main__":
    unittest.main()
