"""Risk-coverage behaviour, checked against hand-computed curves."""

import unittest

from system1_audit.selective import risk_coverage_curve, threshold_feasible_coverage


class TestRiskCoverageCurve(unittest.TestCase):
    def test_perfect_model_has_zero_risk_everywhere(self):
        report = risk_coverage_curve([0.9, 0.8, 0.7], [True, True, True])
        self.assertAlmostEqual(report.full_coverage_risk, 0.0)
        self.assertAlmostEqual(report.aurc, 0.0)

    def test_curve_has_one_point_per_observation(self):
        report = risk_coverage_curve([0.9, 0.8, 0.7, 0.6], [True, False, True, False])
        self.assertEqual(len(report.curve), 4)
        self.assertAlmostEqual(report.curve[-1].coverage, 1.0)

    def test_points_are_ordered_by_descending_confidence(self):
        report = risk_coverage_curve([0.2, 0.9, 0.5], [True, True, False])
        thresholds = [p.threshold for p in report.curve]
        self.assertEqual(thresholds, sorted(thresholds, reverse=True))

    def test_risk_matches_a_hand_computed_curve(self):
        # Sorted by confidence: 0.9 right, 0.8 wrong, 0.7 right, 0.6 wrong.
        # Cumulative risk: 0/1, 1/2, 1/3, 2/4.
        report = risk_coverage_curve([0.9, 0.8, 0.7, 0.6], [True, False, True, False])
        self.assertEqual([p.risk for p in report.curve], [0.0, 0.5, 1 / 3, 0.5])
        self.assertAlmostEqual(report.aurc, (0.0 + 0.5 + 1 / 3 + 0.5) / 4)

    def test_full_coverage_risk_equals_overall_error_rate(self):
        correct = [True, False, True, True, False]
        report = risk_coverage_curve([0.5, 0.4, 0.3, 0.2, 0.1], correct)
        self.assertAlmostEqual(report.full_coverage_risk, 2 / 5)

    def test_abstention_beats_answering_everything_when_confidence_ranks_well(self):
        # Confidence is perfectly informative: every error sits at the bottom.
        confidences = [0.99, 0.98, 0.97, 0.10, 0.05]
        correct = [True, True, True, False, False]
        report = risk_coverage_curve(confidences, correct)
        self.assertAlmostEqual(report.full_coverage_risk, 0.4)
        self.assertAlmostEqual(report.coverage_at_risk(0.0), 0.6)

    def test_uninformative_confidence_buys_no_coverage_at_zero_risk(self):
        # The single most confident item is wrong, so nothing is safe.
        report = risk_coverage_curve([0.9, 0.8, 0.7], [False, True, True])
        self.assertAlmostEqual(report.coverage_at_risk(0.0), 0.0)

    def test_coverage_at_risk_is_monotone_in_the_target(self):
        confidences = [0.9, 0.8, 0.7, 0.6, 0.5]
        correct = [True, True, False, True, False]
        report = risk_coverage_curve(confidences, correct)
        previous = 0.0
        for target in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
            current = report.coverage_at_risk(target)
            self.assertGreaterEqual(current, previous)
            previous = current

    def test_threshold_at_risk_identifies_an_operating_point(self):
        confidences = [0.99, 0.98, 0.97, 0.10]
        correct = [True, True, True, False]
        report = risk_coverage_curve(confidences, correct)
        self.assertAlmostEqual(report.threshold_at_risk(0.0), 0.97)

    def test_threshold_at_risk_is_none_when_unreachable(self):
        report = risk_coverage_curve([0.9, 0.8], [False, False])
        self.assertIsNone(report.threshold_at_risk(0.0))

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            risk_coverage_curve([0.5], [True, False])

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            risk_coverage_curve([], [])


class TestFeasibleOperatingPoint(unittest.TestCase):
    """A reported cutoff has to be one a deployment can actually set.

    The curve walks one item at a time, so it can stop inside a group of
    equally confident items. A cutoff answers all of a tie group or none of it.
    Reporting a coverage from the curve and a threshold from the same point
    therefore describes an operating point that does not exist.
    """

    # Three items tie at 0.9, one of them wrong, and a fourth sits below.
    # A zero-error budget is met by the curve after two items -- but no cutoff
    # can answer two of the three tied items.
    TIED_CONFIDENCES = [0.9, 0.9, 0.9, 0.5]
    TIED_CORRECT = [True, True, False, True]

    def _realised(self, confidences, correct, threshold):
        """What the cutoff actually delivers: coverage and error rate."""
        answered = [i for i in range(len(confidences)) if confidences[i] >= threshold]
        errors = sum(1 for i in answered if not correct[i])
        return len(answered) / len(confidences), errors / len(answered)

    def test_curve_coverage_is_not_reachable_by_any_cutoff(self):
        report = risk_coverage_curve(self.TIED_CONFIDENCES, self.TIED_CORRECT)
        # The bound still reports the prefix figure, which is the point of it.
        self.assertAlmostEqual(report.coverage_at_risk(0.0), 0.5)
        # No cutoff delivers it.
        self.assertAlmostEqual(report.feasible_coverage_at_risk(0.0), 0.0)
        self.assertIsNone(report.operating_point(0.0))

    def test_reported_threshold_never_breaches_the_budget(self):
        # Before this was fixed, threshold_at_risk(0.0) returned 0.9, whose
        # realised error rate is 1/3 -- a cutoff handed back as meeting a
        # zero-error budget while breaching it.
        report = risk_coverage_curve(self.TIED_CONFIDENCES, self.TIED_CORRECT)
        for budget in (0.0, 0.1, 0.2, 0.25, 1 / 3, 0.5, 1.0):
            threshold = report.threshold_at_risk(budget)
            if threshold is None:
                continue
            _, realised_risk = self._realised(
                self.TIED_CONFIDENCES, self.TIED_CORRECT, threshold
            )
            self.assertLessEqual(
                realised_risk,
                budget + 1e-12,
                f"cutoff {threshold} carries risk {realised_risk} at budget {budget}",
            )

    def test_the_three_reported_numbers_describe_the_same_cutoff(self):
        report = risk_coverage_curve(self.TIED_CONFIDENCES, self.TIED_CORRECT)
        point = report.operating_point(0.5)
        self.assertIsNotNone(point)
        coverage, risk = self._realised(
            self.TIED_CONFIDENCES, self.TIED_CORRECT, point.threshold
        )
        self.assertAlmostEqual(point.coverage, coverage)
        self.assertAlmostEqual(point.risk, risk)
        self.assertEqual(point.n_answered, round(coverage * len(self.TIED_CONFIDENCES)))

    def test_a_tie_group_cutoff_is_selectable_when_it_is_the_best_one(self):
        # Abstaining is not the only answer: the 0.9 cutoff carries exactly the
        # tie group's error rate, so a budget of 1/3 admits it. Here the
        # bottom item is also wrong, which puts full coverage at risk 0.5 and
        # out of budget, leaving the tie-group cutoff as the best feasible one.
        report = risk_coverage_curve([0.9, 0.9, 0.9, 0.5], [True, True, False, False])
        point = report.operating_point(1 / 3)
        self.assertIsNotNone(point)
        self.assertAlmostEqual(point.threshold, 0.9)
        self.assertAlmostEqual(point.coverage, 0.75)
        self.assertAlmostEqual(point.risk, 1 / 3)
        self.assertEqual(point.n_answered, 3)

    def test_a_looser_budget_prefers_the_lower_cutoff(self):
        # With budget 1/3 and a correct bottom item, answering everything costs
        # 0.25 and covers all of it, so the tie cutoff is not the best choice.
        report = risk_coverage_curve(self.TIED_CONFIDENCES, self.TIED_CORRECT)
        point = report.operating_point(1 / 3)
        self.assertIsNotNone(point)
        self.assertAlmostEqual(point.threshold, 0.5)
        self.assertAlmostEqual(point.coverage, 1.0)

    def test_feasible_never_exceeds_the_bound(self):
        # Property check over tie patterns the hand-written cases do not cover.
        for seed in range(60):
            rng = __import__("random").Random(seed)
            n = rng.randint(1, 14)
            # A small value set forces ties.
            confidences = [rng.choice([0.2, 0.5, 0.5, 0.8, 0.8, 0.95]) for _ in range(n)]
            correct = [rng.random() < 0.7 for _ in range(n)]
            report = risk_coverage_curve(confidences, correct)
            for budget in (0.0, 0.05, 0.2, 0.5):
                feasible = report.feasible_coverage_at_risk(budget)
                self.assertLessEqual(feasible, report.coverage_at_risk(budget) + 1e-12)
                threshold = report.threshold_at_risk(budget)
                if threshold is not None:
                    _, realised = self._realised(confidences, correct, threshold)
                    self.assertLessEqual(realised, budget + 1e-12)

    def test_distinct_confidences_make_the_two_agree(self):
        # With no ties every curve point is reachable, so nothing is lost.
        for seed in range(40):
            rng = __import__("random").Random(1000 + seed)
            n = rng.randint(1, 16)
            confidences = rng.sample([i / 100.0 for i in range(1, 100)], n)
            correct = [rng.random() < 0.6 for _ in range(n)]
            report = risk_coverage_curve(confidences, correct)
            for budget in (0.0, 0.1, 0.3, 1.0):
                self.assertAlmostEqual(
                    report.feasible_coverage_at_risk(budget),
                    report.coverage_at_risk(budget),
                )

    def test_feasible_points_are_the_tie_group_boundaries(self):
        report = risk_coverage_curve([0.9, 0.9, 0.9, 0.5], [True] * 4)
        self.assertEqual(
            [p.n_answered for p in report.feasible_points()], [3, 4]
        )

    def test_feasible_points_is_the_whole_curve_without_ties(self):
        report = risk_coverage_curve([0.9, 0.8, 0.7], [True, True, True])
        self.assertEqual(len(report.feasible_points()), 3)

    def test_feasible_coverage_is_monotone_in_the_budget(self):
        confidences = [0.9, 0.9, 0.7, 0.7, 0.4]
        correct = [True, False, True, True, False]
        report = risk_coverage_curve(confidences, correct)
        previous = 0.0
        for budget in (0.0, 0.1, 0.25, 0.5, 1.0):
            current = report.feasible_coverage_at_risk(budget)
            self.assertGreaterEqual(current + 1e-12, previous)
            previous = current

    def test_free_function_matches_the_method(self):
        confidences = [0.9, 0.9, 0.9, 0.5]
        correct = [True, True, False, True]
        report = risk_coverage_curve(confidences, correct)
        for budget in (0.0, 1 / 3, 1.0):
            self.assertAlmostEqual(
                threshold_feasible_coverage(confidences, correct, budget),
                report.feasible_coverage_at_risk(budget),
            )

    def test_rejects_a_budget_outside_the_unit_interval(self):
        report = risk_coverage_curve([0.9, 0.8], [True, False])
        for bad in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                report.operating_point(bad)
            with self.assertRaises(ValueError):
                threshold_feasible_coverage([0.9, 0.8], [True, False], bad)


if __name__ == "__main__":
    unittest.main()
