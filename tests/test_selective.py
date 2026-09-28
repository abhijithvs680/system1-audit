"""Risk-coverage behaviour, checked against hand-computed curves."""

import unittest

from system1_audit.selective import risk_coverage_curve


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


if __name__ == "__main__":
    unittest.main()
