"""Calibration metrics checked against values computed by hand."""

import math
import unittest

from system1_audit.calibration import (
    apply_temperature,
    brier_score,
    calibration_report,
    equal_mass_bins,
    equal_width_bins,
    expected_calibration_error,
    fit_temperature,
    maximum_calibration_error,
    negative_log_likelihood,
)


class TestEqualWidthBins(unittest.TestCase):
    def test_perfect_calibration_gives_zero_ece(self):
        # Ten items at confidence 0.9, exactly nine of them correct.
        confidences = [0.9] * 10
        correct = [True] * 9 + [False]
        bins = equal_width_bins(confidences, correct, n_bins=10)
        self.assertAlmostEqual(expected_calibration_error(bins), 0.0, places=12)

    def test_full_overconfidence_gives_ece_equal_to_confidence(self):
        # Always 100% sure, always wrong: the gap is the confidence itself.
        bins = equal_width_bins([1.0] * 4, [False] * 4, n_bins=10)
        self.assertAlmostEqual(expected_calibration_error(bins), 1.0)
        self.assertAlmostEqual(maximum_calibration_error(bins), 1.0)

    def test_ece_is_count_weighted_across_bins(self):
        # Bin [0.8,0.9): 3 items at 0.85, 1 correct -> accuracy 1/3, gap 0.85-1/3
        # Bin [0.5,0.6): 1 item at 0.55, correct    -> accuracy 1.0, gap 0.45
        confidences = [0.85, 0.85, 0.85, 0.55]
        correct = [True, False, False, True]
        bins = equal_width_bins(confidences, correct, n_bins=10)
        expected = (3 * abs(0.85 - 1 / 3) + 1 * abs(0.55 - 1.0)) / 4
        self.assertAlmostEqual(expected_calibration_error(bins), expected)

    def test_boundary_confidence_of_one_lands_in_last_bin(self):
        bins = equal_width_bins([1.0], [True], n_bins=5)
        self.assertEqual(bins[-1].count, 1)
        self.assertEqual(sum(b.count for b in bins), 1)

    def test_empty_bins_do_not_contribute(self):
        bins = equal_width_bins([0.95, 0.95], [True, True], n_bins=10)
        self.assertEqual(sum(1 for b in bins if b.count == 0), 9)
        self.assertAlmostEqual(expected_calibration_error(bins), 0.05)


class TestEqualMassBins(unittest.TestCase):
    def test_bins_hold_equal_counts_when_divisible(self):
        confidences = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        bins = equal_mass_bins(confidences, [True] * 6, n_bins=3)
        self.assertEqual([b.count for b in bins], [2, 2, 2])

    def test_every_observation_is_assigned_exactly_once(self):
        confidences = [0.1, 0.15, 0.7, 0.71, 0.72, 0.9, 0.91]
        bins = equal_mass_bins(confidences, [True] * 7, n_bins=4)
        self.assertEqual(sum(b.count for b in bins), 7)

    def test_adaptive_ece_exposes_a_crowded_bin_that_equal_width_hides(self):
        # 100 items crammed at 0.91-0.99, half wrong. Equal-width buries them
        # in one bucket; equal-mass spreads them and the miscalibration shows.
        confidences = [0.91 + 0.0008 * i for i in range(100)]
        correct = [i % 2 == 0 for i in range(100)]
        width = expected_calibration_error(equal_width_bins(confidences, correct, 10))
        mass = expected_calibration_error(equal_mass_bins(confidences, correct, 10))
        self.assertGreater(mass, 0.0)
        self.assertGreater(width, 0.0)
        self.assertGreaterEqual(mass, width - 1e-9)


class TestProperScoringRules(unittest.TestCase):
    def test_brier_is_zero_for_a_perfect_confident_prediction(self):
        self.assertAlmostEqual(brier_score([[1.0, 0.0]], [0]), 0.0)

    def test_brier_multiclass_form_is_sum_of_squares(self):
        # (0.7-1)^2 + (0.2-0)^2 + (0.1-0)^2 = 0.09 + 0.04 + 0.01 = 0.14
        self.assertAlmostEqual(brier_score([[0.7, 0.2, 0.1]], [0]), 0.14)

    def test_brier_worst_case_is_two(self):
        self.assertAlmostEqual(brier_score([[0.0, 1.0]], [0]), 2.0)

    def test_nll_of_certainty_is_zero(self):
        self.assertAlmostEqual(negative_log_likelihood([[1.0, 0.0]], [0]), 0.0)

    def test_nll_matches_log_of_assigned_probability(self):
        self.assertAlmostEqual(
            negative_log_likelihood([[0.25, 0.75]], [0]), -math.log(0.25)
        )

    def test_nll_is_finite_for_a_zero_probability_truth(self):
        self.assertTrue(math.isfinite(negative_log_likelihood([[0.0, 1.0]], [0])))


class TestTemperature(unittest.TestCase):
    def test_temperature_one_is_the_identity(self):
        probs = (0.6, 0.3, 0.1)
        for got, want in zip(apply_temperature(probs, 1.0), probs):
            self.assertAlmostEqual(got, want)

    def test_high_temperature_flattens_toward_uniform(self):
        flat = apply_temperature((0.9, 0.05, 0.05), 20.0)
        self.assertLess(max(flat) - min(flat), 0.2)

    def test_low_temperature_sharpens_toward_the_argmax(self):
        sharp = apply_temperature((0.5, 0.3, 0.2), 0.05)
        self.assertGreater(max(sharp), 0.99)

    def test_temperature_preserves_ranking(self):
        probs = (0.5, 0.3, 0.2)
        for t in (0.3, 1.0, 5.0):
            scaled = apply_temperature(probs, t)
            self.assertEqual(
                sorted(range(3), key=lambda i: -probs[i]),
                sorted(range(3), key=lambda i: -scaled[i]),
            )

    def test_output_is_a_probability_distribution(self):
        self.assertAlmostEqual(sum(apply_temperature((0.4, 0.4, 0.2), 3.0)), 1.0)

    def test_rejects_non_positive_temperature(self):
        with self.assertRaises(ValueError):
            apply_temperature((0.5, 0.5), 0.0)

    def test_fitting_softens_an_overconfident_model(self):
        # Confidently asserts the first option every time, but is only right
        # 60% of the time. The fit must raise T above 1 to spread mass out.
        probs = [[0.99, 0.01] for _ in range(100)]
        truth = [0 if i % 10 < 6 else 1 for i in range(100)]
        self.assertGreater(fit_temperature(probs, truth), 1.0)

    def test_fitting_sharpens_an_underconfident_model(self):
        probs = [[0.55, 0.45] for _ in range(100)]
        truth = [0] * 100
        self.assertLess(fit_temperature(probs, truth), 1.0)

    def test_fitted_temperature_does_not_increase_nll(self):
        probs = [[0.9, 0.1] for _ in range(50)] + [[0.1, 0.9] for _ in range(50)]
        truth = [0] * 35 + [1] * 15 + [1] * 35 + [0] * 15
        before = negative_log_likelihood(probs, truth)
        t = fit_temperature(probs, truth)
        after = negative_log_likelihood([apply_temperature(p, t) for p in probs], truth)
        self.assertLessEqual(after, before + 1e-9)


class TestCalibrationReport(unittest.TestCase):
    def test_report_reflects_accuracy_and_confidence(self):
        probs = [[0.8, 0.2]] * 5 + [[0.2, 0.8]] * 5
        truth = [0] * 5 + [1] * 5
        report = calibration_report(probs, truth, n_bins=10)
        self.assertEqual(report.n, 10)
        self.assertAlmostEqual(report.accuracy, 1.0)
        self.assertAlmostEqual(report.mean_confidence, 0.8)
        self.assertAlmostEqual(report.overconfidence, -0.2)

    def test_overconfidence_is_positive_when_confidence_exceeds_accuracy(self):
        probs = [[0.95, 0.05]] * 10
        truth = [0] * 5 + [1] * 5
        report = calibration_report(probs, truth, n_bins=10)
        self.assertAlmostEqual(report.accuracy, 0.5)
        self.assertAlmostEqual(report.overconfidence, 0.45)
        self.assertAlmostEqual(report.ece, 0.45)

    def test_report_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            calibration_report([[0.5, 0.5]], [0, 1])


if __name__ == "__main__":
    unittest.main()
