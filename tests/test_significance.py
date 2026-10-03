"""Statistical significance layer, checked against exactly computable values.

The arithmetic here decides whether the rest of the harness has found anything,
so it is checked against closed forms and exhaustive enumeration rather than
against itself. Where a quantity is only available by Monte Carlo, the test
asserts a property that must hold (monotonicity, scaling, determinism) instead
of a fixture number that would drift with the seed.

Standard library only, like the rest of the suite.
"""

import math
import pathlib
import subprocess
import sys
import unittest

from system1_audit.calibration import equal_width_bins, expected_calibration_error
from system1_audit.deciders.mock import SyntheticDecider
from system1_audit.permutation import audit_dataset
from system1_audit.significance import (
    Interval,
    OrderSensitivitySignificance,
    _item_position_means,
    _percentile,
    binomial_tail_at_least,
    binomial_tail_at_most,
    binomial_test_greater,
    bootstrap_interval,
    clopper_pearson_interval,
    ece_noise_floor,
    normal_quantile,
    order_sensitivity_significance,
    selective_coverage_interval,
    wilson_interval,
)
from system1_audit.types import ChoiceQuestion

LABELS = ("alpha", "beta", "gamma", "delta")


def make_questions(n: int) -> list[ChoiceQuestion]:
    """``n`` questions with distinct state text and rotating correct labels.

    Distinct state matters: repeating identical state under new identifiers
    would make the items repeated measurements of one question, which is the
    thing the resampling unit in this module is chosen to avoid.
    """
    return [
        ChoiceQuestion(
            state=f"case {i}: the account shows an unexpected balance of {i * 37}",
            labels=LABELS,
            correct=LABELS[i % len(LABELS)],
            item_id=f"q{i:03d}",
        )
        for i in range(n)
    ]


class TestInterval(unittest.TestCase):
    def test_contains_and_excludes_are_complementary(self):
        iv = Interval(point=0.2, lower=0.1, upper=0.3, confidence=0.95, method="test")
        for value in (0.0, 0.1, 0.2, 0.3, 0.4):
            self.assertNotEqual(iv.contains(value), iv.excludes(value))

    def test_bounds_are_inclusive(self):
        iv = Interval(point=0.2, lower=0.1, upper=0.3, confidence=0.95, method="test")
        self.assertTrue(iv.contains(0.1))
        self.assertTrue(iv.contains(0.3))
        self.assertFalse(iv.excludes(0.1))

    def test_width(self):
        iv = Interval(point=0.2, lower=0.1, upper=0.3, confidence=0.95, method="test")
        self.assertAlmostEqual(iv.width, 0.2, places=12)

    def test_inverted_bounds_rejected(self):
        with self.assertRaises(ValueError):
            Interval(point=0.2, lower=0.3, upper=0.1, confidence=0.95, method="test")

    def test_confidence_must_be_a_proper_probability(self):
        for bad in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                Interval(point=0.2, lower=0.1, upper=0.3, confidence=bad, method="t")

    def test_str_reports_the_method(self):
        iv = Interval(point=0.25, lower=0.1, upper=0.3, confidence=0.95, method="Wilson")
        text = str(iv)
        self.assertIn("0.2500", text)
        self.assertIn("95%", text)
        self.assertIn("Wilson", text)


class TestNormalQuantile(unittest.TestCase):
    def test_median_is_zero(self):
        self.assertAlmostEqual(normal_quantile(0.5), 0.0, places=9)

    def test_known_quantiles(self):
        # Standard published values.
        self.assertAlmostEqual(normal_quantile(0.975), 1.959963985, places=6)
        self.assertAlmostEqual(normal_quantile(0.995), 2.575829304, places=6)
        self.assertAlmostEqual(normal_quantile(0.95), 1.644853627, places=6)

    def test_symmetry(self):
        for p in (0.6, 0.75, 0.9, 0.99):
            self.assertAlmostEqual(normal_quantile(p), -normal_quantile(1.0 - p), places=8)

    def test_round_trips_through_the_cdf(self):
        for p in (0.01, 0.2, 0.5, 0.8, 0.999):
            z = normal_quantile(p)
            cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            self.assertAlmostEqual(cdf, p, places=9)

    def test_out_of_range_rejected(self):
        for bad in (0.0, 1.0, -0.5, 2.0):
            with self.assertRaises(ValueError):
                normal_quantile(bad)


class TestBinomialTails(unittest.TestCase):
    def test_fair_coin_by_hand(self):
        # P(X >= 2 | n=3, p=0.5) = (3 + 1) / 8
        self.assertAlmostEqual(binomial_tail_at_least(2, 3, 0.5), 0.5, places=12)
        # P(X >= 3 | n=3, p=0.5) = 1/8
        self.assertAlmostEqual(binomial_tail_at_least(3, 3, 0.5), 0.125, places=12)

    def test_biased_coin_by_hand(self):
        # P(X >= 1 | n=2, p=0.3) = 1 - 0.7^2
        self.assertAlmostEqual(binomial_tail_at_least(1, 2, 0.3), 1.0 - 0.49, places=12)

    def test_tails_are_complementary(self):
        for k in range(0, 6):
            self.assertAlmostEqual(
                binomial_tail_at_least(k, 5, 0.4) + binomial_tail_at_most(k - 1, 5, 0.4),
                1.0,
                places=12,
            )

    def test_degenerate_thresholds(self):
        self.assertEqual(binomial_tail_at_least(0, 5, 0.3), 1.0)
        self.assertEqual(binomial_tail_at_least(6, 5, 0.3), 0.0)
        self.assertEqual(binomial_tail_at_most(-1, 5, 0.3), 0.0)
        self.assertEqual(binomial_tail_at_most(5, 5, 0.3), 1.0)

    def test_degenerate_probabilities(self):
        self.assertEqual(binomial_tail_at_least(1, 4, 0.0), 0.0)
        self.assertEqual(binomial_tail_at_least(1, 4, 1.0), 1.0)

    def test_monotone_in_p(self):
        previous = -1.0
        for step in range(11):
            value = binomial_tail_at_least(3, 8, step / 10.0)
            self.assertGreaterEqual(value, previous)
            previous = value

    def test_test_greater_matches_the_tail(self):
        self.assertAlmostEqual(
            binomial_test_greater(7, 10, 0.5),
            binomial_tail_at_least(7, 10, 0.5),
            places=12,
        )

    def test_test_greater_validates_counts(self):
        with self.assertRaises(ValueError):
            binomial_test_greater(11, 10, 0.5)
        with self.assertRaises(ValueError):
            binomial_test_greater(-1, 10, 0.5)
        with self.assertRaises(ValueError):
            binomial_test_greater(0, 0, 0.5)


class TestWilsonInterval(unittest.TestCase):
    def test_against_the_closed_form(self):
        successes, n, z = 4, 20, 1.959963985
        p_hat = successes / n
        denom = 1.0 + z * z / n
        center = (p_hat + z * z / (2.0 * n)) / denom
        margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4.0 * n * n)) / denom
        iv = wilson_interval(successes, n, 0.95)
        self.assertAlmostEqual(iv.point, 0.2, places=12)
        self.assertAlmostEqual(iv.lower, center - margin, places=8)
        self.assertAlmostEqual(iv.upper, center + margin, places=8)

    def test_zero_successes_still_has_width(self):
        # The normal approximation collapses here; Wilson does not. This is the
        # case an order-sensitivity audit hits on an order-invariant model.
        iv = wilson_interval(0, 30, 0.95)
        self.assertEqual(iv.point, 0.0)
        self.assertEqual(iv.lower, 0.0)
        self.assertGreater(iv.upper, 0.0)

    def test_bounds_stay_in_the_unit_interval(self):
        for successes in (0, 1, 19, 20):
            iv = wilson_interval(successes, 20, 0.99)
            self.assertGreaterEqual(iv.lower, 0.0)
            self.assertLessEqual(iv.upper, 1.0)

    def test_narrows_as_n_grows(self):
        widths = [wilson_interval(n // 5, n, 0.95).width for n in (20, 200, 2000)]
        self.assertGreater(widths[0], widths[1])
        self.assertGreater(widths[1], widths[2])

    def test_higher_confidence_is_wider(self):
        self.assertGreater(
            wilson_interval(5, 40, 0.99).width, wilson_interval(5, 40, 0.90).width
        )


class TestClopperPearson(unittest.TestCase):
    def test_bounds_satisfy_their_defining_tail_equations(self):
        successes, n, confidence = 3, 25, 0.95
        half = (1.0 - confidence) / 2.0
        iv = clopper_pearson_interval(successes, n, confidence)
        self.assertAlmostEqual(binomial_tail_at_least(successes, n, iv.lower), half, places=6)
        self.assertAlmostEqual(binomial_tail_at_most(successes, n, iv.upper), half, places=6)

    def test_no_successes_gives_a_zero_lower_bound(self):
        iv = clopper_pearson_interval(0, 40, 0.95)
        self.assertEqual(iv.lower, 0.0)
        self.assertTrue(iv.contains(0.0))
        # The classic rule of three: the upper bound sits near 3/n.
        self.assertAlmostEqual(iv.upper, 1.0 - 0.025 ** (1.0 / 40), places=6)

    def test_all_successes_gives_a_one_upper_bound(self):
        iv = clopper_pearson_interval(12, 12, 0.95)
        self.assertEqual(iv.upper, 1.0)

    def test_a_single_success_already_excludes_zero_at_any_n(self):
        # Why the verdict in OrderSensitivitySignificance takes a threshold
        # rather than testing against zero: one observed success excludes a zero
        # rate just as firmly at n=40 as at n=1,000,000, so a "non-zero" verdict
        # carries no information about n or about effect size. What does change
        # with n is the interval's width, and that is the useful part.
        widths = []
        for n in (40, 200, 1000):
            iv = clopper_pearson_interval(1, n, 0.95)
            self.assertGreater(iv.lower, 0.0)
            self.assertTrue(iv.excludes(0.0))
            widths.append(iv.width)
        self.assertGreater(widths[0], widths[1])
        self.assertGreater(widths[1], widths[2])
        self.assertAlmostEqual(clopper_pearson_interval(1, 40, 0.95).point, 0.025, places=12)

    def test_conservative_relative_to_wilson(self):
        exact = clopper_pearson_interval(5, 30, 0.95)
        approximate = wilson_interval(5, 30, 0.95)
        self.assertGreaterEqual(exact.width, approximate.width)

    def test_validates_counts(self):
        with self.assertRaises(ValueError):
            clopper_pearson_interval(5, 4, 0.95)
        with self.assertRaises(ValueError):
            clopper_pearson_interval(0, 0, 0.95)


class TestPercentile(unittest.TestCase):
    def test_endpoints(self):
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(_percentile(values, 0.0), 1.0)
        self.assertEqual(_percentile(values, 1.0), 4.0)

    def test_interpolates_linearly(self):
        # position = 0.5 * 3 = 1.5, halfway between 2.0 and 3.0
        self.assertAlmostEqual(_percentile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5, places=12)

    def test_single_value(self):
        self.assertEqual(_percentile([7.0], 0.3), 7.0)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            _percentile([], 0.5)
        with self.assertRaises(ValueError):
            _percentile([1.0, 2.0], 1.5)


class TestBootstrapInterval(unittest.TestCase):
    def test_point_estimate_is_the_unresampled_statistic(self):
        values = [0.0, 1.0, 2.0, 3.0, 10.0]
        mean = lambda sample: sum(sample) / len(sample)
        iv = bootstrap_interval(values, mean, 0.95, 500, seed=7)
        self.assertAlmostEqual(iv.point, mean(values), places=12)

    def test_is_deterministic_for_a_given_seed(self):
        values = [0.1, 0.4, 0.9, 0.2, 0.7, 0.3]
        mean = lambda sample: sum(sample) / len(sample)
        first = bootstrap_interval(values, mean, 0.95, 400, seed=11)
        second = bootstrap_interval(values, mean, 0.95, 400, seed=11)
        self.assertEqual((first.lower, first.upper), (second.lower, second.upper))

    def test_different_seeds_move_the_bounds(self):
        values = [0.1, 0.4, 0.9, 0.2, 0.7, 0.3]
        mean = lambda sample: sum(sample) / len(sample)
        first = bootstrap_interval(values, mean, 0.95, 400, seed=11)
        second = bootstrap_interval(values, mean, 0.95, 400, seed=12)
        self.assertNotEqual((first.lower, first.upper), (second.lower, second.upper))

    def test_constant_data_gives_a_zero_width_interval(self):
        values = [2.5] * 12
        iv = bootstrap_interval(values, lambda s: sum(s) / len(s), 0.95, 200, seed=3)
        self.assertAlmostEqual(iv.width, 0.0, places=12)

    def test_brackets_the_mean_of_a_symmetric_sample(self):
        values = [float(i) for i in range(1, 41)]
        iv = bootstrap_interval(values, lambda s: sum(s) / len(s), 0.95, 1500, seed=5)
        self.assertTrue(iv.contains(20.5))
        # Roughly the normal-theory interval: 1.96 * sd / sqrt(n) each side.
        sd = math.sqrt(sum((v - 20.5) ** 2 for v in values) / len(values))
        self.assertLess(abs(iv.width - 2 * 1.96 * sd / math.sqrt(len(values))), 1.0)

    def test_narrows_as_the_sample_grows(self):
        mean = lambda sample: sum(sample) / len(sample)
        pattern = [0.0, 1.0]
        small = bootstrap_interval(pattern * 10, mean, 0.95, 800, seed=2)
        large = bootstrap_interval(pattern * 200, mean, 0.95, 800, seed=2)
        self.assertGreater(small.width, large.width)

    def test_skips_resamples_the_statistic_cannot_evaluate(self):
        # A statistic that rejects resamples missing a particular item still
        # yields an interval, as long as enough resamples survive.
        values = list(range(20))

        def picky(sample):
            if 0 not in sample:
                raise ValueError("needs item 0")
            return sum(sample) / len(sample)

        iv = bootstrap_interval(values, picky, 0.95, 600, seed=4)
        self.assertIn("resamples", iv.method)

    def test_refuses_an_interval_when_almost_every_resample_fails(self):
        values = list(range(40))

        def hopeless(sample):
            if len(set(sample)) == len(values):
                return 0.0
            raise ValueError("needs every item exactly once")

        with self.assertRaises(ValueError):
            bootstrap_interval(values, hopeless, 0.95, 200, seed=4)

    def test_validates_input(self):
        with self.assertRaises(ValueError):
            bootstrap_interval([], lambda s: 0.0, 0.95, 100, seed=0)
        with self.assertRaises(ValueError):
            bootstrap_interval([1.0], lambda s: 0.0, 0.95, 1, seed=0)


class TestEceNoiseFloor(unittest.TestCase):
    """The reference value a reported ECE has to beat.

    Binned ECE is positively biased: a model that is exactly as right as it
    says it is still scores above zero, because within-bin accuracy is a finite
    Bernoulli sample and the metric takes an absolute value. Without this
    floor, a raw-versus-fitted ECE comparison cannot be read.
    """

    def test_a_perfectly_calibrated_model_scores_above_zero(self):
        confidences = [0.55 + 0.4 * (i % 9) / 8.0 for i in range(60)]
        floor = ece_noise_floor(confidences, n_bins=10, n_simulations=400, seed=1)
        self.assertGreater(floor.mean, 0.0)
        self.assertGreater(floor.upper_quantile, floor.median)

    def test_floor_shrinks_as_n_grows(self):
        def floor_at(n: int) -> float:
            confidences = [0.55 + 0.4 * (i % 9) / 8.0 for i in range(n)]
            return ece_noise_floor(confidences, n_bins=10, n_simulations=400, seed=1).mean

        small, medium, large = floor_at(40), floor_at(200), floor_at(1000)
        self.assertGreater(small, medium)
        self.assertGreater(medium, large)

    def test_floor_grows_with_bin_count(self):
        confidences = [0.5 + 0.5 * (i % 20) / 19.0 for i in range(200)]
        coarse = ece_noise_floor(confidences, n_bins=2, n_simulations=400, seed=1).mean
        fine = ece_noise_floor(confidences, n_bins=20, n_simulations=400, seed=1).mean
        self.assertGreater(fine, coarse)

    def test_one_bin_over_constant_confidence_has_no_floor_worth_naming(self):
        # With a single bin and identical confidences the gap is one binomial
        # mean against its own parameter, so the floor is the smallest it gets.
        confidences = [0.8] * 500
        one_bin = ece_noise_floor(confidences, n_bins=1, n_simulations=400, seed=1).mean
        ten_bins = ece_noise_floor(confidences, n_bins=10, n_simulations=400, seed=1).mean
        self.assertAlmostEqual(one_bin, ten_bins, places=12)
        self.assertLess(one_bin, 0.05)

    def test_a_calibrated_observation_is_not_evidence_of_miscalibration(self):
        confidences = [0.55 + 0.4 * (i % 9) / 8.0 for i in range(120)]
        floor = ece_noise_floor(confidences, n_bins=10, n_simulations=600, seed=2)
        # An observation at the simulated median must not read as an effect.
        at_median = ece_noise_floor(
            confidences, n_bins=10, observed_ece=floor.median, n_simulations=600, seed=2
        )
        self.assertFalse(at_median.observed_exceeds_floor)
        self.assertGreater(at_median.p_value, 0.2)

    def test_a_large_observed_ece_clears_the_floor(self):
        confidences = [0.95] * 200
        floor = ece_noise_floor(
            confidences, n_bins=10, observed_ece=0.45, n_simulations=400, seed=3
        )
        self.assertTrue(floor.observed_exceeds_floor)
        self.assertEqual(floor.p_value, 0.0)

    def test_recovers_a_planted_overconfidence_above_the_floor(self):
        # A decider claiming 0.9 while being right 0.6 of the time has a true
        # gap of 0.3, which must sit far outside the noise floor at this n.
        confidences = [0.9] * 300
        correct = [i % 10 < 6 for i in range(300)]
        observed = expected_calibration_error(equal_width_bins(confidences, correct, 10))
        self.assertAlmostEqual(observed, 0.3, places=9)
        floor = ece_noise_floor(
            confidences, n_bins=10, observed_ece=observed, n_simulations=400, seed=4
        )
        self.assertTrue(floor.observed_exceeds_floor)
        self.assertLess(floor.upper_quantile, 0.15)

    def test_is_deterministic_for_a_given_seed(self):
        confidences = [0.6, 0.7, 0.8, 0.9] * 20
        first = ece_noise_floor(confidences, n_bins=5, n_simulations=200, seed=9)
        second = ece_noise_floor(confidences, n_bins=5, n_simulations=200, seed=9)
        self.assertEqual(first.mean, second.mean)
        self.assertEqual(first.upper_quantile, second.upper_quantile)

    def test_records_the_parameters_the_floor_depends_on(self):
        floor = ece_noise_floor([0.7] * 50, n_bins=7, n_simulations=100, seed=0)
        self.assertEqual(floor.n, 50)
        self.assertEqual(floor.n_bins, 7)
        self.assertEqual(floor.n_simulations, 100)
        self.assertEqual(floor.quantile, 0.95)

    def test_observed_verdict_requires_an_observation(self):
        floor = ece_noise_floor([0.7] * 50, n_bins=7, n_simulations=100, seed=0)
        self.assertIsNone(floor.p_value)
        with self.assertRaises(ValueError):
            floor.observed_exceeds_floor

    def test_validates_input(self):
        with self.assertRaises(ValueError):
            ece_noise_floor([], n_bins=10)
        with self.assertRaises(ValueError):
            ece_noise_floor([1.5], n_bins=10)
        with self.assertRaises(ValueError):
            ece_noise_floor([0.5], n_bins=0)
        with self.assertRaises(ValueError):
            ece_noise_floor([0.5], n_bins=10, n_simulations=0)
        with self.assertRaises(ValueError):
            ece_noise_floor([0.5], n_bins=10, quantile=1.0)


class TestOrderSensitivitySignificance(unittest.TestCase):
    def test_an_order_invariant_decider_is_not_distinguishable_from_zero(self):
        decider = SyntheticDecider(
            answer_key={q.state: q.correct for q in make_questions(40)},
            skill=0.6,
            position_weight=0.0,
            noise=0.0,
        )
        audit = audit_dataset(decider, make_questions(40), n_permutations=6, seed=0)
        self.assertEqual(audit.unstable_item_rate, 0.0)
        sig = order_sensitivity_significance(audit, n_resamples=400, seed=0)
        self.assertFalse(sig.any_item_flipped)
        self.assertFalse(sig.unstable_rate_exceeds(0.0))
        self.assertTrue(sig.unstable_item_rate.contains(0.0))
        self.assertFalse(sig.position_bias_distinguishable_from_uniform)

    def test_a_planted_position_bias_is_distinguishable(self):
        questions = make_questions(60)
        decider = SyntheticDecider(
            answer_key={q.state: q.correct for q in questions},
            skill=0.05,
            position_weight=0.8,
            noise=0.0,
        )
        audit = audit_dataset(decider, questions, n_permutations=6, seed=0)
        sig = order_sensitivity_significance(audit, n_resamples=400, seed=0)
        self.assertTrue(sig.position_bias_distinguishable_from_uniform)
        # The bias was planted on the first display position, so that position
        # must be the one whose interval sits above 1/K.
        self.assertGreater(sig.position_deviation[0].lower, 0.0)
        for later in sig.position_deviation[1:]:
            self.assertLess(later.upper, 0.0)

    def test_unstable_items_are_resolvable_once_there_are_enough_of_them(self):
        questions = make_questions(60)
        decider = SyntheticDecider(
            answer_key={q.state: q.correct for q in questions},
            skill=0.0,
            position_weight=0.02,
            noise=0.6,
            seed=1,
        )
        audit = audit_dataset(decider, questions, n_permutations=6, seed=0)
        self.assertGreater(audit.unstable_item_rate, 0.5)
        sig = order_sensitivity_significance(audit, n_resamples=400, seed=0)
        self.assertTrue(sig.any_item_flipped)
        self.assertTrue(sig.unstable_item_rate.excludes(0.0))
        self.assertTrue(sig.mean_flip_rate.excludes(0.0))
        # A rate above half clears any threshold a caller would plausibly set.
        self.assertTrue(sig.unstable_rate_exceeds(0.25))
        self.assertFalse(sig.unstable_rate_exceeds(0.99))

    def test_point_estimates_agree_with_the_audit(self):
        questions = make_questions(40)
        decider = SyntheticDecider(
            answer_key={q.state: q.correct for q in questions},
            skill=0.3,
            position_weight=0.1,
            noise=0.3,
            seed=2,
        )
        audit = audit_dataset(decider, questions, n_permutations=6, seed=0)
        sig = order_sensitivity_significance(audit, n_resamples=300, seed=0)
        self.assertAlmostEqual(
            sig.unstable_item_rate.point, audit.unstable_item_rate, places=12
        )
        self.assertAlmostEqual(sig.mean_flip_rate.point, audit.mean_flip_rate, places=12)
        self.assertAlmostEqual(
            sig.vote_gain.point,
            audit.accuracy_modal_vote - audit.accuracy_first_order,
            places=12,
        )

    def test_position_deviations_match_the_audit_and_sum_to_zero(self):
        questions = make_questions(40)
        decider = SyntheticDecider(
            answer_key={q.state: q.correct for q in questions},
            skill=0.3,
            position_weight=0.2,
            noise=0.0,
        )
        audit = audit_dataset(decider, questions, n_permutations=6, seed=0)
        sig = order_sensitivity_significance(audit, n_resamples=200, seed=0)
        # Every item got the same number of display orders, so the item-weighted
        # mean equals the audit's display-weighted one.
        uniform = 1.0 / len(LABELS)
        for position, iv in enumerate(sig.position_deviation):
            self.assertAlmostEqual(
                iv.point, audit.position_bias[position] - uniform, places=9
            )
            # Built at the Bonferroni-corrected level, not the requested one.
            self.assertAlmostEqual(iv.confidence, 1.0 - 0.05 / len(LABELS), places=12)
        # Mass is a probability vector per display, so the deviations cancel.
        self.assertAlmostEqual(sum(iv.point for iv in sig.position_deviation), 0.0, places=9)

    def test_no_interval_is_offered_for_a_maximum(self):
        # A percentile bootstrap interval for max |deviation| does not contain
        # its own point estimate on an order-invariant decider, because the
        # maximum of K absolute deviations is positively biased under
        # resampling. Tested against zero it reported bias on a model with none,
        # so the field was removed. This test exists to keep it removed.
        self.assertFalse(
            hasattr(OrderSensitivitySignificance, "max_abs_position_deviation")
        )
        self.assertNotIn(
            "max_abs_position_deviation",
            OrderSensitivitySignificance.__dataclass_fields__,
        )

    def test_the_biased_maximum_is_reproduced_here_as_the_reason_why(self):
        # Demonstrating the defect that removed the field: an order-invariant
        # decider, whose true max deviation is at the noise level, gets a
        # bootstrap interval for the maximum that sits entirely above its own
        # point estimate.
        questions = make_questions(40)
        decider = SyntheticDecider(
            answer_key={q.state: q.correct for q in questions},
            skill=0.6,
            position_weight=0.0,
            noise=0.0,
        )
        audit = audit_dataset(decider, questions, n_permutations=6, seed=0)
        results = list(audit.results)
        uniform = 1.0 / len(LABELS)

        def max_abs_dev(sample):
            width = len(LABELS)
            means = [
                sum(_item_position_means(r)[j] for r in sample) / len(sample)
                for j in range(width)
            ]
            return max(abs(m - uniform) for m in means)

        iv = bootstrap_interval(results, max_abs_dev, 0.95, 600, seed=0)
        self.assertGreater(iv.lower, iv.point)

    def test_mixed_option_counts_report_no_position_intervals(self):
        questions = [
            ChoiceQuestion("three way case", ("a", "b", "c"), "a", "q1"),
            ChoiceQuestion("four way case", ("a", "b", "c", "d"), "b", "q2"),
            ChoiceQuestion("another three way case", ("a", "b", "c"), "c", "q3"),
        ]
        decider = SyntheticDecider(
            answer_key={q.state: q.correct for q in questions}, skill=0.5
        )
        audit = audit_dataset(decider, questions, n_permutations=4, seed=0)
        sig = order_sensitivity_significance(audit, n_resamples=200, seed=0)
        self.assertEqual(sig.position_deviation, ())
        # Absent position mass must not read as "no bias found".
        self.assertFalse(sig.position_bias_distinguishable_from_uniform)

    def test_is_deterministic_for_a_given_seed(self):
        questions = make_questions(30)
        decider = SyntheticDecider(
            answer_key={q.state: q.correct for q in questions},
            skill=0.3,
            position_weight=0.1,
            noise=0.3,
            seed=2,
        )
        audit = audit_dataset(decider, questions, n_permutations=6, seed=0)
        first = order_sensitivity_significance(audit, n_resamples=300, seed=5)
        second = order_sensitivity_significance(audit, n_resamples=300, seed=5)
        self.assertEqual(first.family_confidence, second.family_confidence)
        self.assertEqual(
            first.mean_flip_rate.lower, second.mean_flip_rate.lower
        )
        self.assertEqual(
            [iv.upper for iv in first.position_deviation],
            [iv.upper for iv in second.position_deviation],
        )

    def test_intervals_narrow_with_more_items(self):
        def flip_width(n: int) -> float:
            questions = make_questions(n)
            decider = SyntheticDecider(
                answer_key={q.state: q.correct for q in questions},
                skill=0.0,
                position_weight=0.02,
                noise=0.6,
                seed=1,
            )
            audit = audit_dataset(decider, questions, n_permutations=6, seed=0)
            return order_sensitivity_significance(
                audit, n_resamples=400, seed=0
            ).mean_flip_rate.width

        self.assertGreater(flip_width(20), flip_width(160))

    def test_a_weak_planted_bias_is_recovered_in_sign_even_when_underpowered(self):
        # At position_weight=0.02 against noise=0.6, 60 items x 6 display orders
        # cannot resolve the effect: most trials produce no significant
        # position, and one misattributes it. What is stable is the *sign* of
        # the deviation on the position the bias was planted on. This is the
        # honest statement of what the harness recovers at this sample size, and
        # it is asserted so that a regression in the position accounting shows
        # up as a sign flip rather than as a slightly different number.
        for trial in range(4):
            offset = trial * 1000
            questions = [
                ChoiceQuestion(
                    state=f"case {i + offset}: the balance moved by {(i + offset) * 37}",
                    labels=LABELS,
                    correct=LABELS[i % len(LABELS)],
                    item_id=f"q{i + offset:04d}",
                )
                for i in range(60)
            ]
            decider = SyntheticDecider(
                answer_key={q.state: q.correct for q in questions},
                skill=0.0,
                position_weight=0.02,
                noise=0.6,
                seed=1,
            )
            audit = audit_dataset(decider, questions, n_permutations=6, seed=trial)
            sig = order_sensitivity_significance(audit, n_resamples=600, seed=trial)
            self.assertGreater(
                sig.position_deviation[0].point,
                0.0,
                f"trial {trial}: planted first-position bias lost its sign",
            )

    def test_family_wise_correction_is_conservative(self):
        # The corrected per-position interval must be at least as wide as one
        # built at the requested level, or the correction is not doing anything.
        questions = make_questions(40)
        decider = SyntheticDecider(
            answer_key={q.state: q.correct for q in questions},
            skill=0.3,
            position_weight=0.05,
            noise=0.2,
            seed=3,
        )
        audit = audit_dataset(decider, questions, n_permutations=6, seed=0)
        sig = order_sensitivity_significance(audit, confidence=0.95, n_resamples=800, seed=0)
        results = list(audit.results)
        uniform = 1.0 / len(LABELS)
        uncorrected = bootstrap_interval(
            results,
            lambda sample: sum(_item_position_means(r)[0] for r in sample) / len(sample)
            - uniform,
            0.95,
            800,
            seed=1,
        )
        self.assertGreaterEqual(sig.position_deviation[0].width, uncorrected.width)
        self.assertEqual(sig.family_confidence, 0.95)

    def test_rejects_an_impossible_confidence(self):
        questions = make_questions(10)
        decider = SyntheticDecider(
            answer_key={q.state: q.correct for q in questions}, skill=0.4
        )
        audit = audit_dataset(decider, questions, n_permutations=4, seed=0)
        for bad in (0.0, 1.0, 1.5):
            with self.assertRaises(ValueError):
                order_sensitivity_significance(audit, confidence=bad, n_resamples=50)

    def test_unstable_rate_threshold_is_validated(self):
        questions = make_questions(10)
        decider = SyntheticDecider(
            answer_key={q.state: q.correct for q in questions}, skill=0.4
        )
        audit = audit_dataset(decider, questions, n_permutations=4, seed=0)
        sig = order_sensitivity_significance(audit, n_resamples=100, seed=0)
        for bad in (-0.1, 1.0, 2.0):
            with self.assertRaises(ValueError):
                sig.unstable_rate_exceeds(bad)

    def test_n_items_is_the_resampling_unit_not_the_call_count(self):
        questions = make_questions(25)
        decider = SyntheticDecider(
            answer_key={q.state: q.correct for q in questions}, skill=0.4
        )
        audit = audit_dataset(decider, questions, n_permutations=8, seed=0)
        sig = order_sensitivity_significance(audit, n_resamples=200, seed=0)
        self.assertEqual(sig.n_items, 25)


class TestSelectiveCoverageInterval(unittest.TestCase):
    def test_a_perfect_ranker_covers_everything_at_its_own_risk(self):
        confidences = [i / 20.0 for i in range(20)]
        correct = [True] * 20
        iv = selective_coverage_interval(
            confidences, correct, target_risk=0.0, n_resamples=300, seed=0
        )
        self.assertAlmostEqual(iv.point, 1.0, places=12)
        self.assertAlmostEqual(iv.lower, 1.0, places=12)

    def test_an_all_wrong_model_covers_nothing_under_a_zero_budget(self):
        confidences = [i / 20.0 for i in range(20)]
        correct = [False] * 20
        iv = selective_coverage_interval(
            confidences, correct, target_risk=0.0, n_resamples=300, seed=0
        )
        self.assertAlmostEqual(iv.point, 0.0, places=12)
        self.assertAlmostEqual(iv.upper, 0.0, places=12)

    def test_interval_bounds_the_achievable_coverage_not_the_curve_bound(self):
        # Few distinct confidences, as a vote share or a coarse score produces.
        # The curve's bound stops inside a tie group; no cutoff can, so the
        # interval has to be built on the number a cutoff delivers.
        import random

        from system1_audit.selective import risk_coverage_curve

        rng = random.Random(7)
        confidences = [rng.choice([0.5, 0.625, 0.75, 0.875]) for _ in range(60)]
        correct = [rng.random() < (c + 0.1) for c in confidences]
        report = risk_coverage_curve(confidences, correct)
        bound = report.coverage_at_risk(0.10)
        achievable = report.feasible_coverage_at_risk(0.10)
        # Confirm the fixture actually exercises the gap, so the test cannot
        # pass vacuously if the curve and the cutoff happen to agree.
        self.assertGreater(bound, achievable)
        iv = selective_coverage_interval(
            confidences, correct, target_risk=0.10, n_resamples=800, seed=3
        )
        self.assertAlmostEqual(iv.point, achievable, places=12)
        self.assertLess(iv.point, bound)

    def test_point_estimate_matches_the_curve(self):
        confidences = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.55, 0.5]
        correct = [True, True, True, False, True, False, False, True]
        from system1_audit.selective import risk_coverage_curve

        expected = risk_coverage_curve(confidences, correct).coverage_at_risk(0.25)
        iv = selective_coverage_interval(
            confidences, correct, target_risk=0.25, n_resamples=300, seed=0
        )
        self.assertAlmostEqual(iv.point, expected, places=12)

    def test_interval_is_wide_at_a_tight_budget_on_a_small_sample(self):
        # The point estimate maximises over a noisy curve, so on few items it
        # cannot be read without the interval. This is the whole reason the
        # function exists.
        confidences = [0.9 - 0.02 * i for i in range(24)]
        correct = [i % 4 != 0 for i in range(24)]
        iv = selective_coverage_interval(
            confidences, correct, target_risk=0.05, n_resamples=600, seed=0
        )
        self.assertGreater(iv.width, 0.1)

    def test_bounds_stay_within_zero_and_one(self):
        confidences = [0.9 - 0.02 * i for i in range(24)]
        correct = [i % 3 != 0 for i in range(24)]
        iv = selective_coverage_interval(
            confidences, correct, target_risk=0.2, n_resamples=400, seed=0
        )
        self.assertGreaterEqual(iv.lower, 0.0)
        self.assertLessEqual(iv.upper, 1.0)

    def test_a_looser_budget_never_reduces_the_point_estimate(self):
        confidences = [0.9 - 0.02 * i for i in range(24)]
        correct = [i % 4 != 0 for i in range(24)]
        tight = selective_coverage_interval(
            confidences, correct, 0.05, n_resamples=200, seed=0
        )
        loose = selective_coverage_interval(
            confidences, correct, 0.30, n_resamples=200, seed=0
        )
        self.assertLessEqual(tight.point, loose.point)

    def test_is_deterministic_for_a_given_seed(self):
        confidences = [0.9 - 0.02 * i for i in range(24)]
        correct = [i % 4 != 0 for i in range(24)]
        first = selective_coverage_interval(confidences, correct, 0.1, n_resamples=300, seed=6)
        second = selective_coverage_interval(confidences, correct, 0.1, n_resamples=300, seed=6)
        self.assertEqual((first.lower, first.upper), (second.lower, second.upper))

    def test_validates_input(self):
        with self.assertRaises(ValueError):
            selective_coverage_interval([0.5], [True, False], 0.1)
        with self.assertRaises(ValueError):
            selective_coverage_interval([], [], 0.1)


class TestReproducibleAcrossProcesses(unittest.TestCase):
    """No interval here may depend on the interpreter's hash seed.

    ``splits.py`` earned this check on milestone 3 and the same argument
    applies: an audit that reports a different interval on a second run is not
    an audit.
    """

    SCRIPT = """
from system1_audit.deciders.mock import SyntheticDecider
from system1_audit.permutation import audit_dataset
from system1_audit.significance import ece_noise_floor, order_sensitivity_significance
from system1_audit.types import ChoiceQuestion

LABELS = ("alpha", "beta", "gamma", "delta")
questions = [
    ChoiceQuestion(
        state="case {0}: the account shows an unexpected balance of {1}".format(i, i * 37),
        labels=LABELS,
        correct=LABELS[i % len(LABELS)],
        item_id="q{0:03d}".format(i),
    )
    for i in range(30)
]
decider = SyntheticDecider(
    answer_key={q.state: q.correct for q in questions},
    skill=0.3,
    position_weight=0.1,
    noise=0.3,
    seed=2,
)
audit = audit_dataset(decider, questions, n_permutations=6, seed=0)
sig = order_sensitivity_significance(audit, n_resamples=300, seed=5)
floor = ece_noise_floor([0.6, 0.7, 0.8, 0.9] * 20, n_bins=5, n_simulations=200, seed=9)
print("{0:.12f} {1:.12f} {2:.12f} {3:.12f}".format(
    sig.mean_flip_rate.lower,
    sig.mean_flip_rate.upper,
    sig.position_deviation[0].lower,
    floor.upper_quantile,
))
"""

    def test_identical_under_three_hash_seeds(self):
        repo_root = pathlib.Path(__file__).resolve().parent.parent
        outputs = []
        for hash_seed in ("1", "999", "12345"):
            proc = subprocess.run(
                [sys.executable, "-c", self.SCRIPT],
                capture_output=True,
                text=True,
                cwd=str(repo_root),
                # Built from scratch so PYTHONHASHSEED is the only difference
                # between runs; PYTHONPATH is restored because that also drops
                # it. See the same note in test_splits.py.
                env={
                    "PYTHONHASHSEED": hash_seed,
                    "PATH": "/usr/bin:/bin:/usr/local/bin",
                    "PYTHONPATH": str(repo_root / "src"),
                },
                check=True,
            )
            outputs.append(proc.stdout.strip())
        self.assertEqual(len(set(outputs)), 1, outputs)


if __name__ == "__main__":
    unittest.main()
