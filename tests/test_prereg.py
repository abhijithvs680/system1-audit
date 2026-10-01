"""Pre-registration and power, checked against exact binomial arithmetic.

Every analytic claim in ``prereg`` reduces to a binomial tail, so the tests
below recompute the tail independently with ``math.comb`` wherever that is
representable and assert agreement, rather than asserting the implementation's
own output back at it.

Three tests exist because measurement contradicted an assumption:

- ``test_power_is_not_monotone_in_n`` pins a concrete pair where collecting one
  *more* item loses the power the plan registered.
- ``test_large_n_does_not_overflow`` pins the ``OverflowError`` that the exact
  tail raised at ``n >= 1030`` before this milestone.
- ``test_null_false_positive_rate_is_controlled`` measures the per-position
  verdict under a decider with no planted bias, because that verdict rests on
  a bootstrap and has no closed form to check.

Standard library only, like the rest of the suite.
"""

import math
import pathlib
import subprocess
import sys
import unittest

from system1_audit.deciders.mock import SyntheticDecider
from system1_audit.permutation import audit_dataset
from system1_audit.prereg import (
    PlanOutcome,
    PreregisteredPlan,
    achieved_power,
    detectable_rate,
    empirical_power,
    minimum_unstable_items,
    plan_for_rate,
    power_curve,
    required_items_for_rate,
)
from system1_audit.significance import (
    Interval,
    OrderSensitivitySignificance,
    binomial_tail_at_least,
    binomial_tail_at_most,
    clopper_pearson_interval,
    order_sensitivity_significance,
)
from system1_audit.types import ChoiceQuestion

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SUBPROCESS_ENV_PATH = "/usr/local/bin:/usr/bin:/bin"

LABELS = ("alpha", "beta", "gamma", "delta")


def reference_tail_at_least(k, n, p):
    """``P(X >= k)`` summed directly with exact integer coefficients.

    Only usable while ``math.comb(n, n // 2)`` stays inside the float range,
    which is the whole reason the implementation stopped doing it this way.
    """
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return min(sum(math.comb(n, i) * p**i * (1.0 - p) ** (n - i) for i in range(k, n + 1)), 1.0)


def questions(trial, n_items):
    """Independent item sets, one per simulated audit."""
    return [
        ChoiceQuestion(
            state=f"trial {trial} item {i}: distinct state text {i}",
            labels=LABELS,
            correct=LABELS[i % len(LABELS)],
            item_id=f"t{trial}-q{i}",
        )
        for i in range(n_items)
    ]


def significance_from_counts(n_unstable, n_items, confidence=0.95):
    """An ``OrderSensitivitySignificance`` carrying only what a plan reads.

    The unstable-item interval is built with the real Clopper-Pearson routine,
    so the verdict logic under test is the production one. The other fields are
    placeholders a plan never consults, kept valid so the dataclass invariants
    still hold.
    """
    nil = Interval(0.0, 0.0, 0.0, confidence, "fixture")
    return OrderSensitivitySignificance(
        n_items=n_items,
        unstable_item_rate=clopper_pearson_interval(n_unstable, n_items, confidence),
        mean_flip_rate=nil,
        position_deviation=(),
        family_confidence=confidence,
        vote_gain=nil,
    )


class TestExactTail(unittest.TestCase):
    """The rewritten tail, against the integer sum it replaced."""

    def test_agrees_with_integer_reference(self):
        worst = 0.0
        for n in (1, 2, 5, 17, 60, 200, 500, 900):
            for p in (0.01, 0.05, 0.2, 0.5, 0.8, 0.99):
                for k in range(0, n + 1, max(1, n // 11)):
                    worst = max(
                        worst,
                        abs(binomial_tail_at_least(k, n, p) - reference_tail_at_least(k, n, p)),
                    )
        self.assertLess(worst, 1e-11, f"largest disagreement was {worst}")

    def test_large_n_does_not_overflow(self):
        # math.comb(1030, 515) exceeds the largest representable float, so the
        # previous implementation raised OverflowError here on a correct call.
        for n in (1030, 1500, 5000):
            value = binomial_tail_at_least(n // 2, n, 0.5)
            self.assertGreater(value, 0.5)
            self.assertLess(value, 0.55)

    def test_degenerate_bounds(self):
        self.assertEqual(binomial_tail_at_least(0, 10, 0.3), 1.0)
        self.assertEqual(binomial_tail_at_least(11, 10, 0.3), 0.0)
        self.assertEqual(binomial_tail_at_least(1, 10, 0.0), 0.0)
        self.assertEqual(binomial_tail_at_least(1, 10, 1.0), 1.0)

    def test_complements_at_most(self):
        for k in range(0, 13):
            self.assertAlmostEqual(
                binomial_tail_at_least(k + 1, 12, 0.37),
                1.0 - binomial_tail_at_most(k, 12, 0.37),
                places=12,
            )

    def test_underflowing_tail_returns_zero_rather_than_raising(self):
        self.assertEqual(binomial_tail_at_least(4000, 5000, 0.01), 0.0)


class TestMinimumUnstableItems(unittest.TestCase):
    """The duality shortcut, against inverting the interval directly."""

    def test_matches_inverting_clopper_pearson(self):
        for n in (10, 25, 40, 97):
            for threshold in (0.0, 0.02, 0.05, 0.1, 0.25):
                for confidence in (0.9, 0.95, 0.99):
                    expected = next(
                        (
                            k
                            for k in range(1, n + 1)
                            if clopper_pearson_interval(k, n, confidence).lower > threshold
                        ),
                        None,
                    )
                    self.assertEqual(
                        minimum_unstable_items(n, threshold, confidence),
                        expected,
                        f"n={n} threshold={threshold} confidence={confidence}",
                    )

    def test_is_the_exact_boundary(self):
        needed = minimum_unstable_items(60, 0.05, 0.95)
        self.assertEqual(needed, 8)
        self.assertGreater(clopper_pearson_interval(needed, 60, 0.95).lower, 0.05)
        self.assertLessEqual(clopper_pearson_interval(needed - 1, 60, 0.95).lower, 0.05)

    def test_none_when_the_sample_cannot_reach_the_threshold(self):
        # Even 5 of 5 unstable leaves a lower bound under 0.6 at 95%.
        self.assertIsNone(minimum_unstable_items(5, 0.6, 0.95))
        self.assertLess(clopper_pearson_interval(5, 5, 0.95).lower, 0.6)

    def test_rejects_out_of_range_arguments(self):
        with self.assertRaises(ValueError):
            minimum_unstable_items(0, 0.05)
        with self.assertRaises(ValueError):
            minimum_unstable_items(10, 1.0)
        with self.assertRaises(ValueError):
            minimum_unstable_items(10, 0.05, 1.0)


class TestAchievedPower(unittest.TestCase):
    def test_matches_an_independently_summed_tail(self):
        needed = minimum_unstable_items(10, 0.1, 0.95)
        self.assertAlmostEqual(
            achieved_power(10, 0.1, 0.5, 0.95),
            reference_tail_at_least(needed, 10, 0.5),
            places=12,
        )

    def test_increases_with_the_assumed_rate(self):
        values = [achieved_power(80, 0.05, rate) for rate in (0.08, 0.12, 0.2, 0.4, 0.8)]
        self.assertEqual(values, sorted(values))
        self.assertLess(values[0], values[-1])

    def test_zero_when_no_outcome_could_fire(self):
        self.assertEqual(achieved_power(5, 0.6, 0.9), 0.0)

    def test_refuses_an_effect_no_larger_than_the_null(self):
        with self.assertRaises(ValueError):
            achieved_power(100, 0.1, 0.1)
        with self.assertRaises(ValueError):
            achieved_power(100, 0.1, 0.05)


class TestRequiredItems(unittest.TestCase):
    def test_returns_the_first_qualifying_sample_size(self):
        size = required_items_for_rate(0.05, 0.20, 0.95, 0.8)
        self.assertGreaterEqual(size.achieved_power, 0.8)
        self.assertLess(achieved_power(size.n_items - 1, 0.05, 0.20, 0.95), 0.8)
        self.assertEqual(
            size.minimum_unstable, minimum_unstable_items(size.n_items, 0.05, 0.95)
        )

    def test_power_is_not_monotone_in_n(self):
        # The finding this milestone was built around. The rejection count is an
        # integer: at n=33 five unstable items suffice, at n=34 six are needed,
        # so one extra item costs more power than it buys. A plan that
        # registered n=33 and collected 34 would be underpowered by its own
        # criterion without any step being wrong.
        self.assertGreaterEqual(achieved_power(33, 0.05, 0.20), 0.8)
        self.assertLess(achieved_power(34, 0.05, 0.20), 0.8)
        self.assertEqual(minimum_unstable_items(33, 0.05, 0.95), 5)
        self.assertEqual(minimum_unstable_items(34, 0.05, 0.95), 6)

    def test_stable_from_holds_for_every_larger_sample(self):
        size = required_items_for_rate(0.05, 0.20, 0.95, 0.8, max_items=400)
        self.assertIsNotNone(size.stable_from)
        self.assertGreater(size.stable_from, size.n_items)
        for point in power_curve(range(size.stable_from, 401), 0.05, 0.20, 0.95):
            self.assertGreaterEqual(point.power, 0.8, f"dipped at n={point.n_items}")

    def test_raises_when_the_budget_cannot_reach_the_power(self):
        with self.assertRaises(ValueError) as caught:
            required_items_for_rate(0.05, 0.055, 0.95, 0.8, max_items=50)
        self.assertIn("unreachable", str(caught.exception))

    def test_rejects_out_of_range_arguments(self):
        with self.assertRaises(ValueError):
            required_items_for_rate(0.05, 0.2, 0.95, 1.0)
        with self.assertRaises(ValueError):
            required_items_for_rate(0.05, 0.2, 0.95, 0.8, max_items=0)


class TestDetectableRate(unittest.TestCase):
    def test_is_the_boundary_of_the_requested_power(self):
        rate = detectable_rate(100, 0.05, 0.95, 0.8)
        self.assertIsNotNone(rate)
        self.assertGreaterEqual(achieved_power(100, 0.05, rate, 0.95), 0.8)
        self.assertLess(achieved_power(100, 0.05, rate - 0.005, 0.95), 0.8)

    def test_falls_as_the_sample_grows(self):
        rates = [detectable_rate(n, 0.05) for n in (40, 100, 300, 500)]
        self.assertEqual(rates, sorted(rates, reverse=True))
        self.assertGreater(rates[0], rates[-1])

    def test_none_when_the_sample_cannot_deliver(self):
        self.assertIsNone(detectable_rate(5, 0.6))

    def test_rejects_out_of_range_arguments(self):
        with self.assertRaises(ValueError):
            detectable_rate(10, 1.0)
        with self.assertRaises(ValueError):
            detectable_rate(10, 0.05, 0.95, 0.0)
        with self.assertRaises(ValueError):
            detectable_rate(10, 0.05, 0.95, 0.8, tolerance=0.0)


class TestEmpiricalPower(unittest.TestCase):
    def test_recovers_a_known_firing_fraction(self):
        interval = empirical_power(lambda index: index % 4 == 0, 40)
        self.assertAlmostEqual(interval.point, 0.25, places=12)
        self.assertTrue(interval.contains(0.25))

    def test_bounds_a_verdict_that_never_fires(self):
        interval = empirical_power(lambda index: False, 50)
        self.assertEqual(interval.point, 0.0)
        self.assertGreater(interval.upper, 0.0)  # Wilson does not collapse

    def test_is_reproducible(self):
        trial = lambda index: index % 3 == 0  # noqa: E731 - one-line fixture
        self.assertEqual(str(empirical_power(trial, 30)), str(empirical_power(trial, 30)))

    def test_rejects_zero_trials(self):
        with self.assertRaises(ValueError):
            empirical_power(lambda index: True, 0)

    def test_null_false_positive_rate_is_controlled(self):
        """The per-position verdict, on a decider with no planted bias.

        The jitter is on, so the audit is not trivially invariant -- it varies
        with display order, it just does not favour any position. Small trial
        count, so the assertion is deliberately loose: it catches a verdict
        firing most of the time under the null, which is the failure mode that
        would invalidate every position result the harness reports.
        """

        def trial(index):
            items = questions(index, 24)
            decider = SyntheticDecider(
                answer_key={q.state: q.correct for q in items},
                skill=0.6,
                position_weight=0.0,
                noise=0.3,
                seed=500 + index,
            )
            audit = audit_dataset(decider, items, n_permutations=6, seed=11)
            significance = order_sensitivity_significance(audit, 0.95, 200, 3)
            return significance.position_bias_distinguishable_from_uniform

        rate = empirical_power(trial, 12, 0.95)
        self.assertLess(rate.point, 0.5, f"verdict fired too often under the null: {rate}")


class TestPreregisteredPlan(unittest.TestCase):
    def test_rejects_an_effect_at_or_below_the_threshold(self):
        with self.assertRaises(ValueError):
            PreregisteredPlan("p", "h", 0.1, 0.1, 100, 8)
        with self.assertRaises(ValueError):
            PreregisteredPlan("p", "h", 0.1, 0.05, 100, 8)

    def test_rejects_a_single_display_order(self):
        # One display order cannot produce a flip, so it cannot test H2.
        with self.assertRaises(ValueError):
            PreregisteredPlan("p", "h", 0.05, 0.2, 100, 1)

    def test_rejects_other_out_of_range_fields(self):
        for kwargs in (
            {"name": ""},
            {"n_items": 0},
            {"confidence": 1.0},
            {"power": 0.0},
            {"unstable_rate_threshold": 1.0},
        ):
            base = {
                "name": "p",
                "hypothesis": "h",
                "unstable_rate_threshold": 0.05,
                "assumed_unstable_rate": 0.2,
                "n_items": 100,
                "n_permutations": 8,
            }
            base.update(kwargs)
            with self.assertRaises(ValueError, msg=f"accepted {kwargs}"):
                PreregisteredPlan(**base)

    def test_plan_id_changes_with_every_field_that_moves_the_verdict(self):
        base = PreregisteredPlan("p", "h", 0.05, 0.2, 100, 8)
        variants = [
            PreregisteredPlan("q", "h", 0.05, 0.2, 100, 8),
            PreregisteredPlan("p", "other", 0.05, 0.2, 100, 8),
            PreregisteredPlan("p", "h", 0.06, 0.2, 100, 8),
            PreregisteredPlan("p", "h", 0.05, 0.21, 100, 8),
            PreregisteredPlan("p", "h", 0.05, 0.2, 101, 8),
            PreregisteredPlan("p", "h", 0.05, 0.2, 100, 9),
            PreregisteredPlan("p", "h", 0.05, 0.2, 100, 8, confidence=0.99),
            PreregisteredPlan("p", "h", 0.05, 0.2, 100, 8, power=0.9),
        ]
        ids = {base.plan_id} | {variant.plan_id for variant in variants}
        self.assertEqual(len(ids), len(variants) + 1)

    def test_plan_id_is_identical_across_processes(self):
        program = (
            "from system1_audit.prereg import PreregisteredPlan\n"
            "print(PreregisteredPlan('p', 'h', 0.05, 0.2, 100, 8).plan_id)\n"
        )

        def run(hash_seed):
            return subprocess.run(
                [sys.executable, "-c", program],
                capture_output=True,
                text=True,
                check=True,
                cwd=str(REPO_ROOT),
                env={
                    "PYTHONHASHSEED": hash_seed,
                    "PATH": SUBPROCESS_ENV_PATH,
                    "PYTHONPATH": str(REPO_ROOT / "src"),
                },
            ).stdout.strip()

        first, second = run("1"), run("999")
        self.assertTrue(first, "subprocess produced no output")
        self.assertEqual(first, second)
        self.assertEqual(first, PreregisteredPlan("p", "h", 0.05, 0.2, 100, 8).plan_id)

    def test_planned_power_reports_the_sample_it_registered(self):
        plan = PreregisteredPlan("p", "h", 0.05, 0.2, 33, 8)
        self.assertAlmostEqual(plan.planned_power, achieved_power(33, 0.05, 0.2, 0.95), places=12)
        self.assertTrue(plan.is_adequately_powered)
        self.assertFalse(PreregisteredPlan("p", "h", 0.05, 0.2, 34, 8).is_adequately_powered)


class TestPlanEvaluation(unittest.TestCase):
    def test_supported_when_the_rate_clears_the_threshold(self):
        plan = PreregisteredPlan("p", "h", 0.05, 0.2, 60, 8)
        outcome = plan.evaluate(significance_from_counts(20, 60))
        self.assertEqual(outcome.conclusion, "supported")
        self.assertTrue(outcome.rate_verdict)
        self.assertEqual(outcome.item_shortfall, 0)
        self.assertFalse(outcome.falsifies_hypothesis)

    def test_not_supported_only_when_the_sample_had_the_power(self):
        plan = PreregisteredPlan("p", "h", 0.05, 0.2, 60, 8)
        outcome = plan.evaluate(significance_from_counts(2, 60))
        self.assertEqual(outcome.conclusion, "not_supported")
        self.assertFalse(outcome.rate_verdict)
        self.assertGreaterEqual(outcome.power_at_observed_n, plan.power)
        self.assertTrue(outcome.falsifies_hypothesis)

    def test_underpowered_null_is_not_a_falsification(self):
        """The distinction this module exists for.

        Both this audit and the one above failed to clear the threshold. The
        interval is the same shape in both cases. Only the one with the power
        to have found the effect says anything about the model.
        """
        plan = PreregisteredPlan("p", "h", 0.05, 0.2, 60, 8)
        outcome = plan.evaluate(significance_from_counts(0, 12))
        self.assertEqual(outcome.conclusion, "inconclusive_underpowered")
        self.assertFalse(outcome.rate_verdict)
        self.assertLess(outcome.power_at_observed_n, plan.power)
        self.assertFalse(outcome.falsifies_hypothesis)
        self.assertEqual(outcome.item_shortfall, 48)

    def test_quotes_the_plan_it_answers(self):
        plan = PreregisteredPlan("p", "h", 0.05, 0.2, 60, 8)
        outcome = plan.evaluate(significance_from_counts(20, 60))
        self.assertEqual(outcome.plan_id, plan.plan_id)
        self.assertIn(plan.plan_id, str(outcome))
        self.assertIsInstance(outcome, PlanOutcome)

    def test_refuses_a_confidence_the_plan_did_not_register(self):
        plan = PreregisteredPlan("p", "h", 0.05, 0.2, 60, 8, confidence=0.95)
        with self.assertRaises(ValueError) as caught:
            plan.evaluate(significance_from_counts(20, 60, confidence=0.99))
        self.assertIn("confidence", str(caught.exception))

    def test_reads_the_position_verdict_without_deciding_on_it(self):
        # Position bias is reported, never folded into the conclusion: the plan
        # registers a threshold for the rate only, and the per-position test is
        # a bootstrap with no pre-registered power.
        plan = PreregisteredPlan("p", "h", 0.05, 0.2, 60, 8)
        outcome = plan.evaluate(significance_from_counts(20, 60))
        self.assertFalse(outcome.position_verdict)


class TestPlanForRate(unittest.TestCase):
    def test_computes_a_sample_size_that_survives_over_collection(self):
        plan = plan_for_rate("H2-rate", "unstable-item rate exceeds 5%", 0.05, 0.2)
        size = required_items_for_rate(0.05, 0.2, 0.95, 0.8)
        self.assertEqual(plan.n_items, size.stable_from)
        self.assertTrue(plan.is_adequately_powered)
        for point in power_curve(range(plan.n_items, plan.n_items + 40), 0.05, 0.2):
            self.assertGreaterEqual(point.power, 0.8)


if __name__ == "__main__":
    unittest.main()
