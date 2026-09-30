"""Disjoint fit/report splits, checked against hand-computed values.

Two things are under test. First, that the split is a split: disjoint,
covering, stable across processes, and stable when the dataset grows. Second,
that fitting a temperature on the items you then report it on is measurably
flattering -- which is the reason the module exists.

Standard library only, like the rest of the suite. A harness that needs a test
runner installed before its arithmetic can be checked is one nobody checks.
"""

import hashlib
import pathlib
import subprocess
import sys
import unittest

from system1_audit.calibration import fit_temperature
from system1_audit.deciders.mock import SyntheticDecider
from system1_audit.splits import (
    HeldOutCalibration,
    Split,
    _unit_hash,
    check_disjoint,
    deterministic_split,
    held_out_calibration,
    split_questions,
)
from system1_audit.types import ChoiceQuestion

LABELS = ("a", "b", "c", "d")
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SUBPROCESS_ENV_PATH = "/usr/bin:/bin:/usr/local/bin"


def make_ids(n, prefix="q"):
    return [f"{prefix}{i}" for i in range(n)]


def make_questions(n):
    return [
        ChoiceQuestion(
            state=f"state-{i}",
            labels=LABELS,
            correct=LABELS[i % len(LABELS)],
            item_id=f"q{i}",
        )
        for i in range(n)
    ]


def decide_all(questions, sharpness=1.0, skill=0.7, noise=0.0):
    """Run a synthetic decider over a question set in canonical option order."""
    answer_key = {q.state: q.correct for q in questions}
    decider = SyntheticDecider(
        answer_key=answer_key, skill=skill, sharpness=sharpness, noise=noise, seed=7
    )
    probabilities = []
    correct_index = []
    for question in questions:
        decision = decider.decide_choice(question.state, question.labels)
        probabilities.append(decision.probabilities)
        correct_index.append(question.labels.index(question.correct))
    return probabilities, correct_index


def run_in_subprocess(program, hash_seed):
    """Run a snippet in a fresh interpreter with a chosen PYTHONHASHSEED."""
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(REPO_ROOT),
        env={
            "PYTHONHASHSEED": hash_seed,
            "PATH": SUBPROCESS_ENV_PATH,
            # The child environment is built from scratch so that
            # PYTHONHASHSEED is the only thing varying between runs. That also
            # drops PYTHONPATH, so ``src`` has to be put back explicitly:
            # without it the child can only import the package when it happens
            # to be installed into site-packages, and the command the README
            # documents does not install anything.
            "PYTHONPATH": str(REPO_ROOT / "src"),
        },
    )
    return result.stdout.strip()


class TestUnitHash(unittest.TestCase):
    def test_matches_an_independently_computed_sha256(self):
        # Recompute the mapping here rather than trusting the implementation.
        payload = b"salty\x1fq42"
        expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / (1 << 64)
        self.assertEqual(_unit_hash("q42", "salty"), expected)

    def test_stays_in_the_unit_interval(self):
        values = [_unit_hash(item_id, "") for item_id in make_ids(500)]
        self.assertTrue(all(0.0 <= v < 1.0 for v in values))

    def test_salt_changes_the_value(self):
        self.assertNotEqual(_unit_hash("q1", "a"), _unit_hash("q1", "b"))


class TestDeterministicSplit(unittest.TestCase):
    def test_split_is_disjoint_and_covers_every_item(self):
        split = deterministic_split(make_ids(200))
        self.assertEqual(set(split.fit) & set(split.report), set())
        self.assertEqual(sorted([*split.fit, *split.report]), list(range(200)))

    def test_split_is_deterministic_within_a_process(self):
        ids = make_ids(100)
        self.assertEqual(deterministic_split(ids), deterministic_split(ids))

    def test_split_is_identical_across_processes_with_different_hash_seeds(self):
        # A split that moves between runs invalidates every number reported off it.
        program = (
            "from system1_audit.splits import deterministic_split;"
            "s = deterministic_split([f'q{i}' for i in range(200)]);"
            "print(','.join(map(str, s.fit)))"
        )
        first = run_in_subprocess(program, "1")
        second = run_in_subprocess(program, "999")
        self.assertTrue(first, "subprocess produced no output")
        self.assertEqual(first, second)

    def test_adding_items_does_not_move_existing_assignments(self):
        # The property a shuffle would not give: assignment is per-item, not positional.
        small = deterministic_split(make_ids(50))
        large = deterministic_split(make_ids(500))
        small_fit = {f"q{i}" for i in small.fit}
        large_fit = {f"q{i}" for i in large.fit}
        for index in range(50):
            item = f"q{index}"
            self.assertEqual(item in small_fit, item in large_fit, item)

    def test_fit_fraction_of_zero_sends_everything_to_report(self):
        split = deterministic_split(make_ids(50), fit_fraction=0.0)
        self.assertEqual((split.n_fit, split.n_report), (0, 50))

    def test_fit_fraction_of_one_sends_everything_to_fit(self):
        split = deterministic_split(make_ids(50), fit_fraction=1.0)
        self.assertEqual((split.n_fit, split.n_report), (50, 0))

    def test_realised_fraction_is_close_to_the_target(self):
        split = deterministic_split(make_ids(4000), fit_fraction=0.25)
        self.assertLess(abs(split.fit_fraction - 0.25), 0.03)

    def test_salt_changes_the_assignment(self):
        ids = make_ids(300)
        self.assertNotEqual(
            deterministic_split(ids, salt="a"), deterministic_split(ids, salt="b")
        )

    def test_fraction_outside_the_unit_interval_is_rejected(self):
        for fraction in (-0.01, 1.01, 2.0):
            with self.subTest(fraction=fraction):
                with self.assertRaisesRegex(ValueError, "fit_fraction"):
                    deterministic_split(make_ids(10), fit_fraction=fraction)

    def test_duplicate_identifiers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            deterministic_split(["q1", "q2", "q1"])

    def test_blank_identifiers_are_rejected(self):
        # ChoiceQuestion.item_id defaults to empty; splitting on it would collapse.
        with self.assertRaisesRegex(ValueError, "non-empty identifier"):
            deterministic_split(["q1", "", "q3"])

    def test_empty_item_list_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one item"):
            deterministic_split([])


class TestSplit(unittest.TestCase):
    def test_rejects_overlapping_sides(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            Split(fit=(0, 1, 2), report=(2, 3))

    def test_fit_fraction_of_an_empty_split_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            Split(fit=(), report=()).fit_fraction

    def test_require_both_sides_rejects_an_empty_fit_side(self):
        with self.assertRaisesRegex(ValueError, "fit side is empty"):
            Split(fit=(), report=(0, 1)).require_both_sides()

    def test_require_both_sides_rejects_an_empty_report_side(self):
        with self.assertRaisesRegex(ValueError, "report side is empty"):
            Split(fit=(0, 1), report=()).require_both_sides()

    def test_require_both_sides_accepts_a_populated_split(self):
        Split(fit=(0,), report=(1,)).require_both_sides()


class TestSplitQuestions(unittest.TestCase):
    def test_keys_on_item_id(self):
        questions = make_questions(120)
        self.assertEqual(
            split_questions(questions),
            deterministic_split([q.item_id for q in questions]),
        )

    def test_rejects_default_blank_item_ids(self):
        questions = [ChoiceQuestion(state="s", labels=LABELS, correct="a")]
        with self.assertRaisesRegex(ValueError, "non-empty identifier"):
            split_questions(questions)


class TestCheckDisjoint(unittest.TestCase):
    def test_accepts_disjoint_identifier_sets(self):
        check_disjoint(["a", "b"], ["c", "d"])

    def test_reports_the_shared_identifiers(self):
        with self.assertRaisesRegex(ValueError, "leakage on 2 item"):
            check_disjoint(["a", "b", "c"], ["b", "c", "d"])


class TestHeldOutCalibration(unittest.TestCase):
    def setUp(self):
        self.questions = make_questions(400)
        self.split = split_questions(self.questions)

    def audit(self, sharpness=1.0, skill=0.7, noise=0.0):
        probabilities, correct_index = decide_all(
            self.questions, sharpness=sharpness, skill=skill, noise=noise
        )
        return probabilities, correct_index, held_out_calibration(
            probabilities, correct_index, self.split
        )

    def test_temperature_comes_from_the_fit_side_only(self):
        probabilities, correct_index, result = self.audit(sharpness=3.0)
        expected = fit_temperature(
            [probabilities[i] for i in self.split.fit],
            [correct_index[i] for i in self.split.fit],
        )
        self.assertEqual(result.temperature, expected)

    def test_in_sample_temperature_is_refitted_on_the_report_side(self):
        probabilities, correct_index, result = self.audit(sharpness=3.0)
        expected = fit_temperature(
            [probabilities[i] for i in self.split.report],
            [correct_index[i] for i in self.split.report],
        )
        self.assertEqual(result.in_sample_temperature, expected)

    def test_reported_metrics_cover_the_report_side_only(self):
        _probabilities, _correct_index, result = self.audit(sharpness=3.0)
        self.assertEqual(result.n_fit, self.split.n_fit)
        self.assertEqual(result.n_report, self.split.n_report)
        self.assertEqual(result.raw.n, self.split.n_report)
        self.assertEqual(result.calibrated.n, self.split.n_report)
        self.assertEqual(result.in_sample.n, self.split.n_report)
        self.assertEqual(result.n_fit + result.n_report, len(self.questions))

    def test_in_sample_fitting_cannot_lose_on_nll(self):
        # The guarantee: in_sample_temperature minimises NLL over exactly the
        # report-side items, so no other temperature can score lower on them.
        # That is what makes an in-sample post-fit figure unfalsifiable rather
        # than merely noisy, and it is the reason the split exists. Checked over
        # several planted defects, since a guarantee holding for one fixture is
        # not a guarantee.
        for skill, noise, sharpness in [
            (0.05, 0.6, 1.0),
            (0.05, 0.6, 4.0),
            (0.12, 0.7, 5.0),
            (0.30, 1.5, 8.0),
        ]:
            with self.subTest(skill=skill, noise=noise, sharpness=sharpness):
                _p, _c, result = self.audit(sharpness=sharpness, skill=skill, noise=noise)
                self.assertGreaterEqual(result.nll_optimism, -1e-9)
                self.assertLessEqual(result.in_sample.nll, result.calibrated.nll + 1e-9)

    def test_ece_optimism_is_not_guaranteed_non_negative(self):
        # Pins the documented caveat instead of leaving it as a claim.
        # ece_optimism compares a binned statistic the temperature fit does not
        # optimise, so the in-sample shortcut can land on a worse ECE than the
        # held-out temperature. This fixture is one that does. If a change made
        # ECE optimism guaranteed, this test should fail and the docstring on
        # HeldOutCalibration.ece_optimism should be corrected.
        _p, _c, result = self.audit(sharpness=5.0, skill=0.12, noise=0.7)
        self.assertGreater(result.nll_optimism, 0.0)
        self.assertLess(result.ece_optimism, 0.0)

    def test_fitted_temperature_is_proportional_to_the_planted_sharpening(self):
        # Planted-defect recovery with an exactly derivable expected value.
        # The decider raises its weights to `sharpness`, and apply_temperature
        # divides log p by T, so both enter the softmax only through
        # sharpness / T. Doubling the planted defect must double the recovered
        # temperature.
        temperatures = {}
        for sharpness in (1.0, 2.0, 4.0, 8.0):
            _p, _c, result = self.audit(sharpness=sharpness, skill=0.05, noise=0.6)
            temperatures[sharpness] = result.temperature
        base = temperatures[1.0]
        self.assertGreater(base, 0.0)
        for sharpness in (2.0, 4.0, 8.0):
            with self.subTest(sharpness=sharpness):
                self.assertAlmostEqual(temperatures[sharpness] / base, sharpness, delta=0.02)

    def test_held_out_temperature_removes_planted_overconfidence(self):
        # skill is low and noise high on purpose: a decider that is always right
        # cannot be overconfident, because accuracy caps the gap at zero.
        _p, _c, result = self.audit(sharpness=4.0, skill=0.05, noise=0.6)
        self.assertLess(result.raw.accuracy, 0.5)
        self.assertGreater(result.raw.accuracy, 0.0)
        self.assertGreater(result.raw.overconfidence, 0.2)
        self.assertGreater(result.temperature, 1.0)
        self.assertGreater(result.ece_reduction, 0.0)
        self.assertLess(result.calibrated.ece, result.raw.ece)

    def test_an_unsharpened_decider_needs_almost_no_scaling(self):
        # The control: with no planted sharpening there is little to fix.
        _p, _c, result = self.audit(sharpness=1.0, skill=0.05, noise=0.6)
        self.assertLess(result.raw.ece, 0.05)
        self.assertGreater(result.temperature, 0.5)
        self.assertLess(result.temperature, 1.5)

    def test_rejects_a_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            held_out_calibration([(0.5, 0.5), (0.5, 0.5)], [0], Split(fit=(0,), report=(1,)))

    def test_rejects_no_observations(self):
        with self.assertRaisesRegex(ValueError, "at least one observation"):
            held_out_calibration([], [], Split(fit=(0,), report=(1,)))

    def test_rejects_an_empty_side(self):
        questions = make_questions(40)
        probabilities, correct_index = decide_all(questions)
        split = deterministic_split([q.item_id for q in questions], fit_fraction=0.0)
        with self.assertRaisesRegex(ValueError, "fit side is empty"):
            held_out_calibration(probabilities, correct_index, split)

    def test_rejects_an_out_of_range_split_index(self):
        with self.assertRaisesRegex(ValueError, "out of range"):
            held_out_calibration([(0.5, 0.5), (0.5, 0.5)], [0, 1], Split(fit=(0,), report=(5,)))

    def test_is_reproducible_across_processes(self):
        program = (
            "from tests.test_splits import decide_all, make_questions;"
            "from system1_audit.splits import held_out_calibration, split_questions;"
            "q = make_questions(200);"
            "p, c = decide_all(q, sharpness=3.0);"
            "r = held_out_calibration(p, c, split_questions(q));"
            "print(f'{r.temperature!r} {r.calibrated.ece!r} {r.nll_optimism!r}')"
        )
        first = run_in_subprocess(program, "1")
        second = run_in_subprocess(program, "999")
        self.assertTrue(first, "subprocess produced no output")
        self.assertEqual(first, second)


class TestPackageSurface(unittest.TestCase):
    def test_new_names_are_exported(self):
        import system1_audit

        self.assertIs(system1_audit.held_out_calibration, held_out_calibration)
        self.assertIs(system1_audit.HeldOutCalibration, HeldOutCalibration)
        self.assertIs(system1_audit.deterministic_split, deterministic_split)
        self.assertIs(system1_audit.split_questions, split_questions)
        self.assertIs(system1_audit.check_disjoint, check_disjoint)
        self.assertIs(system1_audit.Split, Split)

    def test_all_is_sorted_and_resolvable(self):
        import system1_audit

        self.assertEqual(list(system1_audit.__all__), sorted(system1_audit.__all__))
        for name in system1_audit.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(system1_audit, name))


if __name__ == "__main__":
    unittest.main()
