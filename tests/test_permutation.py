"""Order-sensitivity audit, validated against deciders with known defects."""

import unittest

from system1_audit.deciders import SyntheticDecider
from system1_audit.permutation import audit_dataset, audit_question, sample_permutations
from system1_audit.types import ChoiceDecision, ChoiceQuestion


class FixedOrderDecider:
    """Always prefers whatever is shown first. Maximum order sensitivity."""

    def decide_choice(self, state, labels):
        labels = tuple(labels)
        head = 0.9
        rest = (1.0 - head) / (len(labels) - 1)
        return ChoiceDecision(labels, (head,) + (rest,) * (len(labels) - 1))


class ContentOnlyDecider:
    """Ignores display order entirely. Zero order sensitivity."""

    def __init__(self, answer_key):
        self.answer_key = answer_key

    def decide_choice(self, state, labels):
        labels = tuple(labels)
        target = self.answer_key[state]
        probs = tuple(0.7 if lab == target else 0.3 / (len(labels) - 1) for lab in labels)
        return ChoiceDecision(labels, probs)


QUESTIONS = (
    ChoiceQuestion("ticket about a refund", ("billing", "bug", "sales"), "billing", "q1"),
    ChoiceQuestion("app crashes on launch", ("billing", "bug", "sales"), "bug", "q2"),
    ChoiceQuestion("asking for a quote", ("billing", "bug", "sales"), "sales", "q3"),
)
ANSWER_KEY = {q.state: q.correct for q in QUESTIONS}


class TestSamplePermutations(unittest.TestCase):
    def test_identity_always_comes_first(self):
        perms = sample_permutations(4, 5, seed=1)
        self.assertEqual(perms[0], (0, 1, 2, 3))

    def test_permutations_are_distinct(self):
        perms = sample_permutations(5, 12, seed=3)
        self.assertEqual(len(perms), len(set(perms)))

    def test_small_option_sets_are_enumerated_exhaustively(self):
        perms = sample_permutations(3, 6, seed=0)
        self.assertEqual(len(perms), 6)
        self.assertEqual(len(set(perms)), 6)

    def test_is_reproducible_across_calls(self):
        self.assertEqual(sample_permutations(6, 10, seed=7), sample_permutations(6, 10, seed=7))

    def test_each_permutation_is_a_valid_rearrangement(self):
        for perm in sample_permutations(5, 8, seed=2):
            self.assertEqual(sorted(perm), [0, 1, 2, 3, 4])


class TestAuditQuestion(unittest.TestCase):
    def test_order_invariant_decider_is_reported_stable(self):
        result = audit_question(ContentOnlyDecider(ANSWER_KEY), QUESTIONS[0], n_permutations=6)
        self.assertTrue(result.is_stable)
        self.assertEqual(result.flip_rate, 0.0)
        self.assertAlmostEqual(result.mean_total_variation, 0.0, places=12)
        self.assertEqual(result.modal_label, "billing")

    def test_first_position_decider_flips_on_almost_every_permutation(self):
        result = audit_question(FixedOrderDecider(), QUESTIONS[0], n_permutations=6)
        self.assertFalse(result.is_stable)
        self.assertGreater(result.flip_rate, 0.5)
        self.assertGreater(result.max_total_variation, 0.0)

    def test_probabilities_are_realigned_before_comparison(self):
        # The content-only decider gives identical aligned vectors under every
        # display order; that only holds if realignment works.
        result = audit_question(ContentOnlyDecider(ANSWER_KEY), QUESTIONS[1], n_permutations=6)
        self.assertEqual(result.canonical_labels, QUESTIONS[1].labels)
        self.assertAlmostEqual(result.max_total_variation, 0.0, places=12)

    def test_rejects_a_decider_that_invents_labels(self):
        class Rogue:
            def decide_choice(self, state, labels):
                return ChoiceDecision(("made", "up"), (0.5, 0.5))

        with self.assertRaises(ValueError):
            audit_question(Rogue(), QUESTIONS[0], n_permutations=2)

    def test_displays_are_captured_once_per_permutation(self):
        result = audit_question(ContentOnlyDecider(ANSWER_KEY), QUESTIONS[0], n_permutations=4)
        self.assertEqual(len(result.displays), result.n_permutations)


class TestAuditDataset(unittest.TestCase):
    def test_clean_decider_shows_no_position_bias(self):
        audit = audit_dataset(ContentOnlyDecider(ANSWER_KEY), QUESTIONS, n_permutations=6)
        self.assertEqual(audit.unstable_item_rate, 0.0)
        self.assertAlmostEqual(audit.max_position_deviation, 0.0, places=12)
        for share in audit.position_bias:
            self.assertAlmostEqual(share, 1 / 3, places=12)

    def test_position_bias_is_detected_and_quantified(self):
        audit = audit_dataset(FixedOrderDecider(), QUESTIONS, n_permutations=6)
        self.assertAlmostEqual(audit.position_bias[0], 0.9, places=12)
        self.assertAlmostEqual(audit.max_position_deviation, 0.9 - 1 / 3, places=12)
        self.assertEqual(audit.unstable_item_rate, 1.0)

    def test_position_bias_shares_sum_to_one(self):
        audit = audit_dataset(FixedOrderDecider(), QUESTIONS, n_permutations=6)
        self.assertAlmostEqual(sum(audit.position_bias), 1.0, places=12)

    def test_majority_vote_recovers_accuracy_a_single_order_loses(self):
        # Skill is real but weak and the decider is jittery, so one display
        # order is a coin flip while voting across orders averages the noise out.
        decider = SyntheticDecider(ANSWER_KEY, skill=0.30, noise=0.55, seed=11)
        audit = audit_dataset(decider, QUESTIONS, n_permutations=6)
        self.assertGreaterEqual(audit.accuracy_modal_vote, audit.accuracy_first_order)

    def test_synthetic_decider_is_reproducible_across_runs(self):
        a = audit_dataset(SyntheticDecider(ANSWER_KEY, skill=0.4, noise=0.3, seed=5), QUESTIONS)
        b = audit_dataset(SyntheticDecider(ANSWER_KEY, skill=0.4, noise=0.3, seed=5), QUESTIONS)
        self.assertEqual(a.mean_flip_rate, b.mean_flip_rate)
        self.assertEqual(a.position_bias, b.position_bias)

    def test_position_weight_dial_increases_measured_bias(self):
        low = audit_dataset(
            SyntheticDecider(ANSWER_KEY, skill=0.3, position_weight=0.0), QUESTIONS
        )
        high = audit_dataset(
            SyntheticDecider(ANSWER_KEY, skill=0.3, position_weight=1.5), QUESTIONS
        )
        self.assertGreater(high.position_bias[0], low.position_bias[0])

    def test_requires_at_least_one_question(self):
        with self.assertRaises(ValueError):
            audit_dataset(ContentOnlyDecider(ANSWER_KEY), [])


if __name__ == "__main__":
    unittest.main()
