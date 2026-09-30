"""End-to-end audit of a synthetic decider with known, injected defects.

Run:
    PYTHONPATH=src python3 examples/demo.py

The decider here is deliberately overconfident and partly reads the option
order rather than the state. The point of the demo is that the three audits
surface exactly that, and that headline accuracy alone does not.
"""

from __future__ import annotations

from system1_audit import (
    audit_dataset,
    calibration_report,
    ece_noise_floor,
    held_out_calibration,
    order_sensitivity_significance,
    risk_coverage_curve,
    split_questions,
)
from system1_audit.deciders import SyntheticDecider
from system1_audit.types import ChoiceQuestion

LABELS = ("billing", "bug", "sales", "account")

STATES = [
    ("refund not received", "billing"),
    ("crash on launch", "bug"),
    ("pricing for 50 seats", "sales"),
    ("cannot reset password", "account"),
    ("double charged this month", "billing"),
    ("stack trace on upload", "bug"),
    ("renewal discount", "sales"),
    ("change registered email", "account"),
]

QUESTIONS = tuple(
    ChoiceQuestion(state, LABELS, correct, f"q{i}")
    for i, (state, correct) in enumerate(STATES)
)
ANSWER_KEY = {q.state: q.correct for q in QUESTIONS}


def expanded_questions(variants_per_state: int = 30) -> tuple[ChoiceQuestion, ...]:
    """The curated states, varied into a set large enough to split.

    Held-out temperature fitting needs more items than a readable example can
    list by hand: eight items split in half leaves a temperature fitted on four
    points, which is noise. Each variant carries distinct state text, so the
    decider produces a distinct decision for it rather than a copy of another
    item's. Repeating identical states under new identifiers would put the same
    decision on both sides of the split, which is the leakage this module
    exists to prevent.
    """
    return tuple(
        ChoiceQuestion(f"{state} (case {variant})", LABELS, correct, f"v{index}-{variant}")
        for index, (state, correct) in enumerate(STATES)
        for variant in range(variants_per_state)
    )


def main() -> None:
    decider = SyntheticDecider(
        answer_key=ANSWER_KEY,
        skill=0.55,
        position_weight=0.35,
        sharpness=2.5,
        noise=0.25,
        seed=17,
    )

    decisions = [decider.decide_choice(q.state, q.labels) for q in QUESTIONS]
    probabilities = [d.probabilities for d in decisions]
    truth = [q.labels.index(q.correct) for q in QUESTIONS]

    report = calibration_report(probabilities, truth, n_bins=5)
    print("CALIBRATION")
    print(f"  items              {report.n}")
    print(f"  accuracy           {report.accuracy:.3f}")
    print(f"  mean confidence    {report.mean_confidence:.3f}")
    print(f"  overconfidence     {report.overconfidence:+.3f}")
    print(f"  ECE (raw)          {report.ece:.3f}")
    print(f"  adaptive ECE       {report.adaptive_ece:.3f}")
    print(f"  Brier              {report.brier:.3f}")
    print(f"  NLL                {report.nll:.3f}")

    wide = expanded_questions()
    wide_decider = SyntheticDecider(
        answer_key={q.state: q.correct for q in wide},
        skill=0.05,
        position_weight=0.35,
        sharpness=4.0,
        noise=0.6,
        seed=17,
    )
    wide_probabilities = [
        wide_decider.decide_choice(q.state, q.labels).probabilities for q in wide
    ]
    wide_truth = [q.labels.index(q.correct) for q in wide]
    split = split_questions(wide, fit_fraction=0.5)
    held_out = held_out_calibration(wide_probabilities, wide_truth, split, n_bins=10)
    print()
    print("CALIBRATION, TEMPERATURE FITTED ON A DISJOINT SPLIT")
    print("  measured on a larger generated set: eight items cannot be split")
    print(f"  fit / report items {held_out.n_fit} / {held_out.n_report}")
    print(f"  accuracy           {held_out.raw.accuracy:.3f}")
    print(f"  overconfidence     {held_out.raw.overconfidence:+.3f}")
    print(f"  held-out T         {held_out.temperature:.3f}")
    print(f"  ECE, raw           {held_out.raw.ece:.3f}")
    print(f"  ECE, held-out T    {held_out.calibrated.ece:.3f}")
    print(f"  NLL, held-out T    {held_out.calibrated.nll:.3f}")
    print()
    print("  what fitting on the reported items instead would have bought:")
    print(f"  in-sample T        {held_out.in_sample_temperature:.3f}")
    print(f"  ECE, in-sample T   {held_out.in_sample.ece:.3f}")
    print(f"  NLL optimism       {held_out.nll_optimism:+.4f} (never negative)")
    print(f"  ECE optimism       {held_out.ece_optimism:+.4f} (sign not guaranteed)")
    print("  note: a post-fit calibration figure that does not say which items the")
    print("        temperature was fitted on may be an in-sample number.")

    audit = audit_dataset(decider, QUESTIONS, n_permutations=8, seed=17)
    print()
    print("OPTION-ORDER SENSITIVITY")
    print(f"  permutations/item  {audit.n_permutations}")
    print(f"  unstable items     {audit.unstable_item_rate:.3f}")
    print(f"  mean flip rate     {audit.mean_flip_rate:.3f}")
    print(f"  mean TV distance   {audit.mean_total_variation:.3f}")
    bias = ", ".join(f"{b:.3f}" for b in audit.position_bias)
    print(f"  position bias      [{bias}] (uniform = {1 / len(LABELS):.3f})")
    print(f"  max deviation      {audit.max_position_deviation:.3f}")
    print(f"  accuracy, 1 order  {audit.accuracy_first_order:.3f}")
    print(f"  accuracy, vote     {audit.accuracy_modal_vote:.3f}")

    confidences = [d.confidence for d in decisions]
    correct = [d.top_label == q.correct for d, q in zip(decisions, QUESTIONS)]
    selective = risk_coverage_curve(confidences, correct)
    print()
    print("SELECTIVE PREDICTION")
    print(f"  risk at full cover {selective.full_coverage_risk:.3f}")
    print(f"  AURC               {selective.aurc:.3f}")
    for target in (0.0, 0.1, 0.25):
        coverage = selective.coverage_at_risk(target)
        threshold = selective.threshold_at_risk(target)
        shown = "none" if threshold is None else f"{threshold:.3f}"
        print(f"  risk <= {target:.2f}       coverage {coverage:.3f} at threshold {shown}")

    print()
    print("IS ANY OF IT DISTINGUISHABLE FROM ZERO?")
    significance = order_sensitivity_significance(audit, n_resamples=2000, seed=17)
    print(f"  items audited      {significance.n_items}")
    print(f"  unstable items     {significance.unstable_item_rate}")
    print(f"  mean flip rate     {significance.mean_flip_rate}")
    print(f"  vote gain          {significance.vote_gain}")
    for position, interval in enumerate(significance.position_deviation):
        print(f"  position {position} vs 1/K   {interval}")
    print(f"  any item flipped   {significance.any_item_flipped}")
    print(f"  rate above 0.05    {significance.unstable_rate_exceeds(0.05)}")
    print(f"  position bias      {significance.position_bias_distinguishable_from_uniform}")
    print("  note: the per-position intervals are Bonferroni-corrected, so asking")
    print("        whether any position is off uniform holds at the stated level.")
    print("  note: a non-zero flip rate is not a finding on its own. One flipped")
    print("        item excludes a zero rate at any n, which is why the verdict")
    print("        above is stated against a threshold somebody would act on.")

    print()
    print("WHAT A PERFECTLY CALIBRATED MODEL WOULD HAVE SCORED")
    floor = ece_noise_floor(
        confidences,
        n_bins=10,
        observed_ece=report.ece,
        n_simulations=2000,
        seed=17,
    )
    print(f"  observed ECE       {floor.observed:.4f} at n={floor.n}, {floor.n_bins} bins")
    print(f"  noise floor, mean  {floor.mean:.4f}")
    print(f"  noise floor, 95th  {floor.upper_quantile:.4f}")
    print(f"  above the floor    {floor.observed_exceeds_floor}")
    print(f"  p-value            {floor.p_value:.3f}")
    print("  note: binned ECE is positively biased, so a perfectly calibrated")
    print("        model scores above zero. An ECE quoted without n and bin count")
    print("        cannot be read at all.")
    for n_items in (40, 200, 1000):
        stretched = [confidences[i % len(confidences)] for i in range(n_items)]
        scaled = ece_noise_floor(stretched, n_bins=10, n_simulations=1000, seed=17)
        print(f"  floor at n={n_items:<5d}   mean {scaled.mean:.4f}, 95th {scaled.upper_quantile:.4f}")


if __name__ == "__main__":
    main()
