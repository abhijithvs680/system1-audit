"""End-to-end audit of a synthetic decider with known, injected defects.

Run:
    PYTHONPATH=src python3 examples/demo.py

The decider here is deliberately overconfident and partly reads the option
order rather than the state. The point of the demo is that the three audits
surface exactly that, and that headline accuracy alone does not.
"""

from __future__ import annotations

from system1_audit import (
    apply_temperature,
    audit_dataset,
    calibration_report,
    fit_temperature,
    risk_coverage_curve,
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

    temperature = fit_temperature(probabilities, truth)
    rescaled = [apply_temperature(p, temperature) for p in probabilities]
    fitted = calibration_report(rescaled, truth, n_bins=5)
    print()
    print("CALIBRATION AFTER TEMPERATURE FITTING")
    print(f"  fitted temperature {temperature:.3f}")
    print(f"  ECE                {fitted.ece:.3f}")
    print(f"  NLL                {fitted.nll:.3f}")
    print("  note: fitted on the same items it is reported on, which flatters")
    print("        the number. A real audit fits and reports on disjoint splits.")

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


if __name__ == "__main__":
    main()
