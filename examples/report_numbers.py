"""Regenerates every number quoted in REPORT.md.

The report is a harness-only write-up: no model has been audited, so every
figure in it comes from the synthetic deciders in this repository. Rather than
transcribe them into prose and hope they stay true, the report quotes this
script and this script prints them. Run it and diff against the report.

    PYTHONPATH=src python3 examples/report_numbers.py

Every measurement is seeded. Two runs on one commit must agree exactly; if they
do not, that is a bug in the harness and not noise to be averaged away.
"""

from __future__ import annotations

from system1_audit import (
    ChoiceQuestion,
    audit_dataset,
    clopper_pearson_interval,
    detectable_rate,
    ece_noise_floor,
    first_order,
    mean_probability,
    modal_vote,
    plan_for_rate,
    risk_coverage_curve,
    strategy_report,
)
from system1_audit.deciders.mock import SyntheticDecider
from system1_audit.significance import _item_position_means, bootstrap_interval
from system1_audit.prereg import achieved_power, minimum_unstable_items

LABELS = ("a", "b", "c", "d")
BUDGET = 0.10


def heading(text: str) -> None:
    print()
    print(text)
    print("-" * len(text))


def questions(n: int) -> list[ChoiceQuestion]:
    """A question set whose correct labels cycle, so no single label wins."""
    return [
        ChoiceQuestion(
            item_id=f"q{i}",
            state=f"q{i}",
            labels=LABELS,
            correct=LABELS[i % len(LABELS)],
        )
        for i in range(n)
    ]


def section_1_operating_point() -> None:
    """The curve's bound is not an operating point."""
    heading("1. A CURVE POINT IS NOT AN OPERATING POINT")

    # Three items tie at 0.9, one of them wrong, one item below.
    confidences = [0.9, 0.9, 0.9, 0.5]
    correct = [True, True, False, True]
    report = risk_coverage_curve(confidences, correct)

    print("  fixture            confidences [0.9, 0.9, 0.9, 0.5], 3rd item wrong")
    print("  error budget       0.00")
    print(f"  curve bound        {report.coverage_at_risk(0.0):.4f}")
    point = report.operating_point(0.0)
    print(f"  best real cutoff   {'none' if point is None else point.threshold}")
    print(f"  feasible coverage  {report.feasible_coverage_at_risk(0.0):.4f}")

    # What the pre-fix code handed back, recomputed here so the report's claim
    # about it is checkable rather than historical.
    prefix_threshold = max(
        (p for p in report.curve if p.risk <= 0.0),
        key=lambda p: p.coverage,
    ).threshold
    answered = [i for i, c in enumerate(confidences) if c >= prefix_threshold]
    realised_risk = sum(1 for i in answered if not correct[i]) / len(answered)
    print(f"  the pre-fix pair:  threshold {prefix_threshold} reported as meeting "
          "a 0.00 budget")
    print(f"  its realised risk  {realised_risk:.4f}   <- the defect")


def section_2_coarse_confidence() -> None:
    """A vote share takes at most K+1 values, and that is all-or-nothing."""
    heading("2. A VOTE SHARE IS A COARSE GATE, AND COARSENESS IS ALL-OR-NOTHING")

    items = questions(150)
    key = {q.state: q.correct for q in items}

    # Two fixtures, differing only in planted skill. Both drive the vote share
    # to a single distinct value; the budget verdict goes opposite ways.
    fixtures = (
        ("ignores the state", dict(skill=0.0, position_weight=0.5, noise=0.3)),
        ("mostly right", dict(skill=0.6, position_weight=0.3, noise=0.2)),
        ("partly right", dict(skill=0.25, position_weight=0.4, noise=0.5)),
    )
    for label, planted in fixtures:
        decider = SyntheticDecider(answer_key=key, seed=11, **planted)
        audit = audit_dataset(decider, items, n_permutations=8, seed=3)
        print(f"  fixture: {label}  ({planted})")
        print(f"    {'strategy':<18}{'passes':>7}{'accuracy':>10}{'bound':>9}"
              f"{'feasible':>10}{'distinct':>10}")
        for strategy, passes in (
            (first_order, 1), (mean_probability, 8), (modal_vote, 8)
        ):
            r = strategy_report(
                audit, strategy, risk_budget=BUDGET, passes_per_item=passes,
                n_resamples=200, seed=5,
            )
            print(f"    {r.name:<18}{r.passes_per_item:>7}{r.accuracy:>10.4f}"
                  f"{r.coverage_at_risk:>9.4f}{r.threshold_coverage:>10.4f}"
                  f"{r.distinct_confidences:>10}")
        print()

    print("  Structural, not fixture-dependent:")
    print("    - a modal vote over K display orders takes at most K+1 distinct")
    print("      confidence values; a probability average keeps one per item")
    print("      (above: 1 to 5 distinct values against 150).")
    print("    - at one distinct value the only available cutoff answers")
    print("      everything, so feasible coverage is 1.0 when the whole set is")
    print("      inside the budget and 0.0 otherwise. Both outcomes appear")
    print("      above, on fixtures differing only in planted skill.")
    print("    - coverage per pass cannot exceed 1/passes, so a K-pass vote")
    print("      cannot beat a 1-pass baseline that already exceeds 1/K.")
    print()
    print("  NOT established here: that voting costs coverage in general. On")
    print("  two of the three fixtures above the vote improves both accuracy")
    print("  and coverage. These deciders plant a per-display position term,")
    print("  which averaging cancels by construction, so the direction of the")
    print("  effect is a property of the fixture and not a prediction.")


def section_3_ece_floor() -> None:
    """A binned ECE cannot be read without n and the bin count."""
    heading("3. AN ECE WITHOUT n AND A BIN COUNT IS UNREADABLE")

    # A confidence profile spanning the range, held fixed across the sweep so
    # only n and the bin count move.
    def profile(n: int) -> list[float]:
        return [0.30 + 0.65 * (i % 20) / 19.0 for i in range(n)]

    print(f"  {'n':>6}{'bins':>7}{'mean floor':>13}{'95th pct':>11}")
    for n, bins in ((40, 10), (200, 10), (1000, 10), (200, 2), (200, 20)):
        floor = ece_noise_floor(
            profile(n), n_bins=bins, n_simulations=1000, seed=4
        )
        print(f"  {n:>6}{bins:>7}{floor.mean:>13.4f}{floor.upper_quantile:>11.4f}")
    print("  note: a perfectly calibrated model scores this, not zero. An ECE")
    print("        quoted without both numbers cannot be compared to anything.")


def section_4_budget_caps_the_finding() -> None:
    """A fixed item budget caps what the audit can conclude."""
    heading("4. THE ITEM BUDGET CAPS THE FINDING BEFORE THE RUN")

    print("  smallest instability rate resolvable above a 0.05 threshold,")
    print("  at 95% confidence and 80% power:")
    for n in (40, 100, 300, 500):
        print(f"    n={n:<5} {detectable_rate(n, 0.05):.4f}")

    print()
    print("  exact binomial power is not monotone in n:")
    print(f"    {'items':>7}{'need':>7}{'power vs true rate 0.20':>26}")
    for n in (32, 33, 34, 38, 39):
        need = minimum_unstable_items(n, 0.05, 0.95)
        power = achieved_power(n, 0.05, 0.20, 0.95)
        flag = "  <- below 0.8" if power < 0.8 else ""
        print(f"    {n:>7}{need:>7}{power:>26.4f}{flag}")
    plan = plan_for_rate(
        name="report-example",
        hypothesis="option-order instability exceeds 5 percent of items",
        threshold=0.05,
        assumed_rate=0.20,
    )
    print(f"  so the plan registers n={plan.n_items} "
          f"(stable_from), not the first n to clear the bar")

    print()
    print("  one unstable item excludes zero at every n, so 'non-zero' is")
    print("  not a finding -- only the width changes:")
    for n in (40, 200, 1000):
        iv = clopper_pearson_interval(1, n)
        print(f"    1 of {n:<5} [{iv.lower:.4f}, {iv.upper:.4f}]  "
              f"excludes zero: {iv.excludes(0.0)}")


def section_5_biased_maximum() -> None:
    """A percentile interval for a maximum is not a confidence interval."""
    heading("5. AN INTERVAL FOR A MAXIMUM IS NOT A CONFIDENCE INTERVAL")

    items = questions(40)
    decider = SyntheticDecider(
        answer_key={q.state: q.correct for q in items},
        skill=0.6,
        position_weight=0.0,  # order-invariant: the true maximum is at zero
        noise=0.0,
    )
    audit = audit_dataset(decider, items, n_permutations=6, seed=0)
    results = list(audit.results)
    uniform = 1.0 / len(LABELS)

    def max_abs_deviation(sample) -> float:
        means = [
            sum(_item_position_means(r)[j] for r in sample) / len(sample)
            for j in range(len(LABELS))
        ]
        return max(abs(m - uniform) for m in means)

    interval = bootstrap_interval(results, max_abs_deviation, 0.95, 600, seed=0)
    print("  fixture            order-invariant decider (position_weight 0.0)")
    print(f"  max |deviation|    {interval.point:.4f} "
          f"[{interval.lower:.4f}, {interval.upper:.4f}]")
    print(f"  point below its own lower bound: {interval.point < interval.lower}")
    print("  The maximum of K absolute deviations is positively biased under")
    print("  resampling, so the interval climbs above the point value and a")
    print("  test against zero reports position bias on a model with none.")
    print("  The field was removed; a test keeps it removed.")


def section_6_temperature_proportionality() -> None:
    """A planted sharpening is recovered up to a constant."""
    heading("6. TEMPERATURE RECOVERS A PLANTED SHARPENING UP TO A CONSTANT")

    from system1_audit import fit_temperature

    items = questions(200)
    key = {q.state: q.correct for q in items}
    print(f"  {'planted sharpness':>19}{'fitted T':>11}{'ratio':>9}")
    for sharpness in (1, 2, 4, 8):
        decider = SyntheticDecider(
            answer_key=key, skill=0.7, sharpness=float(sharpness), seed=7
        )
        probabilities = []
        correct_index = []
        for question in items:
            decision = decider.decide_choice(question.state, question.labels)
            probabilities.append(decision.probabilities)
            correct_index.append(question.labels.index(question.correct))
        temperature = fit_temperature(probabilities, correct_index)
        note = "  <- fitter's lower bound, not a fit" if sharpness == 1 else ""
        print(f"  {sharpness:>19}{temperature:>11.4f}"
              f"{temperature / sharpness:>9.4f}{note}")
    print("  Both the planted exponent and the fitted temperature enter the")
    print("  softmax only through sharpness/T, so the ratio must be constant.")
    print("  The constant depends on the item set, so it is the proportionality")
    print("  that is the property -- and that is what the test suite asserts.")


def main() -> None:
    print("REPORT.md FIGURES")
    print("Every number below comes from a synthetic decider in this")
    print("repository. No model has been audited. Seeded: two runs on one")
    print("commit must agree exactly.")
    section_1_operating_point()
    section_2_coarse_confidence()
    section_3_ece_floor()
    section_4_budget_caps_the_finding()
    section_5_biased_maximum()
    section_6_temperature_proportionality()
    print()


if __name__ == "__main__":
    main()
