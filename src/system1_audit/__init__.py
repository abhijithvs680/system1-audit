"""system1-audit: an independent audit harness for typed decision models.

"System-1" decision models (Jev, Laya and similar) answer a question with a
typed, probabilistic value in one forward pass rather than generating text.
The marketing case for them is latency and cost. The engineering case depends
on three things the headline numbers do not cover:

1. Is the confidence calibrated, before anyone fits a temperature to it?
2. Does the answer survive a permutation of the option order?
3. What error rate remains once low-confidence items are escalated?

and a fourth that decides whether answers to the first three mean anything:

4. Is the measured effect distinguishable from zero on this many items?
5. Was this many items decided before the run, and enough to have seen it?

This package measures those three, with no model runtime and no third-party
dependency, so the metrics can be unit-tested against hand-computed values.
"""

from .calibration import (
    CalibrationReport,
    ReliabilityBin,
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
from .permutation import (
    PermutationAudit,
    PermutationResult,
    audit_dataset,
    audit_question,
    sample_permutations,
)
from .prereg import (
    PlanOutcome,
    PowerPoint,
    PreregisteredPlan,
    SampleSize,
    achieved_power,
    detectable_rate,
    empirical_power,
    minimum_unstable_items,
    plan_for_rate,
    power_curve,
    required_items_for_rate,
)
from .selective import (
    OperatingPoint,
    RiskCoveragePoint,
    SelectiveReport,
    risk_coverage_curve,
    threshold_feasible_coverage,
)
from .significance import (
    Interval,
    NoiseFloor,
    OrderSensitivitySignificance,
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
from .splits import (
    HeldOutCalibration,
    Split,
    check_disjoint,
    deterministic_split,
    held_out_calibration,
    split_questions,
)
from .types import ChoiceDecider, ChoiceDecision, ChoiceQuestion
from .voting import (
    AggregatedDecision,
    StrategyReport,
    VotingComparison,
    aggregate,
    aligned_vectors,
    compare_strategies,
    first_order,
    mean_probability,
    modal_vote,
    strategy_report,
)

__version__ = "0.1.0"

__all__ = [
    "AggregatedDecision",
    "CalibrationReport",
    "ChoiceDecider",
    "ChoiceDecision",
    "ChoiceQuestion",
    "HeldOutCalibration",
    "Interval",
    "NoiseFloor",
    "OperatingPoint",
    "OrderSensitivitySignificance",
    "PermutationAudit",
    "PermutationResult",
    "PlanOutcome",
    "PowerPoint",
    "PreregisteredPlan",
    "ReliabilityBin",
    "RiskCoveragePoint",
    "SampleSize",
    "SelectiveReport",
    "Split",
    "StrategyReport",
    "VotingComparison",
    "achieved_power",
    "aggregate",
    "aligned_vectors",
    "apply_temperature",
    "audit_dataset",
    "audit_question",
    "binomial_tail_at_least",
    "binomial_tail_at_most",
    "binomial_test_greater",
    "bootstrap_interval",
    "brier_score",
    "calibration_report",
    "check_disjoint",
    "clopper_pearson_interval",
    "compare_strategies",
    "detectable_rate",
    "deterministic_split",
    "ece_noise_floor",
    "empirical_power",
    "equal_mass_bins",
    "equal_width_bins",
    "expected_calibration_error",
    "first_order",
    "fit_temperature",
    "held_out_calibration",
    "maximum_calibration_error",
    "mean_probability",
    "minimum_unstable_items",
    "modal_vote",
    "negative_log_likelihood",
    "normal_quantile",
    "order_sensitivity_significance",
    "plan_for_rate",
    "power_curve",
    "required_items_for_rate",
    "risk_coverage_curve",
    "sample_permutations",
    "selective_coverage_interval",
    "split_questions",
    "strategy_report",
    "threshold_feasible_coverage",
    "wilson_interval",
]
