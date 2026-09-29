"""system1-audit: an independent audit harness for typed decision models.

"System-1" decision models (Jev, Laya and similar) answer a question with a
typed, probabilistic value in one forward pass rather than generating text.
The marketing case for them is latency and cost. The engineering case depends
on three things the headline numbers do not cover:

1. Is the confidence calibrated, before anyone fits a temperature to it?
2. Does the answer survive a permutation of the option order?
3. What error rate remains once low-confidence items are escalated?

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
from .selective import RiskCoveragePoint, SelectiveReport, risk_coverage_curve
from .splits import (
    HeldOutCalibration,
    Split,
    check_disjoint,
    deterministic_split,
    held_out_calibration,
    split_questions,
)
from .types import ChoiceDecider, ChoiceDecision, ChoiceQuestion

__version__ = "0.1.0"

__all__ = [
    "CalibrationReport",
    "ChoiceDecider",
    "ChoiceDecision",
    "ChoiceQuestion",
    "HeldOutCalibration",
    "PermutationAudit",
    "PermutationResult",
    "ReliabilityBin",
    "RiskCoveragePoint",
    "SelectiveReport",
    "Split",
    "apply_temperature",
    "audit_dataset",
    "audit_question",
    "brier_score",
    "calibration_report",
    "check_disjoint",
    "deterministic_split",
    "equal_mass_bins",
    "equal_width_bins",
    "expected_calibration_error",
    "fit_temperature",
    "held_out_calibration",
    "maximum_calibration_error",
    "negative_log_likelihood",
    "risk_coverage_curve",
    "sample_permutations",
    "split_questions",
]
