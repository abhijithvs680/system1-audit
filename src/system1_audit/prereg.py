"""How many items does the audit need, and was that decided before the run?

Milestone 6 attached an interval to every audited quantity. That made the
project's falsification criteria computable, and immediately exposed the next
gap: an interval that fails to exclude the null is reported identically whether
the effect is absent or the sample was too small to see it. Those are opposite
conclusions. H2 is only falsified by the first.

The project's own roadmap says milestone 2 "should not be run until that
threshold is chosen and written down, since choosing it after seeing the
results is the same defect the disjoint-split rule exists to prevent". This
module is that rule moved out of the notes and into the library:

- :func:`required_items_for_rate` answers "how many items" *before* the run,
  from an exact binomial power calculation rather than a rule of thumb.
- :func:`detectable_rate` answers the question a fixed budget actually poses:
  at the ``n`` I can afford, what is the smallest instability rate I could
  resolve at all?
- :class:`PreregisteredPlan` freezes the threshold, the confidence, the power
  and the sample size, and hands back a ``plan_id`` derived from them. A result
  quoting a ``plan_id`` can be checked against the plan it claims to answer.
- :meth:`PreregisteredPlan.evaluate` refuses to call an underpowered null
  result a falsification. It reports ``inconclusive_underpowered`` instead, and
  reports the shortfall in items.

The tests in this module's suite are the only place where power is treated as
analytic. The per-position bias verdict rests on a percentile bootstrap, which
has no closed form, so its power is measured by simulation through
:func:`empirical_power` and reported with an interval of its own.

Standard library only, like the rest of the package.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Sequence

from .significance import (
    Interval,
    OrderSensitivitySignificance,
    binomial_tail_at_least,
    clopper_pearson_interval,
    wilson_interval,
)

_DEFAULT_MAX_ITEMS = 500


# --------------------------------------------------------------------------
# The exact rejection rule
# --------------------------------------------------------------------------


def minimum_unstable_items(
    n_items: int, threshold: float, confidence: float = 0.95
) -> int | None:
    """Fewest unstable items that would put the rate above ``threshold``.

    This is the decision rule behind
    :meth:`OrderSensitivitySignificance.unstable_rate_exceeds`, computed ahead
    of the run instead of read off the result. That verdict fires when the
    two-sided Clopper-Pearson lower bound clears ``threshold``, and that bound
    is monotone in the success count, so a single integer characterises the
    whole rule: see ``k`` or more unstable items and the verdict fires, see
    fewer and it does not.

    Computed through the exact-test duality rather than by inverting the
    interval: the lower bound exceeds ``p`` exactly when ``P(X >= k)`` under
    ``p`` falls below ``alpha / 2``. One test in the suite asserts the two
    routes agree across a grid, because the duality is the kind of shortcut
    that is correct until an off-by-one makes it quietly wrong.

    Returns:
        The count, or ``None`` when no outcome at this ``n_items`` could clear
        the threshold -- which is itself the answer to "is this sample size
        capable of the finding at all?" and is the honest result for a budget
        too small to run.
    """
    if n_items < 1:
        raise ValueError("n_items must be >= 1")
    if not 0.0 <= threshold < 1.0:
        raise ValueError("threshold must be in [0, 1)")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    half_alpha = (1.0 - confidence) / 2.0
    lo, hi = 1, n_items
    if binomial_tail_at_least(n_items, n_items, threshold) >= half_alpha:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if binomial_tail_at_least(mid, n_items, threshold) < half_alpha:
            hi = mid
        else:
            lo = mid + 1
    return lo


def achieved_power(
    n_items: int,
    threshold: float,
    assumed_rate: float,
    confidence: float = 0.95,
) -> float:
    """Probability the verdict fires, if the true rate really is ``assumed_rate``.

    Exact: the rule is "at least :func:`minimum_unstable_items` of ``n_items``
    were unstable", and the count is binomial under the assumed rate, so the
    power is an exact binomial tail and not a normal approximation. At the
    sample sizes an audit of this kind can afford, the approximation is wrong
    in the direction that matters -- it flatters small ``n``.
    """
    if not 0.0 <= assumed_rate <= 1.0:
        raise ValueError("assumed_rate must be in [0, 1]")
    if assumed_rate <= threshold:
        raise ValueError(
            "assumed_rate must exceed threshold; a plan to detect an effect no "
            "larger than the null is not a plan"
        )
    needed = minimum_unstable_items(n_items, threshold, confidence)
    if needed is None:
        return 0.0
    return binomial_tail_at_least(needed, n_items, assumed_rate)


# --------------------------------------------------------------------------
# Sample size
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleSize:
    """The sample size a stated effect needs, and what it buys.

    Attributes:
        n_items: Smallest item count reaching ``requested_power``.
        minimum_unstable: Unstable items needed at that ``n_items`` for the
            verdict to fire.
        threshold: The null the rate is tested against.
        assumed_rate: The true rate the plan is powered for.
        confidence: Confidence level of the interval the verdict reads.
        requested_power: Power asked for.
        achieved_power: Power actually delivered at ``n_items``. At or above
            ``requested_power``, and usually above it rather than equal to it,
            because the item count and the rejection count are both integers.
        stable_from: Smallest item count at or above which power never again
            drops below ``requested_power`` within the search range, or
            ``None`` when the scan did not establish one. This is the number to
            pre-register when the item budget may slip, because
            ``n_items`` alone does not guarantee that *more* items keep the
            power -- see :func:`power_curve`.
    """

    n_items: int
    minimum_unstable: int
    threshold: float
    assumed_rate: float
    confidence: float
    requested_power: float
    achieved_power: float
    stable_from: int | None

    def __str__(self) -> str:
        return (
            f"{self.n_items} items (>= {self.minimum_unstable} unstable) for "
            f"power {self.achieved_power:.3f} against rate {self.assumed_rate:.3f} "
            f"over threshold {self.threshold:.3f}"
        )


def required_items_for_rate(
    threshold: float,
    assumed_rate: float,
    confidence: float = 0.95,
    power: float = 0.8,
    max_items: int = _DEFAULT_MAX_ITEMS,
) -> SampleSize:
    """Smallest item count whose verdict fires with probability ``power``.

    Scanned upward rather than solved, because exact binomial power is not
    monotone in ``n``: adding one item can move the integer rejection count up
    a step and *lower* the power. The scan therefore continues past the first
    qualifying ``n`` to establish ``stable_from``, the point beyond which the
    requested power holds for every larger sample in range. One test in the
    suite exhibits a concrete non-monotone pair, so the warning in this
    docstring is a measured property and not a hedge.

    Raises:
        ValueError: if ``max_items`` is not enough to reach ``power``. The
            error is the result: it says the effect being planned for is too
            small to resolve within the budget, which is worth knowing before
            a model is ever called rather than after.
    """
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if max_items < 1:
        raise ValueError("max_items must be >= 1")

    powers = [
        achieved_power(n, threshold, assumed_rate, confidence)
        for n in range(1, max_items + 1)
    ]
    first = next((i for i, value in enumerate(powers) if value >= power), None)
    if first is None:
        raise ValueError(
            f"power {power} unreachable for rate {assumed_rate} over threshold "
            f"{threshold} within {max_items} items; best was {max(powers):.4f}"
        )

    stable_from: int | None = None
    for i in range(len(powers) - 1, -1, -1):
        if powers[i] < power:
            stable_from = i + 2 if i + 2 <= max_items else None
            break
    else:
        stable_from = 1

    n_items = first + 1
    needed = minimum_unstable_items(n_items, threshold, confidence)
    assert needed is not None  # power > 0 guarantees a rejection count exists
    return SampleSize(
        n_items=n_items,
        minimum_unstable=needed,
        threshold=threshold,
        assumed_rate=assumed_rate,
        confidence=confidence,
        requested_power=power,
        achieved_power=powers[first],
        stable_from=stable_from,
    )


def detectable_rate(
    n_items: int,
    threshold: float,
    confidence: float = 0.95,
    power: float = 0.8,
    tolerance: float = 1e-4,
) -> float | None:
    """Smallest true rate a fixed budget of ``n_items`` could resolve.

    The question a real audit faces is not "how many items would I like" but
    "I have this many; what can I see?". Power is monotone in the assumed rate
    at fixed ``n_items`` -- a larger true rate can only make the same rejection
    count more likely -- so this bisects.

    Returns:
        The rate, or ``None`` when no rate below 1.0 reaches ``power`` at this
        ``n_items``, meaning the budget cannot deliver the finding at all.
    """
    if not 0.0 <= threshold < 1.0:
        raise ValueError("threshold must be in [0, 1)")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if minimum_unstable_items(n_items, threshold, confidence) is None:
        return None

    lo, hi = threshold, 1.0
    if achieved_power(n_items, threshold, 1.0, confidence) < power:
        return None
    while hi - lo > tolerance:
        mid = (lo + hi) / 2.0
        if mid <= threshold:  # guard the open bound of achieved_power
            lo = mid
            continue
        if achieved_power(n_items, threshold, mid, confidence) >= power:
            hi = mid
        else:
            lo = mid
    return hi


@dataclass(frozen=True)
class PowerPoint:
    """Power at one item count, for plotting or for showing a non-monotone step."""

    n_items: int
    minimum_unstable: int | None
    power: float


def power_curve(
    n_values: Sequence[int],
    threshold: float,
    assumed_rate: float,
    confidence: float = 0.95,
) -> tuple[PowerPoint, ...]:
    """Exact power at each of ``n_values``, in the order given."""
    return tuple(
        PowerPoint(
            n_items=n,
            minimum_unstable=minimum_unstable_items(n, threshold, confidence),
            power=achieved_power(n, threshold, assumed_rate, confidence),
        )
        for n in n_values
    )


# --------------------------------------------------------------------------
# Simulated power, for the verdicts with no closed form
# --------------------------------------------------------------------------


def empirical_power(
    trial: Callable[[int], bool],
    n_trials: int,
    confidence: float = 0.95,
) -> Interval:
    """Measure how often a verdict fires, over independent simulated audits.

    ``trial`` is called with the trial index and returns whether the verdict
    fired. The index is the only source of variation, so a caller that derives
    every seed from it gets a power estimate reproducible across processes --
    the same requirement the rest of this package holds itself to.

    Used for the per-position bias verdict, whose rejection rule is a
    percentile bootstrap and therefore has no exact power. The return is a
    Wilson interval rather than a bare fraction, because a power estimated from
    40 simulated audits is itself a measurement on 40 items and reporting it
    without an interval would repeat the mistake milestone 6 existed to fix.

    The same function measures a *false positive* rate: pass a ``trial`` that
    simulates under the null, and the interval should cover ``1 - confidence``
    of the verdict's own level.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    fired = sum(1 for index in range(n_trials) if trial(index))
    return wilson_interval(fired, n_trials, confidence)


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanOutcome:
    """What a pre-registered plan concludes about one audit.

    Attributes:
        plan_id: The plan this outcome answers.
        n_items_planned: Items the plan called for.
        n_items_observed: Items the audit actually covered.
        rate_verdict: Whether the unstable-item rate cleared the
            pre-registered threshold.
        position_verdict: Whether any display position's mass cleared the
            family-wise interval.
        power_at_observed_n: Exact power the rate test had at the sample
            actually collected, against the pre-registered assumed rate.
        conclusion: One of ``"supported"``,
            ``"not_supported"`` or ``"inconclusive_underpowered"``.
    """

    plan_id: str
    n_items_planned: int
    n_items_observed: int
    rate_verdict: bool
    position_verdict: bool
    power_at_observed_n: float
    conclusion: str

    @property
    def item_shortfall(self) -> int:
        """Items the audit fell short by; zero when the plan was met."""
        return max(0, self.n_items_planned - self.n_items_observed)

    @property
    def falsifies_hypothesis(self) -> bool:
        """Whether this outcome may be read as falsifying the hypothesis.

        Only ``"not_supported"`` qualifies. An underpowered null result is
        evidence about the sample, not about the model, and this property is
        the one place the distinction is enforced rather than described.
        """
        return self.conclusion == "not_supported"

    def __str__(self) -> str:
        return (
            f"plan {self.plan_id}: {self.conclusion} "
            f"({self.n_items_observed}/{self.n_items_planned} items, "
            f"power {self.power_at_observed_n:.3f})"
        )


@dataclass(frozen=True)
class PreregisteredPlan:
    """A threshold, a confidence, a power and a sample size, fixed before the run.

    The point is not ceremony. Every number here could be chosen after seeing
    the audit, and each one chosen that way turns a negative result into a
    positive one: lower the threshold, widen the confidence, or stop collecting
    once the interval happens to clear. ``plan_id`` is a digest of the fields,
    so a result that quotes one can be checked against the plan it answers --
    and a plan edited after the fact gets a different id.

    The digest is a reproducibility aid, not a security control: anyone who can
    edit the plan can recompute the digest. It catches a plan that drifted, not
    a plan that was rewritten on purpose.
    """

    name: str
    hypothesis: str
    unstable_rate_threshold: float
    assumed_unstable_rate: float
    n_items: int
    n_permutations: int
    confidence: float = 0.95
    power: float = 0.8

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a plan needs a name")
        if not 0.0 <= self.unstable_rate_threshold < 1.0:
            raise ValueError("unstable_rate_threshold must be in [0, 1)")
        if not self.unstable_rate_threshold < self.assumed_unstable_rate <= 1.0:
            raise ValueError("assumed_unstable_rate must exceed the threshold")
        if self.n_items < 1:
            raise ValueError("n_items must be >= 1")
        if self.n_permutations < 2:
            raise ValueError("n_permutations must be >= 2 to detect a flip")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")
        if not 0.0 < self.power < 1.0:
            raise ValueError("power must be in (0, 1)")

    @property
    def plan_id(self) -> str:
        """Short digest of every field that could change the verdict."""
        payload = "\x1f".join(
            [
                self.name,
                self.hypothesis,
                f"{self.unstable_rate_threshold!r}",
                f"{self.assumed_unstable_rate!r}",
                f"{self.n_items!r}",
                f"{self.n_permutations!r}",
                f"{self.confidence!r}",
                f"{self.power!r}",
            ]
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]

    @property
    def planned_power(self) -> float:
        """Exact power this plan's own sample size delivers."""
        return achieved_power(
            self.n_items,
            self.unstable_rate_threshold,
            self.assumed_unstable_rate,
            self.confidence,
        )

    @property
    def is_adequately_powered(self) -> bool:
        """Whether the planned sample reaches the planned power."""
        return self.planned_power >= self.power

    def minimum_unstable(self) -> int | None:
        """Unstable items needed for this plan's rate verdict to fire."""
        return minimum_unstable_items(
            self.n_items, self.unstable_rate_threshold, self.confidence
        )

    def evaluate(self, significance: OrderSensitivitySignificance) -> PlanOutcome:
        """Read an audit against this plan, and only against this plan.

        Three outcomes, and the third is the one this module exists for:

        - ``"supported"`` -- the rate cleared the pre-registered threshold.
        - ``"not_supported"`` -- it did not, *and* the sample collected had the
          power to find the assumed effect. This is the only outcome that
          falsifies anything.
        - ``"inconclusive_underpowered"`` -- it did not clear, and the sample
          could not have been relied on to show it. Reported separately because
          the interval looks identical to the case above and means the
          opposite.

        ``significance.confidence`` is not re-derived here; the caller is
        expected to have built it at this plan's confidence level, and
        :meth:`PlanOutcome.plan_id` is what ties the two together. A mismatch
        in confidence is raised rather than absorbed, because silently reading
        a 99% interval against a plan written for 95% would make the whole
        digest pointless.
        """
        if significance.family_confidence != self.confidence:
            raise ValueError(
                f"significance was built at confidence "
                f"{significance.family_confidence!r}, plan {self.plan_id} "
                f"registered {self.confidence!r}"
            )
        observed = significance.n_items
        rate_fired = significance.unstable_rate_exceeds(self.unstable_rate_threshold)
        power_here = achieved_power(
            observed,
            self.unstable_rate_threshold,
            self.assumed_unstable_rate,
            self.confidence,
        )
        if rate_fired:
            conclusion = "supported"
        elif power_here >= self.power:
            conclusion = "not_supported"
        else:
            conclusion = "inconclusive_underpowered"
        return PlanOutcome(
            plan_id=self.plan_id,
            n_items_planned=self.n_items,
            n_items_observed=observed,
            rate_verdict=rate_fired,
            position_verdict=significance.position_bias_distinguishable_from_uniform,
            power_at_observed_n=power_here,
            conclusion=conclusion,
        )


def plan_for_rate(
    name: str,
    hypothesis: str,
    threshold: float,
    assumed_rate: float,
    n_permutations: int = 8,
    confidence: float = 0.95,
    power: float = 0.8,
    max_items: int = _DEFAULT_MAX_ITEMS,
) -> PreregisteredPlan:
    """Build a plan whose sample size is computed rather than guessed.

    Uses ``stable_from`` in preference to the first qualifying item count, so a
    plan that over-collects cannot land on a non-monotone dip and lose the
    power it registered.
    """
    size = required_items_for_rate(threshold, assumed_rate, confidence, power, max_items)
    return PreregisteredPlan(
        name=name,
        hypothesis=hypothesis,
        unstable_rate_threshold=threshold,
        assumed_unstable_rate=assumed_rate,
        n_items=size.stable_from or size.n_items,
        n_permutations=n_permutations,
        confidence=confidence,
        power=power,
    )


__all__ = [
    "PlanOutcome",
    "PowerPoint",
    "PreregisteredPlan",
    "SampleSize",
    "achieved_power",
    "detectable_rate",
    "empirical_power",
    "minimum_unstable_items",
    "plan_for_rate",
    "power_curve",
    "required_items_for_rate",
]
