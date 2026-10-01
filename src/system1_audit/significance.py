"""Is the measured effect distinguishable from zero?

Every other module in this package returns a point estimate. A point estimate
alone cannot settle either of the questions this project set out to answer:

- H2 is falsified "if the flip rate is at chance-level zero and per-position
  mass sits within sampling noise of ``1/K``". Sampling noise is not something
  a single number carries.
- The stop condition is "the flip rate is indistinguishable from zero". A flip
  rate of 0.025 measured on 40 items and the same rate measured on 4,000 items
  are different findings, and the reported figure is identical.

So this module attaches an interval to each audited quantity, and states the
verdict as a comparison between that interval and the null value rather than as
a threshold on the point estimate.

It also supplies the reference value that a reported ECE has to beat. Binned
ECE is a *biased* estimator: a perfectly calibrated model scores above zero,
because within-bin accuracy is a finite sample of a Bernoulli mean and the
metric takes the absolute value of the gap. The bias grows with bin count and
shrinks with sample size, so an ECE quoted without ``n`` and ``n_bins`` cannot
be read at all. ``ece_noise_floor`` measures that floor for a given confidence
profile instead of leaving it as a caveat in prose.

The unit of resampling everywhere here is the *item*, not the model call.
Several calls on the same question under different display orders are not
independent observations, and treating them as such would shrink every
interval by a factor that has nothing to do with the evidence.

Standard library only, in keeping with the rest of the package: the point of a
harness that checks someone else's arithmetic is that its own arithmetic can be
checked by hand.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

from .calibration import equal_width_bins, expected_calibration_error
from .permutation import PermutationAudit, PermutationResult
from .selective import risk_coverage_curve

T = TypeVar("T")

_DEFAULT_RESAMPLES = 2000


# --------------------------------------------------------------------------
# Interval
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval around it.

    ``method`` records how the bounds were produced, because a bootstrap
    percentile interval and an exact binomial interval are not interchangeable
    and a reader should not have to guess which one they are looking at.
    """

    point: float
    lower: float
    upper: float
    confidence: float
    method: str

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")
        if self.lower > self.upper:
            raise ValueError("lower bound exceeds upper bound")

    @property
    def width(self) -> float:
        """How much of the answer is still unresolved."""
        return self.upper - self.lower

    def contains(self, value: float) -> bool:
        """True when ``value`` is inside the interval, bounds included."""
        return self.lower <= value <= self.upper

    def excludes(self, value: float) -> bool:
        """True when ``value`` lies outside the interval.

        This is the operational reading of "distinguishable from ``value``" at
        this confidence level. It is not a p-value and does not become one.
        """
        return not self.contains(value)

    def __str__(self) -> str:
        pct = round(self.confidence * 100)
        return f"{self.point:.4f} [{self.lower:.4f}, {self.upper:.4f}] ({pct}% {self.method})"


# --------------------------------------------------------------------------
# Normal quantile
# --------------------------------------------------------------------------


def normal_quantile(p: float) -> float:
    """Inverse standard normal CDF, by bisection on ``math.erf``.

    Bisection rather than a rational approximation so the function has one
    obvious failure mode (too few iterations) instead of an opaque coefficient
    table. 200 iterations over a bracket of 40 is far below float resolution.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    lo, hi = -20.0, 20.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _z_two_sided(confidence: float) -> float:
    """Two-sided z multiplier for a confidence level."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    return normal_quantile(1.0 - (1.0 - confidence) / 2.0)


# --------------------------------------------------------------------------
# Binomial intervals and tail tests
# --------------------------------------------------------------------------


def _check_counts(successes: int, n: int) -> None:
    if n < 1:
        raise ValueError("n must be >= 1")
    if not 0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")


def _log_binomial_pmf(k: int, n: int, p: float) -> float:
    """``log P(X = k)``, through ``lgamma`` so no large integer is formed."""
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )


def binomial_tail_at_least(k: int, n: int, p: float) -> float:
    """Exact ``P(X >= k)`` for ``X ~ Binomial(n, p)``.

    Summed term by term, anchored at the largest term in ``[k, n]`` and walked
    outward with the pmf ratio ``p(i+1)/p(i) = ((n-i)/(i+1)) * (p/(1-p))``.
    Anchoring matters: the terms of a binomial tail span hundreds of orders of
    magnitude, so starting at ``i = k`` and multiplying upward can underflow to
    zero before reaching the terms that carry the mass, and summing downward
    from ``i = n`` has the same problem at the other end.

    The anchor term itself goes through ``lgamma`` rather than ``math.comb``.
    An earlier version of this function used exact integer coefficients, which
    reads better and is checkable by hand, but ``math.comb(n, n // 2)`` exceeds
    the largest representable float at ``n >= 1030`` and the multiplication
    raised ``OverflowError`` -- on a *correct* call, for a sample size a real
    audit could plausibly collect. One test pins ``n = 1500`` and ``n = 5000``
    against that regression, and another asserts this implementation agrees
    with a direct ``math.comb`` reference wherever that reference is
    computable, so the exactness claim is carried by a test rather than by the
    shape of the code.

    Terms are dropped once they stop moving the sum, which bounds the cost on
    the large ``n`` that a power scan reaches.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0

    anchor = min(n, max(k, int((n + 1) * p)))
    log_anchor = _log_binomial_pmf(anchor, n, p)
    if log_anchor < -745.0:
        # Every term in [k, n] is at or below this one, so the whole tail is
        # below (n - k + 1) * 5e-324 and indistinguishable from zero in float.
        return 0.0

    odds = p / (1.0 - p)
    anchor_term = math.exp(log_anchor)
    total = anchor_term

    term = anchor_term
    for i in range(anchor, n):
        term *= ((n - i) / (i + 1)) * odds
        total += term
        if term < 1e-18 * total:
            break

    term = anchor_term
    for i in range(anchor, k, -1):
        term *= (i / (n - i + 1)) / odds
        total += term
        if term < 1e-18 * total:
            break

    return min(total, 1.0)


def binomial_tail_at_most(k: int, n: int, p: float) -> float:
    """Exact ``P(X <= k)`` for ``X ~ Binomial(n, p)``."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return 1.0 - binomial_tail_at_least(k + 1, n, p)


def binomial_test_greater(successes: int, n: int, p_null: float) -> float:
    """One-sided exact p-value for "the rate exceeds ``p_null``"."""
    _check_counts(successes, n)
    return binomial_tail_at_least(successes, n, p_null)


def wilson_interval(
    successes: int, n: int, confidence: float = 0.95
) -> Interval:
    """Wilson score interval for a binomial proportion.

    Preferred over the textbook normal approximation because it does not
    collapse to a zero-width interval when no successes were observed, which is
    exactly the case an order-sensitivity audit runs into on a stable model.
    """
    _check_counts(successes, n)
    z = _z_two_sided(confidence)
    p_hat = successes / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denom
    margin = (
        z
        * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n))
        / denom
    )
    return Interval(
        point=p_hat,
        lower=max(0.0, center - margin),
        upper=min(1.0, center + margin),
        confidence=confidence,
        method="Wilson",
    )


def clopper_pearson_interval(
    successes: int, n: int, confidence: float = 0.95
) -> Interval:
    """Exact (conservative) binomial interval, by bisection on the exact tails.

    The lower limit is the ``p`` at which ``P(X >= successes) = alpha/2``, the
    upper limit the ``p`` at which ``P(X <= successes) = alpha/2``; both tails
    are monotone in ``p``, so bisection is safe. Guaranteed at-or-above nominal
    coverage, which is the property wanted when the honest answer is "this
    sample cannot resolve it".
    """
    _check_counts(successes, n)
    alpha = 1.0 - confidence
    half = alpha / 2.0

    if successes == 0:
        lower = 0.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if binomial_tail_at_least(successes, n, mid) < half:
                lo = mid
            else:
                hi = mid
        lower = (lo + hi) / 2.0

    if successes == n:
        upper = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if binomial_tail_at_most(successes, n, mid) > half:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2.0

    return Interval(
        point=successes / n,
        lower=lower,
        upper=upper,
        confidence=confidence,
        method="Clopper-Pearson",
    )


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------


def bootstrap_interval(
    items: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    confidence: float = 0.95,
    n_resamples: int = _DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Interval:
    """Seeded percentile bootstrap over the item axis.

    ``items`` is the list of independent units. ``statistic`` is recomputed on
    each resample, so it must be the whole aggregation and not a pre-averaged
    number.

    The seed is part of the signature rather than optional: an audit that
    reports a different interval on a second run is not an audit. Resamples
    that the statistic cannot evaluate are skipped rather than silently scored
    as zero, and an interval is refused outright if too few survive.
    """
    if not items:
        raise ValueError("need at least one item")
    if n_resamples < 2:
        raise ValueError("n_resamples must be >= 2")
    point = statistic(items)
    rng = random.Random(seed)
    n = len(items)
    values: list[float] = []
    for _ in range(n_resamples):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        try:
            values.append(statistic(sample))
        except (ValueError, ZeroDivisionError):
            continue
    if len(values) < max(2, n_resamples // 10):
        raise ValueError(
            f"statistic failed on {n_resamples - len(values)} of {n_resamples} "
            "resamples; interval would not be meaningful"
        )
    values.sort()
    return Interval(
        point=point,
        lower=_percentile(values, (1.0 - confidence) / 2.0),
        upper=_percentile(values, 1.0 - (1.0 - confidence) / 2.0),
        confidence=confidence,
        method=f"percentile bootstrap, {len(values)} resamples",
    )


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted sequence."""
    if not sorted_values:
        raise ValueError("no values")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[int(position)]
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


# --------------------------------------------------------------------------
# ECE noise floor
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NoiseFloor:
    """What a perfectly calibrated model would have scored on this sample.

    Attributes:
        n: Number of observations the floor was computed for.
        n_bins: Bin count, which drives the bias as strongly as ``n`` does.
        n_simulations: Monte Carlo draws behind the quantiles.
        mean: Mean ECE under perfect calibration.
        median: Median ECE under perfect calibration.
        upper_quantile: The ``quantile``-th ECE under perfect calibration. An
            observed ECE below this is not evidence of miscalibration at all.
        quantile: Which quantile ``upper_quantile`` reports.
        observed: The measured ECE, when one was supplied.
        p_value: Share of simulations scoring at or above ``observed``. A large
            value means perfect calibration comfortably explains the number.
    """

    n: int
    n_bins: int
    n_simulations: int
    mean: float
    median: float
    upper_quantile: float
    quantile: float
    observed: float | None
    p_value: float | None

    @property
    def observed_exceeds_floor(self) -> bool:
        """True when the measured ECE is above the simulated upper quantile."""
        if self.observed is None:
            raise ValueError("no observed ECE was supplied")
        return self.observed > self.upper_quantile


def ece_noise_floor(
    confidences: Sequence[float],
    n_bins: int = 10,
    observed_ece: float | None = None,
    n_simulations: int = 1000,
    quantile: float = 0.95,
    seed: int = 0,
) -> NoiseFloor:
    """Simulate the ECE of a perfectly calibrated model on this confidence profile.

    The null is "the model is exactly as right as it says it is": correctness is
    drawn as ``Bernoulli(confidence)`` for each item, keeping the observed
    confidences fixed. Conditioning on the real confidence profile matters,
    because the floor depends on how the confidences are spread across bins --
    a model crowding 90% of its mass into one bin has a different floor from
    one spread evenly, at identical ``n`` and ``n_bins``.

    What this does and does not settle: an observed ECE at or below the
    simulated quantile is consistent with perfect calibration and is not
    evidence of miscalibration. An observed ECE above it is evidence that
    something beyond binning noise is present -- it does not say in which
    direction, and ``CalibrationReport.overconfidence`` is the signed quantity
    for that.
    """
    if not confidences:
        raise ValueError("need at least one confidence")
    for c in confidences:
        if not 0.0 <= c <= 1.0:
            raise ValueError(f"confidence out of range: {c!r}")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if n_simulations < 1:
        raise ValueError("n_simulations must be >= 1")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")

    rng = random.Random(seed)
    confs = list(confidences)
    draws: list[float] = []
    for _ in range(n_simulations):
        correct = [rng.random() < c for c in confs]
        draws.append(expected_calibration_error(equal_width_bins(confs, correct, n_bins)))
    draws.sort()

    p_value = None
    if observed_ece is not None:
        at_or_above = sum(1 for value in draws if value >= observed_ece)
        p_value = at_or_above / len(draws)

    return NoiseFloor(
        n=len(confs),
        n_bins=n_bins,
        n_simulations=n_simulations,
        mean=sum(draws) / len(draws),
        median=_percentile(draws, 0.5),
        upper_quantile=_percentile(draws, quantile),
        quantile=quantile,
        observed=observed_ece,
        p_value=p_value,
    )


# --------------------------------------------------------------------------
# Order sensitivity
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OrderSensitivitySignificance:
    """Intervals for the quantities H2 is stated in terms of.

    Two things measured while building this changed how the verdict is phrased,
    and both are recorded here rather than in prose alone.

    **A strict zero null is not informative.** H2's stop condition was written
    as "the flip rate is indistinguishable from zero". Under exact order
    invariance the flip rate is exactly zero, so a single flipped item already
    excludes zero -- at 1 item in 40 the exact interval is
    ``[0.0006, 0.1316]``, which excludes 0.0 just as firmly as 1 in 1,000,000
    would. A non-zero verdict therefore carries no information about ``n`` and
    none about effect size. The usable question is whether the rate clears a
    rate somebody would act on, which is why the verdict below takes a
    threshold instead of defaulting to zero.

    **A bootstrap interval for a maximum is not a confidence interval.** An
    earlier draft exposed an interval for ``max_position_deviation``. For an
    order-invariant decider it read ``0.0000 [0.0016, 0.0250]``: the point
    estimate outside its own bounds, because the maximum of ``K`` absolute
    deviations is positively biased under resampling and the percentile
    interval sits above the point value. Tested against zero it would have
    reported position bias on a model with none. The field was removed rather
    than documented, since a field of type :class:`Interval` that is not a
    valid interval is a trap. ``PermutationAudit.max_position_deviation``
    still carries the point estimate; the per-position intervals below are the
    part that supports a verdict.

    Attributes:
        n_items: Items the audit covered -- the resampling unit.
        unstable_item_rate: Share of items whose top label moved, with an exact
            binomial interval.
        mean_flip_rate: Mean per-item flip rate, bootstrapped over items. This
            is a magnitude, not the verdict: on a sample where every item has
            an identical flip rate the percentile bootstrap collapses to zero
            width, which would read as precision it does not have. The exact
            interval on ``unstable_item_rate`` is what the verdict rests on.
        position_deviation: Per display position, mean probability mass minus
            ``1/K``, bootstrapped over items. Computed at a Bonferroni-
            corrected level, so that asking "is *any* position off uniform"
            holds family-wise at the requested confidence rather than at
            ``K`` independent chances to be wrong. Each interval's own
            ``confidence`` field reports the corrected level it was built at.
            Empty when the item set has mixed option counts, matching
            ``PermutationAudit.position_bias``.
        family_confidence: The confidence the family-wise verdict holds at.
        vote_gain: Modal-vote accuracy minus first-order accuracy, bootstrapped.
            This is the quantity H3 needs, and it is allowed to be negative.
    """

    n_items: int
    unstable_item_rate: Interval
    mean_flip_rate: Interval
    position_deviation: tuple[Interval, ...]
    family_confidence: float
    vote_gain: Interval

    @property
    def any_item_flipped(self) -> bool:
        """Whether a single item's top label moved at all.

        Deliberately not named after significance. Under exact order invariance
        this being true *is* evidence, but it says nothing about how much, and
        it is true just as readily on 20 items as on 20,000.
        """
        return self.unstable_item_rate.point > 0.0

    def unstable_rate_exceeds(self, threshold: float) -> bool:
        """Whether the unstable-item rate is resolvably above ``threshold``.

        This is the testable form of the project's stop condition: pick the
        rate at which option-order instability would change a decision, and ask
        whether this sample can put the rate above it. ``False`` means the
        sample cannot establish that, which is a result about the sample as
        much as about the model.

        ``threshold`` of 0.0 reduces to the degenerate test described in the
        class docstring and is allowed only because refusing it would hide that
        fact rather than document it.
        """
        if not 0.0 <= threshold < 1.0:
            raise ValueError("threshold must be in [0, 1)")
        return self.unstable_item_rate.lower > threshold

    @property
    def position_bias_distinguishable_from_uniform(self) -> bool:
        """Whether any display position's mass is resolvably away from ``1/K``.

        Holds family-wise at ``family_confidence``. False when the option counts
        are mixed, since position mass is not defined for a heterogeneous item
        set -- absence of the measurement is not absence of the effect.

        Note that the per-display deviations sum to exactly zero, because each
        display's probabilities are a normalised vector. So a bias on one
        position necessarily shows as a matching deficit on the others, and a
        single planted effect lights up every position rather than one.
        """
        if not self.position_deviation:
            return False
        return any(iv.excludes(0.0) for iv in self.position_deviation)


def _item_position_means(result: PermutationResult) -> tuple[float, ...]:
    """Mean probability mass per *display* position for one item."""
    width = len(result.canonical_labels)
    totals = [0.0] * width
    for _shown, probs in result.displays:
        for position, value in enumerate(probs):
            totals[position] += value
    return tuple(t / len(result.displays) for t in totals)


def order_sensitivity_significance(
    audit: PermutationAudit,
    confidence: float = 0.95,
    n_resamples: int = _DEFAULT_RESAMPLES,
    seed: int = 0,
) -> OrderSensitivitySignificance:
    """Attach intervals to a :class:`PermutationAudit`.

    Resampling is over items. Within an item every display order is kept
    together, because the permutations of one question are repeated measurements
    of one unit and not separate evidence.

    Position mass is averaged per item first and then across items, so each
    question carries equal weight. That is the same number as
    ``PermutationAudit.position_bias`` whenever every item got the same number
    of display orders, and a more defensible one when they did not.

    ``confidence`` is the level the *whole* report holds at. The per-position
    intervals are therefore built at ``1 - (1 - confidence) / K``, since asking
    whether any of ``K`` positions is off uniform is ``K`` chances to be wrong.
    The correction is Bonferroni, which is conservative here: the deviations sum
    to zero by construction, so the comparisons are strongly dependent. Erring
    conservative is the right direction for a harness whose job is to stop
    someone over-reading a small sample.
    """
    results = list(audit.results)
    if not results:
        raise ValueError("audit has no results")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    n_unstable = sum(1 for r in results if not r.is_stable)
    unstable = clopper_pearson_interval(n_unstable, len(results), confidence)

    flip = bootstrap_interval(
        results,
        lambda sample: sum(r.flip_rate for r in sample) / len(sample),
        confidence,
        n_resamples,
        seed,
    )

    widths = {len(r.canonical_labels) for r in results}
    deviations: tuple[Interval, ...] = ()
    if len(widths) == 1:
        width = widths.pop()
        uniform = 1.0 / width
        per_position_confidence = 1.0 - (1.0 - confidence) / width

        def position_mean(sample: Sequence[PermutationResult], position: int) -> float:
            return sum(_item_position_means(r)[position] for r in sample) / len(sample)

        deviations = tuple(
            bootstrap_interval(
                results,
                lambda sample, position=position: position_mean(sample, position) - uniform,
                per_position_confidence,
                n_resamples,
                seed + 1 + position,
            )
            for position in range(width)
        )

    vote_gain = bootstrap_interval(
        results,
        lambda sample: (
            sum(1 for r in sample if r.modal_label == r.correct_label)
            - sum(1 for r in sample if r.top_labels[0] == r.correct_label)
        )
        / len(sample),
        confidence,
        n_resamples,
        seed + 2000,
    )

    return OrderSensitivitySignificance(
        n_items=len(results),
        unstable_item_rate=unstable,
        mean_flip_rate=flip,
        position_deviation=deviations,
        family_confidence=confidence,
        vote_gain=vote_gain,
    )


# --------------------------------------------------------------------------
# Selective prediction
# --------------------------------------------------------------------------


def selective_coverage_interval(
    confidences: Sequence[float],
    correct: Sequence[bool],
    target_risk: float,
    confidence: float = 0.95,
    n_resamples: int = _DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Interval:
    """Bootstrap interval for coverage at a fixed error budget.

    ``SelectiveReport.coverage_at_risk`` takes the largest coverage whose
    *empirical* risk is under budget, maximising over a noisy curve. That is an
    optimistically biased estimate of the coverage the same threshold would
    deliver on new traffic, and the bias is worst where it matters most -- at a
    tight error budget, where the qualifying prefix is short. The point
    estimate here inherits that bias; the interval is what says how much of the
    number is real.

    Reported as a bounded quantity: the resampled statistic is a coverage, so
    a lower bound of 0.0 is a genuine answer and not a failure.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have equal length")
    if not confidences:
        raise ValueError("need at least one observation")
    pairs = list(zip(confidences, correct))

    def coverage(sample: Sequence[tuple[float, bool]]) -> float:
        confs = [c for c, _ in sample]
        hits = [h for _, h in sample]
        return risk_coverage_curve(confs, hits).coverage_at_risk(target_risk)

    return bootstrap_interval(pairs, coverage, confidence, n_resamples, seed)
