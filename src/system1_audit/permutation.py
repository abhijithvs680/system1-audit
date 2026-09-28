"""Option-order sensitivity audit.

A typed decision model is supposed to answer a question about *state*. If the
answer changes when the same options are shown in a different order, the model
is partly reading the option list rather than the state. Laya's own README
notes this failure mode for one checkpoint ("noul can follow option labels
instead of state content"), which is what makes it worth measuring directly.

Everything here is a permutation of the *display* order only. Probabilities are
realigned to a canonical label order before any comparison, so the metrics
answer "did the decision move?", not "did the list move?".
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Sequence

from .types import ChoiceDecider, ChoiceQuestion


@dataclass(frozen=True)
class PermutationResult:
    """Order-sensitivity for a single question."""

    item_id: str
    n_permutations: int
    canonical_labels: tuple[str, ...]
    top_labels: tuple[str, ...]
    modal_label: str
    modal_share: float
    flip_rate: float
    mean_total_variation: float
    max_total_variation: float
    correct_label: str
    displays: tuple[tuple[tuple[str, ...], tuple[float, ...]], ...]

    @property
    def is_stable(self) -> bool:
        """True when every permutation produced the same top label."""
        return self.flip_rate == 0.0


@dataclass(frozen=True)
class PermutationAudit:
    """Order-sensitivity aggregated over a question set."""

    n_items: int
    n_permutations: int
    unstable_item_rate: float
    mean_flip_rate: float
    mean_total_variation: float
    position_bias: tuple[float, ...]
    max_position_deviation: float
    accuracy_first_order: float
    accuracy_modal_vote: float
    results: tuple[PermutationResult, ...]


def _total_variation(a: Sequence[float], b: Sequence[float]) -> float:
    """Total variation distance between two aligned probability vectors."""
    return 0.5 * sum(abs(x - y) for x, y in zip(a, b))


def sample_permutations(
    n_labels: int, n_permutations: int, seed: int = 0
) -> list[tuple[int, ...]]:
    """Return distinct display orders, always starting with the identity.

    For small option counts every permutation is enumerable, so the audit is
    exhaustive and the seed is irrelevant. Above that, a seeded sample keeps
    runs reproducible.
    """
    if n_labels < 1:
        raise ValueError("n_labels must be >= 1")
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    identity = tuple(range(n_labels))
    all_count = 1
    for k in range(2, n_labels + 1):
        all_count *= k
        if all_count > n_permutations * 4:
            break
    else:
        if all_count <= n_permutations:
            perms = [identity] + [
                p for p in itertools.permutations(range(n_labels)) if p != identity
            ]
            return perms[:n_permutations]
    rng = random.Random(seed)
    seen = {identity}
    perms = [identity]
    attempts = 0
    limit = n_permutations * 50
    while len(perms) < n_permutations and attempts < limit:
        attempts += 1
        candidate = list(range(n_labels))
        rng.shuffle(candidate)
        tup = tuple(candidate)
        if tup not in seen:
            seen.add(tup)
            perms.append(tup)
    return perms


def audit_question(
    decider: ChoiceDecider,
    question: ChoiceQuestion,
    n_permutations: int = 8,
    seed: int = 0,
) -> PermutationResult:
    """Ask one question under several display orders and compare the answers."""
    canonical = question.labels
    perms = sample_permutations(len(canonical), n_permutations, seed)
    aligned: list[tuple[float, ...]] = []
    top_labels: list[str] = []
    displays: list[tuple[tuple[str, ...], tuple[float, ...]]] = []
    for perm in perms:
        shown = [canonical[i] for i in perm]
        decision = decider.decide_choice(question.state, shown)
        if set(decision.labels) != set(canonical):
            raise ValueError("decider returned labels that were not offered")
        aligned.append(decision.aligned_to(canonical))
        top_labels.append(decision.top_label)
        displays.append((tuple(shown), tuple(decision.probabilities)))

    counts: dict[str, int] = {}
    for label in top_labels:
        counts[label] = counts.get(label, 0) + 1
    modal_label = max(canonical, key=lambda lab: (counts.get(lab, 0), -canonical.index(lab)))
    modal_share = counts[modal_label] / len(top_labels)

    mean_vector = tuple(
        sum(vec[j] for vec in aligned) / len(aligned) for j in range(len(canonical))
    )
    tvs = [_total_variation(vec, mean_vector) for vec in aligned]

    return PermutationResult(
        item_id=question.item_id,
        n_permutations=len(perms),
        canonical_labels=canonical,
        top_labels=tuple(top_labels),
        modal_label=modal_label,
        modal_share=modal_share,
        flip_rate=sum(1 for lab in top_labels if lab != top_labels[0]) / len(top_labels),
        mean_total_variation=sum(tvs) / len(tvs),
        max_total_variation=max(tvs),
        correct_label=question.correct,
        displays=tuple(displays),
    )


def audit_dataset(
    decider: ChoiceDecider,
    questions: Sequence[ChoiceQuestion],
    n_permutations: int = 8,
    seed: int = 0,
) -> PermutationAudit:
    """Run the order-sensitivity audit over a question set.

    ``position_bias`` is the mean probability mass landing on each *display*
    position. A model that ignores order puts 1/K on every position.
    """
    if not questions:
        raise ValueError("need at least one question")
    results = [audit_question(decider, q, n_permutations, seed) for q in questions]

    width = len(questions[0].labels)
    homogeneous = all(len(q.labels) == width for q in questions)
    if homogeneous:
        position_totals = [0.0] * width
        position_count = 0
        for result in results:
            for _shown, probs in result.displays:
                for position, p_value in enumerate(probs):
                    position_totals[position] += p_value
                position_count += 1
        position_bias = tuple(t / position_count for t in position_totals)
        uniform = 1.0 / width
        max_deviation = max(abs(value - uniform) for value in position_bias)
    else:
        position_bias = ()
        max_deviation = float("nan")

    n = len(results)
    return PermutationAudit(
        n_items=n,
        n_permutations=n_permutations,
        unstable_item_rate=sum(1 for r in results if not r.is_stable) / n,
        mean_flip_rate=sum(r.flip_rate for r in results) / n,
        mean_total_variation=sum(r.mean_total_variation for r in results) / n,
        position_bias=position_bias,
        max_position_deviation=max_deviation,
        accuracy_first_order=sum(1 for r in results if r.top_labels[0] == r.correct_label) / n,
        accuracy_modal_vote=sum(1 for r in results if r.modal_label == r.correct_label) / n,
        results=tuple(results),
    )
