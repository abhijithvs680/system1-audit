"""Core data types for auditing typed decision models.

A "System-1" decision model returns a typed, probabilistic answer in one
forward pass instead of generating text. This module defines the minimal
surface an audit needs, so the harness stays independent of any vendor SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class ChoiceDecision:
    """One multiple-choice decision.

    Attributes:
        labels: Option labels in the order they were shown to the model.
        probabilities: Probability per label, aligned to ``labels``.
    """

    labels: tuple[str, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.probabilities):
            raise ValueError("labels and probabilities must have equal length")
        if not self.labels:
            raise ValueError("a decision needs at least one label")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be unique")
        if any(p < 0.0 for p in self.probabilities):
            raise ValueError("probabilities must be non-negative")
        total = sum(self.probabilities)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"probabilities must sum to 1.0, got {total!r}")

    @property
    def top_label(self) -> str:
        """Label with the highest probability; ties break on label order."""
        best = max(range(len(self.labels)), key=lambda i: self.probabilities[i])
        return self.labels[best]

    @property
    def confidence(self) -> float:
        """Probability assigned to the top label."""
        return max(self.probabilities)

    def probability_of(self, label: str) -> float:
        """Probability assigned to ``label``.

        Raises:
            KeyError: if the label was not among the options shown.
        """
        try:
            return self.probabilities[self.labels.index(label)]
        except ValueError as exc:
            raise KeyError(label) from exc

    def aligned_to(self, canonical: Sequence[str]) -> tuple[float, ...]:
        """Re-order probabilities to a canonical label order.

        Needed because a permutation audit shows the same options in different
        orders; comparisons are only meaningful in a fixed frame.
        """
        return tuple(self.probability_of(label) for label in canonical)


@dataclass(frozen=True)
class ChoiceQuestion:
    """A single audit item: some state, the options, and the correct label."""

    state: str
    labels: tuple[str, ...]
    correct: str
    item_id: str = field(default="")

    def __post_init__(self) -> None:
        if self.correct not in self.labels:
            raise ValueError(f"correct label {self.correct!r} is not among labels")


@runtime_checkable
class ChoiceDecider(Protocol):
    """Anything that answers a choice question.

    Implementations wrap a model, an API, or a fixture. Keeping this a Protocol
    means the harness never imports a model runtime it does not need.
    """

    def decide_choice(self, state: str, labels: Sequence[str]) -> ChoiceDecision:
        """Return a decision over ``labels``, in the order given."""
        ...
