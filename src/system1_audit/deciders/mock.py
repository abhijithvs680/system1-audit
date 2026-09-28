"""A synthetic decider with known, injected defects.

This exists to validate the harness itself. Each defect is a dial, so a test
can assert that the audit recovers a property that was deliberately planted.
Without it, a metric bug and a well-behaved model look identical.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Sequence

from ..types import ChoiceDecision


def _stable_seed(seed: int, state: str, labels: Sequence[str]) -> int:
    """Seed that is identical across processes.

    ``hash()`` on strings is salted per interpreter run, so it cannot be used
    here: an audit that reports different numbers on a second run is not an
    audit.
    """
    payload = "\x1f".join([str(seed), state, *labels]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass
class SyntheticDecider:
    """Answers questions with controllable competence, confidence and bias.

    Args:
        answer_key: Maps question state to its correct label.
        skill: Probability mass steered toward the correct label, 0.0 to 1.0.
            0.0 makes the decider ignore the state entirely.
        position_weight: Extra mass given to whichever option is displayed
            first. 0.0 is order-invariant; higher values reproduce the
            "reads the option list, not the state" failure mode.
        sharpness: Temperature-like exponent applied before normalising.
            Above 1.0 the decider is overconfident.
        noise: Scale of seeded per-call jitter, which is what makes a top-1
            answer able to flip between display orders.
        seed: Seed for the jitter, so an audit run is reproducible.
    """

    answer_key: dict[str, str]
    skill: float = 0.7
    position_weight: float = 0.0
    sharpness: float = 1.0
    noise: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.skill <= 1.0:
            raise ValueError("skill must be in [0, 1]")
        if self.position_weight < 0.0:
            raise ValueError("position_weight must be non-negative")
        if self.sharpness <= 0.0:
            raise ValueError("sharpness must be positive")
        if self.noise < 0.0:
            raise ValueError("noise must be non-negative")

    def decide_choice(self, state: str, labels: Sequence[str]) -> ChoiceDecision:
        labels = tuple(labels)
        if not labels:
            raise ValueError("need at least one label")
        correct = self.answer_key.get(state)
        base = 1.0 / len(labels)
        weights = []
        rng = random.Random(_stable_seed(self.seed, state, labels))
        for position, label in enumerate(labels):
            w = base
            if correct is not None and label == correct:
                w += self.skill
            if position == 0:
                w += self.position_weight
            if self.noise:
                w += rng.uniform(0.0, self.noise)
            weights.append(max(w, 1e-9) ** self.sharpness)
        total = sum(weights)
        return ChoiceDecision(labels, tuple(w / total for w in weights))
