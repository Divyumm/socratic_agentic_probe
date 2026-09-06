"""
Rubric expressed as testable propositions.

The previous rubric ("The student must demonstrate critical thinking and logical
consistency") is a *norm*, not a proposition. Entailment is a relation between
statements that can be true or false, so a descriptive transcript can never
entail a "must" statement -- there is no inference to make. Scoring it produced
noise.

Each criterion below is instead a declarative claim about what the evidence
shows. That is a well-posed entailment question: given the documentation and the
dialogue transcript, is this statement supported, unsupported, or contradicted?
Every criterion is:

  - descriptive (states what happened, not what ought to happen)
  - decidable from the transcript + documentation alone, with no field knowledge
  - independent enough to score separately, so a faculty screener sees which
    specific criterion failed rather than one opaque aggregate
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class RubricCriterion:
    id: str
    proposition: str
    weight: float
    description: str


RUBRIC_CRITERIA: List[RubricCriterion] = [
    RubricCriterion(
        id="reasons",
        proposition=(
            "The student gave specific reasons for the design decisions described "
            "in the documentation."
        ),
        weight=1.0,
        description="Justification is present and tied to actual decisions.",
    ),
    RubricCriterion(
        id="evidence",
        proposition=(
            "The student referred to concrete evidence from their own project "
            "work, such as observations, tests, iterations, or materials."
        ),
        weight=1.0,
        description="Claims are anchored in project specifics, not generalities.",
    ),
    RubricCriterion(
        id="consistency",
        proposition=(
            "The student's spoken account of the work agrees with the submitted "
            "documentation."
        ),
        weight=1.2,
        description="Divergence here is the primary integrity signal.",
    ),
    RubricCriterion(
        id="revision",
        proposition=(
            "The student refined, qualified, or revised a position after being "
            "challenged on it."
        ),
        weight=0.8,
        description="Reasoning moves under pressure rather than repeating.",
    ),
    RubricCriterion(
        id="tradeoffs",
        proposition=(
            "The student identified a limitation, constraint, or trade-off in "
            "their approach."
        ),
        weight=0.8,
        description="Awareness of what the approach costs or cannot do.",
    ),
    RubricCriterion(
        id="ownership",
        proposition=(
            "The student described the decisions as their own choices rather than "
            "as general practice in the field."
        ),
        weight=1.0,
        description="First-person authorship rather than textbook recitation.",
    ),
]

TOTAL_WEIGHT = sum(c.weight for c in RUBRIC_CRITERIA)


def rubric_summary_text() -> str:
    """Human-readable rubric, for display in the UI and transcripts."""
    return "\n".join(f"- {c.proposition}" for c in RUBRIC_CRITERIA)


def weighted_mean(scores: dict) -> float:
    """Collapses per-criterion support scores into one weighted score in [0, 1]."""
    if not scores:
        return 0.5
    total = 0.0
    weight_used = 0.0
    for c in RUBRIC_CRITERIA:
        if c.id in scores:
            total += c.weight * scores[c.id]
            weight_used += c.weight
    return float(total / weight_used) if weight_used else 0.5
