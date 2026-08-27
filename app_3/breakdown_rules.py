from typing import Optional
from app_3.schemas import StudentState
from app_3.config import COHERENCE_THRESHOLD, VARIANCE_THRESHOLD

class BreakdownDetector:
    """Implements the Jan 2026 logical phase transition model of abrupt reasoning collapse."""

    def __init__(self, collapse_limit: float = 0.45, unstable_limit: float = 0.65):
        # State boundaries
        # Under 0.45 is reasoning collapse (the cliff)
        # Under 0.65 is the half-life point (unstable plateau transition)
        self.collapse_limit = collapse_limit
        self.unstable_limit = unstable_limit

    def evaluate_state(self, features: dict, collapse_limit: Optional[float] = None, 
                       unstable_limit: Optional[float] = None, variance_threshold: Optional[float] = None) -> StudentState:
        """Determines the reasoning state based on computed features.

        Implements the non-gradual, abrupt phase transition.
        """
        from app_3.config import VARIANCE_THRESHOLD
        
        c_limit = collapse_limit if collapse_limit is not None else self.collapse_limit
        u_limit = unstable_limit if unstable_limit is not None else self.unstable_limit
        v_thresh = variance_threshold if variance_threshold is not None else VARIANCE_THRESHOLD
        
        composite = features["composite_confidence"]
        variance = features["variance_score"]
        
        # Abrupt reasoning collapse cliff trigger:
        # 1. Composite confidence drops below cliff threshold.
        # 2. Or, temperature variance crosses the variance threshold.
        if composite < c_limit or variance > v_thresh * 2.0:
            return StudentState.COLLAPSED
            
        # Superficial Fluency Override (High coherence, low grounding)
        elif features.get("coherence_score", 0.0) > 0.75 and features.get("grounding_score", 0.0) < 0.40:
            return StudentState.UNSTABLE
            
        # Buzzword/Parroting Override (High circularity)
        elif features.get("circularity_score", 0.0) > 0.60:
            return StudentState.UNSTABLE
            
        # Transition/plateau boundary (half-life point):
        elif composite < u_limit or variance > v_thresh:
            return StudentState.UNSTABLE
            
        else:
            return StudentState.GROUNDED

    def get_explanation(self, state: StudentState, features: dict, collapse_limit: Optional[float] = None,
                        unstable_limit: Optional[float] = None, variance_threshold: Optional[float] = None) -> str:
        """Justifies the state assessment based on the feedback-independent features."""
        composite = features["composite_confidence"]
        variance = features["variance_score"]
        circularity = features["circularity_score"]
        c_limit = collapse_limit if collapse_limit is not None else self.collapse_limit
        u_limit = unstable_limit if unstable_limit is not None else self.unstable_limit
        v_thresh = variance_threshold if variance_threshold is not None else VARIANCE_THRESHOLD

        if state == StudentState.COLLAPSED:
            return (
                f"Reasoning Collapse Detected (Cliff crossed). Composite confidence fell to {composite:.2f} "
                f"(below threshold of {c_limit:.2f}) and temperature variance rose to {variance:.2f}. "
                f"The student entered a defensive/circular spiral (circularity: {circularity:.2f})."
            )
        elif state == StudentState.UNSTABLE:
            coherence = features.get("coherence_score", 0.0)
            grounding = features.get("grounding_score", 0.0)
            
            if coherence > 0.75 and grounding < 0.40:
                return (
                    f"Superficial Fluency Detected. The student is presenting fluently (coherence: {coherence:.2f}) "
                    f"but lacks deep connection to the process or artifacts (grounding: {grounding:.2f}). "
                    f"This quality/process mismatch requires deeper probing."
                )
            if circularity > 0.60:
                return (
                    f"Superficial Parroting Detected. The student is repeating buzzwords or question terms "
                    f"without demonstrating genuine depth (circularity: {circularity:.2f}). Intervention recommended."
                )

            # Distinguish what actually tripped this: a low composite score, or the
            # assessor's own reported uncertainty (variance) crossing its threshold
            # despite an otherwise solid composite score. These are different
            # situations and were previously described with the same generic text
            # regardless of which one applied, which could misleadingly claim the
            # composite was "hovering near the cliff" when it wasn't.
            if composite < u_limit:
                return (
                    f"Unstable Reasoning Detected (Half-life point). Composite confidence {composite:.2f} "
                    f"is hovering near the cliff boundary (below {u_limit:.2f}), and assessor variance is "
                    f"{variance:.2f}. Pre-collapse plateau is eroding; intervention is recommended."
                )
            return (
                f"Unstable Reasoning Detected. Composite confidence is solid ({composite:.2f}), but the "
                f"assessor's own certainty in that score is low - variance is {variance:.2f}, above the "
                f"{v_thresh:.2f} threshold. The scoring itself is ambiguous, not necessarily the answer's "
                f"quality; intervention is recommended to resolve that ambiguity."
            )
        else:
            return (
                f"Grounded Reasoning Confirmed. Reasoning holds steady on the plateau. "
                f"Composite confidence is high ({composite:.2f}) with extremely stable evaluations "
                f"(variance: {variance:.2f})."
            )
