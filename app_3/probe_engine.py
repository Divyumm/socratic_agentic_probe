import uuid
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
from app_3.schemas import (
    Claim, EpistemicMap, ProbeTurn, SessionTranscript,
    StudentState, InterventionType, PapanekDimension, ExperimentProfile, ChallengeRecord
)
from app_3.llm_client import LLMClient
from app_3.feature_engine import FeatureEngine
from app_3.breakdown_rules import BreakdownDetector
from app_3.reconstruction import ReconstructionRouter
from app_3.teleprompter import Teleprompter

class ProbingSessionManager:
    """Manages the recursive Socratic sourcing chain, interventions, and state routing."""

    def __init__(self, epistemic_map: EpistemicMap, student_name: str, experiment_profile: Optional[ExperimentProfile] = None):
        self.epistemic_map = epistemic_map
        self.student_name = student_name
        self.session_id = str(uuid.uuid4())
        self.experiment_profile = experiment_profile
        
        # Core engines
        self.llm_client = LLMClient()
        self.feature_engine = FeatureEngine()
        self.breakdown_detector = BreakdownDetector()
        self.reconstruction_router = ReconstructionRouter()
        self.teleprompter = Teleprompter()

        # Session state
        self.turns: List[ProbeTurn] = []
        self.probed_claim_history: List[str] = [] # Claim IDs probed
        self.probed_dimension_history: List[PapanekDimension] = [] # Dimensions visited
        
        # Active probe tracking
        self.active_claim: Optional[Claim] = None
        self.active_depth: int = 0
        self.consecutive_unstable_turns = 0
        self.session_state = StudentState.GROUNDED
        
        # Layer 2 Rubric Store
        self.rubric_scores = []
        self.challenges: List[ChallengeRecord] = []
        from app_3.config import SCORING_VERSION
        self.weight_version = SCORING_VERSION

        # Evidence accumulation for grounding/variance
        self.accumulated_responses: List[str] = []  # Student responses so far
        self.rubric_text = "The student must demonstrate critical thinking and logical consistency."

        # Select first target (most vulnerable ranked claim)
        self._select_next_vulnerable_claim()

    def get_current_question(self) -> str:
        """Generates the probing question for the active claim and depth."""
        if not self.active_claim:
            return "No more claims to probe. The viva session is complete!"
            
        # Get question from client
        return self.llm_client.generate_question(self.active_claim.model_dump(), self.active_depth)

    def submit_response(self, response: str, intervention: Optional[InterventionType] = None,
                        w_bias: Optional[float] = None,
                        w_coherence: Optional[float] = None,
                        w_grounding: Optional[float] = None,
                        w_variance: Optional[float] = None,
                        w_circularity: Optional[float] = None,
                        collapse_limit: Optional[float] = None,
                        unstable_limit: Optional[float] = None,
                        cluster_id: Optional[int] = None) -> Tuple[ProbeTurn, str]:
        """Submits a student's answer, computes features, checks state transitions, and returns the next prompt."""
        if not self.active_claim:
            raise ValueError("No active claim is being probed.")

        current_claim_id = self.active_claim.id
        current_claim_text = self.active_claim.text
        current_claim_theme = self.active_claim.emergent_theme
        question = self.get_current_question()
        turn_index = len(self.turns) + 1

        # Handle Pause/Redirect Interventions directly
        if intervention == InterventionType.PAUSE:
            turn = self._create_intervention_turn(turn_index, question, response, InterventionType.PAUSE, StudentState.PAUSED)
            self.turns.append(turn)
            return turn, "Viva paused. What would you like to do? Options: [1] Redirect to another claim, [2] Challenge current score, [3] Resume session."
            
        # Extract features using the new NLI matrix
        from app_3.nli_auditor import NLIAuditor
        from app_3.simulation import AdvocateAgent, EvaluatorAgent, QualityAuditorAgent
        import numpy as np
        
        nli_auditor = NLIAuditor()
        ai_student = AdvocateAgent()
        
        # 1. Coherence: Question vs Student Response (single score)
        coherence = nli_auditor.compute_entailment(question, response)

        # 2. Accumulate responses for evidence portfolio
        if response and response.strip():
            self.accumulated_responses.append(response)

        # 3. Build combined evidence: documentation + epistemic map + accumulated responses
        epistemic_dict = self.epistemic_map.model_dump() if hasattr(self.epistemic_map, 'model_dump') else None
        combined_evidence = self.feature_engine.format_combined_evidence(
            documentation=self.active_claim.source_passage,
            epistemic_map=epistemic_dict,
            responses=self.accumulated_responses
        )

        # 4. Grounding & Variance: MC sampling on combined evidence against rubric
        # Variance measures stability of evidence portfolio across multiple evaluations
        mc_grounding_scores = []
        quality_auditor = QualityAuditorAgent()

        for i in range(3):
            # Generate evaluation of combined evidence against rubric
            audit_resp = quality_auditor.generate_evaluation(combined_evidence, self.rubric_text, 0.5 + i * 0.1)
            # Score: does this evaluation confirm the evidence grounds in rubric?
            score = nli_auditor.compute_entailment(audit_resp, self.rubric_text)
            mc_grounding_scores.append(score)

        # Grounding = mean of MC samples
        grounding = float(np.mean(mc_grounding_scores))
        # Variance = std_dev across samples (stability of evidence portfolio)
        variance = float(np.std(mc_grounding_scores))
        
        # 5. Composite score using simplified model (coherence + grounding - variance penalty)
        # Version A: Hand-set weights
        composite_z = 0.5 + (2.0 * coherence) + (2.5 * grounding) + (-2.0 * variance)
        composite_confidence = float(1.0 / (1.0 + np.exp(-composite_z)))

        features = {
            "coherence_score": coherence,
            "grounding_score": grounding,
            "variance_score": variance,
            "composite_confidence": composite_confidence,
            "accumulated_responses_count": len(self.accumulated_responses),
            "word_count": len(response.split()),
            "is_vague": False
        }
        
        # Process and map claim_id for Layer 2 rubric scores
        if "rubric_scores" in features and features["rubric_scores"]:
            for r_score in features["rubric_scores"]:
                r_score.claim_id = current_claim_id
                self.rubric_scores.append(r_score)
                
        if composite_confidence < 0.4:
            state = StudentState.COLLAPSED
        elif composite_confidence < 0.65:
            state = StudentState.UNSTABLE
        else:
            state = StudentState.GROUNDED
        
        # Keep track of dimension history
        self.probed_dimension_history.append(self.active_claim.dimension)

        # Handle collapse and routing
        next_prompt = ""
        routed_dimension = None
        
        if state == StudentState.COLLAPSED:
            # Reasoning collapse! Trigger Victor Papanek reconstruction routing
            routed_dimension, reconstruction_prompt = self.reconstruction_router.route_collapse(
                self.active_claim.dimension,
                self.probed_dimension_history
            )
            next_prompt = (
                f"\n[REASONING COLLAPSE DETECTED]\n"
                f"Let's explore this from a different angle. {reconstruction_prompt}"
            )
            # Reset active claim depth and pivot to new dimension
            self.active_depth = 0
            self._select_next_vulnerable_claim(exclude_dimensions=[self.active_claim.dimension])
        elif state == StudentState.UNSTABLE:
            self.active_depth += 1
            next_prompt = f"\n[PLATEAU EROSION DETECTED - PROBING DEEPER]\n{self.get_current_question()}"
        else:
            # Grounded! Resolve this claim and move to the next vulnerable one
            self.probed_claim_history.append(self.active_claim.id)
            old_claim_text = self.active_claim.text
            self._select_next_vulnerable_claim()
            
            if self.active_claim:
                # Just ask the next question directly; don't reference the previous answer
                next_prompt = self.get_current_question()
            else:
                next_prompt = "\n[SESSION RESOLVED] All claims have been successfully justified! Congratulations."

        turn = ProbeTurn(
            turn_index=turn_index,
            claim_id=current_claim_id,
            question=question,
            student_response=response,
            coherence_score=features["coherence_score"],
            grounding_score=features["grounding_score"],
            variance_score=features["variance_score"],
            circularity_score=features["circularity_score"],
            composite_confidence=features["composite_confidence"],
            state=state,
            reconstruction_dimension=routed_dimension,
            intervention_type=intervention,
            weight_version=self.weight_version,
            emergent_theme=current_claim_theme
        )
        
        self.turns.append(turn)
        self.session_state = state

        # Successive longitudinal AI rubric scoring on claim resolution
        from app_3.config import VIVA_ENABLE_LAYER_2
        if VIVA_ENABLE_LAYER_2 and state in (StudentState.GROUNDED, StudentState.COLLAPSED):
            claim_turns = [t for t in self.turns if t.claim_id == current_claim_id]
            rubric_res = self.llm_client.evaluate_claim_rubric(current_claim_text, claim_turns)

            from app_3.schemas import RubricScore, RubricConstruct
            rubric_scores = [
                RubricScore(
                    claim_id=current_claim_id,
                    rubric_construct=RubricConstruct.INTERNALISATION,
                    bucket="A",
                    score=rubric_res.internalisation_score,
                    anchor_level=rubric_res.internalisation_anchor,
                    source="auto"
                ),
                RubricScore(
                    claim_id=current_claim_id,
                    rubric_construct=RubricConstruct.ORIGINALITY,
                    bucket="A",
                    score=rubric_res.originality_score,
                    anchor_level=rubric_res.originality_anchor,
                    source="auto"
                ),
                RubricScore(
                    claim_id=current_claim_id,
                    rubric_construct=RubricConstruct.CONFIDENCE_CONVICTION,
                    bucket="B",
                    score=rubric_res.confidence_score,
                    anchor_level=rubric_res.confidence_anchor,
                    source="auto-draft"
                ),
                RubricScore(
                    claim_id=current_claim_id,
                    rubric_construct=RubricConstruct.EMPATHY,
                    bucket="B",
                    score=rubric_res.empathy_score,
                    anchor_level=rubric_res.empathy_anchor,
                    source="auto-draft"
                )
            ]
            self.rubric_scores.extend(rubric_scores)
            # Attach per-turn rubric scores for display in UI
            turn.rubric_scores = rubric_scores

        # Translate next_prompt using Teleprompter for student view
        if state == StudentState.PAUSED or "SESSION RESOLVED" in next_prompt:
            return turn, next_prompt
            
        from app_3.config import USE_TELEPROMPTER_V2
        
        target_claim = None
        if state == StudentState.COLLAPSED:
            target_claim = next((c for c in self.epistemic_map.claims if c.id == current_claim_id), None)
        else:
            target_claim = self.active_claim
            
        if USE_TELEPROMPTER_V2:
            translated_prompt = self.teleprompter.translate_prompt_v2(next_prompt, state, target_claim)
        else:
            dim = target_claim.dimension if target_claim else PapanekDimension.NEED
            theme = target_claim.emergent_theme if target_claim else None
            translated_prompt = self.teleprompter.translate_prompt(next_prompt, state, dim, theme)
            
        return turn, translated_prompt

    def handle_challenge(self, collapse_limit: Optional[float] = None) -> str:
        """Processes the pedagogically significant 'challenge' intervention.

        Returns the agent's detailed justification of its confidence scoring.
        """
        if not self.turns:
            return "No turns have occurred yet. You can only challenge after responding to a probe!"
            
        last_turn = self.turns[-1]
        features = {
            "composite_confidence": last_turn.composite_confidence,
            "variance_score": last_turn.variance_score,
            "circularity_score": last_turn.circularity_score,
            "coherence_score": last_turn.coherence_score,
            "grounding_score": last_turn.grounding_score,
        }

        explanation = self.breakdown_detector.get_explanation(last_turn.state, features, collapse_limit=collapse_limit)

        challenge_prompt = (
            f"\n[STUDENT CHALLENGE INVOKED]\n"
            f"Here is my epistemic justification of your reasoning state on turn {last_turn.turn_index}:\n"
            f"--------------------------------------------------------------------------------\n"
            f"{explanation}\n"
            f"--------------------------------------------------------------------------------\n"
            f"In Socratic vivas, a challenge is where you exercise independent reasoning.\n"
            f"How do you dispute this scoring? Please justify why your response was grounded."
        )
        return challenge_prompt

    def record_challenge(self, system_explanation: str, justification: Optional[str] = None) -> None:
        """Persists a student's challenge/dispute of the last turn's scoring for faculty
        review. Justification is optional - a student may not want to write anything beyond
        invoking the challenge itself, and that's still worth logging."""
        if not self.turns:
            return
        last_turn = self.turns[-1]
        self.challenges.append(ChallengeRecord(
            turn_id=last_turn.id,
            turn_index=last_turn.turn_index,
            disputed_state=last_turn.state,
            system_explanation=system_explanation,
            student_justification=justification.strip() if justification and justification.strip() else None,
        ))

    def handle_redirect(self, target_claim_id: str) -> str:
        """Redirects active probing to a specific claim requested by the student."""
        matched = [c for c in self.epistemic_map.claims if c.id == target_claim_id]
        if not matched:
            return f"Claim ID {target_claim_id} not found in Epistemic Map."
            
        self.active_claim = matched[0]
        self.active_depth = 0
        return f"Probing redirected to claim '{self.active_claim.text}'.\nQuestion: {self.get_current_question()}"

    def build_transcript(self, notes: Optional[str] = None, completed: bool = False) -> SessionTranscript:
        """Compiles the session into a finalized validated SessionTranscript schema."""
        from app_3.config import VIVA_ENABLE_LAYER_2, VARIANCE_THRESHOLD
        review_card = None
        
        if completed and VIVA_ENABLE_LAYER_2 and self.rubric_scores:
            from app_3.schemas import ReviewCard
            confidences = [t.composite_confidence for t in self.turns if t.state != StudentState.PAUSED]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0
            
            v_a_composite = None
            v_b_composite = None
            
            if self.weight_version == "B":
                v_b_composite = round(avg_confidence, 2)
                import math
                v_a_sum = 0
                for t in self.turns:
                    if t.state != StudentState.PAUSED:
                        z = (self.feature_engine.w_bias + 
                             self.feature_engine.w_coherence * t.coherence_score +
                             self.feature_engine.w_grounding * t.grounding_score +
                             self.feature_engine.w_variance * t.variance_score +
                             self.feature_engine.w_circularity * t.circularity_score)
                        v_a_sum += 1.0 / (1.0 + math.exp(-z))
                v_a_composite = round(v_a_sum / len(confidences), 2) if confidences else 1.0
            else:
                v_a_composite = round(avg_confidence, 2)

            review_card = ReviewCard(
                composite_confidence=round(avg_confidence, 2),
                version_a_composite=v_a_composite,
                version_b_composite=v_b_composite,
                rubric_scores=self.rubric_scores,
                session_notes=notes
            )
            
        # Build telemetry export block
        telemetry_export = {
            "advocate": {
                "dialogue_strategy": "Socratic Probing with victor papanek adjacent reconstruction",
                "probed_dimension_history": [d.value for d in self.probed_dimension_history],
                "claim_pivots_history": self.probed_claim_history
            },
            "assessor": {
                "weight_version": self.weight_version,
                "weights": {
                    "w_bias": self.feature_engine.w_bias,
                    "w_coherence": self.feature_engine.w_coherence,
                    "w_grounding": self.feature_engine.w_grounding,
                    "w_variance": self.feature_engine.w_variance,
                    "w_circularity": self.feature_engine.w_circularity
                },
                "thresholds": {
                    "collapse_limit": self.breakdown_detector.collapse_limit,
                    "unstable_limit": self.breakdown_detector.unstable_limit,
                    "variance_threshold": VARIANCE_THRESHOLD
                }
            },
            "evaluator": {
                "turns_ml_telemetry": [
                    {
                        "turn_index": t.turn_index,
                        "claim_id": t.claim_id,
                        "coherence": t.coherence_score,
                        "grounding": t.grounding_score,
                        "variance": t.variance_score,
                        "circularity": t.circularity_score,
                        "composite_confidence": t.composite_confidence,
                        "classified_state": t.state.value
                    }
                    for t in self.turns
                ]
            }
        }
            
        return SessionTranscript(
            session_id=self.session_id,
            student_name=self.student_name,
            document_name=self.epistemic_map.document_name,
            epistemic_map=self.epistemic_map,
            turns=self.turns,
            started_at=datetime.now().isoformat(),
            completed_at=datetime.now().isoformat() if completed else None,
            notes=notes,
            weight_version=self.weight_version,
            review_card=review_card,
            telemetry_export=telemetry_export,
            experiment_profile=self.experiment_profile,
            challenges=self.challenges
        )

    def _select_next_vulnerable_claim(self, exclude_dimensions: Optional[List[PapanekDimension]] = None):
        """Finds the unprobed claim with the highest vulnerability (rank 1 is highest priority)."""
        from app_3.config import MAX_CLAIMS_TO_PROBE

        # Check if we've reached the max number of claims to probe in this session
        if len(self.probed_claim_history) >= MAX_CLAIMS_TO_PROBE:
            self.active_claim = None
            return

        exclude_dims = exclude_dimensions or []
        unprobed = [
            c for c in self.epistemic_map.claims
            if c.id not in self.probed_claim_history and c.dimension not in exclude_dims
        ]

        if unprobed:
            # Claims are pre-sorted by rank in ExtractionEngine, so we pick the first one
            self.active_claim = unprobed[0]
            self.active_depth = 0
        else:
            # If all claims in non-excluded dimensions are probed, try any unprobed claim left
            any_unprobed = [c for c in self.epistemic_map.claims if c.id not in self.probed_claim_history]
            if any_unprobed:
                self.active_claim = any_unprobed[0]
                self.active_depth = 0
            else:
                self.active_claim = None

    def _create_intervention_turn(self, turn_idx: int, question: str, response: str, 
                                 itype: InterventionType, state: StudentState) -> ProbeTurn:
        """Creates a mock/empty turn object for pausing/challenges."""
        return ProbeTurn(
            turn_index=turn_idx,
            claim_id=self.active_claim.id if self.active_claim else "NONE",
            question=question,
            student_response=response,
            coherence_score=1.0,
            grounding_score=1.0,
            variance_score=0.0,
            circularity_score=0.0,
            composite_confidence=1.0,
            state=state,
            intervention_type=itype
        )
