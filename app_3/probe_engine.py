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

    def __init__(self, epistemic_map: EpistemicMap, student_name: str, experiment_profile: Optional[ExperimentProfile] = None,
                 assignment_brief: Optional[str] = None):
        self.epistemic_map = epistemic_map
        self.student_name = student_name
        self.session_id = str(uuid.uuid4())
        self.experiment_profile = experiment_profile
        # Optional coursework brief. Supplied at setup, it gives the auditor the
        # task the work was meant to satisfy, alongside the rubric.
        self.assignment_brief = assignment_brief
        # Populated only by run_final_audit() at the end of the session.
        self.final_audit: Optional[Dict[str, Any]] = None
        
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
        # Consecutive UNSTABLE turns on the ACTIVE claim. Reset on a grounded
        # answer, a threshold collapse, a pivot, or a student redirect.
        self.consecutive_unstable_turns = 0
        self.session_state = StudentState.GROUNDED

        # Claim ids that have already survived a specificity challenge (see the
        # GROUNDED branch below) - tracked so a claim need only prove itself
        # once even if revisited later.
        self.specificity_challenged: set = set()
        # A one-off, dynamically generated question overriding the static
        # per-claim question bank for exactly the next turn. Set by the
        # specificity-challenge step; get_current_question() returns it when
        # present, and it is cleared whenever the active claim changes.
        self.pending_dynamic_question: Optional[str] = None
        
        # Layer 2 Rubric Store
        self.challenges: List[ChallengeRecord] = []
        from app_3.config import SCORING_VERSION
        self.weight_version = SCORING_VERSION

        # Evidence accumulation for grounding/variance
        self.accumulated_responses: List[str] = []  # Student responses so far
        # Rubric as testable propositions rather than a normative "must" statement
        # -- see rubric.py for why the previous phrasing could not be inferred.
        from app_3.rubric import rubric_summary_text
        self.rubric_text = rubric_summary_text()

        # Select first target (most vulnerable ranked claim)
        self._select_next_vulnerable_claim()

    def get_current_question(self) -> str:
        """Generates the probing question for the active claim and depth."""
        if not self.active_claim:
            return "No more claims to probe. The session is complete!"

        if self.pending_dynamic_question:
            return self.pending_dynamic_question

        # Get question from client
        return self.llm_client.generate_question(self.active_claim.model_dump(), self.active_depth)

    def _generate_specificity_challenge(self, claim_text: str, previous_response: str) -> str:
        """One dynamically generated follow-up demanding a concrete, checkable
        specific - not another open-ended question, which can itself be dodged
        with more fluent generality.

        Validated empirically: a topically-fluent but substance-free answer
        cleared the Grounded bar on its first turn (coherence/grounding scored
        it against the CLAIM's general vocabulary, not against whether it
        engaged with the actual question). Once forced to repeatedly produce a
        named example, a number, or a first-hand observation instead of another
        general statement, the same register of answer degraded and collapsed
        within a few rounds. This is the mechanism that makes that pressure
        happen on every claim, not just ones that happen to fail immediately.
        """
        from app_3.simulation import AssessorAgent
        assessor = AssessorAgent()
        prompt = (
            f"You are a strict examiner. The claim under discussion is: \"{claim_text}\"\n\n"
            f"The student just said: \"{previous_response}\"\n\n"
            f"Ask ONE short, sharp follow-up question that demands a concrete, "
            f"checkable specific - a named example, a number, or a particular "
            f"first-hand observation from their own work - and cannot be "
            f"answered with another general statement."
        )
        try:
            q = assessor.client.generate(prompt=prompt, temperature=0.2, max_new_tokens=60)
            if "\n" in q:
                q = q.split("\n")[0]
            q = q.strip()
            return q or "Give one specific, concrete example that supports this claim."
        except Exception:
            return "Give one specific, concrete example that supports this claim."

    def submit_response(self, response: str, intervention: Optional[InterventionType] = None,
                        w_bias: Optional[float] = None,
                        w_coherence: Optional[float] = None,
                        w_grounding: Optional[float] = None,
                        w_variance: Optional[float] = None,
                        w_circularity: Optional[float] = None,
                        collapse_limit: Optional[float] = None,
                        unstable_limit: Optional[float] = None,
                        max_consecutive_unstable: Optional[int] = None,
                        cluster_id: Optional[int] = None) -> Tuple[ProbeTurn, str]:
        """Submits a student's answer, computes features, checks state transitions, and returns the next prompt."""
        if not self.active_claim:
            raise ValueError("No active claim is being probed.")

        # Overridable so the Monte Carlo sweep in simulation.py can calibrate the
        # streak limit the same way it calibrates the composite thresholds.
        from app_3.config import MAX_CONSECUTIVE_UNSTABLE
        max_unstable_streak = (
            max_consecutive_unstable if max_consecutive_unstable is not None
            else MAX_CONSECUTIVE_UNSTABLE
        )

        current_claim_id = self.active_claim.id
        current_claim_text = self.active_claim.text
        current_claim_theme = self.active_claim.emergent_theme
        question = self.get_current_question()
        turn_index = len(self.turns) + 1

        # Handle Pause/Redirect Interventions directly
        if intervention == InterventionType.PAUSE:
            turn = self._create_intervention_turn(turn_index, question, response, InterventionType.PAUSE, StudentState.PAUSED)
            self.turns.append(turn)
            return turn, "Session paused. What would you like to do? Options: [1] Redirect to another claim, [2] Challenge current score, [3] Resume session."
            
        # Extract features using the new NLI matrix
        from app_3.nli_auditor import NLIAuditor
        from app_3.simulation import AdvocateAgent, EvaluatorAgent
        import numpy as np
        
        nli_auditor = NLIAuditor()
        ai_student = AdvocateAgent()
        
        # 1. Coherence: two different questions, two different tools.
        #
        #    A question asserts nothing, so it has no truth value and nothing can
        #    entail or contradict it. Running entailment on a question/answer pair
        #    is a category error, and it was the source of the polarised scores.
        #    What we can ask of that pair is whether the answer is *about* what was
        #    asked, which is relevance -> cosine.
        #
        #    The logical relation lives elsewhere: the question was generated from
        #    a claim, and the claim IS a proposition. Claim vs answer is therefore
        #    a well-posed entailment pair.
        coherence_sts = nli_auditor.compute_sts_similarity(question, response)
        coherence_nli = nli_auditor.compute_support(current_claim_text, response)

        # 2. Grounding: student's answer against their own submitted documentation.
        #    Both sides are propositions, so this is the strongest entailment pair
        #    in the system -- and divergence here is the core integrity signal.
        #    Scored sentence by sentence, because a long passage collapsed into a
        #    single embedding loses the specificity entailment needs.
        grounding_context = self.active_claim.source_passage or current_claim_text
        doc_result = nli_auditor.compute_support_vs_document(
            grounding_context, response, weighting_query=self.active_claim.text
        )
        grounding_nli = doc_result["support"]
        grounding_sts = nli_auditor.compute_sts_similarity(grounding_context, response)

        # Most-contradicted sentence in the documentation, kept for the faculty
        # view: it names which line the answer diverged from, not just that it did.
        contradiction_signal = doc_result["min_support"]
        contradicted_sentence = doc_result["top_sentence"]

        # Discrete contradiction flag - see config.CONTRADICTION_FLAG_THRESHOLD.
        # Independent of contradiction_signal above: that is the ordinal support
        # scale's minimum (P(entail) + 0.5*P(neutral), collapsed), which can sit
        # in the middle of its range for a confident contradiction. This reads
        # the raw P(contradiction) the ordinal scale discards.
        from app_3.config import CONTRADICTION_FLAG_THRESHOLD, CONTRADICTION_FLAG_PENALTY
        contradiction_flag = doc_result.get("max_contradiction_prob", 0.0) >= CONTRADICTION_FLAG_THRESHOLD

        # 3. Build combined evidence: documentation + epistemic map + accumulated responses/dialogue
        if response and response.strip():
            self.accumulated_responses.append(response)

        # Compile dialogue transcript for this claim
        dialogue_lines = []
        for t in self.turns:
            if t.claim_id == self.active_claim.id:
                dialogue_lines.append(f"Assessor: {t.question}")
                dialogue_lines.append(f"Student: {t.student_response}")
        dialogue_lines.append(f"Assessor: {question}")
        dialogue_lines.append(f"Student: {response}")
        dialogue_transcript = "\n".join(dialogue_lines)

        epistemic_dict = self.epistemic_map.model_dump() if hasattr(self.epistemic_map, 'model_dump') else None
        combined_evidence = self.feature_engine.format_combined_evidence(
            documentation=self.active_claim.source_passage,
            epistemic_map=epistemic_dict,
            dialogue_transcript=dialogue_transcript
        )

        # 4. Rubric: score the evidence against each criterion independently.
        #
        #    The old rubric ("the student must demonstrate critical thinking") was
        #    a norm, not a proposition. A descriptive transcript can never entail a
        #    "must" statement -- there is no inference to make, so the score was
        #    noise. rubric.py restates each criterion as a declarative claim that
        #    the documentation and transcript can actually support or refute.
        #
        #    Variance comes from Monte Carlo over the auditor: three independent
        #    readings of the same evidence, spread measuring how stable the
        #    portfolio is under re-examination.
        from app_3.rubric import RUBRIC_CRITERIA, weighted_mean

        propositions = [c.proposition for c in RUBRIC_CRITERIA]

        # Scoring the evidence against each criterion is deterministic, so it is
        # done once rather than inside the sampling loop, and batched so the
        # evidence portfolio is embedded a single time instead of once per
        # criterion.
        criterion_supports = nli_auditor.score_document_against_many(
            combined_evidence, propositions
        )
        rubric_criterion_scores = {
            c.id: score for c, score in zip(RUBRIC_CRITERIA, criterion_supports)
        }
        rubric_support = weighted_mean(rubric_criterion_scores)

        criterion_sts = [
            nli_auditor.compute_sts_similarity(combined_evidence, p)
            for p in propositions
        ]
        rubric_similarity = float(np.mean(criterion_sts))

        # The per-turn Quality Auditor loop was removed here.
        #
        # It called a 0.5B local model three times per turn to write prose, then
        # measured how NLI-similar that prose was to the rubric and called the
        # spread "variance". The model never saw the evidence (source_passage was
        # taken as an argument and never used in the prompt) and the prompt branched
        # on an nli_score that had already decided the verdict, so the number moved
        # more between reruns of the same turn than between different turns.
        #
        # Auditor judgement now happens ONCE at the end of the session, over the
        # whole evidence portfolio, via run_final_audit(). During the session these
        # two terms are therefore provisional: grounding uses the deterministic
        # rubric support alone, and variance is 0 (no uncertainty estimate yet).
        # The live composite is what routing reacts to; the audited composite is
        # written afterwards by apply_final_audit().
        auditor_grounding_nli = rubric_support
        auditor_grounding_sts = rubric_similarity
        variance_nli = 0.0
        variance_sts = 0.0

        # 5. Composite score: hand-tuned logistic over the six variables.
        #
        #    Each support variable is CENTRED at its neutral point (0.5) before
        #    weighting. Without this the weights all pull in the positive
        #    direction from zero, so even an all-neutral turn scores z = 4.25 and
        #    lands at 0.99 confidence -- every session reads GROUNDED and the
        #    thresholds never fire. Centring makes the mapping honest: a turn we
        #    have no evidence about scores exactly 0.5, evidence moves it up,
        #    contradiction moves it down.
        #
        #    Variance is already zero-centred (0 = perfectly stable), so it is
        #    used raw with a negative weight.
        final_grounding_nli = (grounding_nli + auditor_grounding_nli) / 2.0
        final_grounding_sts = (grounding_sts + auditor_grounding_sts) / 2.0

        #    Weights live in config (see the v2-grounding block there for why the
        #    cosine terms were cut and why coherence_nli is negative). Grounding
        #    entailment against the student's own documentation now dominates: it
        #    was the only term that ranked answer quality correctly.
        from app_3.config import (
            W_COHERENCE_NLI, W_COHERENCE_STS, W_GROUNDING_NLI,
            W_GROUNDING_STS, W_VARIANCE_NLI, W_VARIANCE_STS,
            COMPOSITE_WEIGHTS_VERSION,
        )

        NEUTRAL = 0.5
        z = (0.0 +
             W_COHERENCE_NLI * (coherence_nli - NEUTRAL) +
             W_COHERENCE_STS * (coherence_sts - NEUTRAL) +
             W_GROUNDING_NLI * (final_grounding_nli - NEUTRAL) +
             W_GROUNDING_STS * (final_grounding_sts - NEUTRAL) +
             W_VARIANCE_NLI * variance_nli + W_VARIANCE_STS * variance_sts)
        if contradiction_flag:
            z -= CONTRADICTION_FLAG_PENALTY
        composite_confidence = float(1.0 / (1.0 + np.exp(-z)))

        features = {
            "coherence_nli": coherence_nli,
            "coherence_sts": coherence_sts,
            "grounding_nli": final_grounding_nli,
            "grounding_sts": final_grounding_sts,
            "variance_nli": variance_nli,
            "variance_sts": variance_sts,
            "composite_confidence": composite_confidence,
            "accumulated_responses_count": len(self.accumulated_responses),
            "word_count": len(response.split()),
            "is_vague": False,
            "rubric_criterion_scores": rubric_criterion_scores,
            "contradiction_signal": contradiction_signal,
            "contradicted_sentence": contradicted_sentence,
        }
                
        # Average NLI and STS scores for Turn object storage
        coherence_avg = (coherence_nli + coherence_sts) / 2.0
        grounding_avg = (final_grounding_nli + final_grounding_sts) / 2.0
        variance_avg = (variance_nli + variance_sts) / 2.0
        
        features["coherence_score"] = coherence_avg
        features["grounding_score"] = grounding_avg
        features["variance_score"] = variance_avg
        
        # Calculate true circularity score (Jaccard word overlap with question)
        q_words = set(question.lower().split())
        r_words = set(response.lower().split())
        if len(q_words) > 0:
            overlap = len(q_words.intersection(r_words))
            circularity = min(1.0, float(overlap) / len(q_words))
        else:
            circularity = 0.0
            
        features["circularity_score"] = circularity

        # collapse_limit / unstable_limit were accepted by this method but never
        # reached the decision - the cutoffs were hardcoded here. Wiring them up
        # is what lets the Monte Carlo sweep calibrate thresholds against faculty
        # labels instead of hand-tuning weights to fit a fixed arbitrary cutoff.
        from app_3.config import GROUNDED_THRESHOLD, COLLAPSE_THRESHOLD
        collapse_at = collapse_limit if collapse_limit is not None else COLLAPSE_THRESHOLD
        grounded_at = unstable_limit if unstable_limit is not None else GROUNDED_THRESHOLD

        if composite_confidence < collapse_at:
            state = StudentState.COLLAPSED
        elif composite_confidence < grounded_at:
            state = StudentState.UNSTABLE
        else:
            state = StudentState.GROUNDED

        # Sustained-instability escalation.
        #
        # The thresholds above are memoryless: a student hovering just above the
        # collapse line is re-judged from scratch every turn and can stay UNSTABLE
        # indefinitely without ever being recorded as having failed to ground the
        # claim. consecutive_unstable_turns carries that history, so repeated
        # instability on the SAME claim eventually resolves rather than plateauing.
        #
        # raw_state preserves the per-turn verdict so the escalation is auditable
        # and can be separated from a genuine threshold collapse during analysis.
        raw_state = state
        escalated_from_unstable = False

        if state == StudentState.UNSTABLE:
            self.consecutive_unstable_turns += 1
            if (max_unstable_streak > 0
                    and self.consecutive_unstable_turns >= max_unstable_streak):
                state = StudentState.COLLAPSED
                escalated_from_unstable = True
        else:
            # A grounded answer clears the streak; a threshold collapse routes away
            # from this claim, so the streak has no further meaning either way.
            self.consecutive_unstable_turns = 0

        consecutive_unstable_at_turn = self.consecutive_unstable_turns

        if escalated_from_unstable:
            # Routing below pivots to a new claim, so the streak restarts there.
            self.consecutive_unstable_turns = 0

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
            # A streak escalation is a plateau, not a crash. Saying "collapse" to a
            # student who never actually crossed the collapse line misdescribes what
            # happened, so the two routes are labelled distinctly even though both
            # pivot to a reconstructed angle.
            collapse_header = (
                "[SUSTAINED INSTABILITY - CHANGING APPROACH]"
                if escalated_from_unstable else
                "[REASONING COLLAPSE DETECTED]"
            )
            next_prompt = (
                f"\n{collapse_header}\n"
                f"Let's explore this from a different angle. {reconstruction_prompt}"
            )
            # Reset active claim depth and pivot to new dimension
            self.active_depth = 0
            self.pending_dynamic_question = None
            self._select_next_vulnerable_claim(exclude_dimensions=[self.active_claim.dimension])
        elif state == StudentState.UNSTABLE:
            self.active_depth += 1
            next_prompt = f"\n[PLATEAU EROSION DETECTED - PROBING DEEPER]\n{self.get_current_question()}"
        elif self.active_claim.id not in self.specificity_challenged:
            # Grounded - but a single Grounded turn is not proof of understanding.
            # Testing found a fluent, topically-adjacent-but-generic answer can
            # clear this bar without engaging with what was actually asked, since
            # coherence/grounding measure similarity to the CLAIM's vocabulary,
            # not whether the specific question was addressed. Every claim must
            # survive one explicit, SLM-generated demand for a concrete, checkable
            # specific before it is resolved - this is what let the same kind of
            # generic answer collapse within a few rounds once actually pressed.
            self.specificity_challenged.add(self.active_claim.id)
            self.active_depth += 1
            challenge_question = self._generate_specificity_challenge(current_claim_text, response)
            self.pending_dynamic_question = challenge_question
            next_prompt = f"\n[GROUNDED - BUT PROVE IT]\n{challenge_question}"
        else:
            # Grounded, and already survived its specificity challenge - resolve
            # this claim for real and move to the next vulnerable one.
            self.pending_dynamic_question = None
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
            coherence_nli=coherence_nli,
            coherence_sts=coherence_sts,
            grounding_nli=grounding_nli,
            grounding_sts=grounding_sts,
            variance_nli=variance_nli,
            variance_sts=variance_sts,
            coherence_score=coherence_avg,
            grounding_score=grounding_avg,
            variance_score=variance_avg,
            circularity_score=circularity,
            composite_confidence=features["composite_confidence"],
            rubric_criterion_scores=rubric_criterion_scores,
            contradiction_signal=contradiction_signal,
            contradicted_sentence=contradicted_sentence,
            contradiction_flag=contradiction_flag,
            state=state,
            consecutive_unstable_turns=consecutive_unstable_at_turn,
            escalated_from_unstable=escalated_from_unstable,
            raw_state=raw_state,
            reconstruction_dimension=routed_dimension,
            intervention_type=intervention,
            weight_version=self.weight_version,
            composite_weights_version=COMPOSITE_WEIGHTS_VERSION,
            emergent_theme=current_claim_theme
        )
        
        self.turns.append(turn)
        self.session_state = state

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
        """Processes the pedagogically significant 'challenge' intervention."""
        if not self.turns:
            return "No turns have occurred yet. You can only challenge after responding to a probe!"

        challenge_prompt = (
            "In Socratic assessment, a challenge is where you exercise independent reasoning. "
            "How do you dispute this scoring? Please justify why your response was grounded."
        )
        return challenge_prompt

    def record_challenge(self, justification: Optional[str] = None) -> None:
        """Persists a student's challenge/dispute of the last turn's scoring for faculty
        review. Justification is optional - a student may not want to write anything beyond
        invoking the challenge itself, and that's still worth logging."""
        if not self.turns:
            return
        last_turn = self.turns[-1]
        
        # Calculate rationale internally for faculty review logging
        features = {
            "composite_confidence": last_turn.composite_confidence,
            "variance_score": last_turn.variance_score,
            "circularity_score": last_turn.circularity_score,
            "coherence_score": last_turn.coherence_score,
            "grounding_score": last_turn.grounding_score,
        }
        system_explanation = self.breakdown_detector.get_explanation(last_turn.state, features)
        
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
        # The streak is per-claim, so a student-initiated redirect clears it.
        self.consecutive_unstable_turns = 0
        # A pending specificity-challenge question belongs to whatever claim was
        # active before the redirect - clear it so the new claim gets its own
        # normal question rather than an unrelated leftover one.
        self.pending_dynamic_question = None
        return f"Probing redirected to claim '{self.active_claim.text}'.\nQuestion: {self.get_current_question()}"

    def build_evidence_portfolio(self) -> str:
        """The whole session as one portfolio: documentation probed + full dialogue."""
        seen, docs = set(), []
        for t in self.turns:
            c = self.epistemic_map.get_claim(t.claim_id)
            if c and c.id not in seen:
                seen.add(c.id)
                docs.append(f"[{c.id}] {c.text}\n{(c.source_passage or '')[:1200]}")
        dialogue = "\n".join(
            f"Assessor: {t.question}\nStudent: {t.student_response}" for t in self.turns
        )
        return f"DOCUMENTATION:\n{chr(10).join(docs)}\n\nDIALOGUE:\n{dialogue}"

    def run_final_audit(self, n_samples: int = 5,
                        temp: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """One auditor pass over the whole session, at the END of the viva.

        The auditor scores the entire evidence portfolio n_samples times. Its
        disagreement with itself across those samples IS the variance: a portfolio
        that is clearly strong or clearly weak gets the same score every time, while
        a genuinely borderline one moves. That is the self-consistency estimate the
        per-turn local-model loop was trying and failing to produce.

        Deliberately NOT called per turn - it runs once, so the API cost and the
        amount of student text leaving the machine are both bounded, and it is never
        triggered by the per-turn checkpoint autosave.

        Returns None when no sample parsed (no API key, quota, network), leaving the
        session's provisional scores untouched rather than inventing an audit.
        """
        import numpy as np
        if not self.turns:
            return None

        evidence = self.build_evidence_portfolio()
        scores, rationales = [], []
        for _ in range(max(1, n_samples)):
            res = self.llm_client.score_evidence_portfolio(
                evidence, self.rubric_text, brief=self.assignment_brief, temp=temp
            )
            if res and res.get("portfolio_score") is not None:
                scores.append(float(np.clip(res["portfolio_score"], 0.0, 1.0)))
                if res.get("rationale"):
                    rationales.append(res["rationale"])

        if not scores:
            return None

        return {
            "portfolio_scores": scores,
            "portfolio_mean": float(np.mean(scores)),
            "portfolio_variance": float(np.std(scores)),
            "n_samples": len(scores),
            "rationales": rationales,
            "brief_used": bool(self.assignment_brief and self.assignment_brief.strip()),
            "audited_at": datetime.now().isoformat(),
        }

    def apply_final_audit(self, audit: Dict[str, Any]) -> None:
        """Rescore every turn using the session-level auditor result.

        The live composite each turn was scored with had no auditor opinion and no
        uncertainty estimate - both were provisional placeholders. This substitutes
        the audited values and recomputes the composite and state.

        The provisional numbers are PRESERVED (provisional_composite /
        provisional_state) rather than overwritten, because routing during the
        session reacted to them: the path the viva took is only explicable from the
        scores that existed at the time, so both are needed to read a transcript.
        """
        import numpy as np
        from app_3.config import (
            W_COHERENCE_NLI, W_COHERENCE_STS, W_GROUNDING_NLI, W_GROUNDING_STS,
            W_VARIANCE_NLI, W_VARIANCE_STS, GROUNDED_THRESHOLD, COLLAPSE_THRESHOLD,
            COMPOSITE_WEIGHTS_VERSION,
        )
        aud, var = audit["portfolio_mean"], audit["portfolio_variance"]
        NEUTRAL = 0.5
        for t in self.turns:
            if t.intervention_type is not None:
                continue  # pause/challenge stubs carry no measured features
            if t.provisional_composite is None:
                t.provisional_composite = t.composite_confidence
                t.provisional_state = t.state

            g_nli = (t.grounding_nli + aud) / 2.0
            z = (W_COHERENCE_NLI * (t.coherence_nli - NEUTRAL)
                 + W_COHERENCE_STS * (t.coherence_sts - NEUTRAL)
                 + W_GROUNDING_NLI * (g_nli - NEUTRAL)
                 + W_GROUNDING_STS * (t.grounding_sts - NEUTRAL)
                 + W_VARIANCE_NLI * var + W_VARIANCE_STS * var)
            composite = float(1.0 / (1.0 + np.exp(-z)))

            t.variance_nli = t.variance_sts = var
            t.variance_score = var
            t.composite_confidence = round(composite, 3)
            t.state = (StudentState.COLLAPSED if composite < COLLAPSE_THRESHOLD
                       else StudentState.UNSTABLE if composite < GROUNDED_THRESHOLD
                       else StudentState.GROUNDED)
            t.composite_weights_version = f"{COMPOSITE_WEIGHTS_VERSION}+audited"
        self.final_audit = audit

    def build_transcript(self, notes: Optional[str] = None, completed: bool = False) -> SessionTranscript:
        """Compiles the session into a finalized validated SessionTranscript schema."""
        from app_3.config import (
            VIVA_ENABLE_LAYER_2, VARIANCE_THRESHOLD, MAX_CONSECUTIVE_UNSTABLE,
            W_COHERENCE_NLI, W_COHERENCE_STS, W_GROUNDING_NLI,
            W_GROUNDING_STS, W_VARIANCE_NLI, W_VARIANCE_STS,
            COMPOSITE_WEIGHTS_VERSION,
        )
        review_card = None
        
        # Gated on the session being finished and having turns - NOT on Layer 2
        # rubric scores. Those are a nice-to-have the card carries when an API was
        # available; nothing the faculty dashboard renders from the card depends on
        # them (the composite, notes, response-source mix, per-criterion breakdown
        # and contradiction list all come from the turns). Requiring them meant an
        # offline or quota-limited session produced no card at all.
        if completed and self.turns:
            from app_3.schemas import ReviewCard
            # Autonomous pre-flight turns were answered by the Advocate, not the
            # student - they locate where the student needed to take over, and are
            # never part of what the student is graded on.
            confidences = [
                t.composite_confidence for t in self.turns
                if t.state != StudentState.PAUSED and not t.autonomous_preflight
            ]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0
            
            v_a_composite = None
            v_b_composite = None
            
            if self.weight_version == "B":
                v_b_composite = round(avg_confidence, 2)
                v_a_composite = round(avg_confidence, 2)
            else:
                v_a_composite = round(avg_confidence, 2)

            review_card = ReviewCard(
                composite_confidence=round(avg_confidence, 2),
                version_a_composite=v_a_composite,
                version_b_composite=v_b_composite,
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
                    "coherence_nli": W_COHERENCE_NLI,
                    "coherence_sts": W_COHERENCE_STS,
                    "grounding_nli": W_GROUNDING_NLI,
                    "grounding_sts": W_GROUNDING_STS,
                    "variance_nli": W_VARIANCE_NLI,
                    "variance_sts": W_VARIANCE_STS,
                    "bias": 0.0,
                    "composite_weights_version": COMPOSITE_WEIGHTS_VERSION
                },
                "thresholds": {
                    "collapse_limit": self.breakdown_detector.collapse_limit,
                    "unstable_limit": self.breakdown_detector.unstable_limit,
                    "variance_threshold": VARIANCE_THRESHOLD,
                    "max_consecutive_unstable": MAX_CONSECUTIVE_UNSTABLE
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
            assignment_brief=self.assignment_brief,
            final_audit=self.final_audit,
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
