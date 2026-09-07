import os
import time
import random
import textwrap
from typing import List, Dict, Any, Tuple, Optional
import anthropic

from app_3.schemas import EpistemicMap, ProbeTurn, StudentState, PapanekDimension
from app_3.probe_engine import ProbingSessionManager
from app_3.storage import StorageManager
from app_3.config import MOCK_MODE, ANTHROPIC_API_KEY, ADVOCATE_MODEL, ADVOCATE_TEMP, ASSESSOR_TEMP, ADVOCATE_MAX_CHARS
from app_3.llm_client import _clamp_temp


def _cap_advocate_length(text: str, max_chars: Optional[int]) -> str:
    """Hard, word-boundary-safe cap on the Advocate's output.

    The prompt asks for brevity, but a 0.5B model does not reliably count
    characters from an instruction - this is the deterministic backstop that
    actually enforces the limit regardless of what the model produces.

    max_chars=None (config.ADVOCATE_MAX_CHARS's default) disables truncation
    entirely - length is no longer the defense against Advocate text scoring
    too high; see config.py's comment on ADVOCATE_MAX_CHARS.
    """
    text = (text or "").strip()
    if max_chars is None or len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0].rstrip(",.;:- ")
    return (cut + "…") if cut else text[:max_chars].rstrip()


def _strip_stage_directions(text: str) -> str:
    """Removes roleplay stage directions (e.g. "*pauses*") and leftover markdown
    emphasis asterisks. Defense-in-depth for the prompt-level instruction, since
    the model doesn't always follow "no markdown" reliably at high temperature -
    this text is displayed as plain text, so stray asterisks render literally."""
    import re
    # Whole-line stage directions, e.g. a standalone "*shifts slightly, thinking*"
    text = re.sub(r'(?m)^\s*\*[^*\n]+\*\s*$\n?', '', text)
    # Any remaining asterisks (inline emphasis) - drop the markers, keep the words
    text = text.replace('*', '')
    return text.strip()


class AdvocateAgent:
    """Simulates the Advocate Agent (High Temp) that defends student claims creatively.

    Defends claims by creatively pivoting to adjacent Papanek dimensions using the Claude API.
    """

    def __init__(self):
        # Pre-mapped infinite defenses steeled by Papanek pivots (as fallback)
        self.defenses: Dict[str, Dict[PapanekDimension, str]] = {
            "C-01": {
                PapanekDimension.NEED: "I know the method was synthetic, but we really needed to model the user safely in the studio. User safety is a core requirement.",
                PapanekDimension.USE: "We used sixty personas to test the workflow. This gave us a lot of practical usability feedback to validate our approach."
            },
            "C-02": {
                PapanekDimension.TELESIS: "I mapped the tool to show how the vocational stack is changing, which fits the broader societal impact.",
                PapanekDimension.AESTHETICS: "I showed the map interface. The visual design really helps people see the underlying biases."
            },
            "C-03": {
                PapanekDimension.METHOD: "I gave the feedback to the core users first. The technical implementation respects their input.",
                PapanekDimension.TELESIS: "Aligning with core stakeholders fits the broader societal context and ensures the project doesn't get blocked."
            },
            "C-04": {
                PapanekDimension.NEED: "There is no text; we used audio. Accessibility is an issue, so they need sound to learn. It's a fundamental requirement.",
                PapanekDimension.ASSOCIATION: "We used familiar symbols to match their mental models. The cultural perception is much better this way."
            },
            "C-05": {
                PapanekDimension.METHOD: "We ran a small model on edge hardware because network connectivity often fails. A large cloud server wouldn't work technically.",
                PapanekDimension.AESTHETICS: "The offline UI is a simple grid. In a constrained environment, the visual design must prioritize high legibility."
            },
            "C-06": {
                PapanekDimension.USE: "The interactive map isn't just pretty. Its functional usability helps teachers find student mistakes.",
                PapanekDimension.ASSOCIATION: "Users like visual patterns, so this matches their cognitive habits and perception."
            }
        }

        # Use LocalGenerator for on-device inference instead of Claude API
        from app_3.local_llm import LocalGenerator
        self.client = LocalGenerator()

    def generate_response(self, claim_id: str, claim_text: str, source_passage: str,
                          current_dimension: PapanekDimension, question: str, depth: int,
                          advocate_temp: Optional[float] = None) -> Tuple[str, PapanekDimension]:
        """Generates high-temperature rough brainstorming suggestions (intentionally imperfect).

        Uses very high temperature to encourage diverse, varied, uncertain responses that show
        reasoning exploration rather than polished answers. This makes the AI scaffolding role
        explicit and ensures high variance across samples.
        """
        # Find adjacent dimensions to pivot to
        from app_3.reconstruction import ReconstructionRouter
        from app_3.teleprompter import Teleprompter
        router = ReconstructionRouter()
        adj_1, adj_2 = router.adjacencies.get(current_dimension, (PapanekDimension.NEED, PapanekDimension.USE))

        # Stochastically alternate pivot target based on depth
        pivot_dimension = adj_1 if depth % 2 == 0 else adj_2

        # Use plain-language dimension names in the prompt itself, not the raw
        # Papanek enum values - this text is pre-filled directly into the
        # student's editable answer box, so any jargon the model is handed
        # here (e.g. "Telesis") comes straight back out in what looks like
        # the student's own words.
        mask_map = Teleprompter.DIMENSION_MASK_MAP
        current_dim_label = mask_map.get(current_dimension, current_dimension.value)
        pivot_dim_label = mask_map.get(pivot_dimension, pivot_dimension.value)

        # Use Claude at VERY high temperature if client is active
        # High temperature encourages rough, varied brainstorming (not polished answers)
        if self.client is not None:
            # Written as a single flowing instruction rather than a numbered/
            # labelled checklist on purpose: a numbered list with colon-led
            # labels ("Show uncertainty: ...") was echoed back near-verbatim as
            # if it were the nudge itself, rather than followed - a 0.5B model
            # pattern-matches a fill-in-the-blank template far more readily
            # than it follows one. Plain prose guidance doesn't give it a
            # template shape to copy.
            prompt = f"""
You are an external voice giving a student one nudge to react to, not an
answer and not an argument for them - just something to think from.

Context: the claim under discussion is "{claim_text}", drawn from this
original work: "{source_passage}". The question being probed is
"{question}", from the angle of {current_dim_label}.

In your own words, write a single genuinely uncertain thought about this -
something a thoughtful outsider might mutter half to themselves, using words
like "maybe" or "what if" rather than landing on a settled position. Make it
one reflective observation, not a question aimed back at the student, and
don't build a case for them - you are prompting them to think, not answering
for them and not interrogating them in return. Write it as plain prose, with
no markdown, asterisks, stage directions, or numbered points.
"""
            try:
                # Use VERY high temperature for diverse, varied outputs
                high_temp = 1.8  # Force exploration over consistency

                defense_text = self.client.generate(
                    prompt=prompt,
                    temperature=_clamp_temp(high_temp),
                    # Generation budget kept generous rather than tight: a tiny
                    # token budget alongside a brevity instruction previously
                    # made the model spend its whole budget echoing the
                    # constraint itself ("Under 20 words: ...") instead of
                    # producing real content. Neither the prompt nor a post-hoc
                    # cap now limits length (see AdvocateAgent's docstring and
                    # config.ADVOCATE_MAX_CHARS), and without the old word-count
                    # instruction the model runs noticeably longer - raised from
                    # 200 to 350 so a genuinely organic-length nudge doesn't get
                    # cut off mid-sentence.
                    max_new_tokens=350
                )
                defense_text = _strip_stage_directions(defense_text)
                if defense_text:
                    return _cap_advocate_length(defense_text, ADVOCATE_MAX_CHARS), pivot_dimension
            except Exception as e:
                print(f"Local SLM Advocate generation failed: {e}. Falling back to default defense.")

        # Load pre-mapped defense or synthesize a generic high-fidelity defense
        claim_defenses = self.defenses.get(claim_id, {})
        defense_text = claim_defenses.get(pivot_dimension, "")
        
        if not defense_text:
            from app_3.teleprompter import Teleprompter
            mask_map = Teleprompter.DIMENSION_MASK_MAP
            pivot_from = mask_map.get(current_dimension, current_dimension.value)
            pivot_to = mask_map.get(pivot_dimension, pivot_dimension.value)
            
            defense_text = (
                f"I chose this design because it helps with the {pivot_to} of the project. "
                f"I focused here so that users are supported under strict studio constraints. "
                f"This explains why I care about the {pivot_from}."
            )
            
        return _cap_advocate_length(defense_text, ADVOCATE_MAX_CHARS), pivot_dimension

def run_autonomous_preflight(manager: ProbingSessionManager, max_claims: Optional[int] = None) -> Dict[str, Any]:
    """Runs the on-device Advocate against the Assessor's own questions, claim by
    claim, with no human involved, until the FIRST genuine reasoning collapse -
    the same StudentState.COLLAPSED (threshold or sustained-instability) the live
    viva already computes - or until max_claims distinct claims have been
    explored, whichever comes first. The human then takes over from there.

    A Grounded (or not-yet-escalated Unstable) verdict on the Advocate's own
    answer does NOT end probing of that claim here. The Advocate is instructed
    to answer with deliberately hedged, uncertain language (see
    AdvocateAgent.generate_response) - testing showed those non-committal
    answers, and even the Advocate simply reflecting the question back rather
    than answering it, can still score as entailed against strong
    documentation. A single Grounded reading is therefore not trustworthy
    evidence the claim actually held up; this loop keeps pressing the SAME
    claim at increasing depth until it genuinely collapses, or the pre-written
    question bank for that claim runs out (MAX_DEPTH), at which point it is
    conceded - NOT marked resolved - and probing moves to the next candidate.

    Because of that, probed_claim_history is never touched here: the whole
    session's MAX_CLAIMS_TO_PROBE budget is left untouched by the pre-flight
    (submit_response's own bookkeeping for a Grounded verdict is undone below),
    so the human always gets a full, independent allotment of claims regardless
    of what the Advocate did before they joined - testing an earlier version of
    this function found the opposite: a run where every claim came back
    Grounded silently exhausted the shared claim budget and left no active
    claim for the human at all.

    Every turn still goes through the real, current submit_response() pipeline
    (cross-encoder NLI, escalation, everything this session's fixes cover), so
    nothing about the scoring logic is duplicated or re-approximated here. Each
    turn is tagged response_source=ADVOCATE and autonomous_preflight=True: the
    second flag is what excludes it from the graded composite/rubric (see
    build_transcript and rubric_criterion_means) - the Advocate's performance
    only decides where the human needs to step in, never the student's score.

    Scored against the tighter PREFLIGHT_* thresholds (config.py), not the
    human-facing GROUNDED_THRESHOLD/COLLAPSE_THRESHOLD/MAX_CONSECUTIVE_UNSTABLE
    - a borderline reading of the Advocate's deliberately hedged answer should
    count against it, so a genuine breakdown surfaces sooner than MAX_DEPTH.
    """
    from app_3.schemas import ResponseSource, StudentState
    from app_3.config import (
        MAX_CLAIMS_TO_PROBE, MAX_DEPTH,
        PREFLIGHT_GROUNDED_THRESHOLD, PREFLIGHT_COLLAPSE_THRESHOLD,
        PREFLIGHT_MAX_CONSECUTIVE_UNSTABLE,
    )

    claim_limit = max_claims if max_claims is not None else MAX_CLAIMS_TO_PROBE

    advocate = AdvocateAgent()
    collapsed_claim_id: Optional[str] = None
    turns_run = 0
    claims_seen: List[str] = []
    depth_on_claim = 0
    MAX_TURNS = 200  # defensive backstop only; real termination is state/claim-count driven

    while manager.active_claim is not None and turns_run < MAX_TURNS:
        claim = manager.active_claim
        if claim.id not in claims_seen:
            if len(claims_seen) >= claim_limit:
                break
            claims_seen.append(claim.id)
            depth_on_claim = 0
            manager.active_depth = 0

        question = manager.get_current_question()

        defense_text, _pivot_dimension = advocate.generate_response(
            claim_id=claim.id,
            claim_text=claim.text,
            source_passage=claim.source_passage or claim.text,
            current_dimension=claim.dimension,
            question=question,
            depth=manager.active_depth,
        )

        turn, _next_prompt = manager.submit_response(
            response=defense_text,
            collapse_limit=PREFLIGHT_COLLAPSE_THRESHOLD,
            unstable_limit=PREFLIGHT_GROUNDED_THRESHOLD,
            max_consecutive_unstable=PREFLIGHT_MAX_CONSECUTIVE_UNSTABLE,
        )
        turn.response_source = ResponseSource.ADVOCATE
        turn.autonomous_preflight = True
        turns_run += 1

        if turn.state == StudentState.COLLAPSED:
            collapsed_claim_id = turn.claim_id
            break

        # Override whatever submit_response just did for a non-collapse verdict
        # (advance to a new claim on Grounded, or bump depth on Unstable): undo
        # any resolution bookkeeping and keep pressing this exact claim, at a
        # depth we track ourselves rather than trusting the branch taken above.
        if claim.id in manager.probed_claim_history:
            manager.probed_claim_history.remove(claim.id)
        depth_on_claim += 1
        manager.active_claim = claim
        manager.active_depth = depth_on_claim

        if depth_on_claim > MAX_DEPTH:
            # Question bank for this claim is exhausted without a collapse -
            # concede it (still not marked probed/resolved) and move to the
            # next most-vulnerable claim not yet explored this pre-flight.
            # consecutive_unstable_turns is reset too: it is meant to track a
            # streak on ONE claim, and left alone would otherwise carry over
            # into the next claim - or into the human's own first turn, under
            # the very different default (non-PREFLIGHT) thresholds.
            remaining = [c for c in manager.epistemic_map.claims if c.id not in claims_seen]
            manager.active_claim = remaining[0] if remaining else None
            manager.active_depth = 0
            manager.consecutive_unstable_turns = 0
            depth_on_claim = 0

    return {
        "collapsed_claim_id": collapsed_claim_id,
        "turns_run": turns_run,
        "claims_explored": len(claims_seen),
    }


class EvaluatorAgent:
    """Simulates the Evaluator Agent that judges substantive coherence."""
class AssessorAgent:
    """Simulates the Assessor Agent that generates open-ended Socratic probing questions."""
    def __init__(self):
        from app_3.local_llm import LocalGenerator
        self.client = LocalGenerator()
            
    def generate_probe(self, claim_text: str, current_depth: int, previous_response: str = None) -> str:
        if previous_response:
            prompt = f"You are an examiner. The student defended the claim '{claim_text}' by saying: '{previous_response}'. Ask one short, open-ended probing question to challenge their logic."
        else:
            prompt = f"You are an examiner. Ask one short, open-ended probing question to challenge this claim: '{claim_text}'"
        
        try:
            res = self.client.generate(prompt=prompt, temperature=0.1, max_new_tokens=60)
            if "\n" in res:
                res = res.split("\n")[0]
            return res.strip()
        except:
            return f"Can you explain why the assertion '{claim_text}' holds under scrutiny?"

class EvaluatorAgent:
    """Simulates the Evaluator Agent that checks Question vs Answer coherence."""
    def __init__(self):
        from app_3.local_llm import LocalGenerator
        self.client = LocalGenerator()
        
    def generate_evaluation(self, question: str, response: str, nli_score: float) -> str:
        if nli_score > 0.6:
            prompt = f"The student's response '{response}' strongly entails the question '{question}'. In one short sentence, confirm that the answer directly addresses the question."
        else:
            prompt = f"The student's response '{response}' poorly entails the question '{question}'. In one short sentence, critique the student for being evasive or incoherent."
        
        try:
            return self.client.generate(prompt=prompt, temperature=0.4, max_new_tokens=60).strip()
        except:
            return "The response addresses the question."

class QualityAuditorAgent:
    """Scores the evidence portfolio (documentation + dialogue) against the rubric.

    Previously this agent never saw the evidence at all: source_passage was taken
    as an argument and then never referenced in the prompt, and the prompt itself
    branched on an nli_score that had ALREADY decided the verdict, asking the model
    only to write a sentence agreeing with it. The "variance" derived from it was
    therefore the wording jitter of a rubber stamp - it moved more between reruns
    of the same turn than between different turns, and the resulting text was then
    averaged back into grounding as if it were independent evidence, though it had
    been generated from the grounding score in the first place.

    It now does what it is named for: the evidence goes into the prompt, the model
    returns a number, and repeated samples give a genuine self-consistency spread.
    Stays entirely on the local SLM - no coursework text leaves the machine, and
    temperature remains available.
    """

    # Keep prompts inside the small model's usable context.
    MAX_EVIDENCE_CHARS = 1800
    MAX_BRIEF_CHARS = 600

    def __init__(self):
        from app_3.local_llm import LocalGenerator
        self.client = LocalGenerator()

    @staticmethod
    def _parse_score(text: str) -> Optional[float]:
        """Pull the first 0-1 number out of the model's reply.

        A 0.5B model will not reliably emit bare JSON, so anything unparseable
        returns None and is dropped by the caller rather than silently becoming
        a neutral 0.5 - a fabricated midpoint would understate the true spread.
        """
        import re
        for tok in re.findall(r"\d*\.?\d+", text or ""):
            try:
                v = float(tok)
            except ValueError:
                continue
            if 0.0 <= v <= 1.0:
                return v
            if 1.0 < v <= 100.0:      # model answered on a 0-100 scale
                return v / 100.0
        return None

    def build_prompt(self, evidence: str, rubric_text: str,
                     brief: Optional[str] = None) -> str:
        ev = (evidence or "")[: self.MAX_EVIDENCE_CHARS]
        parts = [
            "You are grading a design student's viva evidence.",
            "",
            "EVIDENCE (their documentation and what they said in the dialogue):",
            ev,
            "",
            "RUBRIC (what good evidence looks like):",
            rubric_text,
        ]
        if brief and brief.strip():
            parts += ["", "ASSIGNMENT BRIEF (what the work was asked to do):",
                      brief.strip()[: self.MAX_BRIEF_CHARS]]
        parts += [
            "",
            "How well does the EVIDENCE satisfy the RUBRIC"
            + (" and the BRIEF" if brief and brief.strip() else "") + "?",
            "Reply with ONLY a number between 0.00 and 1.00. No words.",
        ]
        return "\n".join(parts)

    def score_evidence(self, evidence: str, rubric_text: str,
                       brief: Optional[str] = None,
                       n_samples: int = 3, temp: float = 0.7) -> List[float]:
        """Sample the auditor n_samples times; returns the parsed scores.

        All samples run at the SAME temperature - the spread is meant to come from
        sampling the model, so varying temperature between draws would confound
        self-consistency with a temperature sweep.
        """
        prompt = self.build_prompt(evidence, rubric_text, brief)
        scores = []
        for _ in range(max(1, n_samples)):
            try:
                raw = self.client.generate(prompt=prompt, temperature=temp, max_new_tokens=8)
            except Exception:
                continue
            v = self._parse_score(raw)
            if v is not None:
                scores.append(v)
        return scores

    def generate_evaluation(self, source_passage: str, rubric_text: str,
                            nli_score: float = 0.5, temp: float = 0.6) -> str:
        """Deprecated prose path, kept so older callers do not break."""
        prompt = self.build_prompt(source_passage, rubric_text)
        try:
            return self.client.generate(prompt=prompt, temperature=temp, max_new_tokens=60).strip()
        except Exception:
            return "The documentation aligns with the rubric."


def run_adversarial_simulation(epistemic_map: EpistemicMap, max_turns: int = 10):
    """Executes the step-by-step Adversarial Socratic Simulation (Thesis Experiment)."""
    print("\n" + "=" * 80)
    print(" LAUNCHING ADVERSARIAL SOCRATIC SIMULATION (THESIS EXPERIMENT) ".center(80))
    print("=" * 80)
    print(f"  Questioner:   Assessor Agent (Low Temp: {ASSESSOR_TEMP}, Claude Socratic Probing)")
    print(f"  Respondent:   Advocate Agent (High Temp: {ADVOCATE_TEMP}, Claude Papanek Shield)")
    print("  Document:     " + epistemic_map.document_name)
    print("  Max Turns:    " + str(max_turns))
    print("=" * 80)
    print("Simulation starts. Press ENTER after each turn to advance, or type 'exit' to quit.\n")

    # Initialize tutor session manager
    tutor_manager = ProbingSessionManager(epistemic_map, student_name="Advocate Agent")
    # Initialize simulated AI agents
    ai_student = AdvocateAgent()
    assessor_agent = AssessorAgent()
    eval_agent = EvaluatorAgent()
    audit_agent = QualityAuditorAgent()
    
    from app_3.nli_auditor import NLIAuditor
    nli_auditor = NLIAuditor()
    
    # Track the student's previous response to feed into the next probe
    last_response = None
    
    from app_3.clustering_pipeline import ClusteringPipeline
    cluster_pipeline = ClusteringPipeline()
    turn_texts_for_clustering = []
    turn_ids = []

    for turn_idx in range(1, max_turns + 1):
        if not tutor_manager.active_claim:
            print("\n[SIMULATION COMPLETE] All claims successfully defended! Reasoning did not collapse.")
            break
            
        active_claim = tutor_manager.active_claim
        current_depth = tutor_manager.active_depth
        
        # Generative Assessor Probe
        question = assessor_agent.generate_probe(active_claim.text, current_depth, last_response)
        # Update the tutor manager's question getter so submit_response uses the generated probe
        tutor_manager.get_current_question = lambda: question
        
        print("-" * 80)
        print(f" TURN {turn_idx} | ACTIVE CLAIM: {active_claim.id} | DEPTH: {current_depth} | DIMENSION: {active_claim.dimension.value} ")
        print("-" * 80)
        wrapped_q = textwrap.fill(question, width=76)
        print(f"Assessor Agent:\n{textwrap.indent(wrapped_q, '  ')}")
        print("-" * 80)
        
        # Generate defense from high-temp AI student
        response, pivot_dim = ai_student.generate_response(
            claim_id=active_claim.id,
            claim_text=active_claim.text,
            source_passage=active_claim.source_passage,
            current_dimension=active_claim.dimension,
            question=question,
            depth=current_depth
        )
        
        # Simulate typing/thinking delay
        time.sleep(1.0)
        print(f"\nAdvocate Agent: [Pivot: {active_claim.dimension.value} ➞ {pivot_dim.value}]")
        wrapped_r = textwrap.fill(response, width=76)
        print(f"Advocate Agent:\n{textwrap.indent(wrapped_r, '  ')}")
        print("-" * 80)
        
        # Save response for next probe
        last_response = response
        
        # 1. Assessor: is the question relevant to the rubric? A question has no
        #    truth value, so this is a relevance check, not an entailment one.
        from app_3.rubric import rubric_summary_text
        rubric_text = rubric_summary_text()
        assessor_nli = nli_auditor.compute_sts_similarity(rubric_text, question)

        # 2. Evaluator: question vs answer is likewise a relevance relation.
        evaluator_nli = nli_auditor.compute_qa_coherence(question, response)

        # 3. Quality Auditor MC Sampling (Variance based on Auditor NLI: Evaluation vs Rubric)
        print("   [Sampling 3 Quality Auditor evaluations (Bracketed Temp Sweep) for Variance calculation...]")
        base_auditor_nli = nli_auditor.compute_support_vs_document(
            active_claim.source_passage, rubric_text
        )["support"]

        mc_auditor_nli_scores = []
        for t in [0.3, 0.7, 1.0]:
            audit_resp = audit_agent.generate_evaluation(active_claim.source_passage, rubric_text, base_auditor_nli, temp=t)
            # Measure variance in how consistently the Auditor evaluates the document against the rubric
            score = nli_auditor.compute_support_vs_document(audit_resp, rubric_text)["support"]
            mc_auditor_nli_scores.append(score)

        # Documentation vs answer: both propositions, so a genuine entailment pair.
        advocate_nli = nli_auditor.compute_support_vs_document(
            active_claim.source_passage, response
        )["support"]
        
        # Calculate Variance as the standard deviation of the 3 sampled Quality Auditor NLI scores
        import numpy as np
        variance = float(np.std(mc_auditor_nli_scores))
        
        # 4. Quality Auditor NLI: Documentation vs Rubric
        auditor_nli = base_auditor_nli
        
        # Generate Evaluator and Auditor vocalizations
        eval_text = eval_agent.generate_evaluation(question, response, evaluator_nli)
        audit_text = audit_agent.generate_evaluation(active_claim.source_passage, rubric_text, auditor_nli)
        
        print(f"Evaluator Agent: {eval_text}")
        print(f"Quality Auditor: {audit_text}")
        print("-" * 80)
        
        # Pipeline clustering 
        combined_text = cluster_pipeline.extract_turn_features(question, response, eval_text, audit_text)
        turn_texts_for_clustering.append(combined_text)
        turn_ids.append(f"Turn_{turn_idx}")
        cluster_id = cluster_pipeline.predict_cluster(combined_text)
        print(f"[Clustering] Assigned to Cluster ID: {cluster_id}")

        # Submit response to Tutor for evaluation
        turn_result, next_prompt = tutor_manager.submit_response(response, cluster_id=cluster_id)
        
        # Print metrics - replacing heuristic signals with pure NLI Matrix
        print(f" NLI MATRIX SIGNALS:")
        print(f"   Assessor NLI (Rubric vs Question):         {assessor_nli:.2f}")
        print(f"   Evaluator NLI (Question vs Answer):        {evaluator_nli:.2f}")
        print(f"   Advocate NLI (Evidence vs Answer):         {advocate_nli:.2f}")
        print(f"   Quality Auditor NLI (Doc vs Rubric):       {auditor_nli:.2f}")
        
        # Composite calculation from the Symbolic Regression equation
        from app_3.config import SCORING_WEIGHTS_FILE
        import json
        
        try:
            if SCORING_WEIGHTS_FILE.exists():
                with open(SCORING_WEIGHTS_FILE, 'r') as f:
                    config = json.load(f)
                if config.get("version") == "Symbolic_GP":
                    eq = config.get("equation_simplified")
                    # Replace variables safely
                    eq = eq.replace("X0", str(evaluator_nli))
                    eq = eq.replace("X1", str(auditor_nli))
                    eq = eq.replace("X2", str(variance))
                    import sympy
                    # Evaluate the simplified equation
                    z = float(sympy.sympify(eq))
                else:
                    z = (0.5 + (2.0 * evaluator_nli) + (1.5 * auditor_nli) + (-2.0 * variance))
            else:
                z = (0.5 + (2.0 * evaluator_nli) + (1.5 * auditor_nli) + (-2.0 * variance))
        except Exception as e:
            print(f"Failed to load GP equation, using fallback: {e}")
            z = (0.5 + (2.0 * evaluator_nli) + (1.5 * auditor_nli) + (-2.0 * variance))
            
        composite = 1.0 / (1.0 + np.exp(-z))
        
        formatted_scores = [f"{s:.2f}" for s in mc_auditor_nli_scores]
        print(f"   Variance (Monte Carlo StdDev):             {variance:.2f} (from scores: {formatted_scores})")
        print(f"   Composite Score:                           {composite:.2f}")
        
        if composite < 0.4:
            state_str = "COLLAPSED"
        elif composite < 0.65:
            state_str = "UNSTABLE"
        else:
            state_str = "GROUNDED"
            
        print(f"   Reasoning State:                           {state_str}")
        print("-" * 80)
        
        # Pause to let the evaluation sink in
        time.sleep(2.0)
        
        user_input = input("\n[Press Enter to continue, or type 'exit']: ").strip().lower()
        if user_input == "exit":
            print("\nSimulation aborted by user.")
            break
            
    # Export clusters for researcher
    export_path = cluster_pipeline.export_clusters_for_researcher(turn_texts_for_clustering, turn_ids)
    print(f"\nResearcher Clustering Data exported to: {export_path}")

    # Save finalized simulation transcript
    transcript = tutor_manager.build_transcript(notes="Automated Adversarial Socratic Simulation showing reasoning durability.")
    saved_path = StorageManager.save_transcript(transcript)
    print("\n" + "=" * 80)
    print(" EXPERIMENT COMPLETE ".center(80))
    print("=" * 80)
    print(f"Simulation Transcript persisted to:\n -> {saved_path.resolve()}")
    print("=" * 80 + "\n")
