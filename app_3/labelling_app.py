import streamlit as st
import sys
import html
from pathlib import Path
import json
from datetime import datetime

# Ensure project root is on Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app_3.storage import StorageManager
from app_3.schemas import StudentState, FacultyLabel, EvaluatorRating, FacultyExperimentRating
from app_3.config import BASE_DIR
from app_3.wrapper import VivaWrapper
from app_3.theme import inject_theme
from app_3.review_helpers import (
    claim_for_turn as resolve_claim_for_turn,
    composite_to_letter_grade,
    response_source_counts,
    rubric_criterion_means,
    contradiction_flags,
    responsiveness_summary,
    intervention_effort_summary,
)


# Login roles. Access is partitioned so each role only ever sees the surface it
# is meant to judge - in particular, Experiment Rating belongs to Expert
# Alignment alone, keeping the temperature experiment double-blind.
ROLE_PRIMARY = "Primary Evaluator"
ROLE_INDEPENDENT = "Independent Assessor"
ROLE_EXPERT_ALIGNMENT = "Expert Alignment"


def review_card_is_populated(transcript) -> bool:
    """True only when the review card carries a complete set of data.

    A half-built card (no composite score, no notes, or a session with no turns)
    is misleading to grade against, so the Primary Review Card is hidden entirely
    rather than rendered with blanks.
    """
    rc = getattr(transcript, "review_card", None)
    if rc is None:
        return False
    if not transcript.turns:
        return False
    if rc.composite_confidence is None:
        return False
    return True


def evaluator_rating_is_populated(rating) -> bool:
    """True only when a submitted blind rating carries a complete set of data."""
    if rating is None:
        return False
    if not (rating.judgement or "").strip():
        return False
    if rating.rating_score is None:
        return False
    if not (rating.notes or "").strip():
        return False
    return True


def render_source_pane(document_name: str, claim) -> None:
    """Renders a cropped, highlighted PDF page for a claim, falling back to the
    extracted text passage if the source PDF isn't available locally."""
    st.markdown("#### 📄 Source Document")
    if not claim:
        st.info("No claim context available for this turn.")
        return

    pdf_path = BASE_DIR / document_name
    if not pdf_path.exists():
        pdf_path = BASE_DIR / "Misc" / document_name
        
    if not pdf_path.exists():
        st.markdown(f"> *{claim.source_passage}*")
        return

    try:
        image_bytes = VivaWrapper.extract_claim_image(
            pdf_path=str(pdf_path),
            page_number=claim.page,
            search_text=claim.text
        )
        if image_bytes:
            st.image(image_bytes, caption=f"{document_name} — Page {claim.page}")
        else:
            st.info("Could not visually locate the claim text on the page. Showing extracted passage instead.")
            st.markdown(f"> *{claim.source_passage}*")
    except Exception as e:
        st.error("Something went wrong.")
        st.markdown(f"> *{claim.source_passage}*")


def show_reasoning_state_definitions() -> None:
    """Display definitions of reasoning states for faculty reference."""
    with st.expander("📖 Understanding Reasoning States", expanded=False):
        st.markdown("""
**Grounded** 🟢
- Student's reasoning is coherent, well-evidenced, and stands up to follow-up questioning
- Answers directly address the question and connect to their work/document
- Shows genuine understanding of the design decisions and their rationale

**Unstable** 🟡
- Student's reasoning is wavering or inconsistent across multiple probes
- Answers may be partially correct but lack full grounding or clarity
- Student approaches a breakdown but hasn't fully collapsed; deeper probing is revealing gaps
- Shows some understanding but with gaps or hesitation

**Collapsed** 🔴
- Student's reasoning has broken down under probing
- Answers become contradictory, evasive, or reveal fundamental misunderstanding
- Student cannot justify their claimed design decisions
- Shows failure to demonstrate genuine understanding despite repeated questioning

**Use this guide when:**
- Labelling turns in the Data Labelling tab (agree/disagree with system classification)
- Submitting your independent evaluation in Blind Evaluation (form your own judgment first, then compare with system)
        """)


inject_theme()

st.title("Evaluator Dashboard")
st.write("---")

# 1. Capture Labeller ID
col_role, col_labeller, col_change = st.columns([2, 3, 1])
with col_role:
    role = st.selectbox("Login Role:", [ROLE_PRIMARY, ROLE_INDEPENDENT, ROLE_EXPERT_ALIGNMENT])
with col_labeller:
    labeller_id = st.text_input("Evaluator ID:", value="Evaluator_A").strip()
with col_change:
    if st.button("🔄 Change & Restart"):
        st.session_state.turn_step = None
        st.session_state.current_session_id = None
        st.rerun()

# 2. Get Available Transcripts & Checkpoints
session_ids = StorageManager.list_available_transcripts()
checkpoints = StorageManager.list_checkpoints()

# Show recovery banner if checkpoints exist
if checkpoints:

    with st.expander("📋 View & Recover Incomplete Sessions"):
        for ckpt in checkpoints:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.text(f"Session ID: {ckpt['session_id'][:8]}...")
            with col2:
                from datetime import datetime
                ckpt_time = datetime.fromtimestamp(ckpt['modified']).strftime("%Y-%m-%d %H:%M:%S")
                st.caption(ckpt_time)
            with col3:
                if st.button("▶ Recover", key=f"recover_{ckpt['session_id']}"):
                    # Load checkpoint and promote to session
                    ckpt_transcript = StorageManager.load_checkpoint(ckpt['session_id'])
                    if ckpt_transcript:
                        final_path = StorageManager.promote_checkpoint_to_session(ckpt['path'])
                        st.success(f"✅ Recovered! Session now available in main list.")
                        st.rerun()

if not session_ids:
    st.info("No transcripts found in the storage directory.")
else:
    # 3. Session Selection
    selected_session = st.selectbox("Select Session Transcript ID:", session_ids)
    
    if selected_session:
        # Load transcript
        transcript = StorageManager.load_transcript(selected_session)
        
        if transcript:
            st.write(f"**Student:** {transcript.student_name} | **Document:** {transcript.document_name}")
            try:
                formatted_date = datetime.fromisoformat(transcript.started_at).strftime("%B %d, %Y at %I:%M %p")
            except Exception:
                formatted_date = transcript.started_at
            st.write(f"**Started At:** {formatted_date}")
            
            with st.expander("📋 View/Edit Assignment Brief", expanded=False):
                st.write("The brief the coursework was set against. (Updating this will save it to the session transcript)")
                current_brief = transcript.assignment_brief or ""
                new_brief = st.text_area("Assignment Brief:", value=current_brief, height=150, key=f"brief_{selected_session}")
                if st.button("Submit Brief", key=f"submit_btn_{selected_session}"):
                    if new_brief != current_brief:
                        transcript.assignment_brief = new_brief
                        StorageManager.save_transcript(transcript)
                        st.success("Brief updated!")
                    else:
                        st.info("No changes made.")
            
            blind_mode = (role == ROLE_INDEPENDENT)

            tab1 = tab2 = tab3 = tab4 = None

            if blind_mode:
                # Assessor mode is deliberately single-purpose: the blind judgement
                # is only valid if the assessor never sees system scores, review
                # cards or experiment ratings. Nothing else is rendered.
                tabs = st.tabs(["Independent Blind Evaluation"])
                tab3 = tabs[0]
            elif role == ROLE_EXPERT_ALIGNMENT:
                # Experiment Rating is exclusive to the Expert Alignment role.
                tabs = st.tabs(["Experiment Rating"])
                tab4 = tabs[0]
            else:
                # Primary Evaluator: labelling, plus the review card only once the
                # card actually holds a full set of data.
                if review_card_is_populated(transcript):
                    tabs = st.tabs(["Data Labelling", "Primary Review Card"])
                    tab1, tab2 = tabs
                else:
                    tabs = st.tabs(["Data Labelling"])
                    tab1 = tabs[0]
            
            if tab1:
                with tab1:
                    show_reasoning_state_definitions()
                    # Find the first unlabelled turn
                    labelled_turn_indices = {lbl.turn_id for lbl in transcript.faculty_labels if lbl.labeller_id == labeller_id}
                    
                    # Filter valid turns
                    turns_to_label = [t for t in transcript.turns]
                    
                    if not turns_to_label:
                        st.warning("This transcript contains no dialog turns.")
                    else:
                        # Stepper index in session state
                        if "turn_step" not in st.session_state or st.session_state.get("current_session_id") != selected_session:
                            st.session_state.current_session_id = selected_session
                            # Find first unlabelled turn index
                            first_unlabelled = 0
                            for idx, t in enumerate(turns_to_label):
                                if str(t.id) not in labelled_turn_indices:
                                    first_unlabelled = idx
                                    break
                            st.session_state.turn_step = first_unlabelled

                        step = st.session_state.turn_step
                        
                        if step >= len(turns_to_label):
                            st.success("All turns in this session have been labelled by you!")
                            if st.button("Reset Stepper", type="secondary"):
                                st.session_state.turn_step = 0
                                st.rerun()
                        else:
                            current_turn = turns_to_label[step]

                            st.write(f"### Turn {step + 1} of {len(turns_to_label)}")

                            # Surface the underlying claim being probed so labellers aren't
                            # grading a question/response pair in a vacuum.
                            source_claim = resolve_claim_for_turn(transcript, current_turn)

                            main_col, side_col = st.columns([3, 2])

                            with main_col:
                                if source_claim:
                                    st.markdown(f"""
                                    <div class="box">
                                        <b>Claim Under Test ({html.escape(source_claim.id)}):</b><br/>
                                        "{html.escape(source_claim.text)}"
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.warning(f"No claim found in the epistemic map for claim_id '{current_turn.claim_id}'. Grading without source context.")

                                # Display dialog block (without scores to prevent bias)
                                # Response source label
                                source_emoji = "🤖" if current_turn.response_source.value == "Advocate" else "🔀" if current_turn.response_source.value == "Hybrid" else "❔"
                                source_label = f"{source_emoji} {current_turn.response_source.value}"

                                st.markdown(f"""
                                <div class="box">
                                    <b>Assessor Probing Question:</b><br/>
                                    "{html.escape(current_turn.question)}"
                                    <br/><br/>
                                    <b>Student Rationale Response:</b> <small>{source_label}</small><br/>
                                    "{html.escape(current_turn.student_response)}"
                                    <br/><br/>
                                    <b>System Classified Reasoning State:</b>
                                    <span class="system-badge">{current_turn.state.value}</span>
                                </div>
                                """, unsafe_allow_html=True)

                                # Advocate draft vs submitted answer, when a draft was
                                # generated for this turn. Previously the draft was
                                # discarded the instant response_source was decided, so a
                                # labeller had no way to see what the student actually
                                # changed - only the Advocate/Hybrid/Unverified bucket.


                                if current_turn.coherence_sts is not None:
                                    st.caption(
                                        f"Responsiveness to the question asked (not the claim): "
                                        f"{current_turn.coherence_sts:.3f}"
                                    )

                                # Input controls for labelling
                                agree = st.checkbox("I agree with the system classification", value=True)

                                corrected_state = None
                                if not agree:
                                    corrected_state = st.selectbox(
                                        "Corrected Reasoning State:",
                                        [StudentState.GROUNDED, StudentState.UNSTABLE, StudentState.COLLAPSED]
                                    )

                                note = st.text_area("Optional notes/justification:")

                                # Navigation controls
                                nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 1, 1])
                                with nav_col1:
                                    if st.button("⬅ Previous Turn", disabled=(step == 0)):
                                        st.session_state.turn_step = max(0, step - 1)
                                        st.rerun()
                                with nav_col2:
                                    # Jump to specific turn
                                    jump_to = st.number_input("Jump to turn:", min_value=1, max_value=len(turns_to_label), value=step + 1, step=1)
                                    if st.button("Go"):
                                        st.session_state.turn_step = jump_to - 1
                                        st.rerun()
                                with nav_col3:
                                    st.markdown("")  # Spacer
                                with nav_col4:
                                    if st.button("Save Label & Next ➡", type="primary"):
                                        # Create label
                                        label = FacultyLabel(
                                            turn_id=str(current_turn.id),
                                            labeller_id=labeller_id,
                                            agrees_with_system=agree,
                                            corrected_state=corrected_state,
                                            note=note if note.strip() else None,
                                            labelled_at=datetime.now().isoformat()
                                        )

                                        # Remove prior label for this turn by same labeller
                                        transcript.faculty_labels = [
                                            lbl for lbl in transcript.faculty_labels
                                            if not (lbl.turn_id == str(current_turn.id) and lbl.labeller_id == labeller_id)
                                        ]

                                        # Add label and save
                                        transcript.faculty_labels.append(label)
                                        StorageManager.save_transcript(transcript)

                                        st.success(f"Label saved for Turn {current_turn.turn_index}.")

                                        # Increment step
                                        st.session_state.turn_step += 1
                                        st.rerun()

                            with side_col:
                                render_source_pane(transcript.document_name, source_claim)

                    # Basic Agreement Dashboard & Export
                    st.write("---")
                    st.write("### Faculty Agreement Dashboard")
                    
                    my_labels = [lbl for lbl in transcript.faculty_labels if lbl.labeller_id == labeller_id]
                    total_labels = len(my_labels)
                    agreed_labels = sum(1 for lbl in my_labels if lbl.agrees_with_system)
                    agreement_rate = (agreed_labels / total_labels * 100) if total_labels > 0 else 100.0
                    
                    st.write(f"- **Total Labels Filed:** {total_labels}")
                    st.write(f"- **Human-System Agreement Rate:** {agreement_rate:.1f}%")
                    
                    # Export raw JSON data of labels
                    if st.checkbox("Show Flat JSON Export"):
                        flat_export = []
                        for lbl in my_labels:
                            flat_export.append({
                                "session_id": selected_session,
                                "student_name": transcript.student_name,
                                "turn_id": lbl.turn_id,
                                "labeller_id": lbl.labeller_id,
                                "agrees": lbl.agrees_with_system,
                                "corrected_state": lbl.corrected_state.value if lbl.corrected_state else None,
                                "note": lbl.note
                            })
                        st.code(json.dumps(flat_export, indent=2), language="json")

            if tab2:
                with tab2:
                    # Review Card Rendering
                    rc = transcript.review_card
                    st.markdown('<div class="review-card-header"><h2>Primary Review Card</h2></div>', unsafe_allow_html=True)
                    st.write(f"**Generated On:** {rc.created_at}")
                        
                    # Dual-score rendering if Version A and B exist
                    st.write("### Reasoning Foundation")
                    version_a = getattr(rc, "version_a_composite", None)
                    version_b = getattr(rc, "version_b_composite", None)
                    if version_a is not None and version_b is not None:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(label="Version A Grade", value=composite_to_letter_grade(version_a))
                        with col2:
                            st.metric(label="Version B Grade", value=composite_to_letter_grade(version_b))
                    else:
                        st.metric(label="Overall Grade", value=composite_to_letter_grade(rc.composite_confidence))
                            
                    if rc.session_notes:
                        st.write("---")
                        st.write("### Session Notes")
                        st.write(rc.session_notes)

                    # Response source summary for faculty
                    src_counts = response_source_counts(transcript)
                    total_turns = src_counts["total"]

                    if total_turns > 0:
                        st.write("---")
                        st.write("### Response Source Summary")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("❔ Unverified", src_counts["unverified"], f"{src_counts['unverified']/total_turns*100:.0f}%")
                        with col2:
                            st.metric("🤖 Advocate-Suggested", src_counts["advocate"], f"{src_counts['advocate']/total_turns*100:.0f}%")
                        with col3:
                            st.metric("🔀 Hybrid/Edited", src_counts["hybrid"], f"{src_counts['hybrid']/total_turns*100:.0f}%")

                    # Intervention effort: how far each Advocate-assisted submission
                    # diverged from the raw draft. The project's own design notes call
                    # frequent, substantial intervention the richest evidence of tacit
                    # understanding in the session - previously this was computed once
                    # (to bucket the turn as Advocate/Hybrid/Unverified) and discarded,
                    # so that evidence was never actually visible to faculty.
                    effort = intervention_effort_summary(transcript)
                    if effort["n"] > 0:
                        st.write("---")
                        st.write("### Intervention Effort (vs Advocate Draft)")
                        st.caption(
                            "Rewritten % = how much of the submitted answer differs from the "
                            "raw Advocate suggestion for that turn. Higher means more "
                            "independent editing; 0% means the draft was submitted unchanged."
                        )
                        st.metric("Mean Rewritten", f"{effort['mean_rewritten_pct']:.0f}%")
                        for row in effort["rows"]:
                            with st.expander(
                                f"Turn {row['turn_index']} — {row['rewritten_pct']:.0f}% rewritten "
                                f"({row['response_source']})",
                                expanded=False,
                            ):
                                st.markdown(f"**Advocate draft:**\n\n> {row['draft']}")
                                st.markdown(f"**Submitted:**\n\n> {row['submitted']}")

                    # Responsiveness: did the answer address the QUESTION asked, not
                    # just whether it happens to be entailed by the claim/document.
                    # coherence_nli is weighted 0 in the composite precisely because it
                    # rewarded restating the claim over engaging the question, so this
                    # is the composite's only question-facing signal and is otherwise
                    # invisible - a well-targeted answer can score no differently from
                    # one that ignored the question, unless this is shown separately.
                    resp = responsiveness_summary(transcript)
                    if resp["mean"] is not None:
                        st.write("---")
                        st.write("### Responsiveness to Questions Asked")
                        st.caption(
                            "Semantic similarity between each answer and the question it "
                            "was answering (not the claim or documentation). Low values here "
                            "alongside a low composite usually mean the answer went somewhere "
                            "the assessor did not ask; high values alongside a low composite "
                            "usually mean the answer engaged the question but the system could "
                            "not verify it against the documentation."
                        )
                        st.metric("Mean Responsiveness", f"{resp['mean']:.3f}")

                    if transcript.challenges:
                        st.write("---")
                        st.write("### Student Challenges")
                        st.info("Turns where the student disputed the system's scoring decision.")
                        for ch in transcript.challenges:
                            with st.expander(f"Turn {ch.turn_index} - disputed as {ch.disputed_state.value}", expanded=False):
                                st.markdown(f"**System's justification at the time:** {ch.system_explanation}")
                                if ch.student_justification:
                                    st.markdown(f"**Student's reasoning:** {ch.student_justification}")
                                else:
                                    st.markdown("*Student did not provide additional reasoning.*")
                                st.caption(f"Challenged at {ch.challenged_at}")

                    # Per-criterion rubric support. Computed every turn and stored,
                    # but never surfaced until now - an examiner gets far more from
                    # "weak on evidence, strong on ownership" than from one composite.
                    criterion_rows = rubric_criterion_means(transcript)
                    if criterion_rows:
                        st.write("---")
                        st.write("### Rubric Criterion Breakdown")
                        for row in criterion_rows:
                            st.write(f"**{row['id'].capitalize()}** — {row['mean']:.3f}")
                            st.progress(min(max(row["mean"], 0.0), 1.0))
                    else:
                        st.write("---")
                        st.caption(
                            "No per-criterion rubric scores recorded — this session predates "
                            "per-criterion scoring."
                        )

                    # Turns whose answer diverged most from the student's own
                    # documentation. Only meaningful with the cross-encoder backend:
                    # the previous head could not distinguish contradiction from
                    # simply being about a different topic.
                    flagged = contradiction_flags(transcript)
                    if flagged:
                        st.write("---")
                        st.write("### Possible Contradictions With the Documentation")
                        for t in flagged:
                            with st.expander(
                                f"Turn {t.turn_index} — contradiction signal {t.contradiction_signal:.3f}",
                                expanded=False,
                            ):
                                st.markdown(f"**Student said:** {t.student_response}")
                                if t.contradicted_sentence:
                                    st.markdown(
                                        f"**Most relevant line in their documentation:** "
                                        f"*{t.contradicted_sentence}*"
                                    )

                    st.write("---")
                    if st.button("Export Review Card JSON"):
                        st.code(transcript.model_dump_json(include={'review_card', 'student_name', 'session_id'}), language="json")

            if tab3:
                with tab3:
                    st.write("### Independent Blind Evaluation")
                    show_reasoning_state_definitions()
                    st.write("Review the raw transcript below and submit your independent evaluation.")
                    for t in transcript.turns:
                        st.markdown(f"**Assessor:** {t.question}")
                        st.markdown(f"**Student:** {t.student_response}")
                        st.write("---")
                    
                    st.write("#### Submit Rating")
                    judgement = st.selectbox("Your Judgement:", ["Grounded", "Unstable", "Collapsed"])
                    score = st.slider("Rating Score (0.0 to 1.0):", 0.0, 1.0, 0.5)
                    eval_notes = st.text_area("Evaluation Notes:")
                    
                    existing_rating = next((r for r in transcript.evaluator_ratings if r.evaluator_id == labeller_id), None)
                    
                    if existing_rating:
                        st.success(f"You submitted a '{existing_rating.judgement}' rating for this session.")

                        # Only surface the comparison once the rating and the
                        # session it is compared against are both complete - a
                        # partial rating produces a meaningless alignment verdict.
                        if evaluator_rating_is_populated(existing_rating) and transcript.turns:
                            st.write("#### Comparison Matrix")
                            # Calculate alignment rate: AI collapsed states vs Evaluator judgements
                            sys_collapsed = any(t.state == StudentState.COLLAPSED for t in transcript.turns)
                            hum_collapsed = existing_rating.judgement == "Collapsed"

                            alignment = "Aligned" if sys_collapsed == hum_collapsed else "Misaligned"
                            st.metric("System/Human Alignment", alignment)

                            st.write(f"**System flagged collapse?** {sys_collapsed}")
                            st.write(f"**You flagged collapse?** {hum_collapsed}")

                            if not (sys_collapsed == hum_collapsed):
                                st.warning("Disagreement Flagged: Qualitative review required for methodology chapter.")
                    else:
                        if st.button("Submit Blind Evaluation"):
                            rating = EvaluatorRating(
                                session_id=selected_session,
                                evaluator_id=labeller_id,
                                judgement=judgement,
                                rating_score=score,
                                notes=eval_notes,
                                blind=True
                            )
                            transcript.evaluator_ratings.append(rating)
                            StorageManager.save_transcript(transcript)
                            st.success("Evaluation submitted!")
                            st.rerun()

            if tab4:
                with tab4:
                    st.write("### Faculty Experiment Rating (Double-Blind)")
                    st.markdown("Evaluate the system's qualitative performance and provide a Ground Truth grade to calibrate the orchestration parameters.")

                    if transcript.experiment_profile:
                        st.success("🔒 Experimental Profile Assigned (Blind - profile name hidden from faculty)")

                    rubric_default = """| Score Range | Grade | Description |
|---|---|---|
| 0.0 - 0.4 | **F/D** | Unresolved scope; limited originality; basic/incomplete plan. |
| 0.5 - 0.6 | **C** | Adequate/functional but lacks high innovation; reasonable progress. |
| 0.7 | **B** | Mostly meets excellent criteria. |
| 0.8 - 0.9 | **A** | Highly creative; clear/derived aims; detailed realistic plan; strong progress. |
| 1.0 | **A\\*** | Exceptionally goes above and beyond excellent criteria. |"""

                    if not transcript.turns:
                        st.warning("This session has no completed turns.")
                    else:
                        for idx, turn in enumerate(transcript.turns):
                            with st.expander(f"Turn {idx+1} - Claim: {turn.claim_id}", expanded=False):
                                source_claim = resolve_claim_for_turn(transcript, turn)
                                if source_claim:
                                    st.markdown(f"**Claim Under Test:** \"{source_claim.text}\"")
                                    st.markdown(f"*Source document context:* \"{source_claim.source_passage}\"")
                                    st.markdown("---")
                                else:
                                    st.warning(f"No claim found in the epistemic map for claim_id '{turn.claim_id}'. Grading without source context.")

                                st.markdown(f"**Assessor Question:** {turn.question}")
                                # Response source label for faculty
                                source_emoji = "🤖" if turn.response_source.value == "Advocate" else "🔀" if turn.response_source.value == "Hybrid" else "❔"
                                source_label = f"{source_emoji} {turn.response_source.value}"
                                st.markdown(f"**Student Response:** <small>{source_label}</small>  \n{turn.student_response}", unsafe_allow_html=True)

                                if turn.advocate_similarity is not None:
                                    rewritten_pct = (1.0 - turn.advocate_similarity) * 100.0
                                    with st.expander(
                                        f"🤖 In-app suggestion offered for this turn — {rewritten_pct:.0f}% rewritten",
                                        expanded=False,
                                    ):
                                        st.markdown(f"**Suggestion offered:**\n\n> {turn.advocate_draft}")
                                        st.markdown(f"**Submitted:**\n\n> {turn.student_response}")

                                st.markdown("#### Faculty Evaluation")

                                existing_rating = next((r for r in transcript.experiment_ratings if r.turn_index == idx and r.faculty_id == labeller_id), None)

                                if existing_rating:
                                    st.success(f"Rated by {existing_rating.faculty_id} at {existing_rating.rated_at}")
                                    st.json(existing_rating.model_dump())
                                else:
                                    with st.form(key=f"exp_rating_form_{idx}"):
                                        st.markdown("##### 1. Dialogue Quality")
                                        assessor_halluc = st.toggle("Assessor Hallucination (Did it hallucinate facts?)", value=False, key=f"exp_h_{idx}")
                                        assessor_rep = st.toggle("Assessor Repetition (Did it exactly repeat a previous question?)", value=False, key=f"exp_r_{idx}")

                                        st.markdown("##### 2. Scoring Accuracy")
                                        st.info(f"The Evaluator's raw variance score was: {turn.variance_score:.2f}")
                                        eval_var = st.toggle("Did the Variance score correctly reflect the ambiguity?", value=True, key=f"exp_v_{idx}")

                                        st.markdown("##### 3. Expert Alignment")
                                        expert_align = st.toggle("Did the system's response align with expert pedagogical expectations?", value=True, key=f"exp_ea_{idx}")

                                        st.markdown("##### 4. Advocate Quality (if applicable)")
                                        adv_nov = st.toggle("Advocate Pivot Novelty (Was the pivot creative/novel?)", value=True, key=f"exp_an_{idx}")
                                        adv_rel = st.toggle("Advocate Pivot Relevance (Was the pivot relevant?)", value=True, key=f"exp_ar_{idx}")

                                        st.markdown("##### 5. Faculty Overall Grade")
                                        st.markdown("**Reference Rubric:**")
                                        st.markdown(rubric_default)

                                        grade_options = [
                                            "0.0 (F/D)", "0.1 (F/D)", "0.2 (F/D)", "0.3 (F/D)", "0.4 (F/D)",
                                            "0.5 (C)", "0.6 (C)",
                                            "0.7 (B)",
                                            "0.8 (A)", "0.9 (A)",
                                            "1.0 (A*)"
                                        ]
                                        selected_grade = st.select_slider(
                                            "Provide your overall grade for the student's answer",
                                            options=grade_options,
                                            value="0.5 (C)",
                                            key=f"exp_gt_{idx}"
                                        )
                                        ground_truth = float(selected_grade[:3])

                                        submit = st.form_submit_button("Submit Rating")
                                        if submit:
                                            rating = FacultyExperimentRating(
                                                turn_index=idx,
                                                assessor_hallucination=assessor_halluc,
                                                assessor_repetition=assessor_rep,
                                                evaluator_variance_accurate=eval_var,
                                                expert_alignment=expert_align,
                                                advocate_novel=adv_nov,
                                                advocate_relevant=adv_rel,
                                                ground_truth_score=ground_truth,
                                                faculty_id=labeller_id
                                            )
                                            transcript.experiment_ratings.append(rating)
                                            StorageManager.save_transcript(transcript)
                                            st.rerun()
