import streamlit as st
import sys
import html
from pathlib import Path
import json
from datetime import datetime

# Ensure project root is on Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app_3.storage import StorageManager
from app_3.schemas import StudentState, FacultyLabel, EvaluatorRating, FacultyExperimentRating, ResponseSource
from app_3.config import BASE_DIR
from app_3.wrapper import VivaWrapper


def render_source_pane(document_name: str, claim) -> None:
    """Renders a cropped, highlighted PDF page for a claim, falling back to the
    extracted text passage if the source PDF isn't available locally."""
    st.markdown("#### 📄 Source Document")
    if not claim:
        st.info("No claim context available for this turn.")
        return

    pdf_path = BASE_DIR / document_name
    if not pdf_path.exists():
        st.warning(f"Source PDF '{document_name}' not found in the project root. Showing extracted text passage only.")
        st.markdown(f"> *{claim.source_passage}*")
        return

    try:
        image_bytes = VivaWrapper.extract_claim_image(
            pdf_path=str(pdf_path),
            page_number=claim.page,
            search_text=claim.text
        )
        if image_bytes:
            st.image(image_bytes, caption=f"{document_name} — Page {claim.page}", use_container_width=True)
        else:
            st.info("Could not visually locate the claim text on the page. Showing extracted passage instead.")
            st.markdown(f"> *{claim.source_passage}*")
    except Exception as e:
        st.error(f"Error loading PDF preview: {e}")
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


# Plain HTML Web 1.0 aesthetic style overrides
st.markdown("""
<style>
    /* Reset premium styling for Web 1.0 barebones theme */
    *, *:before, *:after {
        font-family: "Times New Roman", Times, serif !important;
    }
    /* Exempt Streamlit's Material Icons (expander arrows etc.) from the
    blanket font-family override above - without this, icon ligature text
    (e.g. "keyboard_arrow_right") renders as literal overlapping text instead
    of the arrow glyph it's supposed to be. */
    [data-testid="stIconMaterial"] {
        font-family: "Material Symbols Rounded" !important;
    }
    html, body, .stApp {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    /* Force all text elements to black */
    h1, h2, h3, h4, h5, h6, p, label, span, li, td, th {
        color: #000000 !important;
    }
    /* Style buttons to classic Web 1.0 grey boxes */
    button, .stButton > button {
        background-color: #f0f0f0 !important;
        color: #000000 !important;
        border: 1px solid #000000 !important;
        border-radius: 0px !important;
        padding: 4px 12px !important;
    }
    button:hover, .stButton > button:hover {
        background-color: #e0e0e0 !important;
        color: #000000 !important;
        border: 1px solid #000000 !important;
    }
    button:focus, .stButton > button:focus {
        color: #000000 !important;
        background-color: #e0e0e0 !important;
    }
    /* Web 1.0 style borders and tables */
    .box {
        border: 1px solid #000000 !important;
        padding: 12px !important;
        background-color: #f0f0f0 !important;
        color: #000000 !important;
        margin-bottom: 15px !important;
    }
    .box b, .box span, .box p {
        color: #000000 !important;
    }
    .system-badge {
        font-weight: bold !important;
        color: #0000ff !important;
    }
    .review-card-header {
        background-color: #e8e8e8 !important;
        padding: 10px !important;
        border: 2px solid black !important;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Faculty Labelling System (Web 1.0 Edition)")
st.write("---")

# Ethics & Faculty Information Gate
if "faculty_consented" not in st.session_state:
    st.session_state.faculty_consented = False

if not st.session_state.faculty_consented:
    st.markdown("""
    ## Faculty Evaluator Information & Consent

    **Study Title:** Agentic Socratic Assessment for Design Reasoning

    **Evaluator Role:** You are being asked to review and label student reasoning transcripts as part of a research study evaluating the effectiveness of an AI-assisted Socratic viva assessment system.

    ### Your Responsibilities:
    1. **Review student responses** to probing questions about design claims
    2. **Label reasoning states** (Grounded, Unstable, Collapsed) to validate system classifications
    3. **Rate system performance** on aspects like hallucination, repetition, and pivot novelty
    4. **Provide independent evaluations** if assigned to the blind evaluation track
    5. **Maintain confidentiality** of student identifiers and session data

    ### Data Handling:
    - All ratings and labels will be securely stored
    - Session data is used solely for research and system improvement
    - Your evaluation is protected and linked only to your evaluator ID
    - You may stop labelling at any time without penalty

    ### Important Notes:
    - This research is conducted under ethical oversight
    - Your participation is voluntary and informed
    - Any questions about the study may be directed to the research team
    """)

    col_consent1, col_consent2 = st.columns(2)
    with col_consent1:
        consent_ack = st.checkbox("I acknowledge that I understand my role as a faculty evaluator in this study")
    with col_consent2:
        data_ack = st.checkbox("I consent to my ratings being used for research purposes")

    if consent_ack and data_ack:
        if st.button("Proceed to Faculty Labelling", type="primary", use_container_width=True):
            st.session_state.faculty_consented = True
            st.rerun()
    else:
        st.info("Please acknowledge both statements to proceed.")
        st.stop()

st.write("---")

# 1. Capture Labeller ID
col_labeller, col_change = st.columns([4, 1])
with col_labeller:
    labeller_id = st.text_input("Faculty ID:", value="Professor_A").strip()
with col_change:
    if st.button("🔄 Change & Restart", use_container_width=True):
        st.session_state.turn_step = None
        st.session_state.current_session_id = None
        st.rerun()

# 2. Get Available Transcripts & Checkpoints
session_ids = StorageManager.list_available_transcripts()
checkpoints = StorageManager.list_checkpoints()

# Show recovery banner if checkpoints exist
if checkpoints:
    st.warning("⚠️ **Incomplete Sessions Detected** - Sessions saved but not finalized (browser closed without clicking 'Save & Exit')")
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
            st.write(f"**Started At:** {transcript.started_at}")
            
            role = st.selectbox("Login Role:", ["Faculty Assessor", "Independent Evaluator"])
            blind_mode = (role == "Independent Evaluator")
            
            if blind_mode:
                tab1, tab2, tab3, tab4 = st.tabs(["[LOCKED] Data", "[LOCKED] Review Card", "Independent Blind Evaluation", "[LOCKED] Experiment Rating"])
            else:
                tab1, tab2, tab3, tab4 = st.tabs(["Data Labelling", "Faculty Jury Review Card", "[LOCKED] Evaluation", "Experiment Rating"])
            
            with tab1:
                if blind_mode:
                    st.error("ACCESS DENIED: Data Labelling is structurally locked in Independent Evaluator mode.")
                else:
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
                            source_claim = next((c for c in transcript.epistemic_map.claims if c.id == current_turn.claim_id), None)

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
                                source_emoji = "🧑" if current_turn.response_source.value == "Student" else "🤖" if current_turn.response_source.value == "Advocate" else "🔀"
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
                                    if st.button("⬅ Previous Turn", use_container_width=True, disabled=(step == 0)):
                                        st.session_state.turn_step = max(0, step - 1)
                                        st.rerun()
                                with nav_col2:
                                    # Jump to specific turn
                                    jump_to = st.number_input("Jump to turn:", min_value=1, max_value=len(turns_to_label), value=step + 1, step=1)
                                    if st.button("Go", use_container_width=True):
                                        st.session_state.turn_step = jump_to - 1
                                        st.rerun()
                                with nav_col3:
                                    st.markdown("")  # Spacer
                                with nav_col4:
                                    if st.button("Save Label & Next ➡", use_container_width=True, type="primary"):
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
                    
                    total_labels = len(transcript.faculty_labels)
                    agreed_labels = sum(1 for lbl in transcript.faculty_labels if lbl.agrees_with_system)
                    agreement_rate = (agreed_labels / total_labels * 100) if total_labels > 0 else 100.0
                    
                    st.write(f"- **Total Labels Filed:** {total_labels}")
                    st.write(f"- **Human-System Agreement Rate:** {agreement_rate:.1f}%")
                    
                    # Export raw JSON data of labels
                    if st.checkbox("Show Flat JSON Export"):
                        flat_export = []
                        for lbl in transcript.faculty_labels:
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

            with tab2:
                if blind_mode:
                    st.error("ACCESS DENIED: The Jury Review Card is structurally locked in Independent Evaluator mode.")
                else:
                    # Review Card Rendering
                    rc = transcript.review_card
                    if not rc:
                        st.warning("Review Card is not available for this session. It has not been completed or Layer 2 scoring was disabled.")
                    else:
                        st.markdown('<div class="review-card-header"><h2>Jury Review Card</h2></div>', unsafe_allow_html=True)
                        st.write(f"**Generated On:** {rc.created_at}")
                        
                        # Dual-score rendering if Version A and B exist
                        st.write("### Reasoning Foundation")
                        version_a = getattr(rc, "version_a_composite", None)
                        version_b = getattr(rc, "version_b_composite", None)
                        if version_a is not None and version_b is not None:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(label="Version A (Heuristics) Score", value=f"{version_a:.2f}")
                            with col2:
                                st.metric(label="Version B (ML Model) Score", value=f"{version_b:.2f}")
                        else:
                            st.metric(label="Composite Confidence Score", value=f"{rc.composite_confidence:.2f}")
                            
                        if rc.session_notes:
                            st.write("---")
                            st.write("### Session Notes")
                            st.write(rc.session_notes)

                        # Response source summary for faculty
                        student_count = sum(1 for t in transcript.turns if t.response_source == ResponseSource.STUDENT)
                        advocate_count = sum(1 for t in transcript.turns if t.response_source == ResponseSource.ADVOCATE)
                        hybrid_count = sum(1 for t in transcript.turns if t.response_source == ResponseSource.HYBRID)
                        total_turns = len(transcript.turns)

                        if total_turns > 0:
                            st.write("---")
                            st.write("### Response Source Summary")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("🧑 Student-Only", student_count, f"{student_count/total_turns*100:.0f}%")
                            with col2:
                                st.metric("🤖 AI-Suggested", advocate_count, f"{advocate_count/total_turns*100:.0f}%")
                            with col3:
                                st.metric("🔀 Hybrid/Edited", hybrid_count, f"{hybrid_count/total_turns*100:.0f}%")
                            st.caption("Shows whether student answers were originally student-written, AI-suggested, or a combination.")

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

                        st.write("---")
                        if st.button("Export Review Card JSON"):
                            st.code(transcript.model_dump_json(include={'review_card', 'student_name', 'session_id'}), language="json")

            with tab3:
                st.write("### Independent Blind Evaluation")
                if not blind_mode:
                    st.error("ACCESS DENIED: Please login as an Independent Evaluator to access blind evaluation.")
                else:
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

            with tab4:
                st.write("### Faculty Experiment Rating (Double-Blind)")
                if blind_mode:
                    st.error("ACCESS DENIED: Experiment Rating is structurally locked in Independent Evaluator mode.")
                else:
                    st.markdown("Evaluate the Socratic AI's qualitative performance and provide a Ground Truth grade to calibrate the orchestration parameters.")

                    if transcript.experiment_profile:
                        st.success("🔒 Experimental Profile Assigned (Blind - profile name hidden from faculty)")
                    else:
                        st.warning("No experimental profile attached to this session.")

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
                                source_claim = next((c for c in transcript.epistemic_map.claims if c.id == turn.claim_id), None)
                                if source_claim:
                                    st.markdown(f"**Claim Under Test:** \"{source_claim.text}\"")
                                    st.markdown(f"*Source document context:* \"{source_claim.source_passage}\"")
                                    st.markdown("---")
                                else:
                                    st.warning(f"No claim found in the epistemic map for claim_id '{turn.claim_id}'. Grading without source context.")

                                st.markdown(f"**Assessor Question:** {turn.question}")
                                # Response source label for faculty
                                source_emoji = "🧑" if turn.response_source.value == "Student" else "🤖" if turn.response_source.value == "Advocate" else "🔀"
                                source_label = f"{source_emoji} {turn.response_source.value}"
                                st.markdown(f"**Student Response:** <small>{source_label}</small>  \n{turn.student_response}", unsafe_allow_html=True)

                                st.markdown("#### Faculty Evaluation")

                                existing_rating = next((r for r in transcript.experiment_ratings if r.turn_index == idx), None)

                                if existing_rating:
                                    st.success(f"Rated by {existing_rating.faculty_id} at {existing_rating.rated_at}")
                                    st.json(existing_rating.model_dump())
                                else:
                                    with st.form(key=f"exp_rating_form_{idx}"):
                                        st.markdown("##### 1. AI Dialogue Quality")
                                        assessor_halluc = st.toggle("Assessor Hallucination (Did it hallucinate facts?)", value=False, key=f"exp_h_{idx}")
                                        assessor_rep = st.toggle("Assessor Repetition (Did it exactly repeat a previous question?)", value=False, key=f"exp_r_{idx}")

                                        st.markdown("##### 2. Scoring Accuracy")
                                        st.info(f"The Evaluator's raw variance score was: {turn.variance_score:.2f}")
                                        eval_var = st.toggle("Did the Variance score correctly reflect the ambiguity?", value=True, key=f"exp_v_{idx}")

                                        st.markdown("##### 3. Advocate Quality (if applicable)")
                                        adv_nov = st.toggle("Advocate Pivot Novelty (Was the pivot creative/novel?)", value=True, key=f"exp_an_{idx}")
                                        adv_rel = st.toggle("Advocate Pivot Relevance (Was the pivot relevant?)", value=True, key=f"exp_ar_{idx}")

                                        st.markdown("##### 4. Faculty Overall Grade")
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
                                                advocate_novel=adv_nov,
                                                advocate_relevant=adv_rel,
                                                ground_truth_score=ground_truth,
                                                faculty_id=labeller_id
                                            )
                                            transcript.experiment_ratings.append(rating)
                                            StorageManager.save_transcript(transcript)
                                            st.rerun()
