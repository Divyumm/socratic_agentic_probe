import streamlit as st
import sys
import os
import html
from pathlib import Path
from typing import Optional
import json
import base64

# Ensure project root is on Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app_3.wrapper import VivaWrapper
from app_3.schemas import EpistemicMap, Claim, PapanekDimension, StudentState, InterventionType, ResponseSource
from app_3.config import BASE_DIR
from difflib import SequenceMatcher

def _composite_to_letter_grade(composite_score: float) -> str:
    """Convert composite confidence score (0-1) to letter grade for student feedback."""
    if composite_score >= 0.85:
        return "A"
    elif composite_score >= 0.75:
        return "B"
    elif composite_score >= 0.65:
        return "C"
    elif composite_score >= 0.50:
        return "D"
    else:
        return "F"

# Set page configuration with a modern title and icon
st.set_page_config(
    page_title="Socratic Viva Assessment",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Plain HTML Web 1.0 aesthetic style overrides
st.markdown("""
<style>
    /* Reset premium styling for Web 1.0 barebones theme */
    html, body, .stApp {
        font-family: "Times New Roman", Times, serif;
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    /* Force all text elements to black, but don't force font-family on everything to save icons */
    h1, h2, h3, h4, h5, h6, p, label, li, td, th {
        color: #000000 !important;
    }
    span {
        color: #000000;
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
    
    /* Header card */
    .header-card {
        border: 2px double #000000;
        padding: 15px;
        margin-bottom: 20px;
        text-align: center;
        background-color: #f0f0f0;
    }
    .header-title {
        color: #000000 !important;
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .header-subtitle {
        color: #333333;
        font-size: 1rem;
    }
    
    /* Card containers */
    .glass-card {
        border: 1px solid #000000;
        padding: 15px;
        margin-bottom: 12px;
        background-color: #fafafa;
    }
    
    .claim-card {
        border-left: 6px solid #000000;
        background-color: #f5f5f5;
    }
    
    /* State badges */
    .badge {
        padding: 4px 8px;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
        border: 1px solid #000000;
    }
    .badge-grounded {
        background-color: #e0ffe0;
        color: #008000 !important;
    }
    .badge-unstable {
        background-color: #fffae0;
        color: #b8860b !important;
    }
    .badge-collapsed {
        background-color: #ffe0e0;
        color: #ff0000 !important;
    }
    .badge-paused {
        background-color: #f0f0f0;
        color: #555555 !important;
    }
    
    /* Chat layout styling */
    .chat-bubble {
        padding: 10px 12px;
        margin-bottom: 10px;
        max-width: 90%;
        line-height: 1.4;
    }
    .assessor-bubble {
        background-color: #f2f2f2;
        color: #000000 !important;
        border: 1px solid #999999;
        border-left: 4px solid #000000;
    }
    .student-bubble {
        background-color: #ffffff;
        color: #000000 !important;
        border: 2px double #000000;
        margin-left: auto;
    }
    .advocate-hint-bubble {
        background-color: #fcf0ff;
        color: #000000 !important;
        border: 1px dotted #800080;
        font-style: italic;
    }
    
    /* Custom metric display */
    .metric-value {
        font-size: 1.6rem;
        font-weight: bold;
        color: #000000 !important;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #555555;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State Variables
if "consent_approved" not in st.session_state:
    st.session_state.consent_approved = False
if "participant_id" not in st.session_state:
    st.session_state.participant_id = ""
if "epistemic_map" not in st.session_state:
    st.session_state.epistemic_map = None
if "session_manager" not in st.session_state:
    st.session_state.session_manager = None
if "advocate_suggestion" not in st.session_state:
    st.session_state.advocate_suggestion = ""
if "advocate_was_generated" not in st.session_state:
    st.session_state.advocate_was_generated = False
if "advocate_pivot" not in st.session_state:
    st.session_state.advocate_pivot = None
if "viva_active" not in st.session_state:
    st.session_state.viva_active = False
if "last_turn_result" not in st.session_state:
    st.session_state.last_turn_result = None
if "next_prompt" not in st.session_state:
    st.session_state.next_prompt = ""
if "response_input" not in st.session_state:
    st.session_state.response_input = ""
if "student_resp_key" not in st.session_state:
    st.session_state.student_resp_key = ""
if "clear_text_area" not in st.session_state:
    st.session_state.clear_text_area = False

# Clear text area before widget instantiation if flag is set
if st.session_state.clear_text_area:
    st.session_state.student_resp_key = ""
    st.session_state.clear_text_area = False

if "show_challenge" not in st.session_state:
    st.session_state.show_challenge = False
if "challenge_response" not in st.session_state:
    st.session_state.challenge_response = ""
if "show_clarification" not in st.session_state:
    st.session_state.show_clarification = False
if "fallback_warning" not in st.session_state:
    st.session_state.fallback_warning = None

# App Title Header
st.markdown("""
<div class="header-card">
    <div class="header-title">Agentic Socratic Assessment</div>
    <div class="header-subtitle">Design Engineering Viva Interface</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SCREEN 1: Consent Gate & ID Entry
# ==========================================
if not st.session_state.consent_approved:
    st.subheader("Ethics Consent Gate & Participant Registration")
    
    st.markdown("""
    ### Participant Information & Consent Form
    **Study Title**: Agentic Socratic Assessment for Design Reasoning
    
    **Supervisor**: Andrew Brand 
    **Student Researcher**: Divyum Maheshwari 
    **Institution**: Imperial College London
    
    Please read the following details carefully before proceeding:
    1. **Purpose**: The purpose of this system is to analyze design reasoning and assess how students justify coursework claims.
    2. **Weights & Parameters Simulation**: You will be able to adjust the scoring weights of the evaluation model (Coherence, Grounding, Variance, Circularity) and see state transitions in real time.
    3. **Advocate Co-creation**: You can co-create responses with an AI Advocate Agent who acts as your defense counsel.
    4. **Data Handling & Anonymity**: All inputs (text entries, dynamic weights, response evaluation metrics, and session timestamps) are anonymized.
    5. **Withdrawal**: You are free to stop the session at any time.
    """)
    
    consent_check = st.checkbox("I consent to participate in this study under the terms outlined above.")
    part_id = st.text_input("Enter Anonymised Participant ID (e.g., P-03, Student-05):", placeholder="P-01").strip()
    
    if st.button("Unlock Assessment Interface", type="primary"):
        if not consent_check:
            st.error("You must agree to the consent form before proceeding.")
        elif not part_id:
            st.error("Please enter a Participant ID to maintain anonymity.")
        else:
            st.session_state.consent_approved = True
            st.session_state.participant_id = part_id
            st.rerun()

# ==========================================
# SCREEN 2: Document Uploader & Setup
# ==========================================
elif st.session_state.epistemic_map is None:
    st.subheader("Document Setup & Epistemic Mapping")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4>Option A: Select Pre-processed Epistemic Map</h4>
            <p>Load an existing document map extracted from prior coursework.</p>
        </div>
        """, unsafe_allow_html=True)
        
        maps = VivaWrapper.list_available_maps()
        if maps:
            selected_map = st.selectbox("Select course document map:", maps)
            if st.button("Load Selected Document", type="primary"):
                loaded_map = VivaWrapper.load_map(selected_map)
                if loaded_map:
                    st.session_state.epistemic_map = loaded_map
                    st.session_state.session_manager = VivaWrapper.start_session(loaded_map, st.session_state.participant_id)
                    st.session_state.viva_active = True
                    manager = st.session_state.session_manager
                    active_c = manager.active_claim
                    raw_q = manager.get_current_question()
                    st.session_state.next_prompt = manager.teleprompter.translate_prompt_v2(
                        raw_q, StudentState.GROUNDED, active_c
                    ) if active_c else raw_q

                    n_fallback = sum(1 for c in loaded_map.claims if c.text.startswith("Fallback extracted claim"))
                    st.session_state.fallback_warning = (
                        f"{n_fallback} of {len(loaded_map.claims)} claims in this saved map are generic "
                        "placeholder claims from a failed extraction, not real content from the document."
                    ) if n_fallback else None
                    st.success(f"Successfully loaded {selected_map}")
                    st.rerun()
                else:
                    st.error("Failed to load map file.")
        else:
            st.info("No pre-processed maps found. Please parse a new PDF coursework file.")

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4>Option B: Parse New Coursework PDF</h4>
            <p>Extract claims and build a new Epistemic Map dynamically using LLM extraction.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Also check local files
        local_pdfs = VivaWrapper.list_available_pdfs()
        local_pdf_names = [p.name for p in local_pdfs]
        
        selected_local_pdf = st.selectbox("Choose a PDF in project folder:", ["-- Select file --"] + local_pdf_names)
        
        uploaded_file = st.file_uploader("Or upload coursework PDF:", type="pdf")
        
        pdf_to_parse = None
        if uploaded_file is not None:
            # Save uploaded file temporarily to root
            temp_path = BASE_DIR / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            pdf_to_parse = str(temp_path)
        elif selected_local_pdf != "-- Select file --":
            pdf_to_parse = str(BASE_DIR / selected_local_pdf)

        if pdf_to_parse and st.button("Parse Coursework and Extract Map"):
            try:
                with st.status("Building epistemic map...", expanded=True) as status:
                    status.update(label="📄 Parsing document...", state="running")
                    parsed_map = VivaWrapper.parse_pdf_to_map(pdf_to_parse)

                    status.update(label="🎯 Extracting key claims...", state="running")
                    # (claims already extracted above, just visual update)

                    status.update(label="🔗 Building reasoning map...", state="running")
                    st.session_state.epistemic_map = parsed_map

                    status.update(label="⚙️ Initializing session...", state="running")
                    st.session_state.session_manager = VivaWrapper.start_session(parsed_map, st.session_state.participant_id)
                    st.session_state.viva_active = True
                    manager = st.session_state.session_manager
                    active_c = manager.active_claim
                    raw_q = manager.get_current_question()
                    st.session_state.next_prompt = manager.teleprompter.translate_prompt_v2(
                        raw_q, StudentState.GROUNDED, active_c
                    ) if active_c else raw_q

                    n_fallback = sum(1 for c in parsed_map.claims if c.text.startswith("Fallback extracted claim"))
                    st.session_state.fallback_warning = (
                        f"{n_fallback} of {len(parsed_map.claims)} claims could not be extracted by the "
                        "LLM (API error, quota, or malformed response) and fell back to generic placeholder "
                        "claims and questions. These will read as vague and won't reflect your document's "
                        "actual content. Consider re-parsing the document."
                    ) if n_fallback else None

                    status.update(label="✅ Map ready!", state="complete")
                    st.success("PDF parsed and Epistemic Map successfully created!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error parsing PDF: {e}")

# ==========================================
# SCREEN 3: Socratic Probing Viva Session
# ==========================================
else:
    manager = st.session_state.session_manager
    active_claim = manager.active_claim

    if st.session_state.fallback_warning:
        st.warning(st.session_state.fallback_warning)

    # ------------------------------------------
    with st.sidebar:
        st.header("Session Controls")

        # Debug mode toggle
        if "debug_mode" not in st.session_state:
            st.session_state.debug_mode = True  # Enable debug mode by default
        st.session_state.debug_mode = st.toggle("🐛 Debug Mode (show scores & badges)", value=st.session_state.debug_mode)

        # Pause
        if st.button("Pause Session", use_container_width=True):
            turn, msg = manager.submit_response("[Intervention: Paused]", InterventionType.PAUSE)
            st.session_state.last_turn_result = turn
            st.session_state.next_prompt = msg

            # Auto-save checkpoint before pause
            checkpoint_transcript = manager.build_transcript(notes=f"Auto-save checkpoint (paused). Participant ID: {st.session_state.participant_id}")
            checkpoint_path = BASE_DIR / "data" / "processed" / f"checkpoint_{manager.session_id}.json"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                f.write(checkpoint_transcript.model_dump_json(indent=2))

            st.info("Session paused.")
            
        # Exit and Save
        if st.button("Save & Exit Viva", use_container_width=True, type="primary"):
            transcript = manager.build_transcript(notes=f"Student-calibrated live run. Participant ID: {st.session_state.participant_id}")
            saved_path = VivaWrapper.save_transcript(transcript)

            # Calculate final grade based on average composite confidence
            if transcript.turns:
                avg_composite = sum(t.composite_confidence for t in transcript.turns) / len(transcript.turns)
                final_grade = _composite_to_letter_grade(avg_composite)
                grade_color = "#2ecc71" if final_grade in ["A", "B"] else "#f39c12" if final_grade == "C" else "#e74c3c"
            else:
                avg_composite = 0.0
                final_grade = "F"
                grade_color = "#e74c3c"

            st.session_state.epistemic_map = None
            st.session_state.session_manager = None
            st.session_state.viva_active = False
            st.session_state.last_turn_result = None

            st.success(f"Transcript saved to {saved_path.name}")
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; border: 3px solid {grade_color}; border-radius: 8px; margin: 20px 0;">
                <h3>Final Grade: <span style="color: {grade_color}; font-size: 48px; font-weight: bold;">{final_grade}</span></h3>
                <p>Average reasoning confidence: {avg_composite:.1%}</p>
                <p style="font-size: 14px; color: #666;">Your viva assessment is complete. Faculty will review your reasoning depth, and you'll receive detailed feedback on your design rationale.</p>
            </div>
            """, unsafe_allow_html=True)
            st.info("Exited session. Thank you!")
            st.rerun()

    # ------------------------------------------
    # MAIN AREA: Interactive Socratic Dialogue
    # ------------------------------------------
    
    # Active Claim card
    if active_claim:
        if not manager.turns:
            state_label = "AWAITING RESPONSE"
            badge_class = "badge-paused"
        else:
            state_label = manager.session_state.value.upper()
            badge_class = "badge-grounded"
            if manager.session_state == StudentState.UNSTABLE:
                badge_class = "badge-unstable"
            elif manager.session_state == StudentState.COLLAPSED:
                badge_class = "badge-collapsed"
            elif manager.session_state == StudentState.PAUSED:
                badge_class = "badge-paused"

        # Clean fallback claims prefix
        import re
        clean_text = re.sub(r"^Fallback extracted claim \([^)]+\):\s*", "", active_claim.text)
        clean_text = html.escape(clean_text)

        # Translate dimension to friendly student mask
        dim_label = "Dynamic Theme" if active_claim.dimension == PapanekDimension.EMERGENT else "Design Aspect"
        if active_claim.dimension == PapanekDimension.EMERGENT:
            # emergent_theme is Optional - the LLM doesn't always populate it
            # despite being instructed to, so fall back rather than crash.
            dim_val = active_claim.emergent_theme or "the situational context of your work"
        else:
            dim_val = manager.teleprompter.DIMENSION_MASK_MAP.get(active_claim.dimension, active_claim.dimension.value)
        dim_val = html.escape(dim_val)

        # Truncate source passage for the quick preview card
        preview_context = active_claim.source_passage
        if len(preview_context) > 120:
            preview_context = preview_context[:120] + "..."

        st.markdown(f"""
        <div class="glass-card claim-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 700; font-size: 1.1rem; color: #8b5cf6;">ACTIVE CLAIM: {html.escape(active_claim.id)}</span>
                <span class="badge {badge_class}">{state_label}</span>
            </div>
            <p style="font-size: 1rem; color: #000000; margin-bottom: 8px;"><b>Extracted Claim:</b> "{clean_text}"</p>
            <div style="display: flex; gap: 20px; font-size: 0.85rem; color: #555555;">
                <span><b>{dim_label}:</b> {dim_val}</span>
                <span><b>Probing Depth:</b> {manager.active_depth} / 3</span>
                <span><b>Vulnerability Rank:</b> {active_claim.vulnerability_rank}</span>
                <span><b>Source Page:</b> PDF Page {active_claim.page} (Physical)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="glass-card">
            <h4>All Claims Justified!</h4>
            <p>Congratulations, your reasoning stood strong across all design claims and dimensions.</p>
        """, unsafe_allow_html=True)

    # Chat Log & Metrics display (full width, scrolls together)
    with st.container():
        st.subheader("Questions")
        
        # Display past turns in chat bubbles
        for turn in manager.turns:
            if turn.intervention_type == InterventionType.PAUSE:
                st.markdown('<div class="chat-bubble student-bubble" style="background-color: #374151; border-color: #4b5563;">[System Action: Session Paused]</div>', unsafe_allow_html=True)
                continue
            
            # Find the dimension of the claim at that turn for accurate translation
            turn_claim = next((c for c in manager.epistemic_map.claims if c.id == turn.claim_id), None)
            turn_dim = turn_claim.dimension if turn_claim else PapanekDimension.NEED
            turn_theme = turn_claim.emergent_theme if turn_claim else None
            translated_question = manager.teleprompter.translate_prompt_v2(turn.question, turn.state, manager.epistemic_map.get_claim(turn.claim_id) if manager.epistemic_map else None)

            st.markdown(f'<div class="chat-bubble assessor-bubble"><b>Assessor (Turn {turn.turn_index}):</b> {html.escape(translated_question)}</div>', unsafe_allow_html=True)

            # Response with source label and badges
            col_resp_label, col_resp_badges = st.columns([3, 1])
            with col_resp_label:
                source_emoji = "🧑" if turn.response_source.value == "Student" else "🤖" if turn.response_source.value == "Advocate" else "🔀"
                st.markdown(f'<div class="chat-bubble student-bubble"><b>Student:</b> {html.escape(turn.student_response)} <small>[{source_emoji} {turn.response_source.value}]</small></div>', unsafe_allow_html=True)

            # Debug mode: show rubric score badges
            if st.session_state.debug_mode and turn.rubric_scores:
                with col_resp_badges:
                    for score in turn.rubric_scores:
                        # Show badge if above threshold (0.65 = "good" level)
                        construct_name = score.rubric_construct.value.replace("_", " ")
                        if score.score >= 0.65:
                            if "Empathy" in construct_name:
                                st.success(f"✓ {construct_name.split()[0]} ({score.score:.2f})")
                            elif "Internalisation" in construct_name:
                                st.success(f"✓ {construct_name.split()[0]} ({score.score:.2f})")
                            elif "Confidence" in construct_name:
                                st.success(f"✓ Conviction ({score.score:.2f})")

            if turn.reconstruction_dimension:
                masked_recon = html.escape(manager.teleprompter.DIMENSION_MASK_MAP.get(turn.reconstruction_dimension, turn.reconstruction_dimension.value))
                st.markdown(f'<div class="chat-bubble advocate-hint-bubble"><small>→ Pivot to {masked_recon}</small></div>', unsafe_allow_html=True)

        # Active turn Socratic question
        if active_claim:
            st.markdown(f'<div class="chat-bubble assessor-bubble"><b>Assessor (Current Probing):</b> {html.escape(st.session_state.next_prompt)}</div>', unsafe_allow_html=True)

        st.markdown("---")
        
        # ------------------------------------------
        # STUDENT INPUT & ADVOCATE CO-CREATION HUB
        # ------------------------------------------
        st.subheader("Your Justification Hub")

        # Advocate Helper Section (one-time only per question)
        if active_claim:
            adv_col1, adv_col2 = st.columns([2, 1])
            with adv_col2:
                # Disable button if advocate was already generated for this question
                advocate_disabled = st.session_state.get("advocate_was_generated", False)
                button_label = "✓ Advocate Suggestion Ready" if advocate_disabled else "Generate Brainstorming Suggestions"

                if st.button(button_label, type="secondary", use_container_width=True, disabled=advocate_disabled):
                    with st.spinner("Generating rough brainstorming suggestions..."):
                        suggestion, pivot_dim = VivaWrapper.get_advocate_defense(
                            claim=active_claim,
                            question=st.session_state.next_prompt,
                            depth=manager.active_depth,
                            advocate_temp=manager.experiment_profile.advocate_temp if manager.experiment_profile else None
                        )
                        st.session_state.advocate_suggestion = suggestion
                        st.session_state.advocate_pivot = pivot_dim
                        st.session_state.advocate_was_generated = True  # Flag that advocate was used (disables button)
                        st.session_state.response_input = suggestion # Pre-populate student answer box
                        st.session_state.student_resp_key = suggestion # Bind directly to text area key to force refresh
                        st.rerun()  # Refresh immediately to show disabled button and suggestions

            # The advocate suggestion text is pre-populated in the text area below.
            # Student response submission form
            response_input_text = st.text_area(
                "Write or co-create your justification (edit the template below):", 
                key="student_resp_key",
                height=300
            )
            
            # Synchronize input to state
            st.session_state.response_input = response_input_text
            
            col_b1, col_b2, col_b3 = st.columns([2, 1, 1])
            with col_b1:
                # Disable button if response already submitted (wait state)
                submit_disabled = st.session_state.get("response_submitted_waiting", False)

                if st.button(
                    "Submit Rationale to Assessor",
                    type="primary",
                    use_container_width=True,
                    disabled=submit_disabled
                ):
                    if not response_input_text.strip():
                        st.error("⚠️ Please enter a response before submitting.")
                    else:
                        # Mark as submitted to disable button during processing
                        st.session_state.response_submitted_waiting = True

                        # Show processing status
                        with st.status("Processing your response...", expanded=True) as status:
                            status.update(label="📝 Received your response", state="running")

                            # Compute features and scores (this takes a few seconds)
                            turn, next_prompt = manager.submit_response(
                                response=response_input_text
                            )

                            status.update(label="✅ Response fully evaluated", state="complete")

                        # Detect response source based on whether advocate was generated
                        if st.session_state.advocate_was_generated:
                            similarity = SequenceMatcher(None, response_input_text.lower(), st.session_state.advocate_suggestion.lower()).ratio()
                            if similarity >= 0.85:
                                turn.response_source = ResponseSource.ADVOCATE
                            elif similarity >= 0.50:
                                turn.response_source = ResponseSource.HYBRID
                            else:
                                turn.response_source = ResponseSource.STUDENT
                        else:
                            turn.response_source = ResponseSource.STUDENT

                        st.session_state.last_turn_result = turn
                        st.session_state.next_prompt = next_prompt

                        # Auto-save checkpoint after each turn
                        checkpoint_transcript = manager.build_transcript(notes=f"Auto-save checkpoint. Participant ID: {st.session_state.participant_id}")
                        checkpoint_path = BASE_DIR / "data" / "processed" / f"checkpoint_{manager.session_id}.json"
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(checkpoint_path, "w", encoding="utf-8") as f:
                            f.write(checkpoint_transcript.model_dump_json(indent=2))

                        # Reset helper variables for next question
                        st.session_state.advocate_suggestion = ""
                        st.session_state.advocate_pivot = None
                        st.session_state.advocate_was_generated = False
                        st.session_state.response_input = ""
                        st.session_state.clear_text_area = True
                        st.session_state.show_challenge = False
                        st.session_state.response_submitted_waiting = False  # Re-enable button

                        st.success("✅ Your response has been recorded and evaluated. Loading next question...")
                        st.rerun()  # Immediately show next question

            with col_b2:
                if st.button("Challenge Score Decision", use_container_width=True, disabled=(not manager.turns)):
                    st.session_state.show_challenge = True
                    st.session_state.challenge_response = manager.handle_challenge()
                    st.rerun()

            with col_b3:
                if st.button("❓ Clarify Question", use_container_width=True, disabled=(not st.session_state.next_prompt)):
                    st.session_state.show_clarification = True
                    st.rerun()

        # Clarification request dialog
        if st.session_state.get("show_clarification", False) and st.session_state.next_prompt:
            st.info("""
            **Question Clarification:**

            The question above is asking you to:
            1. **Explain your reasoning** for the specific design choice
            2. **Ground your answer** in the original document or design principles
            3. **Show how** this claim connects to the broader context of your project

            If you're still uncertain, you can:
            - Re-read the claim and question carefully
            - Use the "Generate Advocate Defense Helper" to see alternative angles
            - Take a moment to think, then try rewording your answer

            If you remain unsure, document what you're uncertain about in your response.
            """)
            if st.button("Got it, ready to answer", use_container_width=True):
                st.session_state.show_clarification = False
                st.rerun()

        st.markdown("---")

        # ------------------------------------------
        # METRICS DISPLAY PANEL (Debug Mode Only)
        # ------------------------------------------
        if st.session_state.debug_mode:
            st.subheader("Real-Time Reasoning Signals")

            last_turn = st.session_state.last_turn_result
            if last_turn:
                st.markdown(f"#### Evaluation for Turn {last_turn.turn_index}")

                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.markdown(f"""
                    <div class="glass-card" style="text-align: center;">
                        <div class="metric-value">{last_turn.coherence_score:.2f}</div>
                        <div class="metric-label">Evaluator NLI</div>
                    </div>
                    <div class="glass-card" style="text-align: center;">
                        <div class="metric-value">{last_turn.circularity_score:.2f}</div>
                        <div class="metric-label">Auditor NLI</div>
                    </div>
                    """, unsafe_allow_html=True)

                with m_col2:
                    st.markdown(f"""
                    <div class="glass-card" style="text-align: center;">
                        <div class="metric-value">{last_turn.grounding_score:.2f}</div>
                        <div class="metric-label">Advocate NLI</div>
                    </div>
                    <div class="glass-card" style="text-align: center;">
                        <div class="metric-value">{last_turn.variance_score:.2f}</div>
                        <div class="metric-label">Variance (MC)</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Composite Confidence Score gauge
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; border-color: rgba(139, 92, 246, 0.4);">
                    <div class="metric-value" style="font-size: 2.5rem; color: #a78bfa;">{last_turn.composite_confidence:.2f}</div>
                    <div class="metric-label" style="font-weight: 600;">COMPOSITE CONFIDENCE SCORE</div>
                </div>
                """, unsafe_allow_html=True)


            else:
                st.info("Submit your first response to see the real-time reasoning metrics dashboard.")
        else:
            st.info("💡 Enable Debug Mode in Session Controls (sidebar) to see real-time reasoning signals.")
            
        # Display the explanation challenge context
        if st.session_state.show_challenge:
            st.markdown("---")
            st.subheader("Challenge This Score")
            st.write(st.session_state.challenge_response)

            challenge_justification = st.text_input(
                "Optional: explain your reasoning for disputing this score (faculty will see this):",
                placeholder="My response is grounded because... (leave blank if you'd rather not add anything)"
            )
            if st.button("Submit Challenge"):
                manager.record_challenge(
                    system_explanation=st.session_state.challenge_response,
                    justification=challenge_justification,
                )
                st.success("Challenge logged for faculty review.")
                st.session_state.show_challenge = False
                st.session_state.challenge_response = ""
                st.rerun()
                    
        # Reference Context Pane (Custom Collapsible) - Only show if there's an active claim
        if active_claim:
            st.markdown("---")
            if "pdf_pane_open" not in st.session_state:
                st.session_state.pdf_pane_open = True

            col1, col2 = st.columns([0.05, 0.95])
            with col1:
                if st.button("▼" if st.session_state.pdf_pane_open else "▶", key="pdf_toggle", use_container_width=True):
                    st.session_state.pdf_pane_open = not st.session_state.pdf_pane_open
                    st.rerun()
            with col2:
                st.markdown("**📄 Source Document Reference**")

            if st.session_state.pdf_pane_open:
                pdf_path = BASE_DIR / manager.epistemic_map.document_name
                if pdf_path.exists():
                    try:
                        # Attempt to extract a cropped image of the specific text
                        image_bytes = VivaWrapper.extract_claim_image(
                            pdf_path=str(pdf_path),
                            page_number=active_claim.page,
                            search_text=active_claim.text
                        )
                        if image_bytes:
                            with st.container(height=600):
                                st.image(image_bytes, caption=f"Extracted from {manager.epistemic_map.document_name} (Physical Page {active_claim.page})", use_container_width=True)
                            st.caption("💡 Tip: Click the arrow above (▼) to collapse this panel and return to full dialogue view.")
                        else:
                            st.info("Could not visually locate the text on the page. Displaying extracted text block instead.")
                            st.markdown(f"> *{active_claim.source_passage}*")
                            st.caption("💡 Tip: Click the arrow above (▼) to collapse this panel.")
                    except Exception as e:
                        st.error(f"Error loading PDF preview: {e}")
                else:
                    st.warning(f"Source PDF '{manager.epistemic_map.document_name}' not found in the project root directory. Context pane unavailable.")
