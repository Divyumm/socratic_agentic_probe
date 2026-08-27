# DEPRECATED: this standalone portal has been merged into labelling_app.py's
# "Experiment Rating" tab, so faculty no longer need to pick a session twice
# across two separate apps. This file is kept only for reference during the
# transition - run `streamlit run app_2/labelling_app.py` for everything now.
import streamlit as st
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app_3.storage import StorageManager
from app_3.schemas import FacultyExperimentRating

st.set_page_config(page_title="Faculty Experiment Portal", layout="wide")

st.title("🧪 Faculty Evaluation Portal (Double-Blind)")
st.markdown("Evaluate the Socratic AI's qualitative performance and provide Ground Truth grades to calibrate the orchestration parameters.")

# Load transcripts
transcripts = StorageManager.list_available_transcripts()
if not transcripts:
    st.info("No session transcripts found. Have students run the viva app first.")
    st.stop()

session_id = st.sidebar.selectbox("Select Student Session", transcripts)
transcript = StorageManager.load_transcript(session_id)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Student:** {transcript.student_name}")
st.sidebar.markdown(f"**Document:** {transcript.document_name}")
# Hide the profile name from the faculty!
if transcript.experiment_profile:
    st.sidebar.success("🔒 Experimental Profile Assigned (Blind)")
else:
    st.sidebar.warning("No experimental profile attached to this session.")

faculty_id = st.sidebar.text_input("Your Faculty ID", value="Faculty-1")

rubric_default = """| Score Range | Grade | Description |
|---|---|---|
| 0.0 - 0.4 | **F/D** | Unresolved scope; limited originality; basic/incomplete plan. |
| 0.5 - 0.6 | **C** | Adequate/functional but lacks high innovation; reasonable progress. |
| 0.7 | **B** | Mostly meets excellent criteria. |
| 0.8 - 0.9 | **A** | Highly creative; clear/derived aims; detailed realistic plan; strong progress. |
| 1.0 | **A\*** | Exceptionally goes above and beyond excellent criteria. |"""

if not transcript.turns:
    st.warning("This session has no completed turns.")
    st.stop()

st.header("Session Dialogue & Evaluation")

# Display each turn and a rating form
for i, turn in enumerate(transcript.turns):
    with st.expander(f"Turn {i+1} - Claim: {turn.claim_id}", expanded=True):
        source_claim = next((c for c in transcript.epistemic_map.claims if c.id == turn.claim_id), None)
        if source_claim:
            st.markdown(f"**Claim Under Test:** \"{source_claim.text}\"")
            st.markdown(f"*Source document context:* \"{source_claim.source_passage}\"")
            st.markdown("---")
        else:
            st.warning(f"No claim found in the epistemic map for claim_id '{turn.claim_id}'. Grading without source context.")

        st.markdown(f"**Assessor Question:** {turn.question}")
        st.markdown(f"**Student Response:** {turn.student_response}")

        st.markdown("### Faculty Evaluation")
        
        # Check if already rated
        existing_rating = next((r for r in transcript.experiment_ratings if r.turn_index == i), None)
        
        if existing_rating:
            st.success(f"Rated by {existing_rating.faculty_id} at {existing_rating.rated_at}")
            st.json(existing_rating.model_dump())
        else:
            with st.form(key=f"rating_form_{i}"):
                st.markdown("#### 1. AI Dialogue Quality")
                assessor_halluc = st.toggle("Assessor Hallucination (Did it hallucinate facts?)", value=False, key=f"h_{i}")
                assessor_rep = st.toggle("Assessor Repetition (Did it exactly repeat a previous question?)", value=False, key=f"r_{i}")
                
                st.markdown("#### 2. Scoring Accuracy")
                st.info(f"The Evaluator's raw variance score was: {turn.variance_score:.2f}")
                eval_var = st.toggle("Did the Variance score correctly reflect the ambiguity?", value=True, key=f"v_{i}")
                
                st.markdown("#### 3. Advocate Quality (if applicable)")
                adv_nov = st.toggle("Advocate Pivot Novelty (Was the pivot creative/novel?)", value=True, key=f"an_{i}")
                adv_rel = st.toggle("Advocate Pivot Relevance (Was the pivot relevant?)", value=True, key=f"ar_{i}")
                
                st.markdown("#### 4. Faculty Overall Grade")
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
                    key=f"gt_{i}"
                )
                ground_truth = float(selected_grade[:3])
                
                submit = st.form_submit_button("Submit Rating")
                if submit:
                    rating = FacultyExperimentRating(
                        turn_index=i,
                        assessor_hallucination=assessor_halluc,
                        assessor_repetition=assessor_rep,
                        evaluator_variance_accurate=eval_var,
                        advocate_novel=adv_nov,
                        advocate_relevant=adv_rel,
                        ground_truth_score=ground_truth,
                        faculty_id=faculty_id
                    )
                    transcript.experiment_ratings.append(rating)
                    StorageManager.save_transcript(transcript)
                    st.rerun()
