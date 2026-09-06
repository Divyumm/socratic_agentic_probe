import streamlit as st

# Add the parent directory of `app_3` to `sys.path` so imports work correctly
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app_3.storage import StorageManager
from app_3.theme import inject_theme
from app_3.review_helpers import composite_to_letter_grade, rubric_criterion_means

st.set_page_config(layout="wide", page_title="Session Feedback Dashboard")
inject_theme()

st.title("Session Feedback Dashboard")
st.write("Your grade and rubric breakdown for a completed session.")

st.write("---")

# 1. Get Available Transcripts
session_ids = StorageManager.list_available_transcripts()

if not session_ids:
    st.info("No transcripts found in the storage directory.")
    st.stop()

# Filter for completed sessions (those with a review card). A session only gets
# a review card once build_transcript(completed=True) has run - see app.py's
# Save & Exit handler.
completed = []
for sid in session_ids:
    t = StorageManager.load_transcript(sid)
    if t and getattr(t, "review_card", None) is not None:
        completed.append((sid, t))

if not completed:
    st.warning("No completed sessions with feedback found.")
    st.stop()

# The selector shows student + document, not the raw session id or a
# timestamp - a student picking their own feedback does not need to see the
# internal identifier, and faculty digging deeper have the raw transcript
# files (and the fuller diagnostic data still recorded inside them - see
# below) for that.
def _label(sid, t):
    return f"{t.student_name} — {t.document_name}"

labels = [_label(sid, t) for sid, t in completed]
# Disambiguate only if two sessions would otherwise show an identical label -
# never silently collapse two different sessions under one entry.
if len(set(labels)) != len(labels):
    seen = {}
    disambiguated = []
    for (sid, t), label in zip(completed, labels):
        seen[label] = seen.get(label, 0) + 1
        disambiguated.append(f"{label} ({sid[:6]})" if seen[label] > 1 else label)
    labels = disambiguated

label_to_session = dict(zip(labels, completed))
selected_label = st.selectbox("Select your session:", labels)
selected_session, transcript = label_to_session[selected_label]

if transcript:
    st.write(f"**Student:** {transcript.student_name} | **Document:** {transcript.document_name}")

    # Only shown when a real brief was actually supplied - no placeholder text
    # for sessions that never had one.
    if (transcript.assignment_brief or "").strip():
        with st.expander("📋 Assignment Brief", expanded=False):
            st.write(transcript.assignment_brief)

    st.write("---")

    rc = transcript.review_card
    st.markdown('<div class="review-card-header"><h2>Final Feedback Report</h2></div>', unsafe_allow_html=True)

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

    # Per-criterion rubric support
    criterion_rows = rubric_criterion_means(transcript)
    if criterion_rows:
        st.write("---")
        st.write("### Rubric Criterion Breakdown")
        for row in criterion_rows:
            st.write(f"**{row['id'].capitalize()}** — {row['mean']:.3f}")
            st.progress(min(max(row["mean"], 0.0), 1.0))
    else:
        st.write("---")
        st.caption("No per-criterion rubric scores recorded — this session predates per-criterion scoring.")
