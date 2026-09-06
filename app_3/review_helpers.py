"""Shared, framework-free helpers for the faculty-facing review surfaces.

labelling_app.py (Primary Review Card) and feedback_app.py (Final Feedback
Report) render the same underlying numbers with slightly different layouts.
Before this module existed, each file computed them independently, which is
how a session's composite score could show a different letter grade
depending which of the two dashboards you opened - the two files had each
grown their own copy of a grading function with different boundaries, and
neither matched the "Reference Rubric" faculty are separately shown when
grading the Experiment Rating tab.

These functions hold the computation once. Both apps import them and render
the result however suits their layout; no Streamlit calls live here.
"""

from typing import Any, Dict, List, Optional


def claim_for_turn(transcript, turn):
    """Find the claim a turn actually probed, tolerating duplicated claim ids.

    Extraction used to number claims per batch, so one map can hold many claims
    sharing an id (a real 91-claim map had only 10 distinct ids). Matching on id
    alone returns whichever duplicate comes first, which showed faculty the wrong
    claim text and the wrong source page - the answer looked ungrounded because it
    was being read against an unrelated passage.

    The question disambiguates it: each claim carries its own probing_questions, so
    the claim whose list contains this turn's question is the one actually probed.
    Falls back to the id match when the question is not found (e.g. reconstruction
    prompts, or maps extracted after the unique-id fix, where ids are unambiguous).
    """
    candidates = [c for c in transcript.epistemic_map.claims if c.id == turn.claim_id]
    if len(candidates) > 1:
        for c in candidates:
            if turn.question in (c.probing_questions or []):
                return c
    return candidates[0] if candidates else None


def composite_to_letter_grade(score: Optional[float]) -> str:
    """The one faculty-facing grade mapping, reused everywhere a card shows a grade.

    Boundaries are the midpoints between the anchors of the "Reference Rubric"
    already shown to faculty in the Experiment Rating tab (0.0-0.4 F/D, 0.5-0.6 C,
    0.7 B, 0.8-0.9 A, 1.0 A*), so a card's grade agrees with the table faculty are
    handed when they grade the same score by hand. Distinct from the 4-band
    A/B/C/D/F scale students see at session end in app.py, which is a different,
    intentionally coarser audience and is left as-is here.
    """
    if score is None:
        return "N/A"
    if score < 0.45:
        return "F/D"
    if score < 0.65:
        return "C"
    if score < 0.75:
        return "B"
    if score < 0.95:
        return "A"
    return "A*"


def response_source_counts(transcript) -> Dict[str, int]:
    """Turn counts by provenance relative to the in-app Advocate.

    UNVERIFIED and the legacy STUDENT value (kept only so old transcripts load)
    are folded together: neither is evidence the student wrote the answer, only
    that no Advocate draft existed to compare against.
    """
    from app_3.schemas import ResponseSource

    unverified = sum(
        1 for t in transcript.turns
        if t.response_source in (ResponseSource.UNVERIFIED, ResponseSource.STUDENT)
    )
    advocate = sum(1 for t in transcript.turns if t.response_source == ResponseSource.ADVOCATE)
    hybrid = sum(1 for t in transcript.turns if t.response_source == ResponseSource.HYBRID)
    return {"unverified": unverified, "advocate": advocate, "hybrid": hybrid,
            "total": len(transcript.turns)}


def rubric_criterion_means(transcript) -> List[Dict[str, Any]]:
    """Mean entailment support per rubric criterion, weakest first.

    Returns [{"id": str, "mean": float, "n": int}, ...]. Empty when the session
    predates per-criterion scoring (rubric_criterion_scores defaults to {}).
    """
    rows: Dict[str, List[float]] = {}
    for t in transcript.turns:
        for cid, val in (t.rubric_criterion_scores or {}).items():
            rows.setdefault(cid, []).append(val)
    out = [{"id": cid, "mean": sum(vals) / len(vals), "n": len(vals)}
           for cid, vals in rows.items()]
    out.sort(key=lambda r: r["mean"])
    return out


def contradiction_flags(transcript, threshold: float = 0.40) -> List[Any]:
    """Turns whose answer scored as contradicting the student's own documentation.

    Only meaningful with the cross-encoder NLI backend - the linear head this
    replaced could not distinguish contradiction from simply being off-topic, so
    this list would previously have been dominated by false positives.
    """
    flagged = [t for t in transcript.turns if t.contradiction_signal < threshold]
    flagged.sort(key=lambda t: t.contradiction_signal)
    return flagged


def responsiveness_summary(transcript) -> Dict[str, Any]:
    """How closely each answer addressed the QUESTION asked, not the claim.

    coherence_sts (question<->answer cosine) is the one signal in the composite
    that measures this; W_COHERENCE_NLI is weighted 0 because claim-entailment
    rewards restating the claim rather than engaging the question, so a strong,
    on-topic answer to a genuinely open question can otherwise score no better
    than one that ignored the question and echoed the claim. This is reported
    separately because the composite alone does not surface it: a griot session
    turn scored 0.164 on document-grounding despite the student directly and
    articulately answering what was asked - the composite made no separate space
    for "did they engage the question" versus "does the document already say this".
    """
    vals = [(t.turn_index, t.coherence_sts) for t in transcript.turns
            if t.coherence_sts is not None]
    if not vals:
        return {"mean": None, "per_turn": []}
    mean = sum(v for _, v in vals) / len(vals)
    return {"mean": mean, "per_turn": vals}


def intervention_effort_summary(transcript) -> Dict[str, Any]:
    """Per-turn Advocate-draft-vs-submission comparison, and its aggregate.

    advocate_similarity is a difflib ratio: 1.0 = submitted the draft verbatim,
    0.0 = unrecognisable from it. Reported as "rewritten" (1 - similarity) so a
    higher number reads as more independent work, matching the reading in the
    project's own design notes that frequent, substantial intervention is a
    signal of genuine tacit understanding rather than of the AI's mistake.

    Only turns where an Advocate draft was actually generated are included -
    advocate_similarity is None otherwise, which is a fact about the turn
    (nothing to compare against), not a rewritten-100% turn.
    """
    rows = []
    for t in transcript.turns:
        if t.advocate_similarity is None:
            continue
        rows.append({
            "turn_index": t.turn_index,
            "draft": t.advocate_draft,
            "submitted": t.student_response,
            "similarity": t.advocate_similarity,
            "rewritten_pct": (1.0 - t.advocate_similarity) * 100.0,
            "response_source": t.response_source.value,
        })
    mean_rewritten = (sum(r["rewritten_pct"] for r in rows) / len(rows)) if rows else None
    return {"rows": rows, "mean_rewritten_pct": mean_rewritten, "n": len(rows)}
