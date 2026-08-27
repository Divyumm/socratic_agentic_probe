# Agentic Socratic Viva — 100-Step Build Breakdown

Covers the full mega-system mapped to the module handbook expectations diagram, including the
Version A/B training loop (para 2) and the independent evaluator comparison (para 5) explicitly
requested as in-scope. Complexity key: XS < 30 min · S ≈ 1–2h · M ≈ half day · L ≈ 1–2 days.

**Not included here** — already tracked separately and still the hard blocker: the interface
application layer and ethics submission (see `dissertation_tasks_next_phase.txt`). Most of what
follows can be built and pilot-tested against synthetic/existing transcripts without waiting on
that blocker — but *running* it with real participants still needs ethics clearance first.

**How to read the context blocks below:** each one states what currently exists in the uploaded
codebase, versus what emerged purely from this conversation and has zero code representation yet.
That gap is the actual scope of this document — it's not refining existing code, it's building a
second system alongside the first.

---

## Phase 1 — Schema & data model extensions (10)

**Context.** `schemas.py` as it stands defines `Claim`, `EpistemicMap`, `ProbeTurn`, and
`SessionTranscript` — nothing else. `FacultyLabel`, `RubricScore`, `ReviewCard`, and
`EvaluatorRating` do not exist anywhere in the codebase; they were designed in conversation over
the last several messages and have never been written down as code until this file. This phase is
pure scaffolding — nothing in Phases 3–9 can be built until these models exist, since every later
phase reads or writes one of them.

1. [XS] Add `id: str` (UUID) to `ProbeTurn` in `schemas.py` so turns can be referenced by labels.
2. [S] Define `FacultyLabel` model (`turn_id`, `labeller_id`, `agrees_with_system`, `corrected_state`, `note`, `labelled_at`).
3. [XS] Add a `RubricConstruct` enum with `bucket: Literal["A","B","C"]` for Layer 2 tagging.
4. [S] Define `RubricScore` model (`claim_id`, `construct`, `bucket`, `score`, `anchor_level`, `source: "auto"|"human"`).
5. [S] Define `ReviewCard` model aggregating `composite_confidence`, `RubricScore` list, session notes.
6. [S] Define `EvaluatorRating` model (`session_id`, `evaluator_id`, `judgement`, `rating_score`, `submitted_at`, `blind: bool`).
7. [XS] Add `weight_version: Literal["A","B"]` to `ProbeTurn`/`SessionTranscript`.
8. [S] Extend `SessionTranscript` with `faculty_labels: List[FacultyLabel]`.
9. [S] Extend `SessionTranscript` with `evaluator_rating: Optional[EvaluatorRating]`.
10. [XS] Write schema validation tests for all new/modified models (mirror `test_parser.py`).

## Phase 2 — Layer 1 verification (5)

**Context.** This phase exists because of a specific contradiction flagged earlier: the uploaded
`feature_engine.py` shows keyword-overlap and word-count heuristics — placeholder logic, not real
scoring — while separate notes describe `variance_score` as wired to genuine multi-temperature API
calls. Both can't be true of the same file at once. Every phase after this one builds on top of
Layer 1's four scores, so this verification has to happen first, or every subsequent phase is
built on an assumption that hasn't been checked.

11. [S] Audit `feature_engine.py` for simulated-placeholder vs real-API-call scores; log which is which.
12. [M] Confirm `variance_score` uses real multi-temperature API calls; add a regression test.
13. [XS] Confirm `composite_confidence` uses the documented weights consistently.
14. [S] Test that `StudentState` thresholds match the phase-transition model in the gateway report.
15. [XS] Document the current Layer 1 formula in a short reference for the dissertation appendix.

## Phase 3 — Layer 2: claim bucket rubric (15)

**Context.** There is no Layer 2 anywhere in the current codebase — `probe_engine.py` and
`feature_engine.py` only implement the four-signal Layer 1. The entire Bucket A/B/C structure came
out of a back-and-forth with a coding agent's "Dual-Pass Architecture" suggestion, which we then
corrected because its example anchors (caste, Ustad hierarchy, artisan constraints) were pulled
from the deprioritized rural-craft study rather than your actual
`Faculty_Jury_Assessment_Questionnaire.docx`. Nothing here has been coded, tested, or even drafted
as a prompt yet — this phase is a from-scratch build, not an extension of existing scoring logic.

16. [M] Write the coding manual (definition + Low/Mid/High anchors) for Bucket A: Internalisation, Originality — sourced from `Faculty_Jury_Assessment_Questionnaire.docx`.
17. [M] Same coding manual pass for Bucket B: Confidence & Conviction, Empathy & Stakeholder Sensitivity.
18. [S] Document Bucket C constructs and draft the limitations-chapter exclusion paragraph.
19. [M] Implement `compute_internalisation()` — first-person/paraphrase heuristic.
20. [M] Implement `compute_originality()` — semantic-distance proxy.
21. [L] Prompt-engineer an LLM scorer for Confidence & Conviction using the anchor manual as system prompt.
22. [L] Prompt-engineer an LLM scorer for Empathy & Stakeholder Sensitivity, same pattern.
23. [S] Wire Bucket A scoring to run automatically after each `ProbeTurn` submission.
24. [S] Wire Bucket B scoring to run but flag `source: "auto-draft"` pending human confirmation.
25. [M] Pilot-code 3–5 existing transcripts manually against the rubric before trusting automated output.
26. [S] Compare pilot manual scores to automated scores; log discrepancy rate.
27. [S] Add a `RubricScore` list field/store per session.
28. [XS] Unit tests for `compute_internalisation`/`compute_originality` on synthetic inputs.
29. [S] Add a `config.py` toggle to enable/disable Layer 2 scoring independently of Layer 1.
30. [XS] Log which claims never received a Bucket A/B score (coverage check).

## Phase 4 — Teleprompter rendering layer (12)

**Context.** This phase is a direct fix, not new territory — the strings it targets are already
sitting in `probe_engine.py` today: "Your defense for this claim has broken down in the
{dimension} dimension", "Here is my epistemic justification of your reasoning state". Right now
there is exactly one rendering path in the codebase, and it doubles as both the internal
diagnostic log and the text the student reads. That conflation is the actual bug this phase fixes;
everything else in this phase (the assumption-surfacing rewrite, the banned-words list) is in
service of splitting that one path into two.

31. [M] Design the internal-state → student-language mapping (assumption + justification prompt pattern) in `prompts.py`.
32. [L] Refactor `probe_engine.py` to separate internal diagnostic strings from student-facing output.
33. [M] Rewrite question-generation prompts in `llm_client.py` to surface the assumption before asking for justification.
34. [S] Remove jargon leakage (e.g. "epistemic justification", "[REASONING COLLAPSE DETECTED]") from student-facing strings.
35. [S] Add `render_for_student(turn) -> str` as the single point of student-facing text generation.
36. [S] Add `render_for_faculty(turn) -> str` passing through raw state/dimension/scores unmediated.
37. [M] Write 10–15 test cases across state × dimension combinations; assert no jargon leaks.
38. [S] Update `main.py` CLI calls to use `render_for_student` instead of printing raw turn objects.
39. [XS] Add a style-guide doc: banned-words list ("epistemic", "defense", "dispute", "vulnerability rank").
40. [S] Add an automated jargon-detection test scanning generated text against the banned-words list.
41. [M] Assert Papanek dimension names are never shown to the student directly.
42. [XS] Log the teleprompter's chosen phrasing per turn for later qualitative analysis.

## Phase 5 — Faculty labelling interface, before phase (15)

**Context.** `main.py` today is the terminal input/print loop flagged from the start of this
project as the interface blocker — but that blocker is specifically about the *participant-facing*
interface, which is what ethics approval depends on. This labelling tool is faculty-only and
internal, so it's decoupled from that blocker: it can be built and even pilot-run without waiting
on ethics, as long as no student data is involved yet. Nothing resembling a labelling screen exists
anywhere in the current code — this is new UI, not a rebuild of the CLI.

43. [S] Decide UI shell: minimal Streamlit page vs. extending the terminal harness (scope decision).
44. [M] Build session-loader screen: load an existing `SessionTranscript` by ID.
45. [M] Build turn-stepper screen: question + response + system state, no numeric scores shown.
46. [S] Add agree/disagree control bound to `FacultyLabel.agrees_with_system`.
47. [S] Add corrected-state picker, shown only on disagree.
48. [S] Add optional free-text justification field.
49. [M] Wire "save label" to `storage.py`, appending to `SessionTranscript.faculty_labels`.
50. [S] Add "next turn" navigation with progress indicator.
51. [S] Add `labeller_id` capture (simple entry, no auth needed for pilot).
52. [M] Generate a seed dataset: 3–5 synthetic sessions via existing mock respondents in `llm_client.py`.
53. [S] Have one pilot labeller run the seed dataset end to end to validate the flow.
54. [XS] Export labelled sessions to flat JSON for inspection.
55. [S] Add a basic label-count/agreement-rate dashboard.
56. [XS] Document the labelling protocol for the methodology chapter.
57. [S] Add resume-where-left-off handling for partially labelled sessions.

## Phase 6 — Training pipeline: Version A vs B (15)

**Context.** "Version A" already exists in the codebase in a loose sense — it's whatever hand-set
weight constants currently live in `feature_engine.py`'s composite score function — but it has
never been named or treated as a formal variant to compare against anything. "Version B" does not
exist in any form: there is no labelled dataset, no call to `LogisticRegression.fit()`, and no
storage for learned weights anywhere in the project. This phase is where the conversation's "Dual
Pass" and "part of the data pipeline" discussion turns into the first line of actual training code.

58. [S] Extract labelled `(features, corrected_state)` pairs from all `faculty_labels` into a flat table.
59. [M] Write feature extraction pulling the four scores per labelled turn.
60. [S] Encode the target label (binary/multiclass — confirm scope) from `corrected_state`.
61. [M] Implement `train_version_b()` using `LogisticRegression.fit()`.
62. [S] Persist Version B's fitted weights (+ intercept) to storage/config.
63. [XS] Keep Version A's hand-set weights untouched and clearly separated.
64. [M] Implement `score_with_version(turn, version)` supporting both A and B.
65. [S] Add a minimum-sample-size guard before training Version B (define N, e.g. 30–50 labels).
66. [M] Run A and B against the same held-out labelled turns; compute agreement rate.
67. [S] Log where A/B diverge most (dimension, state) for the discussion chapter.
68. [S] Add a config flag to choose which version powers a live session.
69. [XS] Unit test: `fit()` runs cleanly on a synthetic minimal dataset.
70. [S] Add k-fold cross-validation to check Version B isn't overfitting to a handful of labels.
71. [XS] Document training data provenance (sessions, label count, labellers) for methodology.
72. [S] Write the A/B comparison summary generator (table/short report) for the dissertation.

## Phase 7 — After phase: jury review card (10)

**Context.** Nothing like a review card exists yet — the closest thing in the current codebase is
`simulation.py` printing raw scores to the terminal ("Coherence Score: 0.72", etc.) during the
adversarial simulation demo. That's a debug console, not a professor-facing artifact. This phase
turns that same underlying data (now including Layer 2's `RubricScore`s and, if Phase 6 is done,
both weight versions) into something actually presentable, which is a genuinely new component.

73. [M] Build `ReviewCard` assembly: composite_confidence (both versions if available) + RubricScore list + notes.
74. [S] Build a display screen rendering `render_for_faculty` output plus Bucket A/B scores side by side.
75. [S] Add a toggle showing Version A vs B scores on the same card when both exist.
76. [XS] Ensure `ReviewCard` never triggers the teleprompter path.
77. [S] Add export-to-PDF/text for a completed card.
78. [XS] Add a "Bucket C — excluded" transparency note on the card.
79. [S] Auto-trigger `ReviewCard` generation when `SessionTranscript.completed_at` is set.
80. [XS] Unit test: assembly handles a session with missing Bucket B scores gracefully.
81. [S] Basic styling/layout pass for a non-technical faculty reader.
82. [XS] Save a sample card output as a dissertation appendix figure.

## Phase 8 — Independent evaluator flow & blind comparison (10)

**Context.** This has the biggest gap of any phase: it exists entirely as a methodological
requirement from the gateway report's evaluation design (compare system collapse-detection to an
independent human assessor's judgement) and has zero code representation — no screen, no schema
usage, no blindness enforcement. It's also the phase most directly gated by ethics approval for
real use, since it involves an outside assessor reviewing actual student work — though, as with
Phase 5, it can be built and dry-run against synthetic transcripts first.

83. [M] Build an evaluator screen showing only the raw transcript — no scores, no state.
84. [S] Add a free-text/rating input bound to `EvaluatorRating`.
85. [S] Enforce blindness in code: evaluator screen imports zero `ProbeTurn` scoring fields.
86. [M] After submission, unlock the comparison view (system state alongside evaluator judgement).
87. [S] Implement the primary metric: alignment rate between system collapse events and evaluator-flagged weak points.
88. [S] Log disagreement cases for qualitative discussion.
89. [XS] Guard against an evaluator viewing a session's `ReviewCard` before submitting their rating.
90. [S] Add `evaluator_id` capture and session-to-evaluator assignment logic.
91. [S] Unit test: comparison view inaccessible until `submitted_at` is set.
92. [XS] Document the blind-rating protocol for the ethics application and methodology chapter.

## Phase 9 — Integration, testing, persistence (8)

**Context.** Current automated test coverage is thin — the walkthrough log shows exactly five
tests, all in `test_parser.py`, covering document parsing only. None of Layers 2–3, the labelling
flow, training, the review card, or the evaluator flow have any tests today because none of them
exist yet. This phase is where all eight prior phases get proven to actually work together, not
just individually — which is also usually where integration bugs between phases surface for the
first time.

93. [M] End-to-end test: parse → session → label → train B → review card → evaluator → compare.
94. [S] Verify `storage.py` persists all new schema types correctly.
95. [S] Add a migration/backfill script for `SessionTranscript`s predating these schema additions.
96. [S] Run the full `pytest` suite (existing + new); fix breakage from schema changes.
97. [XS] Update README/CLI help text for the new labelling and evaluator flows.
98. [S] Smoke-test the full pipeline against a fresh synthetic document.
99. [XS] Tag this milestone in version control with a clear commit message.
100. [S] Write a short internal status note: what's built vs. still theorised (the "during" phase stays excluded) — for Andrew.
