import os
from pathlib import Path
from dotenv import load_dotenv

# Load local .env file if it exists
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LABELS_DIR = DATA_DIR / "labels"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, LABELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Orchestration & LLM Settings (Claude / Anthropic API)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MOCK_MODE = os.getenv("VIVA_MOCK_MODE", "False" if ANTHROPIC_API_KEY else "True").lower() == "true"

# Per-agent model tiering. Kept as separate env vars (rather than one shared
# LLM_MODEL) so each agent's cost/quality tradeoff can be tuned independently:
# - Extraction ("epistemic mapping") and Evaluator both need `temperature`
#   support for the existing temperature-driven experiment design (see
#   ADVOCATE_TEMP/ASSESSOR_TEMP/EVALUATOR_TEMP below), which rules out
#   current-gen models (Opus 5 / Sonnet 5) - they reject `temperature` outright.
#   Sonnet 4.6 is the most capable model that still accepts it.
# - Advocate is high-volume/low-stakes (student-defense simulation), so it
#   runs on the cheapest temperature-capable tier.
EXTRACTION_MODEL = os.getenv("VIVA_EXTRACTION_MODEL", "claude-sonnet-4-6")
EVALUATOR_MODEL = os.getenv("VIVA_EVALUATOR_MODEL", "claude-sonnet-4-6")
ADVOCATE_MODEL = os.getenv("VIVA_ADVOCATE_MODEL", "claude-haiku-4-5")

# Socratic Probing Settings
MAX_DEPTH = int(os.getenv("VIVA_MAX_DEPTH", "5"))
MAX_CLAIMS_TO_PROBE = int(os.getenv("VIVA_MAX_CLAIMS_TO_PROBE", "5"))  # Limit to 5 key assumptions per session

# Agent Temperature Configurations
# NOTE: Claude's temperature range is 0.0-1.0 (Gemini's was 0.0-2.0). These
# defaults, and the randomized "Dynamic Extreme/Moderate" ranges in
# wrapper.py, have been rescaled into Claude's range - see wrapper.py's
# start_session() for the rescaled ranges.
ASSESSOR_TEMP = float(os.getenv("ASSESSOR_TEMP", "0.1"))     # Extreme Low: Assessor is highly deterministic and strict
ADVOCATE_TEMP = float(os.getenv("ADVOCATE_TEMP", "0.5"))     # Moderated: High temp destroys tiny 0.5B models; lowered to 0.5 so it stays relevant
EVALUATOR_TEMP = float(os.getenv("EVALUATOR_TEMP", "1.0"))   # High: Evaluator sits in the middle but is highly stochastic

# Feedback-Independent Thresholds
# Empirically validated via Faculty Interview Extraction (Phase D):
# - Coherence < 0.6 generally indicates pure evasion, but Coherence > 0.75 coupled with low grounding (<0.40) is flagged as "Superficial Fluency"
COHERENCE_THRESHOLD = float(os.getenv("VIVA_COHERENCE_THRESHOLD", "0.6"))

# - Variance > 0.25 detects when stochastic generation becomes unstable (Half-life point)
VARIANCE_THRESHOLD = float(os.getenv("VIVA_VARIANCE_THRESHOLD", "0.25"))

# - Circularity > 0.60 correctly catches "buzzword parroting" without generating false positives on normal terminology reuse
CIRCULARITY_THRESHOLD = float(os.getenv("VIVA_CIRCULARITY_THRESHOLD", "0.6"))

# - Sustained instability on a single claim is itself a collapse signal. Per-turn
#   thresholds alone are memoryless: a student can sit just above the collapse
#   line indefinitely and never be judged to have failed to ground the claim.
#   After this many CONSECUTIVE unstable turns on the same claim, the state is
#   escalated to COLLAPSED and reconstruction routing fires. Default 3 sits below
#   MAX_DEPTH (5), so escalation happens while there is still depth budget left to
#   probe the reconstructed angle. Set to 0 to disable escalation entirely.
MAX_CONSECUTIVE_UNSTABLE = int(os.getenv("VIVA_MAX_CONSECUTIVE_UNSTABLE", "3"))

# Advocate response length cap.
#
# Was a hard character truncation (started 128, widened to 256) defending
# against Advocate-sourced turns scoring higher than genuine student turns
# purely from length/claim-similarity (measured: 112 words avg vs 71, 0.777
# vs 0.702 composite). That vulnerability is now handled at the scoring/
# interrogation layer instead: no claim resolves on a single Grounded turn
# (see the specificity-challenge step in probe_engine.py's submit_response),
# COLLAPSE_THRESHOLD was raised, and the contradiction flag above catches a
# confidently-wrong answer regardless of length. With those in place, capping
# the Advocate's own text is no longer needed to keep the score honest, and
# was costing the project's actual intent - a genuinely useful "external voice
# to react to" - so the cap is lifted. None (the default) disables truncation;
# set VIVA_ADVOCATE_MAX_CHARS to an int to reinstate one.
_advocate_max_chars_env = os.getenv("VIVA_ADVOCATE_MAX_CHARS")
ADVOCATE_MAX_CHARS = int(_advocate_max_chars_env) if _advocate_max_chars_env else None

# Entailment backend.
#
# "linear-head" is the calibrated logistic regression over frozen all-MiniLM-L6-v2
# embeddings (see train_nli_head.py). It was adopted because the previous
# cross-encoder saturated to 0.99/0.01. It does spread scores smoothly - but it
# measures SIMILARITY, not entailment, because MiniLM embeddings are trained for
# similarity and discard negation. Measured on a controlled probe:
#
#     premise "pageR uses a Raspberry Pi Zero W and SPH0645 mic"
#       identical  -> 0.885     (correct)
#       negated    -> 0.531     (should be ~0.0)
#       "uses an Arduino Nano and INMP441"  -> 0.822  (should be ~0.0)
#       unrelated  -> 0.353     (should be ~0.5)
#
# A flat contradiction scored HIGHER than an unrelated sentence. No amount of
# sampling or reweighting downstream can recover a logical relation the embedding
# already threw away.
#
# "cross-encoder" reads premise and hypothesis jointly, so negation is visible.
# On the same probe it returns 0.998 / 0.000 / 0.003 / 0.508 - all four correct,
# including the neutral case that the older nli-deberta-v3-small got wrong (it
# called unrelated text a contradiction, which is what the saturation complaint
# was really about). Runs locally on CPU/MPS; no API, no data leaves the machine.
NLI_BACKEND = os.getenv("VIVA_NLI_BACKEND", "cross-encoder")
NLI_CROSS_ENCODER_MODEL = os.getenv(
    "VIVA_NLI_CROSS_ENCODER", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
)

# Composite Confidence Weights (logit terms)
#
# REVERTED to the original values. A reweighting (coherence_nli -2.0,
# coherence_sts 0.0, grounding_nli 4.0, grounding_sts 1.5) was trialled to stop
# fluent-but-ungrounded text scoring Grounded. It validated well on a synthetic
# probe set but FAILED on a real exemplar session (1f9eea46), marking all 13 turns
# of a candid, high-understanding student as Unstable. Two reasons, both mine:
#
#   1. The synthetic corpus defined a "genuine" answer as one that closely tracks
#      the source passage. Real strong answers reason BEYOND the document -
#      analogy, synthesis, reflection - none of which the passage entails. Raising
#      W_GROUNDING_NLI therefore optimised for document restatement and punished
#      exactly the originality a viva is meant to reward.
#   2. Zeroing W_COHERENCE_STS deleted the only term measuring whether the student
#      answered the QUESTION ASKED. On that session answer~question similarity was
#      0.763 versus answer~passage 0.644 - responsiveness was the signal that
#      tracked quality, and it was the one discarded.
#
# Do not re-tune these until the upstream claim/question mismatch is fixed: the
# question generator drifts off the claim it is attached to (C-07's questions were
# about gesture-detection latency; the questions asked were about the device's
# name), so grounding is frequently scored against a passage unrelated to what was
# asked. No weighting can be calibrated on top of mislabelled turns.
# W_COHERENCE_NLI is 0: the term asked "does the CLAIM entail the ANSWER", which is
# well-posed but tests the wrong thing. A good answer adds evidence the claim does
# not contain, so it is not entailed; a restatement is. Worse, when a question asks
# how the student's design AVOIDS a problem the claim states, the answer reads as a
# contradiction - griot turn 11 (claim: "devices generate the illusion of agency";
# answer: "our interface has no notifications and relies on genuine intent") scored
# 0.011 and collapsed a good answer.
#
# Measured over the griot session and the archetype corpus:
#   claim->answer (1.8): 10/12 grounded, turn 11 wrongly COLLAPSED at 0.253
#   answer->claim (1.8): 11/12 grounded, but waffle rises to 0.886 - reversing the
#                        direction rewards restatement outright
#   dropped   (0.0):     11/12 grounded, turn 11 recovers to 0.449
#
# The only thing the term did well was flag answers contradicting the document, and
# W_GROUNDING_NLI already does that far harder (0.056 on the same case), with
# consistency.py covering it exhaustively. Set non-zero to restore the old term.
W_COHERENCE_NLI = float(os.getenv("VIVA_W_COHERENCE_NLI", "0.0"))
W_COHERENCE_STS = float(os.getenv("VIVA_W_COHERENCE_STS", "1.5"))
W_GROUNDING_NLI = float(os.getenv("VIVA_W_GROUNDING_NLI", "2.2"))
W_GROUNDING_STS = float(os.getenv("VIVA_W_GROUNDING_STS", "2.0"))
W_VARIANCE_NLI = float(os.getenv("VIVA_W_VARIANCE_NLI", "-1.8"))
W_VARIANCE_STS = float(os.getenv("VIVA_W_VARIANCE_STS", "-1.5"))

# Contradiction flag: a discrete override, not another continuous weighted term.
#
# grounding_nli/grounding_sts are continuous, and testing found the underlying
# cross-encoder's raw probabilities can be noisy on genuinely degenerate text
# (a word-salad sample scored P(entail)=0.81 in one run, 0.09 in another, for
# equally meaningless input). Rather than trust that continuous magnitude
# further, this reads the cross-encoder's raw P(contradiction) directly - not
# the ordinal support scale, which already blends it away - and applies a
# fixed penalty ONLY when the model is confidently (>50%) calling the answer a
# contradiction of the student's own documentation. This rides alongside the
# existing grounding terms rather than replacing them: removing NLI's
# continuous contribution entirely would leave grounding as pure cosine
# similarity, which cannot detect contradiction at all (see nli_auditor.py's
# degraded-path comment).
#
# Applied in logit space before the sigmoid, the same way variance is, so it
# composes with the rest of z rather than clamping composite_confidence
# directly. Hand-set, not fitted - no independent labels exist yet to
# calibrate against (see the threshold caveat above).
CONTRADICTION_FLAG_THRESHOLD = float(os.getenv("VIVA_CONTRADICTION_FLAG_THRESHOLD", "0.5"))
CONTRADICTION_FLAG_PENALTY = float(os.getenv("VIVA_CONTRADICTION_FLAG_PENALTY", "2.0"))

# Reasoning-state decision thresholds on the composite.
#
# These were hardcoded in probe_engine (0.40 / 0.65) and had never been fitted to
# anything, while BreakdownDetector separately defaulted to 0.45 / 0.65 - two
# different sets of numbers for the same decision. They are hoisted here so they
# CAN be calibrated; the defaults are unchanged.
#
# THEY HAVE NOT BEEN CALIBRATED, and cannot be yet. The only labelled session
# (1f9eea46) carries 13 faculty_labels of which 12 are the labelling app's default
# "I agree with the system classification" checkbox - i.e. copies of the system's
# own output, itself produced by the uncentred composite bug that scored an
# all-neutral turn at 0.94. Exactly 1 label is an independent human judgement, and
# there are 0 blind evaluator ratings anywhere in the corpus. Fitting anything to
# those labels calibrates the system to reproduce its own known-bad output.
#
# Before touching these numbers: collect labels that are independent of the system
# (blind to its classification, from a rater who is not the system's author), then
# fit thresholds first and weights second.
GROUNDED_THRESHOLD = float(os.getenv("VIVA_GROUNDED_THRESHOLD", "0.65"))
# Raised from 0.40 after testing showed 0.40 was too low to ever fire on a
# single turn: even a genuinely incoherent, word-salad Advocate answer (grounding_nli
# crashed to 0.091) only pulled the composite down to 0.455, because grounding_sts/
# coherence_sts (cosine terms, weighted 3.5 combined vs grounding_nli's 2.2) stay
# moderate as long as SOME topical vocabulary survives, even in nonsense text. 0.40
# meant single-turn collapse was effectively unreachable regardless of how bad the
# answer was - only the multi-turn escalation streak could ever fire.
#
# 0.50 was chosen, not the 0.455 breakeven point exactly, to keep Unstable a real,
# distinct band rather than a sliver: measured scores for vague/evasive (0.57),
# contradictory (0.55), off-topic (0.55), and fluent-but-shallow (0.59-0.68) answers
# all sit ABOVE 0.50, so they still land Unstable - genuinely different, milder
# failure modes than outright incoherence, not immediately Collapsed. Only the
# gibberish case (0.455) crosses below it. Still hand-set, not calibrated against
# independent labels - same caveat as above.
COLLAPSE_THRESHOLD = float(os.getenv("VIVA_COLLAPSE_THRESHOLD", "0.50"))

# Tighter thresholds used ONLY by the autonomous pre-flight (run_autonomous_
# preflight in simulation.py), never by a human-answered turn. The Advocate is
# deliberately instructed to answer with hedged, uncertain language and no
# committed position - testing found the entailment scorer still reads that
# kind of answer, and even a reflected question with no assertion in it at
# all, as comfortably Grounded against strong documentation. Giving the
# Advocate the same benefit of the doubt as a human means the pre-flight
# rarely finds a genuine breakdown before exhausting MAX_DEPTH on a claim.
# These narrow the Unstable band and shorten the escalation streak so a
# borderline reading counts against the Advocate rather than for it, and a
# claim fails sooner without the verdict becoming arbitrary. Hand-set, not
# calibrated - same caveat as the thresholds above.
PREFLIGHT_GROUNDED_THRESHOLD = float(os.getenv("VIVA_PREFLIGHT_GROUNDED_THRESHOLD", "0.75"))
PREFLIGHT_COLLAPSE_THRESHOLD = float(os.getenv("VIVA_PREFLIGHT_COLLAPSE_THRESHOLD", "0.55"))
PREFLIGHT_MAX_CONSECUTIVE_UNSTABLE = int(os.getenv("VIVA_PREFLIGHT_MAX_CONSECUTIVE_UNSTABLE", "2"))

# Stamped onto every turn so analysis can separate transcripts scored under
# different weight sets. Bump this whenever the weights above change.
COMPOSITE_WEIGHTS_VERSION = os.getenv("VIVA_COMPOSITE_WEIGHTS_VERSION", "v2-crossencoder")

# Logging Settings
VERBOSE_LOGGING = os.getenv("VIVA_VERBOSE_LOGGING", "True").lower() == "true"

# Rubric Layer 2 Settings
VIVA_ENABLE_LAYER_2 = os.getenv("VIVA_ENABLE_LAYER_2", "True").lower() == "true"

# Teleprompter Settings
USE_TELEPROMPTER_V2 = os.getenv("USE_TELEPROMPTER_V2", "True").lower() == "true"

# Training Pipeline Settings
SCORING_VERSION = os.getenv("VIVA_SCORING_VERSION", "A").upper()
SCORING_WEIGHTS_FILE = DATA_DIR / "scoring_weights.json"

import json
from typing import Dict, Any

def load_scoring_config() -> Dict[str, Any]:
    """Loads scoring version and model weights from config."""
    if not SCORING_WEIGHTS_FILE.exists():
        return {
            "version": SCORING_VERSION,
            "version_b_weights": None,
            "version_b_intercepts": None,
            "version_b_classes": None
        }
    with open(SCORING_WEIGHTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_scoring_config(version: str, weights: Any, intercepts: Any, classes: Any):
    """Saves the learned Logistic Regression weights to config."""
    config = {
        "version": version,
        "version_b_weights": weights,
        "version_b_intercepts": intercepts,
        "version_b_classes": classes
    }
    SCORING_WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SCORING_WEIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
