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
