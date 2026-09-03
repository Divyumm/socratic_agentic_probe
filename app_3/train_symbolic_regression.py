import json
import logging
from typing import Tuple
import numpy as np
import sympy
from gplearn.genetic import SymbolicRegressor
from app_3.storage import StorageManager
from app_3.config import SCORING_WEIGHTS_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MINIMUM_TRAINING_SAMPLES = 10

def extract_training_data() -> Tuple[np.ndarray, np.ndarray]:
    """Extracts X (6-variable feature set) and y (corrected_state) from faculty labels."""
    X = []
    y = []

    session_ids = StorageManager.list_available_transcripts()
    for session_id in session_ids:
        transcript = StorageManager.load_transcript(session_id)
        if not transcript or not transcript.faculty_labels:
            continue

        for label in transcript.faculty_labels:
            turn = next((t for t in transcript.turns if str(t.id) == label.turn_id), None)
            if turn:
                target_state = label.corrected_state if not label.agrees_with_system else turn.state

                # All six independent variables from hybrid NLI + STS scoring
                features = [
                    turn.coherence_nli,      # X0: Coherence NLI (logical consistency)
                    turn.coherence_sts,      # X1: Coherence STS (semantic similarity)
                    turn.grounding_nli,      # X2: Grounding NLI (logical consistency)
                    turn.grounding_sts,      # X3: Grounding STS (semantic similarity)
                    turn.variance_nli,       # X4: Variance NLI (evidence stability)
                    turn.variance_sts        # X5: Variance STS (evidence stability)
                ]

                # The dependent variable y needs to be continuous for standard regression,
                # so we map Grounded=1.0, Unstable=0.5, Collapsed=0.0
                state_val = 1.0
                if target_state.name == "COLLAPSED":
                    state_val = 0.0
                elif target_state.name == "UNSTABLE":
                    state_val = 0.5

                X.append(features)
                y.append(state_val)

    return np.array(X), np.array(y)

def train_symbolic_equation():
    """Uses Genetic Programming to discover the composite score formula."""
    X, y = extract_training_data()
    
    if len(X) < MINIMUM_TRAINING_SAMPLES:
        logger.warning(f"Not enough data to train Symbolic Regression. Found {len(X)}, need {MINIMUM_TRAINING_SAMPLES}.")
        logger.warning("Please generate data via the Streamlit frontend and have faculty label it first!")
        return False, len(X)
        
    logger.info("Initializing gplearn SymbolicRegressor for 6-variable search...")
    # Configure the genetic program to search for a clean algebraic equation
    # Increased for 6 variables (vs. 3 in previous version): larger search space
    est_gp = SymbolicRegressor(population_size=5000,
                               generations=30,
                               stopping_criteria=0.01,
                               p_crossover=0.7, p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05, p_point_mutation=0.1,
                               max_samples=0.9, verbose=1,
                               parsimony_coefficient=0.01, random_state=42)
    
    est_gp.fit(X, y)
    
    raw_equation = str(est_gp._program)
    logger.info(f"Raw discovered equation: {raw_equation}")

    # Sympy mapping to simplify the expression algebraically
    converter = {
        'add': lambda x, y : x + y,
        'sub': lambda x, y : x - y,
        'mul': lambda x, y : x * y,
        'div': lambda x, y : x / y
    }

    try:
        # Evaluate using sympify
        simplified_equation = str(sympy.sympify(raw_equation, locals=converter))
    except Exception as e:
        logger.warning(f"Sympy simplification failed: {e}. Using raw equation.")
        simplified_equation = raw_equation

    logger.info(f"Simplified Equation: {simplified_equation}")

    # Save the discovered mathematical formula
    config = {
        "version": "Symbolic_GP_v2",
        "equation_raw": raw_equation,
        "equation_simplified": simplified_equation,
        "variables": {
            "X0": "coherence_nli",
            "X1": "coherence_sts",
            "X2": "grounding_nli",
            "X3": "grounding_sts",
            "X4": "variance_nli",
            "X5": "variance_sts"
        }
    }
    
    SCORING_WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SCORING_WEIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
        
    logger.info(f"Symbolic Regression complete! Equation saved to {SCORING_WEIGHTS_FILE.name}")
    return True, len(X)

if __name__ == "__main__":
    train_symbolic_equation()
