import json
import logging
from typing import Tuple, List, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from app_3.storage import StorageManager
from app_3.schemas import StudentState
from app_3.config import SCORING_WEIGHTS_FILE

logger = logging.getLogger(__name__)

MINIMUM_TRAINING_SAMPLES = 30

def extract_training_data() -> Tuple[np.ndarray, np.ndarray]:
    """Extracts X (feature scores) and y (corrected_state) from faculty labels."""
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
                
                # Features: Coherence, Grounding, Variance, Circularity
                features = [
                    turn.coherence_score,
                    turn.grounding_score,
                    turn.variance_score,
                    turn.circularity_score
                ]
                X.append(features)
                y.append(target_state.value)
                
    return np.array(X), np.array(y)

def train_version_b() -> Tuple[bool, int]:
    """Fits LogisticRegression on all faculty labels and saves the model parameters."""
    X, y = extract_training_data()
    
    if len(X) < MINIMUM_TRAINING_SAMPLES:
        logger.warning(f"Not enough data to train Version B. Found {len(X)}, need {MINIMUM_TRAINING_SAMPLES}.")
        return False, len(X)
        
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    
    # 5-Fold Cross Validation for sanity check
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf)
    logger.info(f"Version B CV Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
    
    # Final training on all data
    model.fit(X, y)
    
    # Extract weights for the Config JSON
    weights = model.coef_.tolist()
    intercepts = model.intercept_.tolist()
    classes = model.classes_.tolist()
    
    config = {
        "version": "B",
        "version_b_weights": weights,
        "version_b_intercepts": intercepts,
        "version_b_classes": classes,
        "cv_accuracy_mean": scores.mean(),
        "cv_accuracy_std": scores.std() * 2
    }
    
    SCORING_WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SCORING_WEIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
        
    logger.info(f"Version B trained successfully on {len(X)} samples. Weights saved to {SCORING_WEIGHTS_FILE.name}")
    return True, len(X)
