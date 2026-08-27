import os
import json
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from app_3.config import PROCESSED_DATA_DIR

class ClusteringPipeline:
    """Semi-supervised clustering pipeline for 4-agent conversational states.
    Uses TF-IDF + KMeans to cluster the combined Assessor, Advocate, Evaluator, 
    and Quality Auditor interactions into distinct tension states."""

    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.is_fitted = False

    def extract_turn_features(self, assessor_q: str, advocate_r: str, evaluator_j: str, auditor_j: str) -> str:
        """Concatenates the four agent responses into a single document for clustering."""
        return f"Assessor: {assessor_q}\nAdvocate: {advocate_r}\nEvaluator: {evaluator_j}\nAuditor: {auditor_j}"

    def fit_clusters(self, turns: List[str]) -> None:
        """Fits the KMeans model on a corpus of historical turns."""
        if not turns:
            return
        X = self.vectorizer.fit_transform(turns)
        self.kmeans.fit(X)
        self.is_fitted = True
        print(f"[Clustering] Fitted KMeans with {self.n_clusters} clusters on {len(turns)} turns.")

    def predict_cluster(self, turn_text: str) -> int:
        """Predicts the cluster for a new turn. 
        If not fitted, returns a random mock cluster for the POC until enough data is gathered."""
        if not self.is_fitted:
            # Fallback for POC if we haven't trained it on historical data yet
            return np.random.randint(0, self.n_clusters)
            
        X = self.vectorizer.transform([turn_text])
        return int(self.kmeans.predict(X)[0])

    def export_clusters_for_researcher(self, turns: List[str], turn_ids: List[str], filepath: str = "cluster_export.json"):
        """Exports the turns and their assigned clusters so a researcher can manually label them."""
        if not self.is_fitted:
            self.fit_clusters(turns)
            
        X = self.vectorizer.transform(turns)
        predictions = self.kmeans.predict(X)
        
        export_data = []
        for idx, turn in enumerate(turns):
            export_data.append({
                "turn_id": turn_ids[idx] if idx < len(turn_ids) else str(idx),
                "text": turn,
                "cluster_id": int(predictions[idx])
            })
            
        export_path = os.path.join(PROCESSED_DATA_DIR, filepath)
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=4)
            
        print(f"Exported {len(turns)} clustered turns to {export_path} for researcher labelling.")
        return export_path
