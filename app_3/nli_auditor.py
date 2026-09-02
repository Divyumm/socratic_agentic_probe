import warnings
from transformers import pipeline

# Suppress Hugging Face warnings for cleaner console output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class NLIAuditor:
    """
    Hybrid NLI + STS Auditor for calculating coherence and grounding scores.

    Uses two complementary signals:
    - NLI (cross-encoder/nli-deberta-v3-small): Logical consistency (entailment/contradiction detection)
    - STS (sentence-transformers): Semantic similarity for alignment measurement

    Combined approach ensures both logical soundness and semantic relevance.
    """
    _instance = None

    def __new__(cls):
        # Singleton pattern to prevent reloading the model in memory multiple times
        if cls._instance is None:
            cls._instance = super(NLIAuditor, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        print("[NLIAuditor] Loading on-device NLI model (cross-encoder/nli-deberta-v3-small)...")
        self.classifier = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-small")
        print("[NLIAuditor] NLI model loaded successfully.")

        print("[NLIAuditor] Loading STS model (all-MiniLM-L6-v2) for semantic similarity...")
        from sentence_transformers import SentenceTransformer, util
        self.sts_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sts_util = util
        print("[NLIAuditor] STS model loaded successfully.")

    def compute_sts_similarity(self, text1: str, text2: str) -> float:
        """
        Computes Semantic Textual Similarity between two texts using embeddings.

        Returns a float between 0.0 and 1.0 representing semantic alignment:
        - 1.0: Highly similar / semantically aligned
        - 0.5: Moderately similar
        - 0.0: Completely dissimilar / semantically unrelated
        """
        try:
            emb1 = self.sts_model.encode(text1, convert_to_tensor=True)
            emb2 = self.sts_model.encode(text2, convert_to_tensor=True)
            similarity = self.sts_util.pytorch_cos_sim(emb1, emb2).item()
            return round(float(similarity), 2)
        except Exception as e:
            print(f"[NLIAuditor] Error computing STS similarity: {e}")
            return 0.5

    def _compute_nli_non_contradiction(self, premise: str, hypothesis: str) -> float:
        """
        Extracts the non-contradiction score from NLI model.
        Returns: 1.0 - contradiction_probability

        This represents logical consistency: high score means the hypothesis
        does not logically contradict the premise.
        """
        try:
            all_scores_raw = self.classifier({"text": premise, "text_pair": hypothesis}, top_k=None)

            # Flatten if it's a list of lists
            if isinstance(all_scores_raw, list) and len(all_scores_raw) > 0 and isinstance(all_scores_raw[0], list):
                all_scores = all_scores_raw[0]
            elif isinstance(all_scores_raw, list):
                all_scores = all_scores_raw
            else:
                all_scores = [all_scores_raw]

            # Dynamically check model config for the contradiction label mapping
            contradiction_label_id = None
            if hasattr(self.classifier.model, "config") and hasattr(self.classifier.model.config, "id2label"):
                for idx, label_str in self.classifier.model.config.id2label.items():
                    if 'contradiction' in label_str.lower():
                        contradiction_label_id = label_str
                        break

            contradiction_score = 1.0  # Default to full contradiction if parsing fails
            for score_dict in all_scores:
                label = score_dict['label'].lower()
                # If we dynamically found the exact label, match it
                if contradiction_label_id and label == contradiction_label_id.lower():
                    contradiction_score = score_dict['score']
                    break
                # Fallback heuristics if id2label was missing
                elif not contradiction_label_id:
                    if 'contradiction' in label:
                        contradiction_score = score_dict['score']
                        break
                    # standard fallback: cross-encoder/nli-deberta-v3-small often uses LABEL_0 for contradiction
                    if label == 'label_0':
                        contradiction_score = score_dict['score']
                        break

            return round(float(1.0 - contradiction_score), 2)

        except Exception as e:
            print(f"[NLIAuditor] Error during NLI inference: {e}")
            return 0.5

    def compute_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Hybrid NLI + STS scoring for coherence/grounding measurement.

        Combines two signals:
        1. NLI Non-Contradiction (0.35 weight): Logical consistency check
           - Penalizes logically contradictory pairs
           - Returns: 1.0 - contradiction_probability

        2. STS Similarity (0.65 weight): Semantic alignment measurement
           - Measures how semantically similar the two texts are
           - Returns: cosine similarity of embeddings [0, 1]

        Combined score provides:
        - Logical soundness (NLI): Ensures no contradictions
        - Semantic relevance (STS): Measures actual alignment/coherence

        Returns a float between 0.0 and 1.0.
        """
        try:
            # Get NLI signal: logical consistency (non-contradiction)
            nli_score = self._compute_nli_non_contradiction(premise, hypothesis)

            # Get STS signal: semantic similarity
            sts_score = self.compute_sts_similarity(premise, hypothesis)

            # Combine signals with learned weights
            # STS weighted higher (0.65) because it directly measures semantic alignment
            # NLI weighted lower (0.35) as a consistency check to avoid contradictions
            combined_score = (sts_score * 0.65) + (nli_score * 0.35)

            return round(float(combined_score), 2)

        except Exception as e:
            print(f"[NLIAuditor] Error during hybrid scoring: {e}")
            # Fallback to neutral score on catastrophic failure
            return 0.5
