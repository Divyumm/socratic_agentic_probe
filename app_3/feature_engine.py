import math
from typing import Dict, Any, Optional

class FeatureEngine:
    """Computes field-knowledge-independent reasoning features from student responses."""

    def __init__(self):
        # Learned single-neuron classifier weights (logistic regression)
        # Incorporates Grounding (Fact-check), Coherence, Variance, and Circularity
        self.w_bias = 0.5
        self.w_coherence = 2.0
        self.w_grounding = 2.5
        self.w_variance = -2.0
        self.w_circularity = -1.5

        from app_3.config import MOCK_MODE
        from app_3.llm_client import LLMClient
        self.mock_mode = MOCK_MODE
        self.llm_client = LLMClient()

    def compute_circularity(self, question: str, response: str) -> float:
        """Measures self-referential repeating of the question's content words using a static word overlap heuristic."""
        q_clean = self._clean_text(question)
        r_clean = self._clean_text(response)
        
        q_words = set(w for w in q_clean.split() if len(w) > 4)
        r_words = set(w for w in r_clean.split() if len(w) > 4)
        
        if not q_words or not r_words:
            return 0.0
            
        repeated = q_words.intersection(r_words)
        ratio = len(repeated) / len(r_words)
        
        if len(r_clean.split()) < 8 and len(repeated) > 0:
            ratio = min(ratio * 1.5, 1.0)
            
        return round(ratio, 2)

    def compute_coherence(self, question: str, response: str) -> float:
        """Computes a static heuristic for conversational coherence (using word counts and contraction standardisation). Does not call an LLM."""
        r_clean = response.strip()
        word_count = len(r_clean.split())
        
        if word_count == 0:
            return 0.0
            
        if word_count > 25:
            base_coherence = 0.85
        elif word_count > 12:
            base_coherence = 0.65
        else:
            base_coherence = 0.40
            
        # Standardise contractions in check (e.g. "don't" -> "dont")
        response_norm = r_clean.lower().replace("'", "").replace("’", "")
        evasive_terms = ["i dont know", "dont know", "dont", "whatever", "maybe", "not sure", "not key", "irrelevant", "dont ask"]
        
        if any(term in response_norm for term in evasive_terms) or word_count < 4:
            base_coherence = max(base_coherence - 0.35, 0.05)
            
        return round(base_coherence, 2)

    def compute_grounding_score(self, response: str, claim_text: Optional[str]) -> float:
        """
        Calculates grounding using the on-device NLI model.
        Returns the entailment probability between the claim (evidence) and the response.
        """
        response_norm = response.strip().lower().replace("'", "").replace("’", "")
        if "dont know" in response_norm or "dont" in response_norm or len(response_norm.split()) < 4:
            return 0.0
            
        if not claim_text:
            return 0.80 # Default/high baseline if no claim context is provided
            
        try:
            from app_3.nli_auditor import NLIAuditor
            auditor = NLIAuditor()
            # We treat the original claim context as the Evidence, and the student's new response as the Claim to verify
            score = auditor.compute_grounding(evidence=claim_text, claim=response)
            return score
        except Exception as e:
            print(f"[FeatureEngine] NLI Auditor failed: {e}. Falling back to default.")
            return 0.35

    def compute_variance(self, word_count: int, is_vague: bool) -> float:
        """Computes a mathematical variance heuristic based on word count (sine/cosine curves). Does not simulate multi-temperature runs."""
        if word_count > 20 and not is_vague:
            return round(0.05 + (0.05 * math.sin(word_count)), 2)
        elif is_vague or word_count < 8:
            return round(0.45 + (0.3 * math.cos(word_count)), 2)
        else:
            return round(0.20 + (0.15 * math.sin(word_count)), 2)

    def compute_composite_confidence(self, coherence: float, grounding: float, variance: float, circularity: float,
                                     w_bias: Optional[float] = None,
                                     w_coherence: Optional[float] = None,
                                     w_grounding: Optional[float] = None,
                                     w_variance: Optional[float] = None,
                                     w_circularity: Optional[float] = None) -> float:
        """Implements the single-neuron logistic learned classifier or loads Version B multiclass."""
        from app_3.config import SCORING_VERSION, load_scoring_config
        
        if SCORING_VERSION == "B":
            config = load_scoring_config()
            weights = config.get("version_b_weights")
            intercepts = config.get("version_b_intercepts")
            classes = config.get("version_b_classes")
            
            if weights and intercepts and classes:
                # We have a trained Multiclass Logistic Regression model
                features = [coherence, grounding, variance, circularity]
                
                # Multiclass Logistic Regression: z = weights @ features + intercepts
                z_scores = []
                for i in range(len(classes)):
                    z = intercepts[i] + sum(weights[i][j] * features[j] for j in range(4))
                    z_scores.append(z)
                    
                # Softmax
                max_z = max(z_scores)
                exp_z = [math.exp(z - max_z) for z in z_scores]
                sum_exp_z = sum(exp_z)
                probs = [ez / sum_exp_z for ez in exp_z]
                
                # Find probability of "Grounded"
                if "Grounded" in classes:
                    grounded_idx = classes.index("Grounded")
                    return round(probs[grounded_idx], 2)
                    
        # Fallback to Version A (Heuristics)
        bias = w_bias if w_bias is not None else self.w_bias
        coherence_w = w_coherence if w_coherence is not None else self.w_coherence
        grounding_w = w_grounding if w_grounding is not None else self.w_grounding
        variance_w = w_variance if w_variance is not None else self.w_variance
        circularity_w = w_circularity if w_circularity is not None else self.w_circularity

        z = (bias + 
             (coherence_w * coherence) + 
             (grounding_w * grounding) + 
             (variance_w * variance) + 
             (circularity_w * circularity))
        
        sigmoid = 1.0 / (1.0 + math.exp(-z))
        return round(sigmoid, 2)

    def extract_features(self, question: str, response: str, claim_text: Optional[str] = None,
                          w_bias: Optional[float] = None,
                          w_coherence: Optional[float] = None,
                          w_grounding: Optional[float] = None,
                          w_variance: Optional[float] = None,
                          w_circularity: Optional[float] = None,
                          collapse_limit: Optional[float] = None,
                          unstable_limit: Optional[float] = None,
                          evaluator_temp: Optional[float] = None,
                          cluster_id: Optional[int] = None) -> Dict[str, Any]:
        """Runs the offline or online feature extraction pipeline on a turn."""
        response_clean = response.lower().strip()
        word_count = len(response_clean.split())

        # Shortcut for empty/sub-minimum answers AND canonical non-answers ("I don't
        # know", "not sure"...). These must be caught before live LLM grading: a
        # confidently-bad answer graded 3x tends to get near-zero variance (the
        # assessor agrees with itself), and since w_variance is negative that gives
        # it no penalty at all, letting genuine non-answers slip past collapse
        # detection depending on how the other scores land.
        response_norm = response_clean.replace("'", "").replace("’", "")
        non_answer_phrases = ["dont know", "no idea", "no clue", "not sure", "idk", "cant answer", "no answer"]
        is_non_answer = any(p in response_norm for p in non_answer_phrases)

        if word_count < 3 or is_non_answer:
            return {
                "coherence_score": 0.1,
                "grounding_score": 0.0,
                "variance_score": 0.5,
                "circularity_score": 0.0,
                "composite_confidence": 0.15,
                "is_vague": True,
                "word_count": word_count
            }
        vague_markers = ["i think", "maybe", "not sure", "perhaps", "probably", "essentially", "vague"]
        is_vague = any(marker in response_clean for marker in vague_markers) or "dont know" in response_clean.replace("'", "")

        bias = w_bias if w_bias is not None else self.w_bias
        coherence_w = w_coherence if w_coherence is not None else self.w_coherence
        grounding_w = w_grounding if w_grounding is not None else self.w_grounding
        variance_w = w_variance if w_variance is not None else self.w_variance
        circularity_w = w_circularity if w_circularity is not None else self.w_circularity

        if not self.mock_mode and self.llm_client.client is not None:
            # Delegate to live LLMClient intermediate-temp evaluation
            eval_data = self.llm_client.evaluate_response(
                question, response, claim_text or "",
                w_bias=bias,
                w_coherence=coherence_w,
                w_grounding=grounding_w,
                w_variance=variance_w,
                w_circularity=circularity_w,
                collapse_limit=collapse_limit,
                unstable_limit=unstable_limit,
                evaluator_temp=evaluator_temp
            )
            features_dict = {
                "coherence_score": eval_data["coherence_score"],
                "grounding_score": eval_data["grounding_score"],
                "variance_score": eval_data["variance_score"],
                "circularity_score": eval_data["circularity_score"],
                "composite_confidence": eval_data["composite_confidence"],
                "is_vague": is_vague,
                "word_count": word_count
            }
        else:
            circularity = self.compute_circularity(question, response)
            grounding = self.compute_grounding_score(response, claim_text)
            coherence = self.compute_coherence(question, response)
            if grounding <= 0.05:
                coherence = 0.05
            variance = self.compute_variance(word_count, is_vague)
            
            composite = self.compute_composite_confidence(
                coherence, grounding, variance, circularity,
                w_bias=bias,
                w_coherence=coherence_w,
                w_grounding=grounding_w,
                w_variance=variance_w,
                w_circularity=circularity_w
            )
            
            features_dict = {
                "coherence_score": coherence,
                "grounding_score": grounding,
                "variance_score": variance,
                "circularity_score": circularity,
                "composite_confidence": composite,
                "is_vague": is_vague,
                "word_count": word_count,
                "cluster_id": cluster_id
            }

        # POC Clustering Override
        if cluster_id is not None:
            # Map cluster to a base confidence score (this would normally be learned)
            # cluster 0 -> high, cluster 1 -> medium, cluster 2 -> low
            cluster_base_scores = {0: 0.8, 1: 0.5, 2: 0.2}
            base_score = cluster_base_scores.get(cluster_id, 0.5)
            # Combine cluster base score with structural variance parameter
            actual_variance = features_dict.get("variance_score", 0.5)
            new_composite = round((base_score * 0.7) + (actual_variance * 0.3), 2)
            features_dict["composite_confidence"] = new_composite
            features_dict["cluster_id"] = cluster_id

        return features_dict

    def _clean_text(self, text: str) -> str:
        """Removes punctuation and standardise spacing."""
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
