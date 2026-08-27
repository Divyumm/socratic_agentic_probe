import warnings
from transformers import pipeline

# Suppress Hugging Face warnings for cleaner console output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class NLIAuditor:
    """
    On-Device Natural Language Inference proxy for calculating Grounding Scores.
    Uses cross-encoder/nli-deberta-v3-small to determine if a source document
    logically entails the student's response claim.
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
        # task "text-classification" handles NLI with entailment, neutral, contradiction
        self.classifier = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-small")
        print("[NLIAuditor] Model loaded successfully.")

    def compute_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Calculates the entailment probability between the premise and hypothesis.
        Returns a float between 0.0 and 1.0 representing the NLI score.
        """
        try:
            # DeBERTa-v3 cross-encoder format is essentially passing the pair
            # For pipeline text-classification with cross encoders, we pass a dict or string with sep token
            result = self.classifier({"text": premise, "text_pair": hypothesis})
            
            # DeBERTa-v3 output labels: LABEL_0 (Contradiction), LABEL_1 (Entailment), LABEL_2 (Neutral)
            
            entailment_score = 0.0
            
            # Get all scores
            all_scores = self.classifier({"text": premise, "text_pair": hypothesis}, top_k=None)
            
            # Some models map LABEL_1 to entailment.
            for score_dict in all_scores:
                label = score_dict['label'].lower()
                # nli-deberta-v3-small maps: 'entailment' or 'LABEL_1'
                if 'entailment' in label or label == 'label_1':
                    entailment_score = score_dict['score']
                    break
                    
            return round(float(entailment_score), 2)
            
        except Exception as e:
            print(f"[NLIAuditor] Error during inference: {e}")
            # Fallback to a neutral score on catastrophic failure
            return 0.35
