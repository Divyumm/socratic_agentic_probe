import warnings
from transformers import pipeline

# Suppress Hugging Face warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class LocalGenerator:
    """
    On-Device Generative LLM using a Small Language Model (SLM).
    Used as a drop-in replacement for the Claude API for the Advocate,
    Evaluator, and Quality Auditor agents.
    """
    _instance = None

    def __new__(cls):
        # Singleton pattern to prevent reloading the heavy generative model multiple times
        if cls._instance is None:
            cls._instance = super(LocalGenerator, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        print("[LocalGenerator] Loading on-device SLM (Qwen/Qwen2.5-0.5B-Instruct)...")
        # We use a tiny instruction-tuned model for text generation
        # It's < 1GB and runs comfortably on a standard CPU
        self.generator = pipeline(
            "text-generation", 
            model="Qwen/Qwen2.5-0.5B-Instruct",
            device_map="auto" # Will use MPS (Apple Silicon GPU) if available, else CPU
        )
        print("[LocalGenerator] Model loaded successfully.")

    def generate(self, prompt: str, temperature: float = 0.7, max_new_tokens: int = 150) -> str:
        """
        Generates text using the local SLM with the specified temperature.
        """
        try:
            # Format as a strict user message for the instruction-tuned model
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Apply temperature logic. Transformers pipeline requires do_sample=True for temp > 0.0
            # If temp is extremely low (e.g., 0.1), it's close to greedy decoding.
            do_sample = temperature > 0.05
            
            outputs = self.generator(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            
            # The pipeline returns the full conversation history. We extract just the assistant's reply.
            response_text = outputs[0]["generated_text"][-1]["content"].strip()
            return response_text
        except Exception as e:
            print(f"[LocalGenerator] Error during generation: {e}")
            return "Local model generation failed."
