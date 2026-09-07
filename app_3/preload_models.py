"""
Run this once on a new machine, before starting any Streamlit page, to
download and cache every model the app needs. Without this, the first live
session pauses mid-conversation while Qwen/MiniLM/the cross-encoder download.

    python -m app_3.preload_models
"""
from app_3.local_llm import LocalGenerator
from app_3.nli_auditor import NLIAuditor


def main():
    print("=== 1/3: Advocate/Assessor SLM (Qwen2.5-0.5B-Instruct) ===")
    LocalGenerator()

    print("\n=== 2/3: Sentence embeddings (all-MiniLM-L6-v2) ===")
    auditor = NLIAuditor()  # loads MiniLM eagerly in __init__

    print("\n=== 3/3: Cross-encoder entailment backend (DeBERTa-v3-mnli) ===")
    ok = auditor._ensure_cross_encoder()
    if not ok:
        print("Cross-encoder failed to load - check network access and the "
              "model name in config.NLI_CROSS_ENCODER_MODEL.")
        return

    print("\nAll models downloaded and cached. Subsequent runs load from "
          "the local Hugging Face cache (~/.cache/huggingface) with no "
          "network required.")


if __name__ == "__main__":
    main()
