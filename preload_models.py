import os

print("Pre-loading Sentence Transformer...")
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")

print("Pre-loading Qwen Instruct...")
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

print("Pre-loading Cross Encoder...")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-small")
AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-small")

print("All models successfully pre-loaded to cache!")
