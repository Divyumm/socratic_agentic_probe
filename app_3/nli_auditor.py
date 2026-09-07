import re
import warnings
from typing import Dict, List, Optional

import numpy as np

# Suppress Hugging Face warnings for cleaner console output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_MIN_SENTENCE_WORDS = 4


class NLIAuditor:
    """
    Entailment and similarity auditor over frozen sentence embeddings.

    Two signals, each answering a different question:

    - Support (calibrated 3-class head): does the hypothesis follow from,
      sit neutral to, or contradict the premise? Returned as one continuous
      score in [0, 1] -- the expectation over an ordinal label scale, where
      1.0 is full entailment, 0.5 genuinely neutral, 0.0 flat contradiction.

    - Similarity (cosine): are the two texts about the same thing? This is a
      relevance measure, not a logical one, and is used where the pair is not
      a well-posed entailment problem (notably question-answer pairs, since a
      question asserts nothing and so cannot be entailed or contradicted).

    Two backends provide support, selected by config.NLI_BACKEND:

    - "cross-encoder" (default): reads premise and hypothesis jointly, so negation
      and substitution are visible. This is the only backend that actually does
      entailment.
    - "linear-head": a calibrated logistic regression over frozen MiniLM
      embeddings (train_nli_head.py). Retained for comparison and as a fallback,
      but MiniLM is trained for SIMILARITY and discards negation, so this backend
      measures topical overlap wearing the name "support". On a controlled probe
      it scored a flat contradiction (0.822) HIGHER than an unrelated sentence
      (0.353), and rated a falsified claim above the true one it was derived from.

    The earlier objection to a cross-encoder was saturation (0.99/0.01). That was
    really a symptom of nli-deberta-v3-small mislabelling unrelated text as
    contradiction; a model trained on more diverse NLI data returns a genuine
    neutral (0.508 on that probe), so the ordinal scale spreads as intended.

    IMPORTANT: support answers an entailment question, not a quality question. A
    passage does not entail a good answer any more than a mediocre one - both are
    new text, and neutral is the correct verdict. Use it to detect contradiction
    and foundedness; use the end-of-session auditor to judge quality.
    """

    _instance = None

    def __new__(cls):
        # Singleton pattern to prevent reloading the model in memory multiple times
        if cls._instance is None:
            instance = super(NLIAuditor, cls).__new__(cls)
            try:
                instance._initialize()
                cls._instance = instance
            except Exception:
                raise
        return cls._instance

    def _initialize(self):
        print("[NLIAuditor] Loading embedding model (all-MiniLM-L6-v2)...")
        from sentence_transformers import SentenceTransformer, util

        self.sts_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.sts_util = util
        print("[NLIAuditor] Embedding model loaded.")

        self._head = None
        # Optional cross-encoder entailment backend. Loaded lazily on first use so
        # importing NLIAuditor stays cheap when the linear head is selected.
        self._ce_tok = None
        self._ce_model = None
        self._ce_label_idx = None
        self._ordinal = None
        self._warned_no_head = False
        self._load_head()

    def _load_head(self):
        """Loads the calibrated entailment head if it has been trained."""
        import joblib

        from app_3.train_nli_head import NLI_HEAD_FILE

        if not NLI_HEAD_FILE.exists():
            print(
                "[NLIAuditor] No trained entailment head found at "
                f"{NLI_HEAD_FILE.name}. Support scores will fall back to a "
                "similarity proxy, which cannot detect contradiction.\n"
                "             Train it with:  python -m app_3.train_nli_head"
            )
            return

        try:
            bundle = joblib.load(NLI_HEAD_FILE)
            self._head = bundle["classifier"]
            self._ordinal = np.asarray(bundle["ordinal_values"])
            print(
                f"[NLIAuditor] Entailment head loaded "
                f"({bundle.get('n_train', '?'):,} training pairs)."
            )
        except Exception as e:
            print(f"[NLIAuditor] Failed to load entailment head: {e}")
            self._head = None

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _encode(self, texts):
        """Encodes one string or a list of strings to numpy embeddings."""
        return self.sts_model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )

    @staticmethod
    def _pair_features(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """[u, v, |u-v|, u*v] -- see train_nli_head.pair_features for rationale."""
        u = np.atleast_2d(u)
        v = np.atleast_2d(v)
        return np.hstack([u, v, np.abs(u - v), u * v])

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Splits a passage into scoreable sentences, dropping fragments."""
        if not text:
            return []
        parts = _SENTENCE_SPLIT.split(text.strip())
        return [
            p.strip()
            for p in parts
            if p and len(p.split()) >= _MIN_SENTENCE_WORDS
        ]

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def compute_sts_similarity(self, text1: str, text2: str) -> float:
        """
        Cosine similarity between two texts, rescaled from [-1, 1] to [0, 1].

        Measures topical relevance. It cannot represent contradiction: a
        statement and its negation share almost all their vocabulary and so sit
        high on this scale.
        """
        try:
            emb = self._encode([text1, text2])
            cos = float(np.dot(emb[0], emb[1]) / (
                np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-9
            ))
            return round((cos + 1.0) / 2.0, 4)
        except Exception as e:
            print(f"[NLIAuditor] Error computing similarity: {e}")
            return 0.5

    def _ensure_cross_encoder(self) -> bool:
        """Load the cross-encoder on first use. False if unavailable."""
        from app_3.config import NLI_CROSS_ENCODER_MODEL
        if self._ce_model is not None:
            return True
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            print(f"[NLIAuditor] Loading cross-encoder ({NLI_CROSS_ENCODER_MODEL})...")
            self._ce_tok = AutoTokenizer.from_pretrained(NLI_CROSS_ENCODER_MODEL)
            self._ce_model = AutoModelForSequenceClassification.from_pretrained(NLI_CROSS_ENCODER_MODEL)
            self._ce_model.eval()
            # Label order differs between checkpoints, so resolve it by name.
            self._ce_label_idx = {v.lower(): k for k, v in self._ce_model.config.id2label.items()}
            print("[NLIAuditor] Cross-encoder loaded.")
            return True
        except Exception as e:
            print(f"[NLIAuditor] Cross-encoder unavailable ({e}); using the linear head.")
            self._ce_model = None
            return False

    def _support_cross_encoder(self, premises: List[str], hypotheses: List[str]) -> np.ndarray:
        """Ordinal support for text pairs: P(entail) + 0.5 * P(neutral).

        Same scale the linear head emits, so every caller is unchanged: 1.0 the
        hypothesis follows, 0.5 the evidence is silent on it, 0.0 it is contradicted.
        """
        import torch
        out = []
        BATCH = 16
        with torch.no_grad():
            for i in range(0, len(premises), BATCH):
                enc = self._ce_tok(
                    premises[i:i + BATCH], hypotheses[i:i + BATCH],
                    return_tensors="pt", truncation=True, max_length=512, padding=True,
                )
                probs = torch.softmax(self._ce_model(**enc).logits, dim=-1).numpy()
                e = probs[:, self._ce_label_idx["entailment"]]
                n = probs[:, self._ce_label_idx["neutral"]]
                out.extend(e + 0.5 * n)
        return np.asarray(out, dtype=float)

    def _support_pairs(self, premises: List[str], hypotheses: List[str]) -> np.ndarray:
        """Support for aligned text pairs, via whichever backend is configured."""
        from app_3.config import NLI_BACKEND
        if NLI_BACKEND == "cross-encoder" and self._ensure_cross_encoder():
            return self._support_cross_encoder(premises, hypotheses)
        emb_p = self._encode(premises)
        emb_h = self._encode(hypotheses)
        return self._support_from_embeddings(emb_p, emb_h)

    def _support_from_embeddings(self, emb_p: np.ndarray, emb_h: np.ndarray) -> np.ndarray:
        """
        Batched support scores for aligned premise/hypothesis embedding arrays.

        Returns the expectation over the ordinal label scale, which is
        equivalently P(entail) + 0.5 * P(neutral) -- the same expression you get
        from rescaling P(entail) - P(contradict) into [0, 1].
        """
        if self._head is None:
            # Degraded path: cosine cannot separate neutral from contradiction,
            # so compress into the upper half of the range to avoid asserting a
            # contradiction we have no way of detecting.
            emb_p = np.atleast_2d(emb_p)
            emb_h = np.atleast_2d(emb_h)
            num = np.sum(emb_p * emb_h, axis=1)
            den = np.linalg.norm(emb_p, axis=1) * np.linalg.norm(emb_h, axis=1) + 1e-9
            cos = num / den
            return 0.5 + 0.5 * np.clip(cos, 0.0, 1.0)

        X = self._pair_features(emb_p, emb_h)
        probs = self._head.predict_proba(X)
        return probs @ self._ordinal

    def compute_support(self, premise: str, hypothesis: str) -> float:
        """
        Continuous support score in [0, 1] for a single premise/hypothesis pair.

        1.0 = the hypothesis follows from the premise
        0.5 = neutral, the premise neither supports nor refutes it
        0.0 = the hypothesis contradicts the premise
        """
        try:
            emb = self._encode([premise, hypothesis])
            return round(float(self._support_pairs([premise], [hypothesis])[0]), 4)
        except Exception as e:
            print(f"[NLIAuditor] Error computing support: {e}")
            return 0.5

    def compute_support_vs_document(
        self, document: str, hypothesis: str, weighting_query: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Scores a statement against a multi-sentence document.

        Embedding a long passage as one vector averages it into mush and loses
        the specificity entailment needs, so the document is split and scored
        sentence by sentence. Sentences are then combined by a relevance-weighted
        mean rather than a max or a min: hard selection operators make the score
        jump whenever the winning sentence changes, whereas softmax weighting
        over cosine relevance varies smoothly as the answer changes, and lets
        irrelevant sentences fall out of the average instead of dragging it
        toward neutral.

        Returns:
            support        -- relevance-weighted mean support in [0, 1]
            min_support    -- most-contradicted sentence, the contradiction signal
            top_sentence   -- the sentence carrying the most weight, for display
            n_sentences    -- how many sentences were scored
        """
        sentences = self.split_sentences(document)
        if not sentences:
            return {
                "support": self.compute_support(document, hypothesis),
                "min_support": self.compute_support(document, hypothesis),
                "top_sentence": document[:200],
                "n_sentences": 0,
            }

        try:
            emb_sent = self._encode(sentences)
            emb_hyp = self._encode([hypothesis])[0]
            
            query_str = weighting_query if weighting_query else hypothesis
            emb_query = self._encode([query_str])[0] if weighting_query else emb_hyp

            emb_hyp_tiled = np.tile(emb_hyp, (len(sentences), 1))
            supports = self._support_pairs(sentences, [hypothesis] * len(sentences))

            # Relevance weights: softmax over cosine similarity to the query.
            num = emb_sent @ emb_query
            den = np.linalg.norm(emb_sent, axis=1) * np.linalg.norm(emb_query) + 1e-9
            cos = num / den
            # Restrict to the most relevant sentences before weighting. A long,
            # messy PDF passage (tables, headers, fragments) otherwise dilutes a
            # decisive entailment against dozens of irrelevant sentences, pulling
            # every verdict toward neutral. Top-k keeps the sentences that are
            # actually about the hypothesis, and the softmax then weights among
            # those. k scales with passage length so short passages are unaffected.
            k = max(3, min(len(sentences), int(np.ceil(len(sentences) * 0.25))))
            if k < len(sentences):
                keep = np.argsort(cos)[-k:]
                cos = cos[keep]
                supports = supports[keep]

            logits = cos / 0.1  # temperature: sharp enough to ignore noise
            weights = np.exp(logits - logits.max())
            weights = weights / weights.sum()

            weighted = float(np.sum(weights * supports))
            return {
                "support": round(weighted, 4),
                "min_support": round(float(supports.min()), 4),
                "top_sentence": sentences[int(weights.argmax())],
                "n_sentences": len(sentences),
            }
        except Exception as e:
            print(f"[NLIAuditor] Error scoring against document: {e}")
            return {
                "support": 0.5,
                "min_support": 0.5,
                "top_sentence": "",
                "n_sentences": len(sentences),
            }

    def score_document_against_many(
        self, document: str, hypotheses: List[str]
    ) -> List[float]:
        """
        Scores several hypotheses against one document, embedding the document's
        sentences a single time.

        Scoring N hypotheses with N separate compute_support_vs_document calls
        re-splits and re-embeds the same document N times. For the rubric that is
        six passes over the full evidence portfolio per turn, so this batches
        them into one.
        """
        sentences = self.split_sentences(document)
        if not sentences or not hypotheses:
            return [self.compute_support(document, h) for h in hypotheses]

        try:
            emb_sent = self._encode(sentences)
            emb_hyps = self._encode(list(hypotheses))
            sent_norms = np.linalg.norm(emb_sent, axis=1)

            results = []
            for hyp, emb_hyp in zip(hypotheses, emb_hyps):
                supports = self._support_pairs(sentences, [hyp] * len(sentences))

                cos = (emb_sent @ emb_hyp) / (
                    sent_norms * np.linalg.norm(emb_hyp) + 1e-9
                )
                logits = cos / 0.1
                weights = np.exp(logits - logits.max())
                weights = weights / weights.sum()
                results.append(round(float(np.sum(weights * supports)), 4))
            return results
        except Exception as e:
            print(f"[NLIAuditor] Error in batched document scoring: {e}")
            return [0.5] * len(hypotheses)

    def score_rubric(self, evidence: str) -> Dict[str, float]:
        """
        Scores evidence against each rubric criterion independently.

        Each criterion is a declarative proposition (see rubric.py), so this is a
        well-posed entailment question in a way that scoring against a "must"
        statement never was. Per-criterion scores also tell a faculty screener
        which specific criterion failed rather than handing them one aggregate.
        """
        from app_3.rubric import RUBRIC_CRITERIA

        supports = self.score_document_against_many(
            evidence, [c.proposition for c in RUBRIC_CRITERIA]
        )
        return {c.id: s for c, s in zip(RUBRIC_CRITERIA, supports)}

    # ------------------------------------------------------------------
    # Backwards-compatible entry point
    # ------------------------------------------------------------------

    def compute_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Blended support + similarity score, retained for existing call sites.

        Prefer compute_support (logical relation) or compute_sts_similarity
        (topical relevance) directly, since they answer different questions and
        blending them makes the result harder to interpret.
        """
        try:
            support = self.compute_support(premise, hypothesis)
            similarity = self.compute_sts_similarity(premise, hypothesis)
            return round(float(0.65 * support + 0.35 * similarity), 4)
        except Exception as e:
            print(f"[NLIAuditor] Error during blended scoring: {e}")
            return 0.5

    def compute_qa_coherence(self, question: str, answer: str) -> float:
        """
        Coherence between a question and an answer.

        Deliberately similarity-only. A question has no truth value, so nothing
        can entail or contradict it; running entailment here is a category error
        and was the source of the polarised scores. What we can legitimately ask
        is whether the answer is *about* what was asked, which is relevance.

        For the logical relation, score the answer against the claim the question
        was generated from -- that pair is proposition against proposition.
        """
        return self.compute_sts_similarity(question, answer)
