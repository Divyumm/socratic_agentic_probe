"""
Trains a calibrated 3-class entailment head over frozen sentence embeddings.

Why this exists
---------------
The cross-encoder (nli-deberta-v3-small) is a fine-tuned 100M-parameter network
optimised against one-hot labels. It is architecturally overconfident: it emits
0.99 / 0.01 and very little in between, which is why the derived scores came out
effectively binary.

This replaces it with the standard Sentence-BERT pair-classification setup
(Reimers & Gurevych, 2019): freeze the embeddings, build the pair feature vector
[u, v, |u-v|, u*v], and fit an L2-regularised multinomial logistic regression on
top. The regularisation penalty directly suppresses the large weights that
produce saturated logits, so the output spreads across [0, 1] by construction
rather than by post-hoc squashing. Isotonic calibration on held-out data then
makes the probabilities mean what they say (Guo et al., 2017).

Two further benefits: inference becomes a single matrix multiply over embeddings
we already compute for STS (so the deberta forward pass disappears entirely),
and the fitted coefficients are inspectable.

Usage
-----
    pip install datasets
    python -m app_3.train_nli_head
"""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, classification_report
from sklearn.model_selection import train_test_split

from app_3.config import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

NLI_HEAD_FILE = DATA_DIR / "nli_head.joblib"

# MNLI label order is fixed by the dataset: 0=entailment, 1=neutral, 2=contradiction
LABEL_NAMES = ["entailment", "neutral", "contradiction"]

# Ordinal values used to collapse the three probabilities into one support score.
# Reading: full support = 1.0, genuinely neutral = 0.5, flat contradiction = 0.0.
ORDINAL_VALUES = np.array([1.0, 0.5, 0.0])

N_TRAIN_PAIRS = 40_000
EMBED_BATCH = 256


def pair_features(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Standard SBERT pair encoding.

    Cosine similarity alone cannot represent contradiction: "I consulted users"
    and "I did not consult users" sit at ~0.85 cosine because they share a topic.
    The contradiction lives in the difference vector, which cosine discards.
    |u - v| keeps it; u * v carries elementwise agreement.

    Accepts single vectors (384,) or batches (n, 384).
    """
    u = np.atleast_2d(u)
    v = np.atleast_2d(v)
    return np.hstack([u, v, np.abs(u - v), u * v])


def _load_mnli(n_pairs: int):
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "The 'datasets' package is required to train the head.\n"
            "Install it with:  pip install datasets"
        )

    logger.info("Loading MultiNLI...")
    ds = load_dataset("nyu-mll/multi_nli", split="train")
    ds = ds.shuffle(seed=42).select(range(min(n_pairs, len(ds))))

    premises = [r for r in ds["premise"]]
    hypotheses = [r for r in ds["hypothesis"]]
    labels = np.array(ds["label"], dtype=int)

    # A handful of MNLI rows carry label -1 (no gold consensus); drop them.
    keep = labels >= 0
    premises = [p for p, k in zip(premises, keep) if k]
    hypotheses = [h for h, k in zip(hypotheses, keep) if k]
    labels = labels[keep]

    logger.info(f"Loaded {len(labels):,} labelled pairs.")
    return premises, hypotheses, labels


def _embed(model, texts):
    return model.encode(
        texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def train_nli_head(n_pairs: int = N_TRAIN_PAIRS):
    from sentence_transformers import SentenceTransformer

    premises, hypotheses, y = _load_mnli(n_pairs)

    logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    logger.info("Embedding premises...")
    emb_p = _embed(model, premises)
    logger.info("Embedding hypotheses...")
    emb_h = _embed(model, hypotheses)

    X = pair_features(emb_p, emb_h)
    logger.info(f"Feature matrix: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    logger.info("Fitting L2-regularised multinomial logistic regression...")
    base = LogisticRegression(C=1.0, max_iter=2000)
    base.fit(X_train, y_train)

    logger.info("Calibrating (isotonic, 3-fold)...")
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)
    preds = probs.argmax(axis=1)

    logger.info("\n" + classification_report(y_test, preds, target_names=LABEL_NAMES))

    # The number that actually matters here is the spread, not the accuracy:
    # confirm the head is not reproducing the cross-encoder's saturation.
    support = probs @ ORDINAL_VALUES
    logger.info("Support score distribution on held-out data:")
    logger.info(f"  min / max      : {support.min():.3f} / {support.max():.3f}")
    logger.info(f"  mean / std     : {support.mean():.3f} / {support.std():.3f}")
    for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        frac = float(((support >= lo) & (support < hi)).mean())
        logger.info(f"  [{lo:.1f}, {hi:.1f}) : {frac:6.1%} {'#' * int(frac * 50)}")

    saturated = float(((support < 0.05) | (support > 0.95)).mean())
    logger.info(f"  saturated (<0.05 or >0.95): {saturated:.1%}")

    brier = np.mean([
        brier_score_loss((y_test == i).astype(int), probs[:, i])
        for i in range(3)
    ])
    logger.info(f"  mean Brier score (lower is better): {brier:.4f}")

    NLI_HEAD_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "classifier": clf,
            "label_names": LABEL_NAMES,
            "ordinal_values": ORDINAL_VALUES,
            "embedding_model": "all-MiniLM-L6-v2",
            "n_train": int(len(y_train)),
            "feature_layout": ["u", "v", "abs(u-v)", "u*v"],
        },
        NLI_HEAD_FILE,
    )
    logger.info(f"\nSaved head to {NLI_HEAD_FILE}")
    return True


if __name__ == "__main__":
    train_nli_head()
