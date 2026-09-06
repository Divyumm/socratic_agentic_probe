"""Foundedness and self-consistency analysis over a document's epistemic map.

This is the half of the system that entailment can actually answer. It asks two
questions, both of which are logical relations between propositions:

  1. FOUNDEDNESS  - does a claim's own source passage entail it, stay silent on
     it, or contradict it? A claim the document never supports is an unfounded
     assumption, which is the thing the temperature-sampling experiments were
     reaching for and could not reach, because the backend they ran on measured
     similarity rather than entailment.

  2. SELF-CONSISTENCY - do any two claims in the document contradict each other?
     Long generated documents drift: specs change between sections, numbers stop
     reconciling. A person who actually built the thing has one underlying
     reality to refer back to, so their claims agree.

Deliberately NOT a quality judgement. A passage does not entail a good answer any
more than a mediocre one, so entailment cannot rank quality - that belongs to the
end-of-session auditor. Everything here is exhaustive and deterministic: it reads
every claim and every related pair in one pass, so unlike the 5-of-91 dialogue
sampling there is nothing to get lucky against.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# Verdict bands on the ordinal support scale (1.0 entailed / 0.5 neutral / 0.0
# contradicted). Deliberately asymmetric and conservative: "contradicted" is the
# accusatory verdict, so it needs to be clearly below neutral, while anything
# merely unsupported is reported as UNSUPPORTED rather than as an error.
FOUNDED_AT = 0.55
CONTRADICTED_AT = 0.40

# Two claims must be about the same thing before disagreement means anything.
# Without this gate the lowest-support pairs in a document are simply unrelated
# ones - measured on a real 91-claim map, the "most contradictory" pairs were
# Raspberry Pi vs. retrieval-augmented generation.
RELATEDNESS_AT = 0.55

FOUNDED = "FOUNDED"
UNSUPPORTED = "UNSUPPORTED"
CONTRADICTED = "CONTRADICTED"

# Viva evidence is read ONE-SIDED: it can raise a verdict, never lower one.
#
# Entailment of a claim by the dialogue is the right test for "did the student
# ground this", but it fails asymmetrically. When a question asks how the student's
# design AVOIDS a problem the claim states, a good answer describes the design and
# does not assert the claim - and the model reads the mismatch as contradiction.
# griot turn 11 is the case: claim "digital devices generate cues that provide the
# illusion of agency", answer "our interface has no notifications and relies on
# genuine intent", viva support 0.164. The student is right; the measurement is not.
#
# So a low viva score means "the dialogue was not shown to support this claim",
# never "the student contradicted it". VIVA_NOT_SHOWN exists so the UI cannot
# render that number as evidence against the student.
VIVA_SUPPORTS = "SUPPORTS"
VIVA_NOT_SHOWN = "NOT_SHOWN"


@dataclass
class ClaimVerdict:
    claim_id: str
    text: str
    support: float
    verdict: str
    nearest_sentence: Optional[str] = None
    # Foundedness has two independent sources, kept separate on purpose. A claim
    # the documentation never supports may still be defended in the viva - that is
    # precisely what an oral examination is for - and collapsing the two would
    # erase the distinction between "documented" and "defended under questioning".
    doc_support: Optional[float] = None
    viva_support: Optional[float] = None
    founded_by: Optional[str] = None      # "document" | "viva" | None
    # VIVA_SUPPORTS or VIVA_NOT_SHOWN - never a contradiction verdict. See the
    # constants above for why viva evidence is only ever read one-sided.
    viva_verdict: Optional[str] = None


@dataclass
class ContradictionPair:
    claim_a_id: str
    claim_b_id: str
    text_a: str
    text_b: str
    relatedness: float
    support: float


@dataclass
class DocumentReport:
    document_name: str
    claims: List[ClaimVerdict] = field(default_factory=list)
    contradictions: List[ContradictionPair] = field(default_factory=list)
    self_consistency: float = 0.0
    pairs_considered: int = 0
    pairs_related: int = 0

    @property
    def counts(self) -> Dict[str, int]:
        out = {FOUNDED: 0, UNSUPPORTED: 0, CONTRADICTED: 0}
        for c in self.claims:
            out[c.verdict] += 1
        return out

    @property
    def foundedness_rate(self) -> float:
        return self.counts[FOUNDED] / len(self.claims) if self.claims else 0.0

    def unfounded(self) -> List[ClaimVerdict]:
        """Claims the document does not support, worst first."""
        return sorted(
            [c for c in self.claims if c.verdict != FOUNDED],
            key=lambda c: c.support,
        )


def _verdict(support: float) -> str:
    if support >= FOUNDED_AT:
        return FOUNDED
    if support <= CONTRADICTED_AT:
        return CONTRADICTED
    return UNSUPPORTED


def assess_claim(auditor, claim: Dict[str, Any],
                 dialogue: Optional[str] = None) -> ClaimVerdict:
    """Is this claim supported by its source passage, or failing that, by the viva?

    dialogue, when given, is what the student actually said about THIS claim. It is
    scored as a second, independent body of evidence: a claim the document leaves
    unsupported but the student grounds convincingly under questioning is founded -
    just founded by the viva rather than by the write-up. The stronger of the two
    sets the verdict, and founded_by records which one carried it.
    """
    text = claim["text"]
    passage = (claim.get("source_passage") or "").strip()

    doc_support, nearest = None, None
    if passage:
        res = auditor.compute_support_vs_document(passage, text)
        doc_support = round(float(res["support"]), 4)
        nearest = res.get("top_sentence")

    viva_support = None
    if dialogue and dialogue.strip():
        viva_support = round(
            float(auditor.compute_support_vs_document(dialogue, text)["support"]), 4
        )

    # The verdict is anchored on the DOCUMENT, and the viva may only lift it.
    # Written as an explicit floor rather than relying on max() over both, so the
    # one-sidedness above is a property of the code and not an accident of which
    # combiner happens to be used here.
    best = doc_support if doc_support is not None else 0.5
    founded_by = "document" if _verdict(best) == FOUNDED else None

    viva_verdict = None
    if viva_support is not None:
        viva_verdict = (VIVA_SUPPORTS if viva_support >= FOUNDED_AT
                        else VIVA_NOT_SHOWN)
        if viva_support > best:
            best = viva_support
            if _verdict(best) == FOUNDED:
                founded_by = "viva"

    return ClaimVerdict(
        claim_id=claim["id"], text=text,
        support=round(float(best), 4), verdict=_verdict(best),
        nearest_sentence=nearest,
        doc_support=doc_support, viva_support=viva_support, founded_by=founded_by,
        viva_verdict=viva_verdict,
    )


def find_contradictions(
    auditor, claims: List[Dict[str, Any]],
    relatedness_at: float = RELATEDNESS_AT,
) -> Tuple[List[ContradictionPair], int, int, float]:
    """Cross-claim contradictions, gated by relatedness.

    The gate is not just for precision - it is what makes this affordable. The
    cross-encoder is a forward pass per pair, and a 91-claim map has 4,095 pairs;
    cosine over cached embeddings is nearly free, so relatedness is screened
    first and only the surviving handful reach the entailment model.
    """
    texts = [c["text"] for c in claims]
    n = len(texts)
    if n < 2:
        return [], 0, 0, 1.0

    emb = auditor._encode(texts)
    unit = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    cos = unit @ unit.T

    iu = np.triu_indices(n, 1)
    related = [(int(i), int(j)) for i, j in zip(*iu) if cos[i, j] >= relatedness_at]
    total_pairs = len(iu[0])
    if not related:
        return [], total_pairs, 0, 1.0

    # Symmetric reading: A may entail B without B entailing A, and a genuine
    # contradiction shows up in either direction, so take the weaker of the two.
    prem = [texts[i] for i, _ in related] + [texts[j] for _, j in related]
    hyp = [texts[j] for _, j in related] + [texts[i] for i, _ in related]
    sup = auditor._support_pairs(prem, hyp)
    m = len(related)
    pair_support = np.minimum(sup[:m], sup[m:])

    pairs = [
        ContradictionPair(
            claim_a_id=claims[i]["id"], claim_b_id=claims[j]["id"],
            text_a=texts[i], text_b=texts[j],
            relatedness=round(float(cos[i, j]), 4),
            support=round(float(s), 4),
        )
        for (i, j), s in zip(related, pair_support)
        if s <= CONTRADICTED_AT
    ]
    pairs.sort(key=lambda p: p.support)
    return pairs, total_pairs, m, float(np.mean(pair_support))


def dialogue_by_claim(transcript) -> Dict[int, str]:
    """What the student said about each claim, keyed by the claim's INDEX in the map.

    Keyed by index, not id, on purpose. Maps extracted before ids were made unique
    hold many claims per id - the griot map has 91 claims sharing 10 ids - so an
    id-keyed mapping attributes one turn's dialogue to every claim that happens to
    share its id, inventing viva evidence for claims that were never probed.

    The question disambiguates: each claim carries its own probing_questions, so the
    claim whose list contains this turn's question is the one that was probed. A
    turn whose question matches no claim is dropped rather than guessed at.
    """
    turns = transcript["turns"] if isinstance(transcript, dict) else transcript.turns
    claims = (transcript["epistemic_map"]["claims"] if isinstance(transcript, dict)
              else [c.model_dump() for c in transcript.epistemic_map.claims])

    out: Dict[int, List[str]] = {}
    for t in turns:
        t = t if isinstance(t, dict) else t.model_dump()
        if not (t.get("student_response") or "").strip():
            continue
        idxs = [i for i, c in enumerate(claims) if c["id"] == t["claim_id"]]
        if not idxs:
            continue
        target = idxs[0]
        if len(idxs) > 1:
            matched = [i for i in idxs
                       if t["question"] in (claims[i].get("probing_questions") or [])]
            if len(matched) != 1:
                # Ambiguous: several identically-named claims, none or many matching
                # the question. Attributing it anywhere would fabricate evidence.
                continue
            target = matched[0]
        out.setdefault(target, []).append(t["student_response"])
    return {k: "\n".join(v) for k, v in out.items()}


def to_serialisable(report: DocumentReport) -> Dict[str, Any]:
    """Plain-dict form of a DocumentReport, safe for JSON storage on a transcript.

    consistency.py's types are plain dataclasses, not pydantic models, so they do
    not serialise through SessionTranscript.model_dump_json() on their own. This
    is the boundary: compute once with analyse_document, flatten with this
    function, store the dict on transcript.document_integrity. Cheap to re-render
    from the dict on every dashboard rerun; the expensive part (the actual
    cross-encoder pass) never runs twice for the same session.
    """
    return {
        "document_name": report.document_name,
        "self_consistency": report.self_consistency,
        "pairs_considered": report.pairs_considered,
        "pairs_related": report.pairs_related,
        "counts": report.counts,
        "foundedness_rate": report.foundedness_rate,
        "claims": [
            {
                "claim_id": c.claim_id, "text": c.text, "support": c.support,
                "verdict": c.verdict, "nearest_sentence": c.nearest_sentence,
                "doc_support": c.doc_support, "viva_support": c.viva_support,
                "founded_by": c.founded_by, "viva_verdict": c.viva_verdict,
            }
            for c in report.claims
        ],
        "contradictions": [
            {
                "claim_a_id": p.claim_a_id, "claim_b_id": p.claim_b_id,
                "text_a": p.text_a, "text_b": p.text_b,
                "relatedness": p.relatedness, "support": p.support,
            }
            for p in report.contradictions
        ],
    }


def analyse_document(epistemic_map, auditor=None, transcript=None) -> DocumentReport:
    """Full exhaustive pass: every claim, every related pair.

    Pass `transcript` to let the viva count as evidence: a claim the write-up
    leaves unsupported but the student grounds under questioning is reported as
    FOUNDED with founded_by="viva".
    """
    from app_3.nli_auditor import NLIAuditor

    auditor = auditor or NLIAuditor()
    claims = [
        c if isinstance(c, dict) else c.model_dump()
        for c in (
            epistemic_map["claims"] if isinstance(epistemic_map, dict)
            else epistemic_map.claims
        )
    ]
    name = (
        epistemic_map["document_name"] if isinstance(epistemic_map, dict)
        else epistemic_map.document_name
    )

    dialogue = dialogue_by_claim(transcript) if transcript is not None else {}
    verdicts = [assess_claim(auditor, c, dialogue.get(i)) for i, c in enumerate(claims)]
    contradictions, total, related, mean_support = find_contradictions(auditor, claims)

    return DocumentReport(
        document_name=name,
        claims=verdicts,
        contradictions=contradictions,
        self_consistency=round(mean_support, 4),
        pairs_considered=total,
        pairs_related=related,
    )
