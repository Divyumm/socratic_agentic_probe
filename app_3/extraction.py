from typing import List
from app_3.schemas import Claim, EpistemicMap, PapanekDimension
from app_3.llm_client import MockLLMClient
from app_3.document_parser import DocumentParser

class ExtractionEngine:
    """Coordinates Document Parsing and Epistemic Map Extraction."""

    def __init__(self):
        self.llm_client = MockLLMClient()

    def extract_epistemic_map(self, file_path: str) -> EpistemicMap:
        """Parses a document, extracts claims, computes vulnerability scores, and builds the EpistemicMap."""
        parser = DocumentParser(file_path)
        chunks = parser.parse()
        
        filename = parser.file_path.name
        raw_claims = self.llm_client.extract_claims(filename, chunks)
        
        # Load claims into Pydantic models
        claims: List[Claim] = []
        for c in raw_claims:
            claims.append(Claim(
                id=c["id"],
                text=c["text"],
                is_implicit=c["is_implicit"],
                dimension=c["dimension"],
                confidence=c["confidence"],
                source_passage=c["source_passage"],
                page=c["page"],
                probing_questions=c.get("probing_questions", []),
                vulnerability_rank=0 # Will be calculated below
            ))

        # Compute dimension frequencies (discussion volume)
        dimension_counts = {dim: 0 for dim in PapanekDimension}
        for claim in claims:
            dimension_counts[claim.dimension] += 1
            
        max_claims_in_any_dim = max(dimension_counts.values()) if claims else 1

        # Calculate Vulnerability Score and rank claims
        # Vulnerability = (1.0 - confidence) * (2.0 - (claims_in_dim / max_claims_in_any_dim))
        # High vulnerability means low confidence in a dimension the student wrote least about.
        scored_claims = []
        for claim in claims:
            dim_ratio = dimension_counts[claim.dimension] / max_claims_in_any_dim
            vulnerability_score = (1.0 - claim.confidence) * (2.0 - dim_ratio)
            scored_claims.append((vulnerability_score, claim))

        # Sort descending by vulnerability score (highest score first = most vulnerable = rank 1)
        scored_claims.sort(key=lambda x: x[0], reverse=True)

        # Assign ranks
        for rank, (_, claim) in enumerate(scored_claims):
            claim.vulnerability_rank = rank + 1

        # Return sorted claims
        sorted_claims = [c for _, c in scored_claims]

        return EpistemicMap(
            document_name=filename,
            claims=sorted_claims
        )
