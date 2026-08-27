from pathlib import Path
from typing import List, Dict, Any
import pypdf
import re

class DocumentParser:
    """Handles loading and chunking PDF or text documents for epistemic claim extraction."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    def _is_reference_page(self, text: str) -> bool:
        """Heuristic to detect if an entire page is primarily a References/Bibliography section using citation density."""
        # Find citation patterns:
        # 1. Bracketed numbers: [1], [42]
        bracket_citations = len(re.findall(r'\[\d+\]', text))
        # 2. APA style years in parentheses: (Smith, 2019)
        apa_citations = len(re.findall(r'\(\D+,\s*(?:19|20)\d{2}\)', text))
        # 3. General year mentions which are highly frequent in bibliographies
        year_mentions = len(re.findall(r'\b(?:19|20)\d{2}\b', text))
        
        # Calculate a density score. A reference page is packed with these patterns.
        density_score = bracket_citations + apa_citations + (year_mentions * 0.5)
        
        # If density is unusually high, classify as a Reference page.
        return density_score > 15

    def parse(self) -> List[Dict[str, Any]]:
        """Parses the document and returns chunks containing text and metadata."""
        suffix = self.file_path.suffix.lower()
        if suffix == ".pdf":
            return self._parse_pdf()
        elif suffix in [".txt", ".md"]:
            return self._parse_text()
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _parse_pdf(self) -> List[Dict[str, Any]]:
        """Reads a PDF and splits text into page-scoped paragraph chunks."""
        chunks = []
        try:
            reader = pypdf.PdfReader(str(self.file_path))
            for page_idx, page in enumerate(reader.pages):
                page_num = page_idx + 1
                text = page.extract_text() or ""
                
                # Split by empty lines or lines with significant spacing to find paragraphs
                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                
                # If double-newline splitting found no real paragraph breaks (common
                # when PDF text extraction emits one \n per line rather than per
                # paragraph), treat the whole page as a single chunk instead of
                # splitting line-by-line - splitting by single "\n" fragments the
                # page into near-meaningless one-sentence chunks (observed: ~15
                # chunks/page averaging ~80 characters each), which multiplies
                # LLM call count and cost for no benefit to extraction quality.
                if len(paragraphs) <= 1 and text:
                    paragraphs = [text.strip()] if text.strip() else []

                # Check density heuristic for the entire page first
                if self._is_reference_page(text):
                    print(f"Citation density threshold exceeded at page {page_num}. Halting parser.")
                    break
                    
                for p_idx, para in enumerate(paragraphs):
                    # Filter out short fragments or page headers/footers
                    if len(para) > 40:
                        chunks.append({
                            "chunk_id": f"P-{page_num:02d}-{p_idx+1:02d}",
                            "text": para,
                            "page": page_num
                        })
        except Exception as e:
            print(f"Error parsing PDF {self.file_path}: {e}")
            # Fail gracefully by treating it as plain text if it's actually readable as text
            raise e
        return chunks

    def _parse_text(self) -> List[Dict[str, Any]]:
        """Reads plain text/markdown and splits by paragraph."""
        chunks = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        for p_idx, para in enumerate(paragraphs):
            if len(para) > 30:
                chunks.append({
                    "chunk_id": f"P-TXT-{p_idx+1:02d}",
                    "text": para,
                    "page": 1
                })
        return chunks
