import sys
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import fitz  # PyMuPDF

# Ensure project root is on path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app_3.schemas import EpistemicMap, Claim, PapanekDimension, SessionTranscript, ExperimentProfile
from app_3.extraction import ExtractionEngine
from app_3.storage import StorageManager
from app_3.probe_engine import ProbingSessionManager
from app_3.simulation import AdvocateAgent
import random

class VivaWrapper:
    """Unified session and parsing coordinator wrapper for the Socratic Viva system."""

    @staticmethod
    def parse_pdf_to_map(pdf_path: str) -> EpistemicMap:
        """Parses a coursework PDF and extracts its Epistemic Map."""
        engine = ExtractionEngine()
        epistemic_map = engine.extract_epistemic_map(pdf_path)

        # Refuse to silently clobber a previously-good saved map with an all-fallback
        # one (e.g. a re-parse attempted while the LLM API is quota-exhausted or down).
        n_claims = len(epistemic_map.claims)
        n_fallback = sum(1 for c in epistemic_map.claims if c.text.startswith("Fallback extracted claim"))
        new_is_degraded = n_claims > 0 and n_fallback == n_claims

        if new_is_degraded:
            existing = StorageManager.load_epistemic_map(epistemic_map.document_name)
            existing_is_good = existing is not None and existing.claims and not all(
                c.text.startswith("Fallback extracted claim") for c in existing.claims
            )
            if existing_is_good:
                raise RuntimeError(
                    f"Extraction failed for all {n_claims} claims (likely an LLM API error or quota limit) "
                    f"while a good existing map for '{epistemic_map.document_name}' is already saved on disk. "
                    "The existing map was left untouched. Try re-parsing again once the API is available."
                )

        StorageManager.save_epistemic_map(epistemic_map)
        return epistemic_map

    @staticmethod
    def load_map(document_name: str) -> Optional[EpistemicMap]:
        """Loads an existing Epistemic Map from the processed data directory."""
        return StorageManager.load_epistemic_map(document_name)

    @staticmethod
    def list_available_maps() -> List[str]:
        """Lists all parsed epistemic maps in the processed data folder."""
        from app_3.config import PROCESSED_DATA_DIR
        maps = list(PROCESSED_DATA_DIR.glob("*_map.json"))
        return [m.name.replace("_map.json", ".pdf") for m in maps]

    @staticmethod
    def list_available_pdfs() -> List[Path]:
        """Lists all PDF coursework documents in the root directory."""
        from app_3.config import BASE_DIR
        return list(BASE_DIR.glob("*.pdf"))

    @staticmethod
    def start_session(epistemic_map: EpistemicMap, student_name: str) -> ProbingSessionManager:
        """Starts a new Probing Session with a randomized, dynamic temperature profile."""
        # 1. Randomly decide if this run should be "Extreme" or "Moderate"
        is_extreme = random.choice([True, False])
        
        # NOTE: Ranges rescaled into Claude's 0.0-1.0 temperature range (Gemini
        # supported 0.0-2.0). Rescaled proportionally rather than clamped, so
        # the Extreme-vs-Moderate ordering on each axis (assessor lowest,
        # advocate highest, evaluator in between) is preserved.
        if is_extreme:
            # 2. Pick extreme ranges but enforce inequality
            a_temp = round(random.uniform(0.0, 0.1), 2)
            e_temp = round(random.uniform(0.6, 0.8), 2)
            adv_temp = round(random.uniform(0.9, 1.0), 2)
            profile_name = "Dynamic Extreme"
        else:
            # 3. Pick moderate ranges but enforce inequality
            a_temp = round(random.uniform(0.1, 0.2), 2)
            e_temp = round(random.uniform(0.35, 0.5), 2)
            adv_temp = round(random.uniform(0.6, 0.8), 2)
            profile_name = "Dynamic Moderate"
            
        assigned_profile = ExperimentProfile(
            profile_name=f"{profile_name} (A:{a_temp} E:{e_temp} V:{adv_temp})",
            assessor_temp=a_temp,
            evaluator_temp=e_temp,
            advocate_temp=adv_temp
        )
        return ProbingSessionManager(epistemic_map, student_name, experiment_profile=assigned_profile)

    @staticmethod
    def get_advocate_defense(claim: Claim, question: str, depth: int, advocate_temp: Optional[float] = None) -> Tuple[str, PapanekDimension]:
        """Calls the simulated Advocate Agent to generate a high-temperature defense/pivot."""
        agent = AdvocateAgent()
        return agent.generate_response(
            claim_id=claim.id,
            claim_text=claim.text,
            source_passage=claim.source_passage,
            current_dimension=claim.dimension,
            question=question,
            depth=depth,
            advocate_temp=advocate_temp
        )

    @staticmethod
    def save_transcript(transcript: SessionTranscript) -> Path:
        """Saves a completed Session Transcript."""
        return StorageManager.save_transcript(transcript)

    @staticmethod
    def extract_claim_image(pdf_path: str, page_number: int, search_text: str, context_padding: int = 50) -> Optional[bytes]:
        """
        Extracts a specific snippet of text from a PDF as a cropped PNG image.
        Uses PyMuPDF to locate the text and render just that region.
        """
        try:
            doc = fitz.open(pdf_path)
            # Physical pages are 0-indexed in PyMuPDF
            page = doc.load_page(page_number - 1)
            
            # Clean up search text a bit to handle line breaks/truncations better
            clean_search = search_text.strip().replace("...", "").strip()
            
            # Try to find the exact snippet, falling back to a substring if it's too long
            if len(clean_search) > 60:
                clean_search = clean_search[:60]
                
            rects = page.search_for(clean_search)
            
            if not rects:
                # If exact search failed, try another substring
                if len(search_text) > 30:
                    rects = page.search_for(search_text[20:50])
            
            if rects:
                # Highlight the found text
                for r in rects:
                    page.add_highlight_annot(r)
                
            # Render the full page
            pix = page.get_pixmap(dpi=150)
            return pix.tobytes("png")
        except Exception as e:
            print(f"Error extracting PDF image: {e}")
            return None
