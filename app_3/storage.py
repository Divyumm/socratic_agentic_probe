import json
from pathlib import Path
from typing import Optional
from app_3.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from app_3.schemas import EpistemicMap, SessionTranscript

class StorageManager:
    """Handles persistence of Epistemic Maps and Session Transcripts in JSON format."""

    @staticmethod
    def save_epistemic_map(epistemic_map: EpistemicMap) -> Path:
        """Saves a validated EpistemicMap to local processed data directory."""
        # Sanitize filename
        safe_name = Path(epistemic_map.document_name).stem.lower().replace(" ", "_")
        target_path = PROCESSED_DATA_DIR / f"{safe_name}_map.json"
        
        with open(target_path, "w", encoding="utf-8") as f:
            # Pydantic v2 json export handles Datetime/Enum serialization seamlessly
            f.write(epistemic_map.model_dump_json(indent=2))
        return target_path

    @staticmethod
    def load_epistemic_map(document_name: str) -> Optional[EpistemicMap]:
        """Loads an EpistemicMap from disk by document stem name."""
        safe_name = Path(document_name).stem.lower().replace(" ", "_")
        target_path = PROCESSED_DATA_DIR / f"{safe_name}_map.json"
        
        if not target_path.exists():
            return None
            
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return EpistemicMap(**data)

    @staticmethod
    def save_transcript(transcript: SessionTranscript) -> Path:
        """Saves a SessionTranscript to disk after viva completion."""
        target_path = PROCESSED_DATA_DIR / f"session_{transcript.session_id}.json"
        
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(transcript.model_dump_json(indent=2))
        return target_path

    @staticmethod
    def load_transcript(session_id: str) -> Optional[SessionTranscript]:
        """Loads a SessionTranscript from disk by UUID session_id."""
        target_path = PROCESSED_DATA_DIR / f"session_{session_id}.json"
        
        if not target_path.exists():
            return None
            
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Backwards compatibility: migrate old 'construct' keys
        if "review_card" in data and data["review_card"] and "rubric_scores" in data["review_card"]:
            for score in data["review_card"]["rubric_scores"]:
                if "construct" in score and "rubric_construct" not in score:
                    score["rubric_construct"] = score.pop("construct")
                    
        return SessionTranscript(**data)

    @staticmethod
    def list_available_transcripts() -> list:
        """Lists IDs of all stored transcripts in PROCESSED_DATA_DIR."""
        transcripts = []
        for p in PROCESSED_DATA_DIR.glob("session_*.json"):
            session_id = p.stem.replace("session_", "")
            transcripts.append(session_id)
        return transcripts

    @staticmethod
    def list_checkpoints() -> list:
        """Lists all incomplete session checkpoints (auto-saved sessions)."""
        checkpoints = []
        for p in PROCESSED_DATA_DIR.glob("checkpoint_*.json"):
            session_id = p.stem.replace("checkpoint_", "")
            checkpoints.append({
                "session_id": session_id,
                "path": str(p),
                "modified": p.stat().st_mtime
            })
        return sorted(checkpoints, key=lambda x: x["modified"], reverse=True)

    @staticmethod
    def load_checkpoint(session_id: str) -> Optional[SessionTranscript]:
        """Loads an auto-saved checkpoint by session UUID."""
        target_path = PROCESSED_DATA_DIR / f"checkpoint_{session_id}.json"

        if not target_path.exists():
            return None

        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SessionTranscript(**data)

    @staticmethod
    def promote_checkpoint_to_session(checkpoint_path: str) -> Path:
        """Converts a checkpoint file to a final session file."""
        import shutil
        checkpoint = Path(checkpoint_path)
        session_id = checkpoint.stem.replace("checkpoint_", "")
        session_path = PROCESSED_DATA_DIR / f"session_{session_id}.json"
        shutil.copy(checkpoint, session_path)
        checkpoint.unlink()  # Delete checkpoint after promotion
        return session_path
