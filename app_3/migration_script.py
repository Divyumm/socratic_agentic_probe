import os
import sys
from pathlib import Path
import json

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app_3.storage import StorageManager
from app_3.config import PROCESSED_DATA_DIR
from app_3.schemas import SessionTranscript

def run_migration():
    """Backfills missing schema fields for legacy transcript JSONs."""
    print("Starting data migration...")
    processed_dir = PROCESSED_DATA_DIR
    
    if not processed_dir.exists():
        print("No processed data directory found.")
        return
        
    count = 0
    for file_path in processed_dir.glob("*.json"):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            modified = False
            
            # 1. Backfill weight_version
            if "weight_version" not in data:
                data["weight_version"] = "A"
                modified = True
                
            # 2. Backfill evaluator_ratings
            if "evaluator_ratings" not in data:
                # Check for legacy singular
                if "evaluator_rating" in data and data["evaluator_rating"] is not None:
                    data["evaluator_ratings"] = [data["evaluator_rating"]]
                else:
                    data["evaluator_ratings"] = []
                
                modified = True
                
            if "evaluator_rating" in data:
                del data["evaluator_rating"]
                modified = True
                
            if modified:
                # Re-validate against Pydantic to ensure safety, then save
                transcript = SessionTranscript.model_validate(data)
                StorageManager.save_transcript(transcript)
                count += 1
                print(f"Migrated: {file_path.name}")
                
        except Exception as e:
            print(f"Failed to migrate {file_path.name}: {e}")
            
    print(f"Migration complete. {count} legacy transcripts updated.")

if __name__ == "__main__":
    run_migration()
