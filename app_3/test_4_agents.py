import os
import sys

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_3.simulation import run_adversarial_simulation
from app_3.extraction import ExtractionEngine
from app_3.schemas import EpistemicMap

def test_poc():
    print("Testing 4-Agent Clustering Proof of Concept...")
    
    # Create a mock epistemic map
    map_data = {
        "document_name": "mock_doc.pdf",
        "claims": [
            {
                "id": "C-01",
                "text": "The design uses a decentralized architecture.",
                "dimension": "Need",
                "source_passage": "A decentralized architecture was chosen because users requested data sovereignty in the interviews.",
                "emergent_theme": "Decentralization",
                "is_implicit": False,
                "confidence": 0.9,
                "page": 1
            },
            {
                "id": "C-02",
                "text": "We used Python because it is fast.",
                "dimension": "Method",
                "source_passage": "Python was utilized due to its fast iteration cycle for our data pipelines.",
                "emergent_theme": "Speed",
                "is_implicit": False,
                "confidence": 0.5,
                "page": 2
            }
        ]
    }
    
    epistemic_map = EpistemicMap(**map_data)
    
    try:
        run_adversarial_simulation(epistemic_map, max_turns=3)
    except KeyboardInterrupt:
        print("Test aborted.")
    
if __name__ == "__main__":
    test_poc()
