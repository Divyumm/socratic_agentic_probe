from typing import Dict, List, Tuple, Optional
from app_3.schemas import PapanekDimension

class ReconstructionRouter:
    """Implements Victor Papanek's dimension routing matrix for reasoning recovery.

    Adjacency Matrix connects:
    - Method <-> Need, Use
    - Use <-> Method, Aesthetics
    - Aesthetics <-> Use, Association
    - Association <-> Aesthetics, Telesis
    - Telesis <-> Association, Need
    - Need <-> Telesis, Method
    """

    def __init__(self):
        # Define the hexagonal connections
        self.adjacencies: Dict[PapanekDimension, Tuple[PapanekDimension, PapanekDimension]] = {
            PapanekDimension.METHOD: (PapanekDimension.NEED, PapanekDimension.USE),
            PapanekDimension.USE: (PapanekDimension.METHOD, PapanekDimension.AESTHETICS),
            PapanekDimension.AESTHETICS: (PapanekDimension.USE, PapanekDimension.ASSOCIATION),
            PapanekDimension.ASSOCIATION: (PapanekDimension.AESTHETICS, PapanekDimension.TELESIS),
            PapanekDimension.TELESIS: (PapanekDimension.ASSOCIATION, PapanekDimension.NEED),
            PapanekDimension.NEED: (PapanekDimension.TELESIS, PapanekDimension.METHOD),
        }

        # Prompts to shift reasoning from a collapsed origin to an adjacent destination
        self.routing_prompts: Dict[Tuple[PapanekDimension, PapanekDimension], str] = {
            # Method collapses
            (PapanekDimension.METHOD, PapanekDimension.NEED): 
                "Let's step back. What fundamental human need does this satisfy? How can that need be met if we completely change this process?",
            (PapanekDimension.METHOD, PapanekDimension.USE):
                "Let's refocus on core utility. How does this specific tool perform its primary functional task under field constraints, independent of the materials you chose?",
            
            # Use collapses
            (PapanekDimension.USE, PapanekDimension.METHOD):
                "Let's look at the underlying technology. What alternative materials or processes could make this tool function reliably?",
            (PapanekDimension.USE, PapanekDimension.AESTHETICS):
                "How does the physical form, balance, or layout of your interface directly support its functional use? Can we simplify the design to improve usability?",
            
            # Aesthetics collapses
            (PapanekDimension.AESTHETICS, PapanekDimension.USE):
                "Let's look at the functional layout. How does the core tool function in practice? What visual details are strictly required for utility?",
            (PapanekDimension.AESTHETICS, PapanekDimension.ASSOCIATION):
                "What psychological or educational conditioning makes this specific aesthetic style meaningful or familiar to the end user?",
            
            # Association collapses
            (PapanekDimension.ASSOCIATION, PapanekDimension.AESTHETICS):
                "Let's examine the form. What traditional shapes, textures, or symbols can be integrated into the tool's interface to build immediate familiarity?",
            (PapanekDimension.ASSOCIATION, PapanekDimension.TELESIS):
                "How does this cultural conditioning reflect the broader industrial and economic transition currently happening in the deployment context?",
            
            # Telesis collapses
            (PapanekDimension.TELESIS, PapanekDimension.ASSOCIATION):
                "Let's look at cultural conditioning. How do the user's habits and prior experiences shape their technological trust?",
            (PapanekDimension.TELESIS, PapanekDimension.NEED):
                "Let's simplify. What is the immediate, foundational need of the user that overrides any complex social or technological trust issues?",
            
            # Need collapses
            (PapanekDimension.NEED, PapanekDimension.TELESIS):
                "Let's look at the wider social context. How does this design align with technological bias and power systems in the target environment?",
            (PapanekDimension.NEED, PapanekDimension.METHOD):
                "Let's rebuild from the physical foundation. What tools and methods are already in place in this context to address this need?"
        }

    def route_collapse(self, failed_dimension: PapanekDimension, current_session_history: List[PapanekDimension],
                       failed_theme: Optional[str] = None) -> Tuple[PapanekDimension, str]:
        """Routes a collapsed claim to an adjacent dimension that has been probed least in this session."""
        if failed_dimension == PapanekDimension.EMERGENT:
            # Route to the standard dimension that has appeared least in current session history
            dims = [
                PapanekDimension.METHOD, PapanekDimension.USE, PapanekDimension.AESTHETICS, 
                PapanekDimension.ASSOCIATION, PapanekDimension.TELESIS, PapanekDimension.NEED
            ]
            counts = {d: current_session_history.count(d) for d in dims}
            target_dim = min(counts, key=counts.get)
            theme_name = failed_theme if failed_theme else "your custom theme"
            prompt = f"Regarding '{theme_name}', let's step back and look at this through another aspect of your design: the {target_dim.value}."
            return target_dim, prompt

        adj_1, adj_2 = self.adjacencies[failed_dimension]

        # Prioritize adjacent dimension that has appeared LESS in the current session history
        count_1 = current_session_history.count(adj_1)
        count_2 = current_session_history.count(adj_2)

        target_dim = adj_1 if count_1 <= count_2 else adj_2
        
        # Get prompt
        prompt = self.routing_prompts.get(
            (failed_dimension, target_dim),
            f"Let's step back and look at this through another aspect of your design: the {target_dim.value}."
        )

        return target_dim, prompt
