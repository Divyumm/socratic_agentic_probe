import sys
import os
from pathlib import Path
from typing import Optional
import textwrap

# Ensure project root is on path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app_3.schemas import InterventionType, PapanekDimension, StudentState
from app_3.extraction import ExtractionEngine
from app_3.probe_engine import ProbingSessionManager
from app_3.storage import StorageManager
from app_3.simulation import run_adversarial_simulation
from app_3.config import BASE_DIR, PROCESSED_DATA_DIR

def print_header(title: str):
    print("\n" + "=" * 80)
    print(f" {title.center(78)} ")
    print("=" * 80)

def print_agent_message(message: str, prefix: str = "Assessor Agent:"):
    # Split message by newline to maintain paragraph layout, then wrap each
    paragraphs = message.split("\n")
    wrapped_paragraphs = []
    for para in paragraphs:
        if not para.strip():
            wrapped_paragraphs.append("")
        else:
            wrapped_paragraphs.append(textwrap.fill(para, width=78))
    
    wrapped_text = "\n".join(wrapped_paragraphs)
    print(f"\n{prefix}\n{textwrap.indent(wrapped_text, '  ')}")

def main_menu():
    while True:
        print_header("SOCRATIC VIVA ORCHESTRATION SYSTEM")
        print(" [1] Parse PDF & Extract Epistemic Map (Component 1 Document Analyser)")
        print(" [2] Conduct Live Socratic Viva Session (Component 2 & 3 Interactive Loop)")
        print(" [3] View Past Session Transcripts")
        print(" [4] Run AI Respondent Simulation (Thesis Experiment)")
        print(" [5] Exit")
        
        choice = input("\nSelect option [1-5]: ").strip()
        
        if choice == "1":
            run_document_analyser()
        elif choice == "2":
            run_viva_session()
        elif choice == "3":
            view_past_sessions()
        elif choice == "4":
            run_simulation_menu()
        elif choice == "5":
            print("\nExiting Socratic Viva Orchestration. Goodbye!\n")
            break
        else:
            print("\nInvalid choice. Please select from 1 to 5.")

def run_document_analyser():
    print_header("COMPONENT 1: DOCUMENT ANALYSER")
    
    # List all PDFs in Tron folder
    tron_dir = BASE_DIR
    pdfs = list(tron_dir.glob("*.pdf"))
    
    if not pdfs:
        print(f"\nNo PDF files found in workspace {tron_dir}.")
        return

    print("Available coursework documents:")
    for idx, pdf in enumerate(pdfs):
        print(f" [{idx+1}] {pdf.name} ({pdf.stat().st_size / 1024:.1f} KB)")
        
    try:
        selection = int(input(f"\nSelect PDF to parse [1-{len(pdfs)}]: ")) - 1
        if selection < 0 or selection >= len(pdfs):
            raise ValueError()
        selected_pdf = pdfs[selection]
    except (ValueError, IndexError):
        print("\nInvalid selection.")
        return

    print(f"\nParsing [ {selected_pdf.name} ] and building epistemic map...")
    engine = ExtractionEngine()
    
    try:
        epistemic_map = engine.extract_epistemic_map(str(selected_pdf))
        saved_path = StorageManager.save_epistemic_map(epistemic_map)
        
        print("\n" + "-" * 80)
        print(" EXTRACTED EPISTEMIC MAP (Ordered by Probing Vulnerability Rank)".center(80))
        print("-" * 80)
        print(f"{'Rank':<5} | {'ID':<5} | {'Dimension':<12} | {'Conf':<5} | {'Page':<5} | {'Claim text excerpt':<40}")
        print("-" * 80)
        
        for claim in epistemic_map.claims:
            excerpt = claim.text
            print(f"{claim.vulnerability_rank:<5} | {claim.id:<5} | {claim.dimension.value:<12} | {claim.confidence:.2f} | {claim.page:<5} | {excerpt:<40}")
            
        print("-" * 80)
        print(f"Successfully saved validated Epistemic Map to:")
        print(f" -> {saved_path.resolve()}")
        print("-" * 80)
        
        input("\nPress Enter to return to main menu...")
    except Exception as e:
        print(f"\nError building epistemic map: {e}")
        import traceback
        traceback.print_exc()

def run_viva_session():
    print_header("COMPONENT 2: LIVE SOCRATIC VIVA ENGINE")
    
    # List processed maps
    processed_dir = PROCESSED_DATA_DIR
    maps = list(processed_dir.glob("*_map.json"))
    
    if not maps:
        print("\nNo extracted Epistemic Maps found. Please run Option [1] first to parse a PDF!")
        input("\nPress Enter to return...")
        return
        
    print("Available Epistemic Maps:")
    for idx, map_file in enumerate(maps):
        doc_name = map_file.name.replace("_map.json", ".pdf")
        print(f" [{idx+1}] {doc_name} (Extracted Map)")
        
    try:
        selection = int(input(f"\nSelect Map to load [1-{len(maps)}]: ")) - 1
        if selection < 0 or selection >= len(maps):
            raise ValueError()
        selected_map_file = maps[selection]
    except (ValueError, IndexError):
        print("\nInvalid selection.")
        return

    student_name = input("\nEnter Student Name: ").strip()
    if not student_name:
        student_name = "Advocate"

    # Load Epistemic Map
    try:
        with open(selected_map_file, "r", encoding="utf-8") as f:
            from app_3.schemas import EpistemicMap
            import json
            map_data = json.load(f)
            epistemic_map = EpistemicMap(**map_data)
    except Exception as e:
        print(f"\nFailed to load epistemic map file: {e}")
        return

    manager = ProbingSessionManager(epistemic_map, student_name)
    
    print("\n" + "=" * 80)
    print(f" VIVA ACTIVE: {student_name.upper()} | TOPIC: {epistemic_map.document_name} ".center(80))
    print("=" * 80)
    print("Commands:")
    print("  /pause          Pause current session")
    print("  /redirect [ID]  Jump probing to a specific claim by ID (e.g. /redirect C-03)")
    print("  /challenge      pedagogically significant dispute of the assessor's state score")
    print("  /exit           Terminate and save the viva session")
    print("=" * 80)
    
    # Initial Socratic question
    print(f"\n[VIVA START] Assessor Agent opens the viva:")
    print_agent_message(manager.get_current_question())
    
    while True:
        response = input(f"\n{student_name}: ").strip()
        
        if not response:
            print("Response cannot be empty. Please type a justification or command.")
            continue
            
        # Parse commands
        if response.startswith("/"):
            parts = response.split(maxsplit=1)
            cmd = parts[0].lower()
            
            if cmd == "/exit":
                notes = input("\nEnter final assessor comments/notes: ").strip()
                transcript = manager.build_transcript(notes=notes)
                saved_path = StorageManager.save_transcript(transcript)
                print(f"\nViva terminated. Validated session transcript saved to:\n -> {saved_path.resolve()}\n")
                break
                
            elif cmd == "/challenge":
                challenge_prompt = manager.handle_challenge()
                # challenge_prompt is pre-formatted, so we print it with empty prefix or default wrapping
                print_agent_message(challenge_prompt, prefix="Assessor Agent:")
                continue
                
            elif cmd == "/pause":
                _, next_prompt = manager.submit_response("[Intervention: Student Paused]", InterventionType.PAUSE)
                print_agent_message(next_prompt)
                continue
                
            elif cmd == "/redirect":
                if len(parts) < 2:
                    print("Usage: /redirect [Claim_ID] (e.g. /redirect C-02)")
                    continue
                target_id = parts[1].strip().upper()
                msg = manager.handle_redirect(target_id)
                print_agent_message(msg)
                continue
                
            else:
                print(f"Unknown command: {cmd}")
                continue
        
        # Submit standard response
        try:
            turn, next_prompt = manager.submit_response(response)
            
            # Print turn evaluation features (Composite Score & Phase Transitions)
            print("\n" + "-" * 60)
            print(f" TURN {turn.turn_index} EVALUATION SIGNALS (Field-Knowledge-Independent)".center(60))
            print("-" * 60)
            print(f" Coherence Score (Probe <-> Answer):   {turn.coherence_score:.2f}")
            print(f" Grounding Score (Fact-Check):         {turn.grounding_score:.2f}")
            print(f" Circularity Score (Self-repeating):   {turn.circularity_score:.2f}")
            print(f" Simulated Temp Variance (N runs):    {turn.variance_score:.2f}")
            print(f" Composite Confidence Score (y-axis):  {turn.composite_confidence:.2f}")
            
            state_color = ""
            if turn.state == StudentState.COLLAPSED:
                state_color = "!!! REASONING COLLAPSE / CLIFF DETECTED !!!"
            elif turn.state == StudentState.UNSTABLE:
                state_color = "?? UNSTABLE REASONING (Plateau boundary reached) ??"
            else:
                state_color = "== GROUNDED REASONING (Robust Plateau) =="
                
            print(f" Reasoning State:                     {state_color}")
            print("-" * 60)
            
            # Print agent's next Socratic step
            print_agent_message(next_prompt)
            
            if not manager.active_claim:
                # All claims probed and resolved
                transcript = manager.build_transcript(notes="All claims justified cleanly.")
                StorageManager.save_transcript(transcript)
                print("\n[VIVA COMPLETE] The viva session has completed successfully. Transcript saved.\n")
                break
                
        except Exception as e:
            print(f"Error processing response: {e}")
            import traceback
            traceback.print_exc()
            break
            
    input("\nPress Enter to return to main menu...")

def run_simulation_menu():
    print_header("COMPONENT 4: ADVERSARIAL SOCRATIC SIMULATION (THESIS EXPERIMENT)")
    
    processed_dir = PROCESSED_DATA_DIR
    maps = list(processed_dir.glob("*_map.json"))
    
    if not maps:
        print("\nNo extracted Epistemic Maps found. Please run Option [1] first to parse a PDF!")
        input("\nPress Enter to return...")
        return
        
    print("Available Epistemic Maps:")
    for idx, map_file in enumerate(maps):
        doc_name = map_file.name.replace("_map.json", ".pdf")
        print(f" [{idx+1}] {doc_name} (Extracted Map)")
        
    try:
        selection = int(input(f"\nSelect Map to simulate [1-{len(maps)}]: ")) - 1
        if selection < 0 or selection >= len(maps):
            raise ValueError()
        selected_map_file = maps[selection]
    except (ValueError, IndexError):
        print("\nInvalid selection.")
        return

    # Load Epistemic Map
    try:
        with open(selected_map_file, "r", encoding="utf-8") as f:
            from app_3.schemas import EpistemicMap
            import json
            map_data = json.load(f)
            epistemic_map = EpistemicMap(**map_data)
    except Exception as e:
        print(f"\nFailed to load epistemic map file: {e}")
        return

    try:
        run_adversarial_simulation(epistemic_map, max_turns=12)
    except Exception as e:
        print(f"\nError running simulation: {e}")
        
    input("\nPress Enter to return to main menu...")

def view_past_sessions():
    print_header("PAST SESSION TRANSCRIPTS")
    processed_dir = PROCESSED_DATA_DIR
    sessions = list(processed_dir.glob("session_*.json"))
    
    if not sessions:
        print("\nNo saved viva sessions found.")
        input("\nPress Enter to return...")
        return
        
    print("Saved Socratic Sessions:")
    for idx, sess_file in enumerate(sessions):
        try:
            with open(sess_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f" [{idx+1}] ID: {data['session_id'][:8]} | Advocate: {data['student_name']:<15} | Doc: {data['document_name']:<25} | Turns: {len(data['turns'])}")
        except Exception:
            print(f" [{idx+1}] File: {sess_file.name} (Corrupt)")
            
    try:
        selection = int(input(f"\nSelect Session to view [1-{len(sessions)}]: ")) - 1
        if selection < 0 or selection >= len(sessions):
            raise ValueError()
        selected_sess_file = sessions[selection]
    except (ValueError, IndexError):
        print("\nInvalid selection.")
        return

    try:
        with open(selected_sess_file, "r", encoding="utf-8") as f:
            import json
            sess = json.load(f)
            
        print_header(f"VIVA TRANSCRIPT: {sess['student_name'].upper()}")
        print(f"Session ID:   {sess['session_id']}")
        print(f"Document:     {sess['document_name']}")
        print(f"Started At:   {sess['started_at']}")
        print(f"Total Turns:  {len(sess['turns'])}")
        print(f"Assessor:     {sess.get('notes', 'None')}")
        print("=" * 80)
        
        for turn in sess["turns"]:
            if turn.get("intervention_type") == "pause":
                print(f"\n[Turn {turn['turn_index']}] INTERVENTION: Student Paused")
                continue
            print(f"\n[Turn {turn['turn_index']}] (Claim: {turn['claim_id']} | State: {turn['state']})")
            print(f"  Assessor Question:  \"{turn['question']}\"")
            print(f"  Student Response: \"{turn['student_response']}\"")
            print(f"  Scores: [Coherence: {turn['coherence_score']:.2f} | Grounding: {turn['grounding_score']:.2f} | Variance: {turn['variance_score']:.2f} | Circularity: {turn['circularity_score']:.2f} | Composite: {turn['composite_confidence']:.2f}]")
            
        print("=" * 80)
        input("\nPress Enter to return...")
    except Exception as e:
        print(f"\nFailed to display transcript: {e}")

if __name__ == "__main__":
    main_menu()
