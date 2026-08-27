# Socratic Viva Orchestration System (Phase 3)

This repository contains the source code for the empirical evaluation architecture of the Socratic Viva dissertation (`app_3`). 

This architecture implements a **4-Agent Epistemic Matrix** (Assessor, Evaluator, Quality Auditor, Advocate) powered entirely by on-device **Local Small Language Models (SLMs)**. This means the core Socratic simulation runs completely offline and for free on any standard machine.

## Setup & Installation

To run this on your own machine, first clone the repository and install the required dependencies:

```bash
git clone https://github.com/Divyumm/socratic_agentic_probe.git
cd socratic_agentic_probe
pip install -r requirements.txt
```

*(Note: The first time you run the application, it will automatically download the `Qwen2.5-0.5B-Instruct` and `cross-encoder/nli-deberta-v3-small` models to your local Hugging Face cache. This may take a few minutes depending on your internet connection).*

## Running the Application

To launch the Streamlit frontend dashboard, run the following command from the root directory:

```bash
python3 -m streamlit run app_3/app.py
```

This dashboard will allow you to:
- Select a pre-parsed Epistemic Map
- Engage in a live Socratic dialogue
- View real-time NLI matrix scores, Epistemic Variance calculations, and composite evaluations.

---

## Important: Parsing New PDFs (API Key Required)

While the **Simulation Phase** runs entirely locally, the **Document Extraction Phase** (parsing a raw coursework PDF into an Epistemic Map of claims) still relies on the Anthropic API (Claude 3.5 Sonnet).

If you want to parse your own brand new PDF documents, you **must** provide an Anthropic API Key. 

**How to add your API Key:**
1. In the root directory of this repository, create a new file named `.env`.
2. Open the file and add your secret API key like this:
   ```env
   ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxx
   ```
3. Save the file and restart the Streamlit application.

*If you do not provide an API key, the Document Extraction engine will gracefully fall back into `MOCK_MODE` and generate dummy placeholder claims instead of reading the PDF.*
