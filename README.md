# Automated Viva Probing & Evaluation System

This repository contains the source code for the empirical evaluation architecture of my dissertation.

## System Architecture

The system is divided into two primary phases:

- **The During-Phase (Probing):** `probe_engine.py` orchestrates a real-time conversational agent matrix that questions a student's thesis claims, detecting reasoning collapse using strict probabilistic NLI signals.
- **The After-Phase (Evaluation):** An offline suite of tools used to label data, train the Symbolic Regression Model to derive composite scores, generate qualitative Jury Review Cards, and conduct blind independent evaluations.

## How to Run

To run this on your own machine, first clone the repository and install the required dependencies:

```bash
git clone https://github.com/Divyumm/socratic_agentic_probe.git
cd socratic_agentic_probe
pip install -r requirements.txt
```
*(Note: The first time you run the application, it will automatically download the `Qwen2.5-0.5B-Instruct` and `cross-encoder/nli-deberta-v3-small` models to your local Hugging Face cache. This may take a few minutes depending on your internet connection).*

### 1. Running the Application (Live Simulation)

To launch the Streamlit frontend dashboard, run the following command from the root directory:

```bash
python3 -m streamlit run app_3/app.py
```
This dashboard will allow you to:
- Select a pre-parsed Epistemic Map
- Engage in a live Socratic dialogue
- View real-time NLI matrix scores, Epistemic Variance calculations, and composite evaluations.

### 2. Data Labelling & Independent Evaluation UI

The entire After-Phase is contained within a single Streamlit dashboard (`app_3/labelling_app.py`).

To launch the dashboard, run:

```bash
python3 -m streamlit run app_3/labelling_app.py
```
This dashboard contains four tabs:
- **Data Labelling:** Faculty can label individual probing turns as Grounded, Unstable, or Collapsed to generate training data.
- **Faculty Jury Review Card:** Shows the composite confidence score, session notes, and any student-initiated score challenges, with claim source context alongside.
- **Independent Blind Evaluation:** A locked-down flow where an external researcher can read the raw transcript (stripped of all AI scores) and file a blind rating. Submitting a rating unlocks a Comparison Matrix that calculates alignment between human judgement and system detection.
- **Experiment Rating:** Per-turn double-blind faculty ratings plus an overall ground-truth grade, feeding the temperature-profile experiment.

*Login Role at the top of the page ("Faculty Assessor" vs "Independent Evaluator") gates which tabs are unlocked — switch it before trying to use Independent Blind Evaluation.*

### 3. Training the ML Pipeline (Symbolic Regression)

Once you have filed enough Faculty Labels in the dashboard, you can run the symbolic regression model to automatically discover the best mathematical equation for composite scoring:

```bash
python3 -m app_3.train_symbolic_regression
```
This will output the newly generated mathematical equation and automatically integrate it into the `probe_engine.py` logic for future sessions.

### Important: Parsing New PDFs (API Key Required)

While the Simulation Phase runs entirely locally, the Document Extraction Phase (parsing a raw coursework PDF into an Epistemic Map of claims) still relies on the Anthropic API (Claude 3.5 Sonnet).

If you want to parse your own brand new PDF documents, you **must** provide an Anthropic API Key.

**How to add your API Key:**
1. In the root directory of this repository, create a new file named `.env`.
2. Open the file and add your secret API key like this:
   ```env
   ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxx
   ```
3. Save the file and restart the Streamlit application.

*If you do not provide an API key, the Document Extraction engine will gracefully fall back into MOCK_MODE and generate dummy placeholder claims instead of reading the PDF.*
