# Tron: Automated Viva Probing & Evaluation System

> **[DEPRECATION WARNING]**
> The `app/` directory contains legacy code from the early Tron iterations. All current development and the active system architecture reside in `app_2/`. Please do not build against `app/`.

This repository contains the source code for the empirical evaluation architecture of the dissertation.

## System Architecture
The system is divided into two primary phases:
1. **The During-Phase (Probing):** `probe_engine.py` orchestrates a real-time conversational agent that interrogates a student's thesis claims, detecting reasoning collapse using cognitive heuristics.
2. **The After-Phase (Evaluation):** An offline suite of tools used to label data, train the Model (Version B), generate qualitative Jury Review Cards, and conduct blind independent evaluations.

## How to Run

### 1. Data Labelling & Independent Evaluation UI
The entire After-Phase is contained within a single Streamlit dashboard (`app_2/labelling_app.py`).

To launch the dashboard, run:
```bash
python3 -m streamlit run app_2/labelling_app.py
```

This dashboard contains four tabs (`labelling_app.py`; the previously-separate `faculty_experiment_portal.py` on its own port has been merged in and is deprecated):
- **Data Labelling:** Faculty can label individual probing turns as Grounded, Unstable, or Collapsed to generate training data.
- **Faculty Jury Review Card:** Shows the composite confidence score (Version A/B comparison where both exist), session notes, and any student-initiated score challenges, with claim source context alongside.
- **Independent Blind Evaluation:** A locked-down flow where an external researcher can read the raw transcript (stripped of all AI scores) and file a blind rating. Submitting a rating unlocks a Comparison Matrix that calculates alignment between human judgement and system detection.
- **Experiment Rating:** Per-turn double-blind faculty ratings (assessor hallucination/repetition, whether the variance score reflected genuine ambiguity, advocate pivot quality) plus an overall ground-truth grade, feeding the temperature-profile experiment.

Login Role at the top of the page ("Faculty Assessor" vs "Independent Evaluator") gates which tabs are unlocked — switch it before trying to use Independent Blind Evaluation.

### 2. Training the ML Pipeline (Version B)
Once you have filed enough Faculty Labels in the dashboard (N=30 minimum), you can train the logistic regression model:

```bash
python3 -m app_2.training_pipeline
```

This will output the weights to `data/weights/` and update `config.py` to route all future probing sessions to use the Version B weights.

### 3. Running the Test Suite
The system is verified by a 19-test suite that guarantees 100% end-to-end integration stability.

```bash
python3 -m pytest tests/
```
