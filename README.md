# VigilAI

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**VigilAI** is an advanced, AI-assisted visual scene review tool designed for security and investigative workflows. By leveraging state-of-the-art vision models, VigilAI provides meticulous, evidence-first scene interpretations—extracting clear facts, identifying suspicious activities, and formulating hypotheses based on explicit visual grounding.

> **Disclaimer:** VigilAI outputs are model-assisted triage and analysis mechanisms. They are not a substitute for formal lab procedures, legal findings, or strict forensic validation. Always verify AI-assisted findings against primary evidence.

## Overview

Modern security operations and investigations require rapid, reliable insight from complex crime scenes or situational imagery. VigilAI bridges the gap between raw data and actionable intelligence. Rather than offering basic object detection, VigilAI constructs a structured, comprehensive narrative of the scene—separating observed facts from inferred logic and providing investigators with a highly interpretable evidence trail.

### Key Features
- **Evidence-First Scene Review:** Systematically categorizes intelligence into what is *observed*, *inferred*, and *speculative*, ensuring reviewers are never misled by AI hallucinations.
- **Multimodal Intelligence:** Upload CCTV stills, crime scene photos, or forensic frames for deep visual analysis.
- **Simulated Narrative Analysis:** Feed textual witness reports or simulated scene descriptions for thorough narrative deconstruction.
- **Robust Output Schema:** Generates actionable, strictly validated JSON reports detailing visible facts, limitations, anomalies, relationships, and risk assessments.
- **Deterministic Offline Capabilities:** In environments without secure API access, the system gracefully falls back to displaying precise, deterministic image metadata. 

## Technical Stack

- **Frontend Application:** [Streamlit](https://streamlit.io/) for rapid, responsive UI delivery.
- **Data Validation:** [Pydantic](https://docs.pydantic.dev/) ensures all model outputs strictly adhere to our forensic reporting schema.
- **Inference Engine:** Backed by **Groq**, utilizing the highly capable `meta-llama/llama-4-scout-17b-16e-instruct` vision model for unparalleled processing speed and multi-modal accuracy.

## Project Structure

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application and UI routing definitions. |
| `analysis.py` | Core logic for API integration, image preparation, and Pydantic schema enforcement. |
| `check_groq.py` | Headless API testing and validation utility. |
| `run_app.ps1` / `setup_venv.ps1` | Automated setup and execution parameters for Windows environments. |
| `requirements.txt` | Explicit project dependencies. |

## Installation & Setup

### Prerequisites
- **Python 3.12+** installed on your host system.
- A **Groq API Key** (available via the [Groq Console](https://console.groq.com/keys)).

### 1. Clone & Configure
Clone the repository to your local environment and initialize the configurations:
```powershell
git clone https://github.com/adhintra28/ARGUS.git
cd ARGUS
copy .env.example .env
```
Edit the newly created `.env` file and append your `GROQ_API_KEY`:
```env
GROQ_API_KEY=your_api_key_here
VIGILAI_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
```

### 2. Automated Installation (Windows)
For rapid deployment on Windows environments, run the provided PowerShell wrappers:
```powershell
.\setup_venv.ps1
.\run_app.ps1
```

### Manual Installation
If deploying on other operating systems or manually mapping environments:
```bash
python -m venv .venv
# Activate the virtual environment
# Windows: .\.venv\Scripts\activate
# Unix/MacOS: source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Usage

1. Launch the application via `streamlit run app.py`. This hosts the UI locally at `http://localhost:8501`.
2. **Visual Evidence Tab:** Upload an image (`.png`, `.jpg`, `.webp`) and supply optional investigator context, such as timestamp or geolocation.
3. **Simulated Scenario Tab:** Paste raw textual testimonies or simulated scenario parameters for semantic analysis.
4. Interact with the generated **Intelligence Report**, analyzing the Risk Level, Suspicious Activities, Evidence limitations, and Investigative follow-up procedures.

## License

This project is distributed under the MIT License.
