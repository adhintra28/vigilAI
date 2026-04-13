# VigilAI

Hackathon prototype: an **AI-assisted visual scene review** tool for security and investigation workflows. Upload an image or describe a simulated scene; the app returns a structured report (visible facts, suspicious activities, objects, anomalies, relationships, hypotheses, risk, follow-ups) with explicit **grounding** and **limitations**.

**Disclaimer:** Outputs are model-assisted triage, not legal or forensic findings. Verify against primary evidence.

---

## Is this “complete”?

**For a hackathon demo:** yes — end-to-end UI, Groq-backed vision + JSON, offline honest mode, optional illustrative sample, validation and repair retry.

**For production:** no — no auth, audit trail, video ingestion, detector boxes, or formal evaluation harness.

---

## How we reached this solution

1. **Problem framing**  
   The brief asked for visual crime / security analysis: detect elements, context, overlooked details, and plausible narratives for investigation support.

2. **Stack choice**  
   **Python + Streamlit** keeps the prototype fast to build and easy to demo without a separate frontend. One `app.py` entrypoint and a small `analysis.py` module keeps the architecture obvious for judges.

3. **Model provider**  
   We use **Groq** with the OpenAI-compatible client (`base_url=https://api.groq.com/openai/v1`) for speed and simple JSON chat. The default vision model is **`meta-llama/llama-4-scout-17b-16e-instruct`** ([Groq vision docs](https://console.groq.com/docs/vision)), which supports images and `response_format: json_object`.

4. **Accuracy / trust**  
   Raw “AI says it’s suspicious” is weak for serious use. We iterated to an **evidence-first schema**:
   - **`visible_facts`** — only what pixels or the narrative explicitly support.
   - **`grounding`** on interpretive rows (`observed` / `inferred` / `speculative`).
   - **`limitations_and_uncertainties`** — required; enforced with **Pydantic** validation.
   - **Repair retry** — if JSON or schema fails, the model gets one correction pass.

5. **Groq image limits**  
   Groq caps base64 payload size (~4MB). Uploads are **re-encoded as JPEG** and scaled in `prepare_image_for_groq()` so typical photos stay under the limit.

6. **Offline and demo behavior**  
   Without `GROQ_API_KEY`, the app does **not** fake analysis of your file: it shows **deterministic image metadata** only, or an **optional clearly labeled illustrative sample** for pitch mode.

7. **Windows friction**  
   Many Windows installs resolve `python` to the **Microsoft Store stub**. Scripts `setup_venv.ps1` and `run_app.ps1` use the real **venv interpreter** so `streamlit` runs reliably (`python -m streamlit run app.py`).

---

## Repository layout

| File | Purpose |
|------|--------|
| `app.py` | Streamlit UI |
| `analysis.py` | Prompts, Groq calls, JPEG prep, Pydantic validation, offline payloads |
| `check_groq.py` | Smoke test API + JSON mode (no UI) |
| `setup_venv.ps1` / `run_app.ps1` | Windows-friendly venv setup and launch |
| `.env.example` | Copy to `.env`; document variables (no secrets) |

---

## Quick start

### 1. Prerequisites

- Python **3.12+** (3.14 works; from [python.org](https://www.python.org/downloads/) or Microsoft Store **full** install — avoid the “app execution alias” stub alone).
- A [Groq API key](https://console.groq.com/keys).

### 2. Configure

```powershell
cd path\to\vigilAI
copy .env.example .env
```

Edit `.env`:

- `GROQ_API_KEY` — your secret key  
- `VIGILAI_MODEL` — optional; default is Llama 4 Scout instruct (vision + JSON)

**Never commit `.env`.** It is listed in `.gitignore`.

### 3. Install and run (Windows)

```powershell
.\setup_venv.ps1
.\run_app.ps1
```

Manual equivalent:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### 4. Test Groq without Streamlit

```powershell
.\.venv\Scripts\python.exe check_groq.py
```

---

## Using the app

- **Image / scene photo:** upload PNG/JPG/WebP/GIF; optional investigator context; **Analyze image**.
- **Simulated scene (text):** paste a witness-style narrative; **Analyze simulated scene**.
- **Sidebar:** model id, force offline, illustrative sample toggle.

To stress **suspicious activities**, use a busy image and/or short neutral context (“evaluate for loitering, tailgating, unattended bags…”), or write explicit behavior in the **simulated** tab.

---

## Pushing to GitHub

1. Confirm **`git status`** does not list `.env` (if it does, unstage and keep it local-only).
2. If `.env` was ever committed, **rotate `GROQ_API_KEY`** and use `git filter-repo` or BFG if history must be cleaned.
3. Commit source only: `app.py`, `analysis.py`, `requirements.txt`, scripts, `README.md`, `.env.example`, `.gitignore`.
4. Push to your remote.

```powershell
git add app.py analysis.py requirements.txt README.md .env.example .gitignore check_groq.py setup_venv.ps1 run_app.ps1
git commit -m "Add VigilAI Streamlit prototype with Groq vision analysis"
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

---

## License

Specify a license for the hackathon repo if required by your event (e.g. MIT).
