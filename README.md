
# PSI 03–19 Evaluator (2025)

A Streamlit app that reads your **compatible Excel**, uses your PSI evaluation engine, and displays **INCLUSION/EXCLUSION** results with a **clickable dropdown** (expander) showing the checklist that led to that result — encounter by encounter.

> **No `NOT_APPLICABLE`:** The UI collapses any `NOT_APPLICABLE` from the core into `EXCLUSION`, per Mr. Paul's instruction.

## 🗂 Repo contents
- `streamlit_app.py` — Streamlit UI
- `psi_core.py` — your provided core (copied from `psi_calculator_2025_03_19_patched.py`)
- `PSI_Code_Sets_2025.json` — code sets
- `requirements.txt`

## ▶️ Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## ☁️ Deploy on Streamlit Cloud
1. Push this folder to a public or private GitHub repo.
2. In Streamlit Cloud, point the app to `streamlit_app.py`.
3. (Optional) Keep `PSI_Code_Sets_2025.json` at repo root or update the sidebar path.

## 📥 Inputs
Upload an Excel (`.xlsx`/`.xls`) with headers compatible with your core. The app normalizes headers, accepting uppercase variants for **AGE/SEX/YEAR** as well.

You should include **EncounterID** to group results; if missing, the app generates a surrogate ID so you can still review.

## 📤 Outputs
- **KPI summary** (counts by PSI and result)
- **Consolidated CSV** to download
- **Encounter-wise expanders** listing each PSI's decision and the checklist table

## 🔧 Notes
- The app uses your core's registry `PSI_LIST` and `_PSI_REGISTRY` to evaluate PSI 03–19.
- If any PSI function raises, the app marks that PSI as `EXCLUSION` with a "Runtime error" rationale — so the UI never breaks.
- `NOT_APPLICABLE` is converted to `EXCLUSION` in the UI and CSV.
