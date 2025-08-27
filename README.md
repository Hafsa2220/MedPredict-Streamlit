# MedPredict-Streamlit

**MedPredict** is a Streamlit web application for predictive maintenance of biomedical equipment.  
It uses AI models (SVM) trained on real and simulated FMEA data to anticipate equipment failures.

## Features
- Upload Excel logs + optional PDF manuals
- AI predictions with failure scores and alerts
- Automatic "Action to take" suggestions from rules and manuals
- History of previous runs with export (CSV / XLSX)
- Equipment management and user management (technician / engineer roles)

## Installation (local)
```bash
git clone https://github.com/Hafsa2220/MedPredict-Streamlit.git
cd MedPredict-Streamlit
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run main.py
