
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader

st.set_page_config(page_title="MedPredict", page_icon="🩺", layout="wide")

# --- simple auth (demo) ---
if "role" not in st.session_state:
    st.session_state["role"] = None
if st.session_state["role"] is None:
    st.title("MedPredict — Login")
    u = st.text_input("Username", value="tech01")
    p = st.text_input("Password", type="password", value="medtech")
    if st.button("Login"):
        if (u, p) == ("tech01","medtech"):
            st.session_state["role"] = "technician"
            st.experimental_rerun()
        elif (u, p) == ("biomed01","biomed"):
            st.session_state["role"] = "engineer"
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

st.sidebar.markdown(f"**Role:** {st.session_state['role']}")
page = st.sidebar.radio("Navigation", ["Upload & Predict","History","Model & Thresholds","Rules & Actions","About","Logout"] if st.session_state["role"]=="engineer" else ["Upload & Predict","History","About","Logout"])

def load_cfg():
    p = Path("history/config.json")
    if p.exists():
        return json.loads(p.read_text())
    return {"threshold":0.5,"prob_mode":False}

def save_cfg(cfg):
    Path("history").mkdir(exist_ok=True, parents=True)
    Path("history/config.json").write_text(json.dumps(cfg))

def page_about():
    st.write("Upload Excel logs + optional PDF manual, predict failures, and see history.")

def predict(df, threshold=None):
    # Fallback heuristic for this minimal bundle
    out = pd.Series(0, index=df.index, dtype=int)
    if "severity" in df.columns:
        s = df["severity"].astype(str).str.lower()
        out = out.where(~s.isin(["high","critical"]), 1)
    if "anomaly_score" in df.columns:
        out = out.where(~(pd.to_numeric(df["anomaly_score"], errors="coerce")>=0.8), 1)
    return out

def page_upload():
    st.header("Upload & Predict")
    c1,c2,c3 = st.columns(3)
    with c1: equip = st.text_input("Equipment Name","Microscope chirurgical ORL")
    with c2: Company = st.text_input("Company","Leica")
    with c3: model = st.text_input("Model","Provido")
    e = st.file_uploader("Excel (.xlsx)", type=["xlsx"])
    p = st.file_uploader("Manual (PDF) — optional", type=["pdf"])
    if st.button("Submit"):
        if not e:
            st.error("Upload an Excel file")
            return
        df = pd.read_excel(e, engine="openpyxl")
        cfg = load_cfg()
        thr = cfg["threshold"] if cfg.get("prob_mode") else None
        preds = predict(df, threshold=thr)
        df["Failure_Predicted"] = preds
        st.success(f"Predicted failures: {int(preds.sum())}")
        st.dataframe(df.head(100), use_container_width=True)

def page_history():
    st.header("History")
    st.info("Full history module is included in the full package; this minimal starter shows navigation only.")

def page_model():
    st.header("Model & Thresholds")
    cfg = load_cfg()
    cfg["prob_mode"] = st.toggle("Probability mode", value=cfg.get("prob_mode",False))
    cfg["threshold"] = st.slider("Decision threshold", 0.0, 1.0, float(cfg.get("threshold",0.5)), 0.01)
    if st.button("Save settings"):
        save_cfg(cfg); st.success("Saved.")

def page_rules():
    st.header("Rules & Actions")
    st.info("Rules editor is included in the full package; this minimal starter shows navigation only.")

if page=="Upload & Predict": page_upload()
elif page=="History": page_history()
elif page=="Model & Thresholds": page_model()
elif page=="Rules & Actions": page_rules()
elif page=="About": page_about()
elif page=="Logout":
    st.session_state["role"] = None
    st.experimental_rerun()
