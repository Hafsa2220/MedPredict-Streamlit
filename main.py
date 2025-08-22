import json
from pathlib import Path
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from app.utils.auth import verify_login, get_user_role


st.set_page_config(page_title="MedPredict", page_icon="🩺", layout="wide")

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

    # Download template
    with open("assets/medpredict_template.xlsx", "rb") as f:
        st.download_button(
            "Download template",
            f,
            file_name="medpredict_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        equip = st.text_input("Equipment Name", value="", placeholder="Microscope chirurgical ORL")
    with c2:
        brand = st.text_input("Brand", value="", placeholder="Leica")
    with c3:
        model = st.text_input("Model", value="", placeholder="Provido")

    e = st.file_uploader("Excel (.xlsx)", type=["xlsx"])
    p = st.file_uploader("Manual (PDF) — optional", type=["pdf"])

    if st.button("Submit"):
        if not e:
            st.error("Upload an Excel file")
            return  # ← bien à l'intérieur de la fonction !
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
# ---------- AUTH + NAV (propre) ----------

# init session
if "auth_user" not in st.session_state:
    st.session_state["auth_user"] = None

# login form
def login_view():
    st.title("MedPredict — Login")
    c1, c2 = st.columns(2)
    with c1:
        username = st.text_input("Username")
    with c2:
        password = st.text_input("Password", type="password")
    if st.button("Login", type="primary"):
        if verify_login(username, password):
            st.session_state["auth_user"] = username
            st.rerun()
        else:
            st.error("Invalid credentials")

# gate
if st.session_state["auth_user"] is None:
    login_view()
    st.stop()

# role courant (PAS stocké en session)
username = st.session_state["auth_user"]
role = get_user_role(username)
st.sidebar.caption(f"Signed in as **{username}** ({role})")

# menu selon le rôle
if role == "technician":
    pages = ["Upload & Predict", "History", "About", "Logout"]
else:
    pages = ["Upload & Predict", "History", "Model & Thresholds", "Rules & Actions", "About", "Logout"]

page = st.sidebar.radio("Navigation", pages, index=0)

# router
if page == "Upload & Predict":
    page_upload()
elif page == "History":
    page_history()
elif page == "Model & Thresholds":
    page_model()
elif page == "Rules & Actions":
    page_rules()
elif page == "About":
    page_about()
elif page == "Logout":
    # option A : ne remettre à zéro que l’utilisateur
    st.session_state["auth_user"] = None
    st.rerun()

    # option B : tout vider si tu stockes d’autres clés et veux repartir à zéro
    # st.session_state.clear()
    # st.rerun()

# ---------- /AUTH + NAV ----------

