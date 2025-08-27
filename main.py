import json
from pathlib import Path
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from PIL import Image
from base64 import b64encode
import numpy as np
from datetime import datetime
import io
import re, unicodedata

# --- utils locaux
from app.utils.auth import (
    verify_login, get_user_role,
    list_users, create_or_update_user,
    delete_user as auth_delete_user,
    change_password as auth_change_password,
    set_password_admin,
)
from app.utils.history import ensure_history, save_run, read_meta
from app.utils.rules import load_rules, save_rules, apply_rules
from app.utils.validate import validate_df
from app.utils.i18n import t
from app.utils.model import (
    load_equipment_registry, get_equipment_entry, list_equipment_labels,
    _select_features, _set_last_proba_error, _coerce_features_for_model,
)

# Liste des équipements depuis le registry (fallback si vide)
EQUIPMENT_NAMES = [e.get("label") for e in load_equipment_registry() if e.get("label")] \
                  or ["Surgical microscope ORL", "Autoclave"]

# --- anti "help() / st.help()" qui pollue l'UI ---
import builtins, streamlit as _st
builtins.help = lambda *a, **k: None
_st.help       = lambda *a, **k: None

# --- Always-on status badge (sidebar)
def ensure_status_badge():
    with st.sidebar:
        ph = st.session_state.get("status_ph")
        if ph is None:
            st.session_state["status_ph"] = st.empty()
            ph = st.session_state["status_ph"]
        c = int(st.session_state.get("last_alert_count", 0) or 0)
        badge = t("sidebar.status.ok") if c == 0 else t("sidebar.status.alert", n=c)
        ph.markdown(
            f"<div class='status-line'><strong>{t('sidebar.status')}</strong> {badge}</div>",
            unsafe_allow_html=True,
        )

def show_status_badge_for(page_name: str):
    """Affiche le badge de statut dans la sidebar uniquement pour certains écrans."""
    allowed = {"Upload & Predict", "History"}

    # créer/récupérer le placeholder dans la sidebar
    if "status_ph" not in st.session_state or st.session_state["status_ph"] is None:
        with st.sidebar:
            st.session_state["status_ph"] = st.empty()

    ph = st.session_state["status_ph"]

    if page_name in allowed:
        n = int(st.session_state.get("last_alert_count") or 0)
        badge = t("sidebar.status.ok") if n == 0 else t("sidebar.status.alert", n=n)
        with st.sidebar:
            ph.markdown(
                f"<div class='status-line'><strong>{t('sidebar.status')}</strong> {badge}</div>",
                unsafe_allow_html=True,
            )
    else:
        # Nettoyer la zone si on n'est pas sur une page autorisée
        with st.sidebar:
            ph.empty()

# --- Helpers anti-bruit ---
import builtins as _blt
import sys as _sys, io as _io, contextlib as _ctx

def _silence_help():
    try:
        _blt.help = lambda *a, **k: None
        _st.help  = lambda *a, **k: None
    except Exception:
        pass

@_ctx.contextmanager
def _mute_stdout():
    old_out, old_err = _sys.stdout, _sys.stderr
    buf_out, buf_err = _io.StringIO(), _io.StringIO()
    try:
        _sys.stdout, _sys.stderr = buf_out, buf_err
        yield
    finally:
        _sys.stdout, _sys.stderr = old_out, old_err

# ---------------- Normalisation colonnes/valeurs (non utilisée pour l’instant) ----------------
def _slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[\W_]+", " ", s).strip().lower()
    return s

def _tokens(s: str) -> set[str]:
    return set(_slug(s).split())

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Gardé pour référence, non utilisé pour éviter de casser les entêtes du template unifié
    return df

def normalize_values(df: pd.DataFrame) -> pd.DataFrame:
    # Gardé pour référence, non utilisé pour éviter de casser les entêtes du template unifié
    return df

# ---------------- PDF helper ----------------
def _extract_pdf_text(pdf_bytes: bytes | None) -> str | None:
    if not pdf_bytes:
        return None
    try:
        r = PdfReader(io.BytesIO(pdf_bytes))
        parts = []
        for pg in r.pages:
            txt = pg.extract_text() or ""
            if txt:
                parts.append(txt)
        return "\n".join(parts) if parts else None
    except Exception:
        return None

# ---------------- CONFIG / SESSION ----------------
LOGO = Image.open("assets/logo.png")
st.set_page_config(page_title="MedPredict", page_icon=LOGO, layout="wide")

def init_auth_state():
    defaults = {
        "auth_user": None,
        "auth_role": None,
        "auth_error": None,
        "auth_expires_at": None,
        "last_alert_count": 0,
        "pred_df": None,
        "pred_nb": 0,
        "pred_meta": None,
        # marqueur de dernière soumission pour contrôler l’affichage de la bannière
        "_upload_last_submit_ts": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
init_auth_state()

# ---------------- Status (stockage du compteur uniquement) ----------------
def draw_status(count: int | None = None):
    """Mets à jour le compteur d’alertes en session. L’affichage est fait par la sidebar."""
    if count is None:
        st.session_state.setdefault("last_alert_count", 0)
    else:
        st.session_state["last_alert_count"] = int(count)

def build_pages_list(role: str) -> list[str]:
    role_lc = (role or "").strip().lower()
    if role_lc == "engineer":
        return [
            "Upload & Predict",
            "History",
            "Equipment Management",   # << ajouté
            "Rules & Actions",
            "Users",
            "About",
            "My account",
            "Logout",
        ]
    # Technicien : pas de Model & Thresholds ni Equipment Management
    return [
        "Upload & Predict",
        "History",
        "About",
        "My account",
        "Logout",
    ]

def render_sidebar_and_nav() -> str:
    """Sidebar rendue à chaque run (logo & titre). Le statut utilise un placeholder réactualisable."""
    username = st.session_state.get("auth_user", "")
    role = get_user_role(username) or "—"
    pages = build_pages_list(role)

    # paramètres d'affichage
    LOGO_WIDTH = 150
    TOP_OFFSET = -20

    with st.sidebar:
        # CSS compact
        st.markdown("""
        <style>
          [data-testid="stSidebar"] section[data-testid="stSidebarContent"]{padding-top:8px !important;}
          [data-testid="stSidebar"] > div:first-child{padding-top:0px !important;}
          [data-testid="stSidebar"] .status-line { margin: 0 0 4px 0; }
          [data-testid="stSidebar"] hr { margin: 6px 0 !important; }
          .sidebar-title{
            margin-top:8px; margin-bottom:6px; font-weight:600;
            letter-spacing:2px; color:#334155; text-align:center;
          }
        </style>
        """, unsafe_allow_html=True)

        # Logo + titre
        logo_b64 = b64encode(open("assets/logo.png", "rb").read()).decode()
        st.markdown(
            f"""
            <div style="display:flex; justify-content:center; align-items:center; margin-top:{TOP_OFFSET}px;">
              <img src="data:image/png;base64,{logo_b64}" style="width:{LOGO_WIDTH}px; height:auto; border-radius:8px;" />
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"<div class='sidebar-title'>{t('app.title')}</div>", unsafe_allow_html=True)

        # --- Statut (placeholder) ---
        if "status_ph" not in st.session_state:
            st.session_state["status_ph"] = st.empty()

        def _render_status_badge():
            c = int(st.session_state.get("last_alert_count", 0) or 0)
            badge = t("sidebar.status.ok") if c == 0 else t("sidebar.status.alert", n=c)
            st.session_state["status_ph"].markdown(
                f"<div class='status-line'><strong>{t('sidebar.status')}</strong> {badge}</div>",
                unsafe_allow_html=True,
            )

        _render_status_badge()
        st.markdown("<hr />", unsafe_allow_html=True)

        # Signed in
        st.caption(t("sidebar.signed_in", user=username or "—", role=role or "—"))

        # Navigation
        if "nav_page" not in st.session_state or st.session_state["nav_page"] not in pages:
            st.session_state["nav_page"] = pages[0]
        current = st.radio(
            "Navigation",
            pages,
            index=pages.index(st.session_state["nav_page"]),
            key="nav_page",
        )

    return current


# ---------------- utilités prédiction ----------------
def _as_series(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return None
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    return obj

def predict_heuristic(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(0, index=df.index, dtype=int)
    sev = _as_series(df, "severity")
    if sev is not None:
        s = sev.astype(str).str.lower()
        out = out.where(~s.isin(["high","critical"]), 1)
    score = _as_series(df, "anomaly_score")
    if score is not None:
        out = out.where(~(pd.to_numeric(score, errors="coerce") >= 0.8), 1)
    return out

def compute_predictions(df, prob_mode, threshold, equipment_type):
    """
    Retourne (df_enrichi, alerts_count).
    - Utilise le SVM si possible (predict_proba / decision_function / predict).
    - Fallback propre si le modèle ne tourne pas (anomaly_score ou zéros).
    - Garantit toujours Failure_Score + Failure_Predicted pour l'UI.
    """
    import numpy as np, pandas as pd, joblib, streamlit as st
    from app.utils.model import (
        get_equipment_entry,
        _select_features,
        _set_last_proba_error,
        _coerce_features_for_model,
    )

    ent = get_equipment_entry(equipment_type) or {}
    model_path = (ent.get("model") or {}).get("path")

    scores = None

    # Flags debug par défaut
    try:
        st.session_state["debug_used_model"] = False
        st.session_state["debug_score_source"] = "fallback:anomaly_score"
    except Exception:
        pass

    # 1) Features strictes pour CET équipement
    X, meta = _select_features(df, ent)

    # Exposer la liste dans le debug
    try:
        st.session_state["debug_selected_features"] = (meta or {}).get("selected_features")
    except Exception:
        pass

    # 2) Si X est None OU s'il manque des colonnes -> on ne tente pas le modèle
    missing_list = (meta or {}).get("features_missing")
    if (X is None) or (missing_list not in (None, [], ())):
        _set_last_proba_error(
            equipment_type,
            f"columns are missing: {set(missing_list or [])}"
        )
    else:
        # 3) Coercition robuste (numérique + NA -> 0)
        X = _coerce_features_for_model(X)

        # 4) Tentative modèle
        try:
            clf = joblib.load(model_path)
            if prob_mode and hasattr(clf, "predict_proba"):
                scores = clf.predict_proba(X)[:, 1]
                src = "model:predict_proba"
            elif hasattr(clf, "decision_function"):
                raw = clf.decision_function(X)
                mn, mx = float(np.min(raw)), float(np.max(raw))
                scores = (raw - mn) / (mx - mn + 1e-9)
                src = "model:decision_function"
            else:
                pred = clf.predict(X)
                scores = np.clip(pred.astype(float), 0.0, 1.0)
                src = "model:predict"

            _set_last_proba_error(equipment_type, None)
            try:
                st.session_state["debug_used_model"] = True
                st.session_state["debug_score_source"] = src
            except Exception:
                pass

        except Exception as e:
            _set_last_proba_error(equipment_type, repr(e))

    # 5) Fallback propre si le modèle n'a pas tourné
    if scores is None:
        if "anomaly_score" in df.columns:
            scores = pd.to_numeric(df["anomaly_score"], errors="coerce").fillna(0.0).to_numpy()
            try:
                st.session_state["debug_score_source"] = "fallback:anomaly_score"
            except Exception:
                pass
        else:
            scores = np.zeros(len(df), dtype=float)
            try:
                st.session_state["debug_score_source"] = "fallback:zeros(no_anomaly_score)"
            except Exception:
                pass

    # 6) Colonnes garanties pour l'UI
    df["Failure_Score"] = scores
    df["Failure_Predicted"] = (df["Failure_Score"] >= float(threshold)).astype(int)

    # 7) Ménage colonnes redondantes (name == type)
    try:
        if "equipment_name" in df.columns and "equipment_type" in df.columns:
            if df["equipment_name"].astype(str).equals(df["equipment_type"].astype(str)):
                df.drop(columns=["equipment_type"], inplace=True)
    except Exception:
        pass

    return df, int(df["Failure_Predicted"].sum())



# ---------------- PAGES ----------------
def load_cfg():
    p = Path("history/config.json")
    if p.exists():
        return json.loads(p.read_text())
    return {"threshold": 0.5, "prob_mode": False}

def save_cfg(cfg):
    Path("history").mkdir(exist_ok=True, parents=True)
    Path("history/config.json").write_text(json.dumps(cfg))

def page_upload():
    """Upload + predict + export (multi-équipement, stable)."""
    from pathlib import Path
    import io
    import numpy as np
    import pandas as pd
    # helpers de debug exposés par app.utils.model
    from app.utils.model import get_last_missing_features, get_last_proba_error
    try:
        from app.utils.model import get_last_selected_features as _get_last_selected_features
    except Exception:
        _get_last_selected_features = lambda equip: None  # fallback silencieux

    _silence_help()
    st.header(t("upload.header"))
    st.session_state.pop("_rendering_upload", None)
    # Toujours afficher le badge de statut dès l’ouverture de la page
    

    # ---- Template
    with open("assets/medpredict_template.xlsx", "rb") as f:
        st.download_button(
            t("upload.download_template"),
            f,
            file_name="medpredict_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_template_btn_v3",
        )

    # =======================
    #  EQUIPMENT / BRAND / MODEL
    # =======================
    c1, c2, c3 = st.columns(3)
    with c1:
        opts = list_equipment_labels() or ["Surgical microscope ORL", "Autoclave"]
        if st.session_state.get("equip_type") not in opts:
            st.session_state["equip_type"] = opts[0]
        equip_type = st.selectbox(
            t("upload.equipment"),
            options=opts,
            index=opts.index(st.session_state["equip_type"]),
            key="equip_type",
        )

    # reset brand/model si l’équipement change
    if st.session_state.get("_last_equip") != equip_type:
        for k in list(st.session_state.keys()):
            if k.startswith("brand_sel_") or k.startswith("model_sel_"):
                st.session_state.pop(k, None)
        st.session_state["_last_equip"] = equip_type

    ent = get_equipment_entry(equip_type) or {}
    brand_opts = ent.get("brand_options") or ent.get("brands") or []
    model_opts = ent.get("model_options") or ent.get("models") or []

    ent_key   = (ent.get("id") or (equip_type or "")).strip().lower().replace(" ", "_")
    brand_key = f"brand_sel_{ent_key}"
    model_key = f"model_sel_{ent_key}"

    if brand_opts and st.session_state.get(brand_key) not in brand_opts:
        st.session_state[brand_key] = brand_opts[0]
    if model_opts and st.session_state.get(model_key) not in model_opts:
        st.session_state[model_key] = model_opts[0]

    with c2:
        if brand_opts:
            st.selectbox(
                t("upload.brand"),
                options=brand_opts,
                index=brand_opts.index(st.session_state.get(brand_key, brand_opts[0])),
                key=brand_key,
            )
        else:
            st.text_input(t("upload.brand"), value="", placeholder="e.g.: Leica", key=brand_key)

    with c3:
        if model_opts:
            st.selectbox(
                t("upload.model"),
                options=model_opts,
                index=model_opts.index(st.session_state.get(model_key, model_opts[0])),
                key=model_key,
            )
        else:
            st.text_input(t("upload.model"), value="", placeholder="e.g.: Provido", key=model_key)

    # =======================
    #  FORMULAIRE
    # =======================
    with st.form("upload_form", clear_on_submit=False):
        excel_file = st.file_uploader(
            t("upload.excel"), type=["xlsx"], key="excel_file", accept_multiple_files=False
        )
        pdf_file = st.file_uploader(
            t("upload.pdf"), type=["pdf"], key="pdf_manual", accept_multiple_files=False
        )
        submitted = st.form_submit_button(t("upload.submit"))
    # ---- Placeholder bannière unique (hors formulaire, emplacement fixe)
    if "_alert_banner_ph" not in st.session_state or st.session_state["_alert_banner_ph"] is None:
        st.session_state["_alert_banner_ph"] = st.empty()
    banner_ph = st.session_state["_alert_banner_ph"]

    # masquer la bannière tant qu’aucune soumission *courante* n’a eu lieu
    if not st.session_state.get("_upload_last_submit_ts"):
        banner_ph.empty()

    # ---------- Affichage du dernier résultat (sans toucher à la bannière/statut) ----------
    pred_df = st.session_state.get("pred_df")
    if isinstance(pred_df, pd.DataFrame):
        st.dataframe(pred_df.head(100), use_container_width=True)

        csv_str = pred_df.to_csv(index=False, sep=";", decimal=",", encoding="utf-8-sig")
        st.download_button(
            "Download predictions (CSV ;)",
            data=csv_str,
            file_name="predictions.csv",
            mime="text/csv",
            key="dl_csv_pred_v2",
        )

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            pred_df.to_excel(writer, index=False, sheet_name="predictions")
        buf.seek(0)
        st.download_button(
            "Download predictions (XLSX)",
            data=buf.getvalue(),
            file_name="predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_xlsx_pred_v2",
        )

        # Clear immédiat : ne touche ni au statut ni à la bannière
        if st.button("Clear result", key="clear_pred_v2"):
            st.session_state["pred_df"] = None
            st.session_state["pred_nb"] = 0
            st.session_state["pred_meta"] = None
            # NE PAS modifier last_alert_count ni _upload_last_submit_ts, ni vider banner_ph
            st.rerun()

    if not submitted:
        return

    # ---------- Lecture fichiers ----------
    if excel_file is None:
        st.warning(t("upload.need_excel"))
        return

    try:
        excel_bytes = excel_file.getvalue()
    except Exception:
        excel_bytes = None
    if not excel_bytes:
        st.error("Uploaded Excel file is empty or unreadable.")
        return

    try:
        raw = pd.read_excel(io.BytesIO(excel_bytes), engine="openpyxl")
    except Exception as ex:
        st.error(f"Could not read Excel: {ex}")
        return

    try:
        pdf_bytes = pdf_file.getvalue() if pdf_file is not None else None
    except Exception:
        pdf_bytes = None

    # ---------- Normalisation & enrichissement ----------
    # IMPORTANT : on conserve les en-têtes EXACTS du template (19 colonnes unifiées)
    df = raw.copy()

    brand       = st.session_state.get(brand_key, "")
    equip_model = st.session_state.get(model_key, "")

    equip_clean = (equip_type or "").strip()
    if "equipment_name" in df.columns:
        if equip_clean:
            df["equipment_name"] = equip_clean
    else:
        df.insert(0, "equipment_name", equip_clean)

    if "equipment_type" not in df.columns:
        df["equipment_type"] = equip_type

    df = df.loc[:, ~df.columns.duplicated()]

    # ---------- Validation (alignée sur le modèle) ----------
    ent = get_equipment_entry(equip_type) or {}
    model_path = (ent.get("model") or {}).get("path")

    expected = []
    try:
        import joblib
        clf_tmp = joblib.load(model_path)
        if getattr(clf_tmp, "feature_names_in_", None) is not None:
            expected = [str(c) for c in clf_tmp.feature_names_in_]
    except Exception:
        pass
    if not expected:
        expected = list((ent.get("model") or {}).get("features") or [])

    missing = [c for c in expected if c not in df.columns]
    if missing:
        st.error(
            f"Missing required feature columns for '{equip_type}': {missing}. "
            f"Use the exact 19 headers from the training template."
        )
        return

    # ---------- Prédiction ----------
    prob_mode = bool(ent.get("prob_mode", False))
    try:
        thr = float(ent.get("threshold", 0.5))
    except Exception:
        thr = 0.5

    with st.spinner(t("upload.predicting")):
        df, alerts_count = compute_predictions(df, prob_mode, thr, equip_type)

    # ---- MAJ bannière + statut (après soumission uniquement) ----
    msg = t("upload.pred.alert", n=int(alerts_count)) if alerts_count > 0 else t("upload.pred.ok")
    (banner_ph.error if alerts_count > 0 else banner_ph.success)(msg)
    draw_status(alerts_count)                                # met à jour la valeur
    show_status_badge_for("Upload & Predict")               # REFRESH visuel immédiat
    st.session_state["_upload_last_submit_ts"] = datetime.now().timestamp()

    # >>> Garantir la présence d'une colonne de score, même en fallback
    if "Failure_Score" not in df.columns:
        if "anomaly_score" in df.columns:
            df["Failure_Score"] = pd.to_numeric(df["anomaly_score"], errors="coerce").fillna(0.0).astype(float)
        else:
            df["Failure_Score"] = 0.0
    if "Failure_Predicted" not in df.columns:
        df["Failure_Predicted"] = (df["Failure_Score"] >= thr).astype(int)

    # ---------- Nettoyage colonnes redondantes ----------
    try:
        if "equipment_name" in df.columns and "equipment_type" in df.columns:
            same = df["equipment_name"].astype(str).equals(df["equipment_type"].astype(str))
            if same:
                df.drop(columns=["equipment_type"], inplace=True)
    except Exception:
        pass

    # ---------- Debug détaillé ----------
    try:
        import joblib
        joblib_ok = True
        joblib_ver = joblib.__version__
    except Exception as e:
        joblib = None
        joblib_ok = False
        joblib_ver = f"not available: {e!r}"

    model_path = (ent.get("model") or {}).get("path")
    p = Path(model_path) if model_path else None
    exists = p.exists() if p else False
    abs_path = str(p.resolve()) if p else None
    size = (p.stat().st_size if exists else None)

    can_load = None
    load_exc = None
    if exists and joblib_ok:
        try:
            _ = joblib.load(p)
            can_load = True
        except Exception as e:
            can_load = False
            load_exc = repr(e)

    if not exists:
        st.error(f"Model file not found at: {model_path} (resolved: {abs_path})")
    elif can_load is False:
        st.error(f"Model file exists but failed to load via joblib: {load_exc}")
    elif can_load is True:
        st.success("Model file exists and can be loaded.")

    features_missing = get_last_missing_features(equip_type)
    proba_err = get_last_proba_error(equip_type)
    selected_feats = _get_last_selected_features(equip_type)

    if features_missing:
        st.warning(
            f"Model expects missing features in the Excel: {features_missing}. "
            f"The 'Failure_Score' column will fallback to anomaly_score until these columns exist (numeric)."
        )
    if proba_err:
        st.info(f"predict_proba() fallback cause: {proba_err}")

    debug_payload = {
        "equip_type": equip_type,
        "registry_model_path": model_path,
        "resolved_abs_path": abs_path,
        "file_exists": exists,
        "file_size_bytes": size,
        "joblib_available": joblib_ok,
        "joblib_version": joblib_ver,
        "can_joblib_load_file": can_load,
        "load_exception": load_exc,
        "prob_mode_USED": prob_mode,
        "threshold_USED": thr,
        "score_column_present": ("Failure_Score" in df.columns),
        # indicateurs runtime (renseignés par compute_predictions)
        "used_model": st.session_state.get("debug_used_model"),
        "score_source": st.session_state.get("debug_score_source"),
        "selected_features_runtime": st.session_state.get("debug_selected_features"),
        # caches utilitaires
        "features_missing": features_missing,
        "selected_features": selected_feats,
        "predict_proba_error": proba_err,
        "df_columns_sample": list(df.columns)[:25],
    }
    st.session_state["last_debug_upload"] = debug_payload  # mémorise


    # ---------- Règles -> Action ----------
    from app.utils.rules import apply_rules, extract_pdf_actions

    # 1) Actions automatiques (JSON + extraction PDF)
    try:
        actions = apply_rules(df.copy(), pdf_bytes=pdf_bytes)   # -> Series (peut contenir NaN)
    except Exception:
        actions = None

    # 2) Écriture dans la colonne (sans convertir NaN en "None")
    if isinstance(actions, pd.Series):
        df["Action to take"] = actions.where(actions.notna(), "—")
    else:
        # Fallback si apply_rules a échoué ou n'a rien renvoyé
        df["Action to take"] = np.where(df["Failure_Predicted"] == 1, "Check within 24h", "—")


    # ---------- Affichage du résultat (run courant, sans rerun) ----------
    with st.expander("Predictions (current run)", expanded=True):
        view_df = df.rename(columns={
            "Failure_Score": "Failure score",
            "Failure_Predicted": "Failure predicted",
        })
        st.dataframe(view_df, use_container_width=True)

    # --- Downloads (current run) ---
    csv_str = view_df.to_csv(index=False, sep=";", decimal=",", encoding="utf-8-sig")
    st.download_button(
        "Download predictions (CSV ;)",
        data=csv_str,
        file_name="predictions.csv",
        mime="text/csv",
        key="dl_csv_pred_current",
    )

    import io, pandas as pd
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        view_df.to_excel(writer, index=False, sheet_name="predictions")
    buf.seek(0)
    st.download_button(
        "Download predictions (XLSX)",
        data=buf.getvalue(),
        file_name="predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_xlsx_pred_current",
    )

    # --- Clear result (current run) ---
    if st.button("Clear result", key="clear_pred_current"):
        st.session_state["pred_df"] = None
        st.session_state["pred_nb"] = 0
        st.session_state["pred_meta"] = None
        # NE PAS toucher au statut ni à la bannière
        st.rerun()


    # ---------- Persistance session ----------
    st.session_state["last_alert_count"] = int(alerts_count)
    st.session_state["pred_df"] = df
    st.session_state["pred_nb"] = int(alerts_count)
    st.session_state["pred_meta"] = {
        "equipment_type": equip_type,
        "equipment_name": equip_clean,
        "brand": brand,
        "model": equip_model,
        "prob_mode": prob_mode,
        "threshold": thr,
    }

    # ---------- Sauvegarde run ----------
    username = st.session_state.get("auth_user", "")
    role = get_user_role(username)
    ensure_history()
    run_id = save_run(
        df,
        {
            "user": username,
            "role": role,
            "equipment_type": equip_type,
            "equipment_name": equip_clean,
            "brand": brand,
            "model": equip_model,
            "prob_mode": prob_mode,
            "threshold": thr,
            "n_predicted": int(alerts_count),
            "n_rows": int(len(df)),
        },
        pdf_bytes=pdf_bytes,
    )
    st.success(t("upload.saved", rid=run_id))
    if st.session_state.get("auto_rerun_after_save"):
        st.rerun()
    # sinon: on reste sur la page et le Debug reste visible





def page_history():
    show_status_badge_for("History")
    _silence_help()
    st.header("History")
    ensure_history()
    meta = read_meta()

    username = st.session_state.get("auth_user", "")
    role = get_user_role(username)

    if meta.empty:
        st.info("No runs yet.")
        return

    if "datetime" in meta.columns:
        meta["datetime"] = pd.to_datetime(meta["datetime"], errors="coerce")
    if role == "technician":
        meta = meta[meta["user"] == username].copy()

    dmin = meta["datetime"].min().date() if meta["datetime"].notna().any() else datetime.today().date()
    dmax = meta["datetime"].max().date() if meta["datetime"].notna().any() else datetime.today().date()

    if "hist_filters" not in st.session_state:
        st.session_state["hist_filters"] = {"start": dmin, "end": dmax, "equip": "", "brand": "", "model": ""}

    with st.form("history_filters", clear_on_submit=False):
        fc1, fc2 = st.columns(2)
        start = fc1.date_input("Start date", value=st.session_state["hist_filters"]["start"])
        end   = fc2.date_input("End date",   value=st.session_state["hist_filters"]["end"])

        fc3, fc4, fc5 = st.columns(3)
        f_equip = fc3.text_input("Equipment name contains", st.session_state["hist_filters"]["equip"])
        f_brand = fc4.text_input("Brand contains",          st.session_state["hist_filters"]["brand"])
        f_model = fc5.text_input("Model contains",          st.session_state["hist_filters"]["model"])

        cA, cB = st.columns(2)
        apply = cA.form_submit_button("Apply filters", type="primary")
        reset = cB.form_submit_button("Reset")

    if apply:
        st.session_state["hist_filters"] = {"start": start, "end": end, "equip": f_equip, "brand": f_brand, "model": f_model}
    elif reset:
        st.session_state["hist_filters"] = {"start": dmin, "end": dmax, "equip": "", "brand": "", "model": ""}

    f = st.session_state["hist_filters"]
    start, end, f_equip, f_brand, f_model = f["start"], f["end"], f["equip"], f["brand"], f["model"]

    df = meta[
        (meta["datetime"] >= pd.Timestamp(start)) &
        (meta["datetime"] <  pd.Timestamp(end) + pd.Timedelta(days=1))
    ].copy()

    for col, val in [("equipment_name", f_equip), ("brand", f_brand), ("model", f_model)]:
        if val:
            df = df[df[col].astype(str).str.contains(val, case=False, na=False)]

    st.subheader(f"{len(df)} run(s)")
    st.dataframe(
        df.sort_values("datetime", ascending=False),
        use_container_width=True,
        hide_index=True,
        height=460,
        column_config={
            "run_id":        st.column_config.TextColumn("Run ID", width="medium"),
            "datetime":      st.column_config.DatetimeColumn("Date/Time", format="YYYY-MM-DD HH:mm:ss", width="large"),
            "user":          st.column_config.TextColumn("User", width="small"),
            "role":          st.column_config.TextColumn("Role", width="small"),
            "equipment_name":st.column_config.TextColumn("Equipment", width="large"),
            "brand":         st.column_config.TextColumn("Brand", width="small"),
            "model":         st.column_config.TextColumn("Model", width="small"),
            "n_rows":        st.column_config.NumberColumn("Rows", format="%d", width="small"),
            "n_predicted":   st.column_config.NumberColumn("Alerts", format="%d", width="small"),
        },
    )

    st.divider()

    if df.empty:
        return

    rid_view = st.selectbox("Preview a run", df["run_id"].tolist(), key="prev_run_id")
    if not rid_view:
        return

    sel = df.set_index("run_id").loc[rid_view]
    try:
        n_pred = int(sel.get("n_predicted", 0) or 0)
    except Exception:
        n_pred = 0

    (st.error if n_pred > 0 else st.success)(
        t("history.preview.alert", n=n_pred) if n_pred > 0 else t("history.preview.ok")
    )

    prev = int(st.session_state.get("last_alert_count", 0) or 0)
    if prev != n_pred:
        st.session_state["last_alert_count"] = n_pred
        st.rerun()

    run_path = Path(sel["run_path"]) if "run_path" in sel else None
    if run_path is not None and (run_path / "preview.csv").exists():
        st.caption(str(run_path))
        st.dataframe(pd.read_csv(run_path / "preview.csv"), use_container_width=True)
    else:
        st.info("No preview.csv found for this run.")


def page_rules():
    st.header(t("rules.header"))
    username = st.session_state.get("auth_user", "")
    role = get_user_role(username)

    if role != "engineer":
        st.info(t("rules.engineer_only"))
        return

    st.markdown(t("rules.format"))
    st.caption(t("rules.example.caption") + ' `{"name":"High severity","match":{"column":"severity","in":["high"]},"action":"Priority intervention"}`')

    rules = load_rules()
    txt = st.text_area(t("rules.json.label"),
                       value=json.dumps(rules, ensure_ascii=False, indent=2),
                       height=320)

    c1, c2 = st.columns(2)
    if c1.button(t("rules.save")):
        try:
            new_rules = json.loads(txt)
            if not isinstance(new_rules, list):
                raise ValueError("JSON must be a list of rules.")
            save_rules(new_rules)
            st.success(t("rules.save.ok"))
        except Exception as ex:
            st.error(t("rules.save.bad", err=str(ex)))

    if c2.button(t("rules.restore")):
        try:
            from app.utils.rules import DEFAULT_RULES
            save_rules(DEFAULT_RULES)
            st.success(t("rules.restore.ok"))
            st.rerun()
        except Exception:
            st.warning("DEFAULT_RULES not found; cannot restore.")

def page_equipment_mgmt():
    from app.utils.model import (
        load_equipment_registry, save_equipment_registry,
        upsert_equipment_entry, delete_equipment_entry, detect_model_features
    )

    _silence_help()
    st.header("Equipment Management")

    # Accès ingénieur uniquement
    username = st.session_state.get("auth_user", "")
    role = (get_user_role(username) or "").strip().lower()
    if role != "engineer":
        st.info("This page is restricted to the engineer.")
        st.stop()

    reg = load_equipment_registry()
    ids = [e.get("id", "") for e in reg]

    cL, cR = st.columns([2, 3])

    # ----- Liste & sélection -----
    with cL:
        st.subheader("Registered equipments")
        if reg:
            st.dataframe(
                pd.DataFrame([{
                    "id": e.get("id"),
                    "label": e.get("label"),
                    "model_path": (e.get("model") or {}).get("path"),
                    "prob_mode": e.get("prob_mode"),
                    "threshold": e.get("threshold"),
                } for e in reg]),
                use_container_width=True, hide_index=True, height=260,
            )
        else:
            st.info("No equipment registered yet.")

        sel_id = st.selectbox("Edit equipment", options=["— select —"] + ids, index=0)
        if sel_id != "— select —":
            # charge l'entrée sélectionnée
            ent = next((e for e in reg if (e.get("id") or "") == sel_id), {})
            st.markdown("---")
            st.subheader(f"Edit: {sel_id}")

            with st.form(f"edit_{sel_id}", clear_on_submit=False):
                label = st.text_input("Label", value=ent.get("label", ""))
                brand_opts = st.text_input("Brand options (comma-separated)",
                                           value=", ".join(ent.get("brand_options") or ent.get("brands") or []))
                model_opts = st.text_input("Model options (comma-separated)",
                                           value=", ".join(ent.get("model_options") or ent.get("models") or []))
                model_path = st.text_input("Model .pkl path", value=(ent.get("model") or {}).get("path",""))
                features = st.text_area("Model features (one per line)",
                                        value="\n".join((ent.get("model") or {}).get("features") or []),
                                        height=120)
                prob_mode = st.toggle("Use probability mode", value=bool(ent.get("prob_mode", True)))
                thr = st.slider("Threshold", min_value=0.05, max_value=0.95, step=0.01,
                                value=float(ent.get("threshold", 0.5)))
                rules_path = st.text_input("Rules file (JSON)", value=ent.get("rules_path", ""))

                c1, c2, c3 = st.columns(3)
                detect = c1.form_submit_button("Detect features from .pkl")
                save   = c2.form_submit_button("Save changes", type="primary")
                delete = c3.form_submit_button("Delete equipment", type="secondary")

            if detect:
                feats = detect_model_features(model_path)
                if feats:
                    st.success(f"Detected {len(feats)} features.")
                    features = "\n".join(feats)
                    # affichage immédiat
                    st.text_area("Detected features", value=features, height=140, key=f"det_feats_{sel_id}")
                else:
                    st.warning("No features found in the model file.")

            if save:
                entry = {
                    "id": sel_id,
                    "label": label,
                    "brand_options": [s.strip() for s in brand_opts.split(",") if s.strip()],
                    "model_options": [s.strip() for s in model_opts.split(",") if s.strip()],
                    "model": {
                        "path": model_path,
                        "type": "sklearn_svm",
                        "features": [s.strip() for s in features.splitlines() if s.strip()],
                    },
                    "schema": ent.get("schema") or {},  # garde tel quel si tu n'en utilises pas
                    "prob_mode": bool(prob_mode),
                    "threshold": float(thr),
                    "rules_path": rules_path,
                }
                ok, msg = upsert_equipment_entry(entry)
                (st.success if ok else st.error)(msg)
                st.rerun()

            if delete:
                ok, msg = delete_equipment_entry(sel_id)
                (st.success if ok else st.error)(msg)
                st.rerun()

    # ----- Création -----
    with cR:
        st.subheader("Add new equipment")
        with st.form("create_equipment", clear_on_submit=True):
            new_id = st.text_input("ID (unique, no spaces, e.g. autoclave_vs)")
            new_label = st.text_input("Label (visible in UI)")
            new_brand = st.text_input("Brand options (comma-separated)", value="")
            new_model = st.text_input("Model options (comma-separated)", value="")
            new_model_path = st.text_input("Model .pkl path", value="")
            new_features_text = st.text_area("Model features (one per line)", value="", height=120)
            new_prob = st.toggle("Use probability mode", value=True)
            new_thr = st.slider("Threshold", 0.05, 0.95, 0.80, 0.01)
            new_rules = st.text_input("Rules file (JSON)", value="")

            c1, c2 = st.columns(2)
            btn_detect = c1.form_submit_button("Detect from .pkl")
            btn_create = c2.form_submit_button("Create", type="primary")

        if btn_detect and new_model_path.strip():
            feats = detect_model_features(new_model_path.strip())
            if feats:
                st.success(f"Detected {len(feats)} features.")
                st.code("\n".join(feats))
            else:
                st.warning("No features found in the model file.")

        if btn_create:
            if not new_id.strip() or not new_label.strip():
                st.error("ID and Label are required.")
            else:
                entry = {
                    "id": new_id.strip(),
                    "label": new_label.strip(),
                    "brand_options": [s.strip() for s in new_brand.split(",") if s.strip()],
                    "model_options": [s.strip() for s in new_model.split(",") if s.strip()],
                    "model": {
                        "path": new_model_path.strip(),
                        "type": "sklearn_svm",
                        "features": [s.strip() for s in new_features_text.splitlines() if s.strip()],
                    },
                    "schema": {},
                    "prob_mode": bool(new_prob),
                    "threshold": float(new_thr),
                    "rules_path": new_rules.strip(),
                }
                ok, msg = upsert_equipment_entry(entry)
                (st.success if ok else st.error)(msg)
                if ok:
                    st.rerun()


def page_users():
    username = st.session_state.get("auth_user", "")
    role = (get_user_role(username) or "").strip().lower()
    if role != "engineer":
        st.info("This page is restricted to the engineer.")
        st.stop()

    st.header("Users")

    tab1, tab2, tab3 = st.tabs(["Create / Update", "Reset password (admin)", "Delete"])

    with tab1:
        u = st.text_input("Username", key="user_u")
        c1, c2 = st.columns(2)
        with c1:
            p1 = st.text_input("Password (leave blank to keep current)", type="password", key="user_p1")
        with c2:
            p2 = st.text_input("Confirm password", type="password", key="user_p2")
        r = st.selectbox("Role", ["technician", "engineer"], index=0, key="user_r")

        if st.button("Save user", type="primary"):
            if not u.strip():
                st.error("Username is required.")
            elif p1 != p2:
                st.error("Passwords do not match.")
            else:
                ok, msg = create_or_update_user(u, p1, r)
                (st.success if ok else st.error)(msg or f"User {u} saved.")

    with tab2:
        others = [x["username"] for x in list_users()
                  if x["username"].strip().lower() != (username or "").strip().lower()]
        if not others:
            st.info("No other user.")
        else:
            u2 = st.selectbox("User", others, key="user_reset")
            n1 = st.text_input("New password", type="password", key="reset_n1")
            n2 = st.text_input("Confirm", type="password", key="reset_n2")
            if st.button("Reset password"):
                if n1 != n2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = set_password_admin(u2, n1)
                    (st.success if ok else st.error)(msg or "Password reset.")

    with tab3:
        deletables = [x["username"] for x in list_users()
                      if x["username"].strip().lower() != (username or "").strip().lower()]
        if not deletables:
            st.info("No other user to delete.")
        else:
            u3 = st.selectbox("User", deletables, key="user_del")
            if st.button("Delete user"):
                ok, msg = auth_delete_user(u3, requester=username)
                (st.success if ok else st.error)(msg or f"User {u3} deleted.")

def page_profile():
    st.header("My account")
    username = st.session_state.get("auth_user", "")
    st.caption(f"Signed in as **{username}**")

    old = st.text_input("Current password", type="password", key="prof_old")
    n1  = st.text_input("New password", type="password", key="prof_n1")
    n2  = st.text_input("Confirm new password", type="password", key="prof_n2")

    if st.button("Update password", type="primary"):
        if len(n1) < 8:
            st.error("New password must be at least 8 characters.")
        elif n1 != n2:
            st.error("Passwords do not match.")
        else:
            ok, msg = auth_change_password(username, old, n1)
            if ok:
                st.success("Password updated. Please log in again.")
                st.session_state["auth_user"] = None
                st.rerun()
            else:
                st.error(msg or "Could not update password.")

def page_about():
    st.markdown(
        """
# About MedPredict

**MedPredict is** an **AI-powered web application** that helps biomedical teams **anticipate equipment failures** and **prioritize interventions** from log data—so you get fewer unexpected outages, safer operating rooms, and less time spent in spreadsheets.

### What it does
- **AI predictions**: machine-learning models (with a robust heuristic fallback) estimate failure risk; an **adjustable threshold** and optional **probability mode** let you tune alerting.
- **Simple ingestion**: upload one **Excel (.xlsx)** file; optionally add a **PDF manual** to enrich context.
- **Normalization & validation**: automatic column harmonization, sanity checks, and format warnings before prediction.
- **Actionable rules**: a configurable rules engine converts risk + domain logic into a clear **“Action to take.”**
- **Traceability**: every run is saved with metadata, quick previews, and **CSV/XLSX exports** for reporting and audit.
- **Role-aware UX**: streamlined for **technicians**; advanced settings and **user management** for the **engineer**.

### Why it matters
- **Reduce downtime** by surfacing at-risk devices before they fail.  
- **Transparent decisions** via scores, thresholds, and editable rules you control.  
- **Save time** on data prep and routine reporting.  
- **Adaptable** to diverse fleets (brands, models, usage) without changing your workflows.

*MedPredict brings AI to your daily maintenance: the data speaks, the model alerts, and your rules turn insight into timely action.*
        """
    )

# ---------------- LOGIN ----------------
def login_view():
    logo_b64 = b64encode(open("assets/logo.png", "rb").read()).decode()
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:12px;">
          <img src="data:image/png;base64,{logo_b64}" style="height:4.5em; border-radius:4px;" />
          <h1 style="margin:0">{t('login.title')}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        username = st.text_input(t("login.username"))
    with c2:
        password = st.text_input(t("login.password"), type="password")
    err = st.empty()
    if st.button(t("login.button"), type="primary"):
        if verify_login(username, password):
            st.session_state["auth_user"] = username
            st.rerun()
        else:
            err.error(t("login.invalid"))

# ---- Gate
if st.session_state.get("auth_user") is None:
    login_view()
    st.stop()

# ---------------- SIDEBAR & ROUTER ----------------
current_page = render_sidebar_and_nav()
show_status_badge_for(current_page)


if current_page == "Upload & Predict":
    page_upload()
elif current_page == "History":
    page_history()
elif current_page == "Equipment Management":
    page_equipment_mgmt()
elif current_page == "Rules & Actions":
    page_rules()
elif current_page == "About":
    page_about()
elif current_page == "My account":
    page_profile()
elif current_page == "Users":
    page_users()
elif current_page == "Logout":
    st.session_state["auth_user"] = None
    st.session_state["last_alert_count"] = 0
    st.session_state["pred_df"] = None
    st.session_state["pred_nb"] = 0
    st.session_state["pred_meta"] = None
    st.rerun()
