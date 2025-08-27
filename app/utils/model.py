# app/utils/model.py
# app/utils/model.py (tout en haut des imports)
import numpy as np
# mémoires simples pour remonter l'info au debug UI
_LAST_PROBA_ERROR: dict[str, str] = {}
_LAST_FEATURES_MISSING: dict[str, list[str]] = {}
_LAST_MISSING = {}
_LAST_PROBA_ERR = {}
_LAST_SELECTED = {}

def get_last_missing_features(equip_key):   return _LAST_MISSING.get(equip_key)
def get_last_proba_error(equip_key):        return _LAST_PROBA_ERR.get(equip_key)
def get_last_selected_features(equip_key):  return _LAST_SELECTED.get(equip_key)
def _set_last_proba_error(equip_key, msg):  _LAST_PROBA_ERR[equip_key] = msg


import json
from pathlib import Path
import pandas as pd

# joblib pour charger les modèles scikit-learn (.pkl)
try:
    import joblib
except Exception:
    joblib = None  # si joblib non installé, on retombe proprement

# <- le registry des équipements
REG_PATH = Path("history/equipment_registry.json")


def load_equipment_registry() -> list[dict]:
    if REG_PATH.exists():
        return json.loads(REG_PATH.read_text(encoding="utf-8"))
    return []

# --- Écriture / CRUD du registry + introspection des features ---

def save_equipment_registry(registry: list[dict]) -> None:
    Path(REG_PATH).parent.mkdir(parents=True, exist_ok=True)
    REG_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

def upsert_equipment_entry(entry: dict) -> tuple[bool, str]:
    """Ajoute ou remplace une entrée par son 'id'."""
    if not entry or not isinstance(entry, dict):
        return False, "Invalid entry."
    eid = (entry.get("id") or "").strip()
    if not eid:
        return False, "Missing 'id'."
    reg = load_equipment_registry()
    # remplace si même id, sinon append
    out, replaced = [], False
    for e in reg:
        if (e.get("id") or "").strip() == eid:
            out.append(entry)
            replaced = True
        else:
            out.append(e)
    if not replaced:
        out.append(entry)
    save_equipment_registry(out)
    return True, ("Updated" if replaced else "Created")

def delete_equipment_entry(equip_id: str) -> tuple[bool, str]:
    eid = (equip_id or "").strip()
    if not eid:
        return False, "Missing id."
    reg = load_equipment_registry()
    new = [e for e in reg if (e.get("id") or "").strip() != eid]
    if len(new) == len(reg):
        return False, f"Not found: {eid}"
    save_equipment_registry(new)
    return True, "Deleted"

def detect_model_features(pkl_path: str) -> list[str]:
    """Retourne feature_names_in_ si disponibles dans le .pkl."""
    try:
        p = Path(pkl_path)
        if not p.exists() or joblib is None:
            return []
        mdl = joblib.load(p)
        if hasattr(mdl, "feature_names_in_"):
            return [str(x) for x in mdl.feature_names_in_]
        if hasattr(mdl, "named_steps"):
            last = mdl
            try:
                for _, step in mdl.named_steps.items():
                    last = step
            except Exception:
                pass
            if hasattr(last, "feature_names_in_"):
                return [str(x) for x in last.feature_names_in_]
    except Exception:
        return []
    return []


def get_equipment_entry(equipment_type: str | None):
    """Retourne l'entrée du registry par id ou label (insensible casse/espaces)."""
    if not equipment_type:
        return None
    key = (equipment_type or "").strip().lower()
    for e in load_equipment_registry():
        if (e.get("id", "").strip().lower() == key) or (e.get("label", "").strip().lower() == key):
            return e
    return None

def list_equipment_labels() -> list[str]:
    """Liste des labels depuis le registry (pour le select Equipment)."""
    try:
        return [e.get("label") for e in load_equipment_registry() if e.get("label")]
    except Exception:
        return []

def load_model(equipment_type: str | None = None):
    ent = get_equipment_entry(equipment_type)
    if not ent:
        return None
    p = ent.get("model", {}).get("path")
    if not p or not Path(p).exists() or joblib is None:
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None


# --- sélection stricte des features attendues par l'équipement ---
def _select_features(df, ent):
    from pathlib import Path
    import pandas as pd

    model_cfg = (ent.get("model") or {})
    expected = list(model_cfg.get("features") or [])
    equip_key = (ent.get("id") or ent.get("label") or "").strip() or "unknown"

    # 1) Si le registry n'a pas de features, on tente de les lire depuis le .pkl
    if not expected:
        path = model_cfg.get("path")
        if path and Path(path).exists() and joblib is not None:
            try:
                mdl = joblib.load(path)
                feats = None
                if hasattr(mdl, "feature_names_in_"):
                    feats = list(map(str, mdl.feature_names_in_))
                elif hasattr(mdl, "named_steps"):  # pipeline / calibré
                    last = mdl
                    try:
                        for _, step in mdl.named_steps.items():
                            last = step
                    except Exception:
                        pass
                    if hasattr(last, "feature_names_in_"):
                        feats = list(map(str, last.feature_names_in_))
                if feats:
                    expected = feats
            except Exception as e:
                _LAST_PROBA_ERR[equip_key] = f"introspect_features_error: {e}"

    # 2) Si on n'a toujours rien, on signale un vrai "missing"
    if not expected:
        miss = ["<no feature list>"]
        _LAST_MISSING[equip_key]  = miss
        _LAST_SELECTED[equip_key] = None
        return None, {"features_missing": miss, "selected_features": None}

    # 3) Vérifie que toutes les colonnes existent dans df
    missing = [c for c in expected if c not in df.columns]
    if missing:
        _LAST_MISSING[equip_key]  = missing
        _LAST_SELECTED[equip_key] = expected
        return None, {"features_missing": missing, "selected_features": expected}

    # 4) OK
    X = df[expected].copy()
    _LAST_MISSING[equip_key]  = None
    _LAST_SELECTED[equip_key] = expected
    return X, {"features_missing": None, "selected_features": expected}


def predict_with_model(df: pd.DataFrame, prob_mode: bool, equipment_type: str | None = None):
    """
    Retourne (preds, scores) :
      - preds: 0/1
      - scores: proba si dispo et prob_mode=True, sinon None
    """
    ent = get_equipment_entry(equipment_type)
    if not ent:
        return None, None

    # 1) Sélection explicite des features du modèle
    X, missing = _select_features(df, ent)
    if missing:
        _LAST_FEATURES_MISSING[equipment_type or ""] = missing
        return None, None  # => fallback heuristique en amont

    # 2) Coercition des types pour coller à l'entraînement
    X = _coerce_features_for_model(X)

    mdl = load_model(equipment_type)
    if mdl is None:
        return None, None

    # 3) Probas si possible
    if prob_mode and hasattr(mdl, "predict_proba"):
        try:
            proba = mdl.predict_proba(X)[:, 1]
            preds = (proba >= float(ent.get("threshold", 0.5))).astype(int)
            # reset les erreurs précédentes pour ce type d'équipement
            _LAST_PROBA_ERROR.pop(equipment_type or "", None)
            return preds, proba
        except Exception as ex:
            _LAST_PROBA_ERROR[equipment_type or ""] = str(ex)

    # 4) Sinon, prédiction directe
    try:
        preds = mdl.predict(X)
        return preds, None
    except Exception as ex:
        _LAST_PROBA_ERROR[equipment_type or ""] = str(ex)
        return None, None


# ⬇️ colle ceci DANS app/utils/model.py (au-dessus de predict_with_model par ex.)
def _coerce_features_for_model(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # Map 'severity' texte -> numérique si besoin
    if "severity" in X.columns and not np.issubdtype(X["severity"].dtype, np.number):
        sev_map = {
            "low": 0, "bas": 0, "faible": 0,
            "medium": 1, "moyen": 1, "modere": 1, "modérée": 1, "moderee": 1,
            "high": 2, "eleve": 2, "élevé": 2, "haute": 2,
            "critical": 3, "critique": 3,
        }
        X["severity"] = (
            X["severity"].astype(str).str.strip().str.lower().map(sev_map)
        )

    # Tout en numérique + NA -> 0
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.fillna(0)
def get_last_proba_error(equipment_type: str | None = None):
    key = equipment_type or ""
    # privilégie _LAST_PROBA_ERR (utilisé par compute_predictions)
    return _LAST_PROBA_ERR.get(key, _LAST_PROBA_ERROR.get(key))

def get_last_missing_features(equipment_type: str | None = None):
    key = equipment_type or ""
    # privilégie _LAST_MISSING (écrit par _select_features)
    return _LAST_MISSING.get(key, _LAST_FEATURES_MISSING.get(key))

