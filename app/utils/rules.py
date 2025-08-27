# app/utils/rules.py
from __future__ import annotations

import io
import re
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd


# =====================================================================
# Stockage des règles JSON (utilisées par la page "Rules & Actions")
# =====================================================================

RULES_PATH = Path("history/rules.json")

DEFAULT_RULES: List[Dict[str, Any]] = [
    # Exemples; modifiables depuis la page "Rules & Actions"
    {
        "name": "Very high RPN",
        "match": {"column": "RPN", "gte": 50},
        "action": "Immediate shutdown & call vendor",
    },
    {
        "name": "High RPN",
        "match": {"column": "RPN", "gte": 24},
        "action": "Priority inspection within 24h",
    },
    {
        "name": "High score or long duration",
        "match": {"any": [
            {"column": "Failure_Score", "gte": 0.90},
            {"column": "Durée(h)", "gte": 2},
        ]},
        "action": "Run self-test and monitor 12h",
    },
    {
        "name": "Low MTBF",
        "match": {"column": "MTBF (h)", "lte": 200},
        "action": "Preventive maintenance: check wear parts",
    },
]


def _ensure_rules_file() -> None:
    RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not RULES_PATH.exists():
        RULES_PATH.write_text(
            json.dumps(DEFAULT_RULES, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def load_rules() -> List[Dict[str, Any]]:
    _ensure_rules_file()
    try:
        return json.loads(RULES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_RULES


def save_rules(rules: List[Dict[str, Any]]) -> None:
    _ensure_rules_file()
    RULES_PATH.write_text(
        json.dumps(rules, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ------------------------- Matching des règles JSON -------------------

def _num(v) -> float:
    try:
        return float(str(v).replace(",", "."))
    except Exception:
        return np.nan


def _match_simple(row: pd.Series, cond: Dict[str, Any]) -> bool:
    """Supporte eq / in / gte / lte sur une colonne."""
    col = cond.get("column")
    if not col:
        return False
    val = row.get(col)

    if "eq" in cond:
        return str(val) == str(cond["eq"])

    if "in" in cond:
        return str(val) in {str(x) for x in cond["in"]}

    if "gte" in cond:
        return _num(val) >= _num(cond["gte"])

    if "lte" in cond:
        return _num(val) <= _num(cond["lte"])

    return False


def _match_rule(row: pd.Series, rule: Dict[str, Any]) -> bool:
    m = rule.get("match", {})
    if not isinstance(m, dict):
        return False

    # any / all de sous-conditions
    if "any" in m and isinstance(m["any"], list):
        return any(_match_simple(row, c) for c in m["any"])
    if "all" in m and isinstance(m["all"], list):
        return all(_match_simple(row, c) for c in m["all"])

    # condition simple
    return _match_simple(row, m)


def _apply_json_rules_first_match(row: pd.Series, rules: List[Dict[str, Any]]) -> str | None:
    for r in rules or []:
        try:
            if _match_rule(row, r):
                act = r.get("action")
                if isinstance(act, str) and act.strip():
                    return act.strip()
        except Exception:
            continue
    return None


# =====================================================================
# Extraction PDF (best-effort). Si PDF scanné, renvoie [].  (utilisé
# uniquement en dernier recours pour ne PAS perturber les règles/heuristiques)
# =====================================================================

def _extract_pdf_text(pdf_bytes: bytes | None) -> str:
    if not pdf_bytes:
        return ""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception:
        return ""


def extract_pdf_actions(pdf_bytes: bytes | None) -> List[str]:
    """
    Renvoie une petite KB d’actions potentiellement utiles trouvées dans le manuel.
    Si le PDF est scanné / vide -> [] (on bascule alors sur les règles locales).
    """
    text = _extract_pdf_text(pdf_bytes)
    if not text.strip():
        return []

    lines = [re.sub(r"\s+", " ", ln).strip(" .;:-") for ln in text.splitlines() if ln.strip()]
    actions: List[str] = []

    # Quelques motifs génériques (restent en dernier recours)
    patterns = [
        r"(?i)\b(corrective\s*action|remedy|solution|action(?:s)?\s*to\s*take|procedure)\b[:\-\s]+(.{5,180})",
        r"(?i)^\s*Action\s*[:\-]\s*(.{5,180})",
        r"^[\u2022•\-]\s*(.{5,180})",
    ]
    for ln in lines:
        for pat in patterns:
            m = re.search(pat, ln)
            if m:
                grp = m.group(2) if (m.lastindex and m.lastindex >= 2) else m.group(1)
                act = re.sub(r"\s+", " ", grp).strip(" .;:-")
                if 8 <= len(act) <= 180 and act not in actions:
                    actions.append(act)

    return actions


# =====================================================================
# Heuristiques texte / mots-clés
# =====================================================================

_KEYWORD_ACTIONS = [
    (r"\bdoor|porte|gasket|joint\b", "Inspect/replace door gasket and interlock"),
    (r"\bpressure|pression|steam|vapeur\b", "Check pressure sensor and relief valve"),
    (r"\bpump|pompe|vacuum\b", "Inspect pump/vacuum system; check tubing"),
    (r"\btemperature|temp[ée]rature|sensor\b", "Verify temperature sensor; run calibration"),
    (r"\bfilter|filtre\b", "Replace/clean filter and run self-test"),
    (r"\bleak|fuite\b", "Leak test; tighten connections"),
    (r"\bvalve|[ée]lectrovanne\b", "Test inlet/solenoid valves"),
    (r"\bpower|alimentation\b", "Check power supply and fuses"),
]


def _row_text(row: pd.Series) -> str:
    """Concatène plusieurs colonnes textuelles du template 19 colonnes."""
    buf = []
    for col in [
        "message", "Module_concerné", "Label", "window_label",
        "module", "equipment_name", "Module_ID", "ID_événement"
    ]:
        if col in row and pd.notna(row[col]):
            buf.append(str(row[col]))
    return " ".join(buf).lower()


def _tok(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def _pick_pdf_action_for_row(row_text: str, kb: List[str]) -> str | None:
    """Choisit l’action PDF la plus proche d’un texte de ligne (overlap de tokens)."""
    if not kb:
        return None
    rt = _tok(row_text)
    best, best_score = None, 0
    for act in kb:
        sc = len(rt & _tok(act))
        if sc > best_score:
            best, best_score = act, sc
    return best or kb[0]


# =====================================================================
# Règles hybrides (JSON -> Heuristiques -> PDF -> Fallback)
# =====================================================================

def apply_rules(df: "pd.DataFrame", pdf_bytes: bytes | None = None) -> "pd.Series":
    """
    Priorité des sources d’action :
      1) Règles JSON (page Rules & Actions) — avec RPN calculé si FMEA dispo
      2) Heuristiques (mots-clés, puis numériques)
      3) PDF (si rien trouvé)
      4) Fallback : 'Check within 24h'

    Si Failure_Predicted != 1 => '—'
    """
    rules = load_rules()
    kb = extract_pdf_actions(pdf_bytes)

    out = []
    for _, row in df.iterrows():
        # Non alerté -> pas d’action
        pred = int(pd.to_numeric(row.get("Failure_Predicted", 0), errors="coerce") or 0)
        if pred != 1:
            out.append("—")
            continue

        action = None

        # 1) Règles JSON (on calcule un RPN si colonnes FMEA présentes)
        c = _num(row.get("C (Criticité)"))
        g = _num(row.get("G (Gravité)"))
        o = _num(row.get("O (Occurrence)"))
        d = _num(row.get("D (Détectabilité)"))
        rpn = np.nan
        if not np.isnan(c) and not np.isnan(g) and not np.isnan(o) and not np.isnan(d):
            rpn = c * g * o * d
        row2 = row.copy()
        row2["RPN"] = rpn

        action = _apply_json_rules_first_match(row2, rules)

        # 2) Heuristiques texte (mots-clés)
        if action is None:
            text = _row_text(row)
            for pat, act in _KEYWORD_ACTIONS:
                if re.search(pat, text):
                    action = act
                    break

        # 2b) Heuristiques numériques
        if action is None:
            mtbf = _num(row.get("MTBF (h)"))
            dur = _num(row.get("Durée(h)"))
            score = _num(row.get("Failure_Score"))

            if not np.isnan(rpn) and rpn >= 50:
                action = "Immediate shutdown & call vendor"
            elif not np.isnan(rpn) and rpn >= 24:
                action = "Priority inspection within 24h"
            elif (not np.isnan(score) and score >= 0.90) or (not np.isnan(dur) and dur >= 2):
                action = "Run self-test and monitor 12h"
            elif not np.isnan(mtbf) and mtbf <= 200:
                action = "Preventive maintenance: check wear parts"

        # 3) PDF en dernier recours
        if action is None and kb:
            action = _pick_pdf_action_for_row(_row_text(row), kb)

        # 4) Fallback final
        if action is None:
            action = "Check within 24h"

        out.append(action)

    return pd.Series(out, index=df.index, dtype=object)
