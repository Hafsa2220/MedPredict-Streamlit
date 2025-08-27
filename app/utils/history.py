# app/utils/history.py
from pathlib import Path
from datetime import datetime
import pandas as pd
import json
import shutil
import streamlit as st

HIST_DIR = Path("history")
RUNS_DIR = HIST_DIR / "runs"
META_PATH = HIST_DIR / "meta.csv"

META_COLS = [
    "run_id","datetime","user","role",
    "equipment_name","brand","model",
    "n_rows","n_predicted",
    "prob_mode","threshold","score_available",
    "run_path"
]

def ensure_history():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if not META_PATH.exists():
        pd.DataFrame(columns=META_COLS).to_csv(META_PATH, index=False)

def _new_run_id(username: str) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}_{username}"

def save_run(df_with_preds: pd.DataFrame, info: dict, pdf_bytes: bytes | None = None) -> str:
    """
    df_with_preds : DataFrame qui contient déjà la colonne 'Failure_Predicted' (+ 'score' si dispo)
    info : dict = {
      user, role, equipment_name, brand, model, prob_mode (bool), threshold (float), score_available (bool)
    }
    """
    ensure_history()
    run_id = _new_run_id(info.get("user","user"))
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Sauvegardes de fichiers du run
    df_with_preds.to_excel(run_dir / "output.xlsx", index=False)
    df_with_preds.head(100).to_csv(run_dir / "preview.csv", index=False)
    if pdf_bytes:
        (run_dir / "manual.pdf").write_bytes(pdf_bytes)

    # 2) Petit report JSON
    report = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_rows": int(len(df_with_preds)),
        "n_predicted": int(df_with_preds["Failure_Predicted"].sum()),
        "prob_mode": bool(info.get("prob_mode", False)),
        "threshold": float(info.get("threshold", 0.5)),
        "meta": {
            "equipment_name": info.get("equipment_name",""),
            "brand": info.get("brand",""),
            "model": info.get("model",""),
            "score_available": bool(info.get("score_available", False)),
            "user": info.get("user",""),
            "role": info.get("role",""),
        }
    }
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) Append dans meta.csv
    row = {
        "run_id": run_id,
        "datetime": report["created_at"],
        "user": info.get("user",""),
        "role": info.get("role",""),
        "equipment_name": info.get("equipment_name",""),
        "brand": info.get("brand",""),
        "model": info.get("model",""),
        "n_rows": report["n_rows"],
        "n_predicted": report["n_predicted"],
        "prob_mode": report["prob_mode"],
        "threshold": report["threshold"],
        "score_available": info.get("score_available", False),
        "run_path": str(run_dir.as_posix()),
    }
    meta = pd.read_csv(META_PATH) if META_PATH.exists() else pd.DataFrame(columns=META_COLS)
    meta = pd.concat([meta, pd.DataFrame([row])], ignore_index=True)
    meta.to_csv(META_PATH, index=False)

    return run_id

def read_meta() -> pd.DataFrame:
    ensure_history()
    if META_PATH.exists():
        return pd.read_csv(META_PATH)
    return pd.DataFrame(columns=META_COLS)

def delete_run(run_id: str):
    """Supprime le dossier du run et la ligne correspondante dans meta.csv"""
    ensure_history()
    run_dir = RUNS_DIR / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    meta = read_meta()
    meta = meta[meta["run_id"] != run_id]
    meta.to_csv(META_PATH, index=False)
