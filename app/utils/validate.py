# app/utils/validate.py
from typing import Tuple, List
import pandas as pd

from app.utils.model import get_equipment_entry

def validate_df(df, equipment_type: str | None = None):
    errors = []
    extras = {}

    # Use registry-driven required features if an equipment is given
    if equipment_type:
        ent = get_equipment_entry(equipment_type) or {}
        expected = (ent.get("model") or {}).get("features") or []
        miss = [c for c in expected if c not in df.columns]
        if miss:
            errors.append(f"Missing required feature columns for '{equipment_type}': {miss}")
        return (len(errors) == 0), errors, extras

    # (legacy generic checks if no equipment_type provided)
    # ... keep your old rules here if needed ...
