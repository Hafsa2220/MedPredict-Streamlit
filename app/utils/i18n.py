# app/utils/i18n.py
from pathlib import Path
import json

# simple loader; default to English
def load_lang(lang: str = "en") -> dict:
    p = Path("assets/i18n") / f"{lang}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

# global dict (English only for now)
_STRINGS = load_lang("en")

def t(key: str, **kw) -> str:
    """
    Translate key -> string; format with optional kwargs.
    If key missing, key itself is returned (safe fallback).
    """
    s = _STRINGS.get(key, key)
    try:
        return s.format(**kw) if kw else s
    except Exception:
        return s
