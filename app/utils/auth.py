# app/utils/auth.py
from pathlib import Path
import json, bcrypt

USERS_PATH = Path(__file__).with_name("users.json")

def _load():
    if USERS_PATH.exists():
        try:
            return json.loads(USERS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"users": []}

def _save(db):
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    USERS_PATH.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")

def _norm(u: str) -> str:
    return (u or "").strip().lower()

def _find(db, username):
    u = _norm(username)
    for i, rec in enumerate(db["users"]):
        if _norm(rec.get("u")) == u:
            return i, rec
    return -1, None

# ---------- login / role ----------
def verify_login(username: str, password: str) -> bool:
    db = _load()
    _, rec = _find(db, username)
    if not rec:
        return False
    h = rec.get("hash", "")
    if not h.startswith("$2"):
        return False
    try:
        return bcrypt.checkpw(password.encode(), h.encode())
    except Exception:
        return False

def get_user_role(username: str):
    db = _load()
    _, rec = _find(db, username)
    return rec.get("role") if rec else None

# ---------- helpers exposés à l'UI ----------
def list_users():
    """Retourne [{username, role}, ...]"""
    db = _load()
    return [{"username": rec.get("u",""), "role": rec.get("role","")} for rec in db["users"]]

def create_or_update_user(username: str, password: str, role: str):
    """Crée ou met à jour (role + mot de passe si fourni)."""
    if role not in ("technician", "engineer"):
        return False, "Invalid role."

    if not username or len(username.strip()) == 0:
        return False, "Username is required."

    db = _load()
    idx, rec = _find(db, username)
    if idx >= 0:
        rec["role"] = role
        if password:
            rec["hash"] = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        db["users"][idx] = rec
    else:
        if not password or len(password) < 4:
            return False, "Password is required (≥ 4 chars)."
        rec = {
            "u": _norm(username),
            "role": role,
            "hash": bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode(),
        }
        db["users"].append(rec)
    _save(db)
    return True, None

def delete_user(username: str, *, requester: str | None = None):
    """Supprime un compte (pas soi-même, ni le dernier ingénieur)."""
    db = _load()
    idx, rec = _find(db, username)
    if idx < 0:
        return False, "User not found."

    if requester and _norm(username) == _norm(requester):
        return False, "You cannot delete your own account."

    if rec.get("role") == "engineer":
        nb_eng = sum(1 for r in db["users"] if r.get("role") == "engineer")
        if nb_eng <= 1:
            return False, "Cannot delete the last engineer account."

    db["users"].pop(idx)
    _save(db)
    return True, None

def change_password(username: str, old_password: str, new_password: str):
    """Un utilisateur change son propre mot de passe (verif ancien)."""
    db = _load()
    idx, rec = _find(db, username)
    if idx < 0:
        return False, "User not found."
    h = rec.get("hash", "")
    if not h.startswith("$2"):
        return False, "Invalid hash format."
    if not bcrypt.checkpw(old_password.encode(), h.encode()):
        return False, "Old password is incorrect."
    rec["hash"] = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    db["users"][idx] = rec
    _save(db)
    return True, None

def set_password_admin(username: str, new_password: str):
    """L’ingénieur change le mot de passe de quelqu’un (sans ancien)."""
    if not new_password:
        return False, "New password required."
    db = _load()
    idx, rec = _find(db, username)
    if idx < 0:
        return False, "User not found."
    rec["hash"] = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    db["users"][idx] = rec
    _save(db)
    return True, None
