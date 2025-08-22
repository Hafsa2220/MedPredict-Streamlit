# app/utils/auth.py
import hashlib
from dataclasses import dataclass

# ⚠️ change-moi : mets une chaîne longue et unique (32–48 caractères)
SALT = "APPLICATION_MEDPREDICT_2025_RANDOM_STRING"

@dataclass
class User:
    username: str
    role: str  # "technician" | "engineer"
    password_hash: str

def _hash(pw: str) -> str:
    return hashlib.sha256((pw + SALT).encode("utf-8")).hexdigest()

# Comptes démo (tu peux changer les mots de passe/identifiants)
USERS = {
    "tech01":   User("tech01",   "technician", _hash("medtech")),
    "biomed01": User("biomed01", "engineer",   _hash("biomed")),
}

def verify_login(username: str, password: str) -> bool:
    u = USERS.get(username)
    return bool(u and u.password_hash == _hash(password))

def get_user_role(username: str) -> str:
    u = USERS.get(username)
    return u.role if u else "unknown"
