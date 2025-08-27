# app/utils/manage_users.py
from pathlib import Path
import json, sys, bcrypt

USERS = Path(__file__).with_name("users.json")

def load():
    if USERS.exists():
        return json.loads(USERS.read_text(encoding="utf-8"))
    return {"users":[]}

def save(obj):
    USERS.parent.mkdir(parents=True, exist_ok=True)
    USERS.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def upsert(username, password, role):
    db = load()
    h = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    for rec in db["users"]:
        if rec["u"].strip().lower() == username.strip().lower():
            rec["hash"] = h
            rec["role"] = role
            break
    else:
        db["users"].append({"u": username, "hash": h, "role": role})
    save(db)
    print(f"OK: {username} ({role})")

def list_users():
    db = load()
    for r in db["users"]:
        print(f"{r['u']} -> {r['role']}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: manage_users.py list | add <user> <password> <role>")
        sys.exit(1)
    if sys.argv[1] == "list":
        list_users()
    elif sys.argv[1] == "add" and len(sys.argv) == 5:
        _, _, u, p, role = sys.argv
        upsert(u, p, role)
    else:
        print("Usage: manage_users.py list | add <user> <password> <role>")
