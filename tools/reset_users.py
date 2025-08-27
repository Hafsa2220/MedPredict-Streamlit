# tools/reset_users.py
import json, bcrypt, pathlib

p = pathlib.Path("app/data/users.json")
p.parent.mkdir(parents=True, exist_ok=True)

def make_user(username, password, role):
    h = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    return {"username": username, "hash": h, "role": role}

data = {
    "users": [
        make_user("biomed01", "biomed", "engineer"),
        make_user("tech01",   "tech",   "technician"),
    ]
}

p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
print("Users file written to:", p.resolve())
