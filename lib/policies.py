import csv
import hashlib
import json

from .config import config


def load_policies():
    with open(config.csv_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    policies = []
    for i, r in enumerate(rows, start=1):
        r = {(k or "").strip(): (v or "").strip() for k, v in r.items()}
        policies.append({
            "id": i,
            "title": r.get("title", ""),
            "department": r.get("department", ""),
            "text": r.get("policy_text", ""),
            "category": r.get("category", ""),
        })
    return policies


def policies_hash(policies):
    blob = json.dumps(policies, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def public_policy(p):
    if not p:
        return None
    return {k: p[k] for k in ("id", "title", "department", "category", "text")}
