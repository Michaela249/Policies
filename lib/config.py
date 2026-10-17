"""Configuration plus a tiny .env loader (no third-party packages needed)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv():
    f = ROOT / ".env"
    if not f.exists():
        return
    for raw in f.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].strip()
        # Real environment variables (e.g. set on a hosting platform) win over .env
        os.environ.setdefault(key, value)


_load_dotenv()



def _list(name, default):
    items = [x.strip() for x in os.environ.get(name, "").split(",") if x.strip()]
    return items or default


class Config:
    port = int(os.environ.get("PORT") or 3000)
    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    api_base = os.environ.get("GEMINI_API_BASE") or "https://generativelanguage.googleapis.com/v1beta"
    # Tried in order. If one is busy or out of quota (429/503/...), the next one is used.
    # Embedding 2 = "gemini-embedding-2", Embedding 1 = "gemini-embedding-001".
    # Each has its own index, because vectors from different embedding models cannot be compared.
    embedding_models = _list("EMBEDDING_MODELS", ["gemini-embedding-2", "gemini-embedding-001"])
    embedding_dims = int(os.environ.get("EMBEDDING_DIMS") or 768)
    # Models that write the answers in the two LLM approaches, tried in order.
    llm_models = _list("GEMINI_LLM_MODELS", ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"])
    top_k = int(os.environ.get("TOP_K") or 3)
    csv_path = ROOT / "data" / "company_policies.csv"
    eval_path = ROOT / "data" / "eval.json"
    benchmark_path = ROOT / "data" / "benchmark-results.json"
    public_dir = ROOT / "public"
    max_question_length = 300
    # Per-IP limit on /api/ask so a public deployment can't drain the API key
    rate_window_s = 5 * 60
    rate_max = int(os.environ.get("RATE_LIMIT_MAX") or 40)
    host = os.environ.get("HOST") or ("0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")

    @staticmethod
    def index_path_for(model):
        return ROOT / "data" / f"index-{model}.json"


config = Config()
