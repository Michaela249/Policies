"""Company Policy Assistant. Start it with:  python app.py   (Python 3.9+, nothing to install)."""
import json
import mimetypes
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

from lib import benchmark, vector_index
from lib.approaches import run_all
from lib.config import config
from lib.policies import load_policies, public_policy
from lib.rules import RulesIndex

if sys.version_info < (3, 9):
    sys.exit("This program needs Python 3.9 or newer.")

POLICIES = load_policies()
RULES = RulesIndex(POLICIES)

# ---- per-IP rate limiter ----
_hits = {}
_hits_lock = threading.Lock()


def limited(ip):
    now = time.time()
    with _hits_lock:
        recent = [t for t in _hits.get(ip, []) if now - t < config.rate_window_s]
        recent.append(now)
        _hits[ip] = recent
        return len(recent) > config.rate_max


def start_index_build():
    if not config.api_key:
        print("GEMINI_API_KEY is not set. Rules-based search works; the two LLM approaches need the key in .env.")
        return

    def work():
        try:
            vector_index.build_index(POLICIES)
        except Exception as err:  # noqa: BLE001
            print("Could not build the vector index:", err)

    threading.Thread(target=work, daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    server_version = "PolicyAssistant/1.0"

    def log_message(self, fmt, *args):  # keep the console readable
        pass

    # ---- helpers ----
    def send_json(self, status, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def client_ip(self):
        fwd = self.headers.get("X-Forwarded-For")
        return (fwd.split(",")[0].strip() if fwd else self.client_address[0])

    def read_json(self, limit=10_000):
        n = int(self.headers.get("Content-Length") or 0)
        if n > limit:
            raise ValueError("Request too large")
        return json.loads(self.rfile.read(n).decode("utf-8") or "{}")

    def serve_static(self, path):
        rel = "index.html" if path in ("", "/") else unquote(path).lstrip("/")
        f = (config.public_dir / rel).resolve()
        if config.public_dir.resolve() not in f.parents or not f.is_file():
            return self.send_json(404, {"error": "Not found"})
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        data = f.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ---- routes ----
    def do_GET(self):
        try:
            path = urlparse(self.path).path
            if path == "/api/status":
                return self.send_json(200, {
                    "has_api_key": bool(config.api_key),
                    "llm_models": config.llm_models,
                    "embedding_models": config.embedding_models,
                    "embedding_dims": config.embedding_dims,
                    "top_k": config.top_k,
                    "policy_count": len(POLICIES),
                    "index": vector_index.index_status(),
                })
            if path == "/api/policies":
                return self.send_json(200, [public_policy(p) for p in POLICIES])
            if path == "/api/benchmark":
                return self.send_json(200, benchmark.state())
            return self.serve_static(path)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as err:  # noqa: BLE001
            print("Error:", err)
            self.send_json(500, {"error": "Something went wrong on the server."})

    def do_POST(self):
        try:
            path = urlparse(self.path).path
            if path == "/api/ask":
                if limited(self.client_ip()):
                    return self.send_json(429, {"error": "Too many questions from this address. Please wait a few minutes."})
                try:
                    body = self.read_json()
                except ValueError:
                    return self.send_json(400, {"error": 'Send JSON like {"question": "..."}.'})
                question = str(body.get("question") or "").strip()
                if not question:
                    return self.send_json(400, {"error": "Type a question first."})
                if len(question) > config.max_question_length:
                    return self.send_json(400, {"error": f"Keep the question under {config.max_question_length} characters."})
                return self.send_json(200, {"question": question, "results": run_all(question, POLICIES, RULES)})
            if path == "/api/benchmark":
                if limited(self.client_ip()):
                    return self.send_json(429, {"error": "Too many requests. Please wait a few minutes."})
                ok, reason = benchmark.start(POLICIES, RULES)
                return self.send_json(202 if ok else 409, {"started": True} if ok else {"error": reason})
            return self.send_json(405, {"error": "Method not allowed"})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as err:  # noqa: BLE001
            print("Error:", err)
            self.send_json(500, {"error": "Something went wrong on the server."})


def main():
    start_index_build()
    server = ThreadingHTTPServer((config.host, config.port), Handler)
    print(f"Policy assistant running on http://localhost:{config.port}  (press Ctrl+C to stop)")
    print(f"  {len(POLICIES)} policies | embeddings: {' > '.join(config.embedding_models)} ({config.embedding_dims} dims)"
          f" | answers: {' > '.join(config.llm_models)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
