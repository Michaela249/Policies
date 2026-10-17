"""Direct REST calls to the Gemini API (standard library only), with automatic switching
between models when one is busy or out of quota."""
import json
import math
import ssl
import time
import urllib.error
import urllib.request

from .config import config


class GeminiError(Exception):
    def __init__(self, message, status):
        super().__init__(message)
        self.status = status


# ---------- model fallback helpers ----------
# 0 = network problem, 403/404 = no access / unknown model, 429 = rate or quota limit, 5xx = busy.
# 502 is our own code for "the model returned nothing usable".
FALLBACK_STATUSES = {0, 403, 404, 429, 500, 502, 503, 504}


def can_fallback(err):
    return isinstance(err, GeminiError) and err.status in FALLBACK_STATUSES


_cooldown = {}  # model -> time until which we skip it


def mark_down(model, err):
    if err.status == 502:
        return  # one bad reply does not mean the model is down
    _cooldown[model] = time.time() + (600 if err.status in (403, 404) else 60)


def order_models(models):
    """Models that are not cooling down come first; if all are cooling down, try them all again."""
    up = [m for m in models if _cooldown.get(m, 0) <= time.time()]
    return up or list(models)


def describe_failure(model, err):
    msg = str(err)
    if msg.startswith("Gemini API ") and ": " in msg:
        msg = msg.split(": ", 1)[1]
    return f"{model} ({err.status or 'network'}): {msg[:140]}"


# ---------- HTTP ----------
def _post(path, body, timeout=60, max_retries=1):
    if not config.api_key:
        raise GeminiError("GEMINI_API_KEY is not set. Add it to the .env file and restart the program.", 401)
    data = json.dumps(body).encode("utf-8")
    retries = 0
    attempt = 0
    while True:
        req = urllib.request.Request(
            f"{config.api_base}/{path}", data=data, method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": config.api_key},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as res:
                out = json.loads(res.read().decode("utf-8"))
                out["_retries"] = retries
                return out
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = json.loads(e.read().decode("utf-8")).get("error", {}).get("message", "")
            except Exception:
                pass
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                retries += 1
                time.sleep(0.5 * 2 ** attempt)
                attempt += 1
                continue
            raise GeminiError(f"Gemini API {e.code}: {detail or e.reason}", e.code)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < max_retries:
                retries += 1
                time.sleep(0.5 * 2 ** attempt)
                attempt += 1
                continue
            reason = getattr(e, "reason", e)
            hint = ""
            if isinstance(reason, ssl.SSLCertVerificationError):
                hint = " (On a Mac, open the Python folder in Applications and double-click 'Install Certificates.command', then restart.)"
            raise GeminiError(f"Network error talking to Gemini: {reason}{hint}", 0)


# ---------- embeddings ----------
def _is_embedding2(model):
    return model.startswith("gemini-embedding-2")


# Gemini Embedding 2 has no task_type parameter: the task goes into the text itself.
# Gemini Embedding 1 (gemini-embedding-001) uses the taskType field instead.
def format_document(policy, model):
    if _is_embedding2(model):
        return f"title: {policy['title']} | text: {policy['text']}"
    return f"{policy['title']}. {policy['text']}"


def format_query(question, model):
    return f"task: question answering | query: {question}" if _is_embedding2(model) else question


def _normalize(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def embed(text, kind, model, max_retries=1):
    """Embed ONE text with ONE model (Embedding 2 merges multiple parts into a single vector,
    so we never batch parts). kind is "document" or "query"."""
    body = {
        "model": f"models/{model}",
        "content": {"parts": [{"text": text}]},
        "output_dimensionality": config.embedding_dims,
    }
    if not _is_embedding2(model):
        body["taskType"] = "RETRIEVAL_DOCUMENT" if kind == "document" else "RETRIEVAL_QUERY"
    data = _post(f"models/{model}:embedContent", body, max_retries=max_retries)
    values = (data.get("embedding") or {}).get("values")
    if not values:
        raise GeminiError("Gemini returned no embedding.", 502)
    reported = (data.get("usageMetadata") or {}).get("promptTokenCount")
    return {
        # Embedding 2 re-normalises truncated vectors itself; Embedding 1 does not.
        "vector": values if _is_embedding2(model) else _normalize(values),
        "tokens": reported if reported is not None else math.ceil(len(text) / 4),
        "tokens_estimated": reported is None,
        "retries": data.get("_retries", 0),
    }


# ---------- answer generation ----------
ANSWER_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "answerable": {"type": "BOOLEAN", "description": "true only if one of the provided policies answers the question"},
        "policy_title": {"type": "STRING", "description": "Exact title of the single most relevant policy, or an empty string if none applies"},
        "evidence": {"type": "STRING", "description": "One sentence copied word-for-word from that policy's text, or an empty string"},
        "answer": {"type": "STRING", "description": "Short answer to the question based only on that policy"},
    },
    "required": ["answerable", "policy_title", "evidence", "answer"],
}


def _generate_with(model, system_instruction, user_prompt):
    data = _post(f"models/{model}:generateContent", {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        # No temperature: Gemini 3.x models ignore it and recommend leaving it out.
        "generationConfig": {"responseMimeType": "application/json", "responseSchema": ANSWER_SCHEMA},
    })
    cand = (data.get("candidates") or [{}])[0]
    parts = (cand.get("content") or {}).get("parts") or []
    raw = "".join(p["text"] for p in parts if isinstance(p.get("text"), str) and not p.get("thought"))
    if not raw:
        why = (data.get("promptFeedback") or {}).get("blockReason") or cand.get("finishReason") or "empty response"
        raise GeminiError(f"Gemini returned no answer ({why}).", 502)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").removeprefix("json").strip()
    try:
        parsed = json.loads(cleaned)
    except ValueError:
        raise GeminiError("Gemini returned malformed JSON.", 502)
    u = data.get("usageMetadata") or {}
    inp, out = u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)
    return {
        "parsed": parsed,
        "usage": {
            "input": inp,
            "output": out,
            "thinking": u.get("thoughtsTokenCount", 0),
            "total": u.get("totalTokenCount", inp + out),
        },
        "retries": data.get("_retries", 0),
    }


def generate_answer(system_instruction, user_prompt):
    """Try each LLM in order until one answers. Returns which model answered and which were skipped."""
    fallbacks = []
    last = None
    for model in order_models(config.llm_models):
        try:
            r = _generate_with(model, system_instruction, user_prompt)
            r["model"] = model
            r["fallbacks"] = fallbacks
            return r
        except GeminiError as err:
            if not can_fallback(err):
                raise
            last = err
            mark_down(model, err)
            fallbacks.append(describe_failure(model, err))
    raise GeminiError(f"Every answer model failed: {'; '.join(fallbacks)}", last.status if last else 502)
