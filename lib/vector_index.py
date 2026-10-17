"""Vector index over the policies, one index per embedding model.

Vectors from Embedding 1 and Embedding 2 live in different spaces and can never be compared with
each other, so a query must be embedded with the same model that built the index it searches.
"""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .config import config
from .gemini import (GeminiError, can_fallback, describe_failure, embed, format_document,
                     format_query, mark_down, order_models)
from .policies import policies_hash

_indexes = {}   # model -> {"model", "dims", "hash", "vectors": [{"id", "vector"}]}
_errors = {}    # model -> last build error
_state = {"building": False}
_lock = threading.Lock()
_retry_timer = None


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _load_from_disk(model, policies):
    try:
        saved = json.loads(config.index_path_for(model).read_text(encoding="utf-8"))
        if (saved.get("model") == model and saved.get("dims") == config.embedding_dims
                and saved.get("hash") == policies_hash(policies)
                and len(saved.get("vectors", [])) == len(policies)):
            return saved
    except (OSError, ValueError):
        pass
    return None


def _build_one(model, policies, log):
    log(f"Building {model} index ({len(policies)} policies)...")

    def one(p):
        # Indexing is not time-critical, so be patient with rate limits here.
        r = embed(format_document(p, model), "document", model, max_retries=5)
        return {"id": p["id"], "vector": r["vector"]}

    with ThreadPoolExecutor(max_workers=4) as pool:
        vectors = list(pool.map(one, policies))
    built = {"model": model, "dims": config.embedding_dims, "hash": policies_hash(policies),
             "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "vectors": vectors}
    config.index_path_for(model).write_text(json.dumps(built), encoding="utf-8")
    return built


def build_index(policies, force=False, log=print):
    """Load or build the index for every configured embedding model.

    Succeeds if at least one model works; models that fail are retried every minute in the background.
    """
    global _retry_timer
    if _retry_timer:
        _retry_timer.cancel()
    missing = []
    for m in config.embedding_models:
        if not force and m in _indexes:
            continue
        if not force:
            cached = _load_from_disk(m, policies)
            if cached:
                _indexes[m] = cached
                _errors.pop(m, None)
                log(f"{m} index loaded from disk.")
                continue
        missing.append(m)

    if missing:
        _state["building"] = True
        try:
            with ThreadPoolExecutor(max_workers=len(missing)) as pool:
                futures = {m: pool.submit(_build_one, m, policies, log) for m in missing}
                for m, fut in futures.items():
                    try:
                        _indexes[m] = fut.result()
                        _errors.pop(m, None)
                        log(f"{m} index built and saved.")
                    except Exception as e:  # noqa: BLE001
                        _errors[m] = str(e)
                        log(f"{m} index failed: {e}")
        finally:
            _state["building"] = False

    still_missing = [m for m in config.embedding_models if m not in _indexes]
    if still_missing and not force:
        _retry_timer = threading.Timer(60, lambda: _safe_rebuild(policies, log))
        _retry_timer.daemon = True
        _retry_timer.start()
    if not _indexes:
        raise RuntimeError("No vector index could be built. " + " | ".join(f"{m}: {e}" for m, e in _errors.items()))


def _safe_rebuild(policies, log):
    try:
        build_index(policies, log=log)
    except Exception:  # noqa: BLE001
        pass


def index_status():
    return {
        "ready": bool(_indexes),
        "building": _state["building"],
        "dims": config.embedding_dims,
        "models": [{"model": m, "ready": m in _indexes,
                    "size": len(_indexes[m]["vectors"]) if m in _indexes else 0,
                    "error": _errors.get(m)} for m in config.embedding_models],
        "error": next(iter(_errors.values())) if (not _indexes and _errors) else None,
    }


def search(question, policies, k=None):
    """Embed the question (Embedding 2 first, then Embedding 1 if needed) and return the top-k
    policies by cosine similarity."""
    k = k or config.top_k
    candidates = order_models([m for m in config.embedding_models if m in _indexes])
    if not candidates:
        raise GeminiError("The vector index is not ready yet. Check the program window, then try again.", 503)
    by_id = {p["id"]: p for p in policies}
    fallbacks, last = [], None
    for model in candidates:
        try:
            t0 = time.perf_counter()
            q = embed(format_query(question, model), "query", model)
            embed_ms = (time.perf_counter() - t0) * 1000
            t1 = time.perf_counter()
            scored = sorted(((by_id[v["id"]], _dot(q["vector"], v["vector"])) for v in _indexes[model]["vectors"]),
                            key=lambda x: x[1], reverse=True)[:k]
            return {
                "hits": [{"policy": p, "score": s} for p, s in scored],
                "model": model,
                "fallbacks": fallbacks,
                "embed_ms": embed_ms,
                "search_ms": (time.perf_counter() - t1) * 1000,
                "embed_tokens": q["tokens"],
                "embed_tokens_estimated": q["tokens_estimated"],
                "retries": q["retries"],
            }
        except GeminiError as err:
            if not can_fallback(err):
                raise
            last = err
            mark_down(model, err)
            fallbacks.append(describe_failure(model, err))
    raise GeminiError(f"Every embedding model failed: {'; '.join(fallbacks)}", last.status if last else 502)
