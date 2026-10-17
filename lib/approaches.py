"""The three ways of answering a policy question, plus the check for unsupported answers."""
import re
import time

from . import vector_index
from .config import config
from .gemini import generate_answer
from .policies import public_policy

SYSTEM = " ".join([
    "You are a company policy assistant. Answer ONLY from the policy text you are given.",
    "Never use outside knowledge, never guess, and never invent numbers, deadlines, amounts or rules.",
    "If none of the given policies answers the question, set answerable to false, leave policy_title and evidence empty,",
    "and say in answer that no policy in the database covers it.",
    "Otherwise pick the single most relevant policy, copy its title exactly, copy ONE sentence from its text word-for-word as evidence,",
    "and answer the question in one or two sentences.",
])


def _fmt(p):
    return f"[{p['id']}] {p['title']} (Department: {p['department']}; Category: {p['category']}): {p['text']}"


# ---------------------------------------------------------------------------
# Grounding check: does the answer stay inside the policy database?
# ---------------------------------------------------------------------------
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


def _norm(s):
    return re.sub(r"\s+", " ", (s or "").lower().replace("‘", "'").replace("’", "'")).strip()


def _numbers_in(text):
    out = set()
    for m in re.finditer(r"\d+(?:\.\d+)?", text.lower()):
        out.add(str(float(m.group(0))).rstrip("0").rstrip("."))
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in NUMBER_WORDS:
            out.add(str(NUMBER_WORDS[w]))
    return out


def check_grounding(parsed, policies):
    """status:
      supported   - cites a real policy, quotes it verbatim, adds no numbers that policy lacks
      unsupported - cites a policy that does not exist / a quote that is not in the policy / invents numbers
      abstained   - says the database does not cover the question
    """
    if not parsed.get("answerable"):
        return {"status": "abstained", "reasons": [], "policy": None}
    reasons = []
    title = _norm(parsed.get("policy_title"))
    policy = next((p for p in policies if _norm(p["title"]) == title), None)
    if not policy:
        shown = parsed.get("policy_title") or "(none)"
        return {"status": "unsupported", "reasons": [f'Cited policy "{shown}" does not exist in the database.'], "policy": None}
    evidence = _norm(parsed.get("evidence"))
    if not evidence or evidence not in _norm(policy["text"]):
        reasons.append("The quoted evidence does not appear word-for-word in the cited policy.")
    invented = sorted(_numbers_in(parsed.get("answer") or "") - _numbers_in(policy["text"]))
    if invented:
        reasons.append("The answer contains numbers that are not in the policy: " + ", ".join(invented) + ".")
    return {"status": "unsupported" if reasons else "supported", "reasons": reasons, "policy": policy}


def _round(n):
    return round(n * 10) / 10


ZERO_TOKENS = {"input": 0, "output": 0, "thinking": 0, "total": 0}


# ---------------------------------------------------------------------------
# 1) Rules-based search
# ---------------------------------------------------------------------------
def run_rules(question, policies, rules_index):
    t0 = time.perf_counter()
    r = rules_index.search(question, config.top_k)
    ms = (time.perf_counter() - t0) * 1000
    base = {"approach": "rules", "metrics": {"ms": _round(ms), "tokens": dict(ZERO_TOKENS)}}
    if not r["match_found"]:
        return {**base, "answer": "No policy matched the words in this question.", "answerable": False,
                "policy": None, "grounding": {"status": "abstained", "reasons": []},
                "details": {"query_terms": r["query_terms"], "candidates": []}}
    p = r["best"]["policy"]
    return {
        **base,
        # A rules engine cannot write an answer; it returns the policy text itself.
        "answer": p["text"],
        "answerable": True,
        "policy": public_policy(p),
        # The text is copied straight from the database, so it can never invent content,
        # but it can still return the WRONG policy: that is measured by the benchmark.
        "grounding": {"status": "supported", "reasons": ["Returns the policy text verbatim, so nothing is invented."]},
        "details": {
            "query_terms": r["query_terms"],
            "candidates": [{"title": c["policy"]["title"], "score": round(c["score"], 2), "matched": c["matched"]} for c in r["top"]],
        },
    }


# ---------------------------------------------------------------------------
# 2) LLM without a vector index: the whole policy database is pasted into the prompt
# ---------------------------------------------------------------------------
def run_llm_full(question, policies, rules_index=None):
    t0 = time.perf_counter()
    prompt = f"POLICY DATABASE ({len(policies)} policies):\n" + "\n".join(_fmt(p) for p in policies) + f"\n\nQUESTION: {question}"
    r = generate_answer(SYSTEM, prompt)
    ms = (time.perf_counter() - t0) * 1000
    parsed = r["parsed"]
    g = check_grounding(parsed, policies)
    return {
        "approach": "llm",
        "answer": parsed.get("answer", ""),
        "answerable": bool(parsed.get("answerable")),
        "policy": public_policy(g["policy"]),
        "cited_title": parsed.get("policy_title") or "",
        "evidence": parsed.get("evidence") or "",
        "grounding": {"status": g["status"], "reasons": g["reasons"]},
        "metrics": {"ms": _round(ms), "tokens": r["usage"], "retries": r["retries"], "model": r["model"], "fallbacks": r["fallbacks"]},
        "details": {"policies_in_prompt": len(policies)},
    }


# ---------------------------------------------------------------------------
# 3) LLM with a vector index: retrieve top-k by embedding similarity, then answer from those
# ---------------------------------------------------------------------------
def run_llm_vector(question, policies, rules_index=None):
    t0 = time.perf_counter()
    s = vector_index.search(question, policies, config.top_k)
    prompt = (f"RETRIEVED POLICIES (top {len(s['hits'])} by vector similarity):\n"
              + "\n".join(_fmt(h["policy"]) for h in s["hits"]) + f"\n\nQUESTION: {question}")
    t1 = time.perf_counter()
    r = generate_answer(SYSTEM, prompt)
    llm_ms = (time.perf_counter() - t1) * 1000
    ms = (time.perf_counter() - t0) * 1000
    parsed = r["parsed"]
    g = check_grounding(parsed, policies)
    return {
        "approach": "vector",
        "answer": parsed.get("answer", ""),
        "answerable": bool(parsed.get("answerable")),
        "policy": public_policy(g["policy"]),
        "cited_title": parsed.get("policy_title") or "",
        "evidence": parsed.get("evidence") or "",
        "grounding": {"status": g["status"], "reasons": g["reasons"]},
        "metrics": {
            "ms": _round(ms),
            "tokens": r["usage"],
            "embedding_tokens": s["embed_tokens"],
            "embedding_tokens_estimated": s["embed_tokens_estimated"],
            "model": r["model"],
            "embedding_model": s["model"],
            "fallbacks": s["fallbacks"] + r["fallbacks"],
            "retries": (r["retries"] or 0) + (s["retries"] or 0),
            "breakdown": {"embed_ms": _round(s["embed_ms"]), "search_ms": _round(s["search_ms"]), "llm_ms": _round(llm_ms)},
        },
        "details": {"retrieved": [{"title": h["policy"]["title"], "score": round(h["score"], 3)} for h in s["hits"]]},
    }


def safe_run(name, question, policies, rules_index):
    """Run one approach and turn any failure into a result object instead of raising."""
    t0 = time.perf_counter()
    try:
        if name == "rules":
            return run_rules(question, policies, rules_index)
        if name == "llm":
            return run_llm_full(question, policies)
        return run_llm_vector(question, policies)
    except Exception as err:  # noqa: BLE001
        return {"approach": name, "error": str(err) or err.__class__.__name__,
                "metrics": {"ms": _round((time.perf_counter() - t0) * 1000), "tokens": dict(ZERO_TOKENS)}}


def run_all(question, policies, rules_index):
    """Run the three approaches at the same time and return them in a dict."""
    from concurrent.futures import ThreadPoolExecutor
    names = ["rules", "llm", "vector"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(safe_run, n, question, policies, rules_index) for n in names]
        return {n: f.result() for n, f in zip(names, futures)}
