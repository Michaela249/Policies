"""Runs the 27 test questions through all three approaches and summarises the results."""
import json
import threading
import time

from .approaches import run_all
from .config import config

NAMES = ["rules", "llm", "vector"]
COOLDOWN_S = 5 * 60

_lock = threading.Lock()
_state = {"status": "idle", "done": 0, "total": 0, "started_at": None, "finished_at": None,
          "results": None, "summary": None, "models": None, "error": None}
_started_ts = 0.0

try:  # show the last saved run right away after a restart
    _saved = json.loads(config.benchmark_path.read_text(encoding="utf-8"))
    _state.update(_saved)
    _state["status"] = "done"
except (OSError, ValueError):
    pass


def _avg(xs):
    return sum(xs) / len(xs) if xs else 0


def _summarize(rows):
    out = {}
    for name in NAMES:
        runs = [{"q": r["q"], "res": r["results"].get(name)} for r in rows]
        ok = [x for x in runs if x["res"] and "error" not in x["res"]]
        answerable = [x for x in ok if x["q"]["expected"]]
        out_of_scope = [x for x in ok if not x["q"]["expected"]]

        def correct(x):
            return x["res"]["answerable"] and (x["res"].get("policy") or {}).get("title") == x["q"]["expected"]

        by_type = {}
        for typ in ("direct", "paraphrase"):
            g = [x for x in answerable if x["q"]["type"] == typ]
            by_type[typ] = {"correct": sum(1 for x in g if correct(x)), "total": len(g)}
        out[name] = {
            "questions_run": len(ok),
            "errors": len(runs) - len(ok),
            "correct_policy": {"n": sum(1 for x in answerable if correct(x)), "of": len(answerable)},
            "by_type": by_type,
            "declined_out_of_scope": {"n": sum(1 for x in out_of_scope if not x["res"]["answerable"]), "of": len(out_of_scope)},
            # Answered although the database has no policy for it
            "answered_out_of_scope": {"n": sum(1 for x in out_of_scope if x["res"]["answerable"]), "of": len(out_of_scope)},
            # Answer cites a policy that does not exist, quotes text that is not in it, or invents numbers
            "unsupported_claims": {"n": sum(1 for x in ok if x["res"]["grounding"]["status"] == "unsupported"), "of": len(ok)},
            # Cited a real policy, but it is the wrong policy for the question
            "wrong_policy": {"n": sum(1 for x in answerable if x["res"]["answerable"] and not correct(x)), "of": len(answerable)},
            "avg_ms": round(_avg([x["res"]["metrics"]["ms"] for x in ok])),
            "avg_tokens": round(_avg([x["res"]["metrics"]["tokens"]["total"] + x["res"]["metrics"].get("embedding_tokens", 0) for x in ok])),
        }
    return out


def state():
    with _lock:
        s = dict(_state)
    cooldown = 0
    if _started_ts and s["status"] != "running":
        cooldown = max(0, round(_started_ts + COOLDOWN_S - time.time()))
    s["cooldown_seconds"] = cooldown
    return s


def start(policies, rules_index):
    global _started_ts
    with _lock:
        if _state["status"] == "running":
            return False, "A benchmark run is already in progress."
    wait = state()["cooldown_seconds"]
    if wait > 0:
        return False, f"Please wait {wait} seconds before running the benchmark again."
    questions = json.loads(config.eval_path.read_text(encoding="utf-8"))
    with _lock:
        _state.update({"status": "running", "done": 0, "total": len(questions),
                       "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "finished_at": None,
                       "results": [], "summary": None, "error": None,
                       "models": {"llm": ", ".join(config.llm_models), "embedding": ", ".join(config.embedding_models)}})
    _started_ts = time.time()

    def work():
        try:
            for q in questions:
                results = run_all(q["question"], policies, rules_index)
                with _lock:
                    _state["results"].append({"q": q, "results": results})
                    _state["done"] += 1
            with _lock:
                _state["summary"] = _summarize(_state["results"])
                _state["status"] = "done"
                _state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                config.benchmark_path.write_text(json.dumps(_state, indent=2), encoding="utf-8")
        except Exception as err:  # noqa: BLE001
            with _lock:
                _state["status"] = "error"
                _state["error"] = str(err)

    threading.Thread(target=work, daemon=True).start()
    return True, ""
