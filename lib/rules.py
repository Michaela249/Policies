"""Approach 1: rules-based search. No AI, no tokens. Pure keyword rules.

Rules, in order:
 1. lowercase, split into words, drop stopwords and question words
 2. reduce words to a crude stem (plural / -ing / -ed removal)
 3. expand a small, fixed list of abbreviations (wfh -> work from home ...)
 4. score each policy: a question word found in the title counts 3x, in the policy text 1x,
    in the category 0.5x, weighted by how rare the word is across all policies
 5. return the best policy only if it covers enough of the question's words, otherwise "no match"
"""
import math
import re
import time

STOPWORDS = set((
    "a an and are as at be by can could do does for from get had has have how i if in into is it its me my "
    "of on or our should so that the their them then there these they this to us was we what when where which "
    "who whom whose why will with would you your am been being did doing done just any all some many much "
    "very also about after before than too per tell please need want know allowed policy policies company "
    "employee employees staff worker workers people must may require required rule rules"
).split())

# Small hand-written abbreviation table (kept deliberately short and generic).
ALIASES = {
    "wfh": "work from home remote",
    "pto": "vacation",
    "hr": "human resources",
    "vpn": "vpn remote access",
    "wifi": "internet",
    "e-mail": "email",
    "phone": "mobile device",
    "cellphone": "mobile device",
    "wfa": "work abroad",
}

WEIGHTS = {"title": 3.0, "text": 1.0, "category": 0.5}
MIN_COVERAGE = 0.34  # share of the question's (rarity-weighted) words that must be found


def stem(w):
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("ing") and len(w) > 5:
        return w[:-3]
    if w.endswith("ed") and len(w) > 4:
        return w[:-2]
    if w.endswith("ly") and len(w) > 5:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def tokenize(text, expand=False):
    words = re.sub(r"[^a-z0-9\-\s]", " ", text.lower()).split()
    if expand:
        words = [x for w in words for x in (ALIASES[w].split() if w in ALIASES else [w])]
    return [s for s in (stem(w) for w in words if w not in STOPWORDS) if len(s) > 1]


class RulesIndex:
    def __init__(self, policies):
        self.docs = [{
            "policy": p,
            "title": set(tokenize(p["title"])),
            "text": set(tokenize(p["text"])),
            "category": set(tokenize(p["category"])),
        } for p in policies]
        df = {}
        for d in self.docs:
            for t in d["title"] | d["text"] | d["category"]:
                df[t] = df.get(t, 0) + 1
        n = len(self.docs)
        self.idf = {t: math.log(1 + n / c) for t, c in df.items()}
        self.max_idf = max(self.idf.values()) if self.idf else 1.0

    def search(self, question, k=3):
        t0 = time.perf_counter()
        q_tokens = list(dict.fromkeys(tokenize(question, expand=True)))
        known = [t for t in q_tokens if t in self.idf]
        unknown = len(q_tokens) - len(known)
        # Words in the question that appear in no policy still count against the match
        # (at half the weight of the rarest known word), so one shared word like "work"
        # cannot make a long, unrelated question look like a perfect match.
        denom = sum(self.idf[t] * WEIGHTS["title"] for t in known) + unknown * self.max_idf * WEIGHTS["title"] * 0.5

        scored = []
        for d in self.docs:
            s, matched = 0.0, []
            for t in known:
                w = (WEIGHTS["title"] if t in d["title"] else
                     WEIGHTS["text"] if t in d["text"] else
                     WEIGHTS["category"] if t in d["category"] else 0)
                if w:
                    s += self.idf[t] * w
                    matched.append(t)
            if s > 0:
                scored.append({"policy": d["policy"], "score": s / denom if denom else 0.0, "matched": matched})
        scored.sort(key=lambda x: x["score"], reverse=True)

        best = scored[0] if scored else None
        found = bool(best) and best["score"] >= MIN_COVERAGE and len(best["matched"]) >= 1
        return {
            "match_found": found,
            "best": best if found else None,
            "top": scored[:k],
            "query_terms": q_tokens,
            "ms": (time.perf_counter() - t0) * 1000,
        }
