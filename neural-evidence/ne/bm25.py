"""A small dependency-free BM25 for re-ranking a handful of sentences."""
import math
import re
from collections import Counter

_STOP = set("""a an the of in on at to for from by with and or but is are was were be been being
this that these those it its as into than then there their his her he she they them we you i
not no do does did has have had which who whom what when where why how""".split())
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def rank(query: str, docs: list, k1: float = 1.5, b: float = 0.75) -> list:
    """Return document indices sorted by BM25 score (best first)."""
    if not docs:
        return []
    toks = [tokenize(d) for d in docs]
    n = len(docs)
    avgdl = sum(len(t) for t in toks) / n or 1.0
    df = Counter(w for t in toks for w in set(t))
    q = tokenize(query)
    scores = []
    for i, t in enumerate(toks):
        tf = Counter(t)
        s = 0.0
        for w in q:
            if w not in tf:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(t) / avgdl))
        scores.append(s)
    return sorted(range(n), key=lambda i: (-scores[i], i))
