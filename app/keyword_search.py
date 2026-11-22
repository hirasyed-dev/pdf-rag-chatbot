"""Small BM25 keyword ranking (no extra library). Used together with the semantic
search ("hybrid search"), because exact words like "SQLite" or "page 4" are
sometimes better found by keywords than by meaning."""
import math
import re
from collections import Counter

STOP_WORDS = frozenset(
    "a an and are as at be by for from how in is it of on or that the this to was "
    "what when where which who why with do does did about can i you we they their "
    "used use using".split()
)
K1, B = 1.5, 0.75


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    # crude plural handling so "databases" matches "database"
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w
            for w in words if w not in STOP_WORDS]


class BM25:
    def __init__(self, documents: list[str]):
        self.tokens = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(c.values()) for c in self.tokens]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        document_frequency = Counter()
        for counter in self.tokens:
            document_frequency.update(counter.keys())
        n = len(documents)
        self.idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        terms = set(tokenize(query))
        result = []
        for counter, length in zip(self.tokens, self.lengths):
            score = 0.0
            for term in terms:
                tf = counter.get(term, 0)
                if tf:
                    norm = tf * (K1 + 1) / (
                        tf + K1 * (1 - B + B * length / (self.avg_length or 1))
                    )
                    score += self.idf[term] * norm
            result.append(score)
        return result


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Merge several ranked id lists into one (standard RRF)."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + position + 1)
    return sorted(fused, key=fused.get, reverse=True)
