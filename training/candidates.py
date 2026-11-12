"""SymSpell-style candidate generation in Python, used to build ranking lists for training.
Mirrors Sources/TypoCore/SymSpell.swift: prefix length 7, OSA verification, top-K by frequency."""
import os, pickle
from rapidfuzz.distance import OSA

PREFIX = 7


def _deletes(s, max_d, out):
    out.add(s)
    if max_d == 0 or len(s) <= 1:
        return
    for i in range(len(s)):
        d = s[:i] + s[i + 1:]
        if d not in out:
            _deletes(d, max_d - 1, out)


class Candidates:
    def __init__(self, words, counts, cache_path=None):
        self.words, self.counts = words, counts
        self.index = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.index = pickle.load(f)
            return
        for i, w in enumerate(words):
            s = set()
            _deletes(w[:PREFIX], 2, s)
            for d in s:
                self.index.setdefault(d, []).append(i)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(self.index, f, protocol=pickle.HIGHEST_PROTOCOL)

    def lookup(self, word, max_dist=None, top_k=12, min_count=300.0):
        if max_dist is None:
            max_dist = 1 if len(word) <= 4 else 2
        s = set()
        _deletes(word[:PREFIX], max_dist, s)
        seen, hits = set(), []
        for d in s:
            for i in self.index.get(d, ()):
                if i in seen:
                    continue
                seen.add(i)
                c = self.words[i]
                if c == word or abs(len(c) - len(word)) > max_dist or self.counts[i] < min_count:
                    continue
                dist = OSA.distance(word, c, score_cutoff=max_dist)
                if dist <= max_dist:
                    hits.append(i)
        hits.sort(key=lambda i: -self.counts[i])
        return [self.words[i] for i in hits[:top_k]]
