"""Synthetic typo generation: keyboard slips, drops, doubles, transpositions, vowel swaps.
Mirrors makeTypo() in Sources/typo-cli/main.swift so Swift eval and Python training agree."""
import random
from keyboard import ADJ

VOWELS = "aeiou"


def make_typo(word: str, rng: random.Random) -> str:
    w = list(word)
    edits = 1 if rng.random() < 0.82 else 2
    for _ in range(edits):
        if len(w) < 2:
            break
        r = rng.random()
        i = rng.randrange(len(w))
        if r < 0.32 and w[i] in ADJ:
            w[i] = rng.choice(ADJ[w[i]])
        elif r < 0.50:
            if len(w) > 3:
                del w[i]
        elif r < 0.64 and w[i] in ADJ:
            w.insert(i, rng.choice(ADJ[w[i]]))
        elif r < 0.82:
            if i + 1 < len(w) and w[i] != w[i + 1]:
                w[i], w[i + 1] = w[i + 1], w[i]
        elif r < 0.88:
            w.insert(i, w[i])
        elif r < 0.94:
            for j in range(len(w) - 1):
                if w[j] == w[j + 1]:
                    del w[j]
                    break
        else:
            if w[i] in VOWELS:
                w[i] = rng.choice(VOWELS)
    s = "".join(w)
    return make_typo(word, rng) if s == word else s


def load_lexicon(path):
    words, counts = [], []
    with open(path) as f:
        for line in f:
            w, c = line.rstrip("\n").split("\t")
            words.append(w); counts.append(float(c))
    return words, counts


def sample_words(words, counts, n, rng, min_count=1000.0, min_len=3):
    pool = [i for i, (w, c) in enumerate(zip(words, counts)) if c >= min_count and len(w) >= min_len and "'" not in w]
    weights = [counts[i] ** 0.5 for i in pool]
    return [words[i] for i in rng.choices(pool, weights=weights, k=n)]
