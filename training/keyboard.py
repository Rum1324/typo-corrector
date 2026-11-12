"""Shared vocabulary, encoding and keyboard geometry. Must stay in sync with
Sources/TypoCore/NeuralRanker.swift (encode) and EditDistance.swift (adjacency)."""

CHARS = "abcdefghijklmnopqrstuvwxyz'-"
PAD, CLS, SEP, UNK = 0, 1, 2, 3
CHAR_TO_ID = {c: i + 4 for i, c in enumerate(CHARS)}
VOCAB_SIZE = 4 + len(CHARS)          # 32
MAX_WORD = 20
MAX_LEN = 2 + 2 * MAX_WORD           # 42

ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]


def build_adjacency():
    adj = {}
    for r, row in enumerate(ROWS):
        for i, c in enumerate(row):
            s = set()
            for rr, ii in [(r, i - 1), (r, i + 1), (r - 1, i), (r - 1, i + 1), (r + 1, i), (r + 1, i - 1)]:
                if 0 <= rr < len(ROWS) and 0 <= ii < len(ROWS[rr]):
                    s.add(ROWS[rr][ii])
            adj[c] = sorted(s)
    return adj


ADJ = build_adjacency()


def encode_pair(typo: str, cand: str):
    """Returns (tokens, segs) lists of length MAX_LEN."""
    tokens = [CLS] + [CHAR_TO_ID.get(c, UNK) for c in typo[:MAX_WORD]]
    segs = [0] * len(tokens)
    tokens.append(SEP); segs.append(1)
    for c in cand[:MAX_WORD]:
        tokens.append(CHAR_TO_ID.get(c, UNK)); segs.append(1)
    pad = MAX_LEN - len(tokens)
    return tokens + [PAD] * pad, segs + [0] * pad
