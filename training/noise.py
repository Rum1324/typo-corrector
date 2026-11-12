"""Error model for simulated typing: how a meant word turns into the word that gets typed.

  .venv/bin/python -I noise.py --fit                  fits noise.json from the real-typing tables
  .venv/bin/python -I noise.py --show the tomorrow    prints sample typos

Two parts:
  * edit rates by position and word length (omit, add, replace, swap a letter), fitted on the
    first-attempt words of the Aalto keystroke study against the words that were meant
    (Dhakal et al., CHI 2018; build_typos.py makes the tables);
  * for words seen in the real data, the typos people actually typed (keystrokes) and the
    misspellings they actually wrote (Birkbeck), sampled as they are.
noise.json holds only the fitted rates. The word-level tables stay in .cache and are optional.
"""
import argparse, json, os, random, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I drops the script directory
from keyboard import ADJ

HERE = Path(__file__).resolve().parent
TYPOS = HERE / ".cache" / "data" / "typos"
PARAMS = HERE / "noise.json"
ALPHA = "abcdefghijklmnopqrstuvwxyz'"
LB_EDGES = (2, 3, 4, 6, 9)            # word-length groups: up to 2, 3, 4, 5-6, 7-9, 10 or more
NB = len(LB_EDGES) + 1
P_SEEN, P_MISSP = 0.5, 0.12           # how often a word's real typos / real misspellings are used


def lb(n):
    for b, e in enumerate(LB_EDGES):
        if n <= e:
            return b
    return len(LB_EDGES)


def pc(i, n):
    """Position of letter i in a word of n letters: 0 first, 1 last, 2 next to last, 3 middle."""
    if i == 0:
        return 0
    if i == n - 1:
        return 1
    if i == n - 2 and n >= 4:
        return 2
    return 3


def sc(j, n):
    """Position of the gap before letter j (j == n is after the last letter): 0 start, 1 end, 2 inside."""
    return 0 if j == 0 else 1 if j == n else 2


def ok(s):
    return bool(s) and all(c in ALPHA for c in s)


def align(typed, meant):
    """Cheapest edit script (OSA) that turns `meant` into `typed`.
    Returns (distance, ops); ops index `meant`: ("del", i), ("ins", gap, x), ("sub", i, x), ("swap", i)."""
    n, m = len(meant), len(typed)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j

    def swapped(i, j):
        return i > 1 and j > 1 and meant[i - 1] == typed[j - 2] and meant[i - 2] == typed[j - 1] and meant[i - 1] != meant[i - 2]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = min(d[i - 1][j - 1] + (meant[i - 1] != typed[j - 1]), d[i - 1][j] + 1, d[i][j - 1] + 1)
            if swapped(i, j):
                c = min(c, d[i - 2][j - 2] + 1)
            d[i][j] = c
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and meant[i - 1] == typed[j - 1] and d[i][j] == d[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif swapped(i, j) and d[i][j] == d[i - 2][j - 2] + 1:
            ops.append(("swap", i - 2)); i, j = i - 2, j - 2
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            ops.append(("sub", i - 1, typed[j - 1])); i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", i - 1)); i -= 1
        else:
            ops.append(("ins", i, typed[j - 1])); j -= 1
    return d[n][m], ops


def rows(name):
    with open(TYPOS / name, encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n").split("\t")


def grid(rows_, cols=NB):
    return [[0.0] * cols for _ in range(rows_)]


def fit():
    clean = {w: int(c) for w, c in rows("keystroke_clean.tsv")}
    pairs = [(t, w, int(c)) for t, w, c, _ in rows("keystroke_pairs.tsv") if ok(t) and ok(w)]
    total = Counter({w: c for w, c in clean.items() if ok(w)})
    for _, w, c in pairs:
        total[w] += c
    opp_c, opp_p, opp_s = grid(4), grid(4), grid(3)        # chances: per letter, per unlike pair, per gap
    opp_ch, opp_dd = Counter(), 0
    for w, N in total.items():
        n, b = len(w), lb(len(w))
        for i, ch in enumerate(w):
            opp_c[pc(i, n)][b] += N
            opp_ch[ch] += N
            if i and w[i - 1] == ch:
                opp_dd += N
            if i + 1 < n and w[i + 1] != ch:
                opp_p[pc(i, n)][b] += N
        for j in range(n + 1):
            opp_s[sc(j, n)][b] += N
    ev = {"del": grid(4), "sub": grid(4), "swap": grid(4), "ins": grid(3)}
    edits = grid(NB, 3)                                    # number of edits in a typo, by length group
    del_ch, del_dd = Counter(), 0
    conf = defaultdict(Counter)                            # meant letter -> typed letter
    rel = grid(3, 3)                                       # added letter: copy of a neighbour, key next to one, other
    other = Counter()
    for t, w, c in pairs:
        dist, ops = align(t, w)
        n, b = len(w), lb(len(w))
        edits[b][min(dist, 3) - 1] += c
        for op in ops:
            if op[0] == "del":
                i = op[1]
                while i + 1 < n and w[i + 1] == w[i]:      # a dropped twin counts on the pair's last letter
                    i += 1
                ev["del"][pc(i, n)][b] += c
                del_ch[w[i]] += c
                if i and w[i - 1] == w[i]:
                    del_dd += c
            elif op[0] == "sub":
                ev["sub"][pc(op[1], n)][b] += c
                conf[w[op[1]]][op[2]] += c
            elif op[0] == "swap":
                ev["swap"][pc(op[1], n)][b] += c
            else:
                j, x = op[1], op[2]
                ev["ins"][sc(j, n)][b] += c
                near = [w[k] for k in (j - 1, j) if 0 <= k < n]
                kind = 0 if x in near else 1 if any(x in ADJ.get(a, ()) for a in near) else 2
                rel[sc(j, n)][kind] += c
                if kind == 2:
                    other[x] += c

    def rates(events, chances, prior=2000.0):
        out = []
        for e_row, o_row in zip(events, chances):
            pooled = sum(e_row) / max(sum(o_row), 1.0)
            out.append([(e + prior * pooled) / (o + prior) for e, o in zip(e_row, o_row)])
        return out

    r_del = sum(map(sum, ev["del"])) / sum(map(sum, opp_c))
    params = {
        "version": 1,
        "source": "Aalto 136M keystrokes (Dhakal et al., CHI 2018), first-attempt words",
        "lengths": list(LB_EDGES),
        "del": rates(ev["del"], opp_c), "sub": rates(ev["sub"], opp_c),
        "swap": rates(ev["swap"], opp_p), "ins": rates(ev["ins"], opp_s),
        "delLetter": {ch: round(min(3.0, max(0.3, ((del_ch[ch] + 20000 * r_del) / (opp_ch[ch] + 20000)) / r_del)), 4)
                      for ch in ALPHA},
        "delTwin": round((del_dd / max(opp_dd, 1)) / r_del, 4),
        "edits": [[x / max(sum(row), 1.0) for x in row] for row in edits],
        "insKind": [[x / max(sum(row), 1.0) for x in row] for row in rel],
        "insOther": {ch: other[ch] for ch in ALPHA if other[ch]},
        "confusions": {a: dict(conf[a].most_common(12)) for a in ALPHA if conf[a]},
    }
    PARAMS.write_text(json.dumps(params, indent=1))
    names = {"del": "omitted", "sub": "replaced", "swap": "swapped", "ins": "added"}
    all_ev = sum(sum(map(sum, g)) for g in ev.values())
    print(f"fitted on {sum(c for *_, c in pairs)} typos in {sum(total.values())} first-attempt words")
    for k, g in ev.items():
        print(f"  {names[k]:9s} {sum(map(sum, g)) / all_ev:6.1%} of edits")
    print("  omit rate per letter by position (first, last, next to last, middle), words of 5-6 letters:",
          [f"{params['del'][p][3]:.4f}" for p in range(4)])
    print(f"  dropped twin letter: x{params['delTwin']}; edits per typo (5-6 letters): "
          f"{[round(x, 3) for x in params['edits'][3]]}")
    print(f"wrote {PARAMS}")


class Noise:
    def __init__(self, tables=True):
        p = json.loads(PARAMS.read_text())
        self.p = p
        self.conf = {a: (list(d), list(d.values())) for a, d in p["confusions"].items()}
        self.other = (list(p["insOther"]), list(p["insOther"].values()))
        self.seen, self.missp = {}, {}
        if tables and (TYPOS / "keystroke_pairs.tsv").exists():
            seen = defaultdict(list)
            for t, w, c, _ in rows("keystroke_pairs.tsv"):
                if int(c) >= 2 and ok(t) and ok(w):
                    seen[w].append((t, int(c)))
            self.seen = {w: ([t for t, _ in v], [c for _, c in v]) for w, v in seen.items()}
        if tables and (TYPOS / "birkbeck.tsv").exists():
            missp = defaultdict(set)
            for t, w, _, _ in rows("birkbeck.tsv"):
                t, w = t.lower(), w.lower()
                if ok(t) and ok(w) and t != w and abs(len(t) - len(w)) <= 2 and align(t, w)[0] <= 2:
                    missp[w].add(t)
            self.missp = {w: sorted(v) for w, v in missp.items()}

    def edit(self, s, rng):
        """One edit of `s`, drawn in proportion to the fitted rates."""
        p, n, b = self.p, len(s), lb(len(s))
        opts, wts = [], []
        for i, ch in enumerate(s):
            c = pc(i, n)
            if n > 1 and not (i + 1 < n and s[i + 1] == ch):
                opts.append(("del", i))
                wts.append(p["del"][c][b] * p["delLetter"].get(ch, 1.0) * (p["delTwin"] if i and s[i - 1] == ch else 1.0))
            if ch in self.conf:
                opts.append(("sub", i)); wts.append(p["sub"][c][b])
            if i + 1 < n and s[i + 1] != ch:
                opts.append(("swap", i)); wts.append(p["swap"][c][b])
        for j in range(n + 1):
            opts.append(("ins", j)); wts.append(p["ins"][sc(j, n)][b])
        kind, i = rng.choices(opts, weights=wts)[0]
        if kind == "del":
            return s[:i] + s[i + 1:]
        if kind == "sub":
            letters, counts = self.conf[s[i]]
            return s[:i] + rng.choices(letters, weights=counts)[0] + s[i + 1:]
        if kind == "swap":
            return s[:i] + s[i + 1] + s[i] + s[i + 2:]
        near = [s[k] for k in (i - 1, i) if 0 <= k < n]
        how = rng.choices((0, 1, 2), weights=p["insKind"][sc(i, n)])[0]
        keys = [k for a in near for k in ADJ.get(a, ())]
        if how == 0:
            x = rng.choice(near)
        elif how == 1 and keys:
            x = rng.choice(keys)
        else:
            x = rng.choices(self.other[0], weights=self.other[1])[0]
        return s[:i] + x + s[i:]

    def sample(self, word, rng):
        """A mistyped form of `word` (first-letter capital kept), or None if none could be made."""
        low = word.lower()
        if not ok(low):
            return None
        seen, missp = self.seen.get(low), self.missp.get(low)
        for _ in range(20):
            r = rng.random()
            if seen and r < P_SEEN:
                t = rng.choices(seen[0], weights=seen[1])[0]
            elif missp and r < (P_SEEN if seen else 0.0) + P_MISSP:
                t = rng.choice(missp)
            else:
                t = low
                for _ in range(rng.choices((1, 2, 3), weights=self.p["edits"][lb(len(low))])[0]):
                    t = self.edit(t, rng)
            if t and t != low:
                break
        else:
            return None
        if word[0].isupper() and (t[0] == low[0] or rng.random() < 0.5):
            t = t[0].upper() + t[1:]
        return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--show", nargs="*")
    args = ap.parse_args()
    if args.fit:
        fit()
    if args.show is not None:
        noise, rng = Noise(), random.Random(3)
        for w in args.show or ["the", "you", "from", "because", "tomorrow", "receive", "Probably", "don't"]:
            print(f"{w:10s}", " ".join(noise.sample(w, rng) or "-" for _ in range(14)))
        gen = Noise(tables=False)
        print("rates only:")
        for w in args.show or ["the", "because", "tomorrow", "necessary"]:
            print(f"{w:10s}", " ".join(gen.sample(w, rng) or "-" for _ in range(14)))


if __name__ == "__main__":
    main()
