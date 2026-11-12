"""Simulated typing for the decider: what was typed for a sample of the instances.

  .venv/bin/python -I build_typed.py [--n 450000] [--typos 0.5]

Half of the sampled words (by default) are mistyped with the fitted error model (noise.py), the
rest are typed as meant. Real typing has far fewer typos (about 3.5% of words); the decision
threshold, tuned on eval/cases, accounts for that.

Output: .cache/data/instances/typed.tsv
  id, typed, sentence_start, start_known, n_ctx, cut, word, source, line
The first six columns are what `typo-cli features` reads; `id` is the row of hidden.npy.
"""
import argparse, os, random, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I drops the script directory
import numpy as np
from noise import Noise
from tok import CUT

INST = Path(__file__).resolve().parent / ".cache" / "data" / "instances"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=450_000)
    ap.add_argument("--typos", type=float, default=0.5)
    ap.add_argument("--fragments", type=float, default=0.5, help="share of first-words-after-a-click that are word tails")
    ap.add_argument("--stretched", type=float, default=0.012, help="share of correctly typed words that are stretched on purpose")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default=str(INST / "typed.tsv"))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    noise = Noise()
    first = np.load(INST / "ctx.npy", mmap_mode="r")[:, 0]
    with open(INST / "instances.tsv", encoding="utf-8") as f:
        next(f)
        rows = [line.rstrip("\n").split("\t") for line in f]
    picked = sorted(rng.sample(range(len(rows)), min(args.n, len(rows))))
    typos = fragments = stretched = 0
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("id\ttyped\tsentence_start\tstart_known\tn_ctx\tcut\tword\tsource\tline\n")
        for i in picked:
            ident, source, line, start_known, sentence_start, _, word, _, _, n_ctx = rows[i]
            typed = (noise.sample(word, rng) if rng.random() < args.typos else None) or word
            # After a click the caret may sit inside a word: what gets typed first is then only the
            # word's tail ("rdimensional"). A tail is typed as meant, whatever it looks like.
            if start_known == "0" and len(word) >= 5 and rng.random() < args.fragments:
                word = typed = word[rng.randint(1, min(4, len(word) - 3)):]
                fragments += 1
            # A word stretched on purpose ("sooo", "heyyy", "crazyyy") is typed as meant too.
            elif typed == word and len(word) >= 2 and word.isalpha() and rng.random() < args.stretched:
                at = len(word) - 1 if rng.random() < 0.6 else rng.choice([k for k, ch in enumerate(word) if ch in "aeiouy"] or [len(word) - 1])
                word = typed = word[:at] + word[at] * (1 + rng.choices((1, 2, 3, 4), weights=(15, 35, 30, 20))[0]) + word[at + 1:]
                stretched += 1
            typos += typed != word
            f.write("\t".join([ident, typed, sentence_start, start_known, n_ctx, str(int(first[i] == CUT)), word, source, line]) + "\n")
    print(f"{len(picked)} words, {typos} mistyped, {fragments} word tails, {stretched} stretched -> {args.out}")


if __name__ == "__main__":
    main()
