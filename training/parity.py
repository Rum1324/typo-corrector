"""Reference outputs for the Swift ports of the tokenizer and the context language model.

  .venv/bin/python -I parity.py [--run runs/lm]
  ../.build/release/typo-cli tok-parity .cache/parity/tok.tsv
  ../.build/release/typo-cli lm-parity .cache/parity/lm.tsv --context-dir runs/lm

tok.tsv: text <TAB> token ids, for held-out corpus lines of every source and a few hand-made lines.
lm.tsv:  context ids <TAB> word ids <TAB> log P(word | context) from PyTorch on the CPU.
"""
import argparse, os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I drops the script directory
import numpy as np
import torch
import lm
from tok import Vocab

HERE = Path(__file__).resolve().parent
CORPUS = HERE / ".cache" / "data" / "corpus"
OUT = HERE / ".cache" / "parity"
HAND = [
    "Hey, are you free tonight? I'm at 3.5 GPA... see github.com/user/repo or file.txt (Dr. Haruto's h264enc)",
    "‘quoted’ don’t MODEL'S model's   tabs\tand  spaces  10,000 12:30 3rd v2 e.g. ?! … — naïve Zoë 東京 ½ x² '90s 'tis rock'n'roll",
    "a;b: c,d. 5.x .5 5. a.5 5.a don't' ''' 'a' it's' O'Brien's",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(lm.RUN))
    ap.add_argument("--lines", type=int, default=4000, help="corpus lines per source")
    ap.add_argument("--contexts", type=int, default=400)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    vocab = Vocab()
    rng = np.random.default_rng(11)

    n = 0
    with open(OUT / "tok.tsv", "w", encoding="utf-8") as f:
        texts = [t.replace("\t", " ") for t in HAND]
        for path in sorted(CORPUS.glob("*.sim.txt")):
            lines = [l for l in path.read_text(encoding="utf-8").split("\n") if l and "\t" not in l and "\r" not in l]
            texts += [lines[i] for i in rng.choice(len(lines), size=min(args.lines, len(lines)), replace=False)]
        for text in texts:
            f.write(text + "\t" + " ".join(map(str, vocab.encode(text))) + "\n")
            n += 1
    print(f"{n} lines -> {OUT / 'tok.tsv'}")

    model, ck = lm.load_checkpoint(Path(args.run) / "ckpt.pt", torch.device("cpu"))
    ctx = np.load(HERE / ".cache" / "data" / "instances" / "ctx.npy", mmap_mode="r")
    bounds = model.bounds
    with open(OUT / "lm.tsv", "w") as f, torch.no_grad():
        for row in rng.choice(len(ctx), size=args.contexts, replace=False):
            ids = [int(t) for t in ctx[row] if t != 0]
            h = model.hidden(torch.tensor([ids]))[0, -1]
            words = np.concatenate([rng.integers(1, bounds[1], 10), rng.integers(bounds[1], bounds[2], 6),
                                    rng.integers(bounds[2], bounds[3], 6)])
            logp = model.log_probs(h, torch.tensor(words))
            f.write(" ".join(map(str, ids)) + "\t" + " ".join(map(str, words.tolist())) + "\t"
                    + " ".join(f"{x:.6f}" for x in logp.tolist()) + "\n")
    print(f"{args.contexts} contexts (checkpoint step {ck['step']}) -> {OUT / 'lm.tsv'}")


if __name__ == "__main__":
    main()
