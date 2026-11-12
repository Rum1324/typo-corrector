"""Samples typing instances for the decision model from the held-out corpus lines.

  .venv/bin/python -I build_instances.py [--scale 1.0]

An instance is one word of a line at the moment the user finishes it: the word that was meant,
what the app would know about its position, and the tokens before it (the language-model context).
Slips are added later (noise model), so the same contexts serve every experiment.

Output, under .cache/data/instances/:
  instances.tsv   id, source, line, start_known, sentence_start, capitalised, word, prefix, boundary, n_ctx
  ctx.npy         int32 [N, 32]: <s> or <cut>, then up to 31 context tokens, padded with 0
The position rules mirror TextTracker / AutocorrectPolicy: a word is analysed when a trigger key
(space . , ! ? ; :) ends it and nothing before it in its chunk marks a link or identifier.
"""
import argparse, os, re, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I drops the script directory
import numpy as np
from tok import Vocab, BOS, CUT, UNK, N_SPECIAL

HERE = Path(__file__).resolve().parent
CORPUS = HERE / ".cache" / "data" / "corpus"
OUT = HERE / ".cache" / "data" / "instances"
L = 32
TARGET = {"opensubtitles": 500_000, "stackexchange": 450_000, "wikitext": 250_000, "enron": 100_000, "arxiv": 70_000}
PER_LINE = 8
CHUNK = re.compile(r"^([^A-Za-z0-9']*)([A-Za-z']+)([^A-Za-z0-9']*)$")
LINK_CHARS = set("/\\@_#=~<>`|{}[$%^*+&.")          # AutocorrectPolicy.linkChars
TRIGGERS = set(".,!?;:")


def candidates(line):
    """(start offset, prefix, word, boundary) for every word the app would analyse."""
    out, at = [], 0
    for chunk in line.split(" "):
        start = at
        at += len(chunk) + 1
        m = CHUNK.match(chunk)
        if not m:
            continue
        prefix, word, suffix = m.groups()
        if not 1 <= len(word) <= 24 or word[0] == "'" or word[-1] == "'":
            continue
        if any(c in LINK_CHARS for c in prefix) or (suffix and suffix[0] not in TRIGGERS):
            continue
        if any(c.isupper() for c in word[1:]):         # ALL CAPS and camelCase are never corrected
            continue
        out.append((start + len(prefix), prefix, word, suffix[0] if suffix else " "))
    return out


def sentence_start(text_before, line_start_known):
    s = text_before.rstrip(" ")
    return line_start_known if not s else s[-1] in ".!?:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    vocab = Vocab()
    rng = np.random.default_rng(args.seed)
    rows, ctxs = [], []
    for source, target in TARGET.items():
        target = int(target * args.scale)
        lines = (CORPUS / f"{source}.sim.txt").read_text(encoding="utf-8").split("\n")
        lines = [l for l in lines if l]
        order = rng.permutation(len(lines))
        made = 0
        for li in order:
            if made >= target:
                break
            line = lines[li]
            cands = candidates(line)
            if not cands:
                continue
            picks = rng.choice(len(cands), size=min(PER_LINE, len(cands)), replace=False)
            for k in sorted(picks):
                start, prefix, word, boundary = cands[k]
                before = line[:start]
                r = rng.random()
                start_known, line_known = True, True
                if r < 0.08:                              # first word after a click: nothing known
                    before, start_known, line_known = "", False, False
                elif r < 0.20 and " " in before.strip():  # clicked earlier in the line
                    words = before.split(" ")
                    keep = int(rng.integers(1, 7))
                    if keep < len(words) - 1:
                        before, line_known = " ".join(words[-keep - 1:]), False
                full = vocab.encode(before)
                ids, truncated = full[-(L - 1):], len(full) > L - 1
                ids = [t if t < N_SPECIAL or rng.random() > 0.03 else UNK for t in ids]   # an unfixed earlier typo
                ctx = [BOS if (line_known and not truncated) else CUT] + ids
                rows.append((source, int(li), int(start_known), int(sentence_start(before, line_known)),
                             int(word[0].isupper()), word, prefix, boundary, len(ctx)))
                ctxs.append(ctx + [0] * (L - len(ctx)))
                made += 1
        print(f"{source}: {made} instances from {len(lines)} lines", flush=True)
    np.save(OUT / "ctx.npy", np.asarray(ctxs, dtype=np.int32))
    with open(OUT / "instances.tsv", "w", encoding="utf-8") as f:
        f.write("id\tsource\tline\tstart_known\tsentence_start\tcapitalised\tword\tprefix\tboundary\tn_ctx\n")
        for i, r in enumerate(rows):
            f.write(str(i) + "\t" + "\t".join(str(x) for x in r) + "\n")
    print(f"{len(rows)} instances -> {OUT}")


if __name__ == "__main__":
    main()
