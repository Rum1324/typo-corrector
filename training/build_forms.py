"""Counts every word form in the language-model training text, for Resources/forms_en.tsv.

  .venv/bin/python -I build_forms.py [--workers 10] [--until 06:55]     count (heavy: night only)
  .venv/bin/python -I build_forms.py --export [--min-count 5]           write Resources/forms_en.tsv

The 80,000-word lexicon misses a lot of real vocabulary ("backprop", "timesteps", "postgres",
names). A typed word outside the lexicon that the corpus has seen many times is probably meant;
one it has never seen, next to a common word, is probably a slip. The decider gets the count of
the typed word and of each candidate as evidence.

Only training lines are counted (the same lines, filters and caps as build_corpus.py): the lines
held out for validation and simulated typing must look as unseen here as text the user will type.
Words inside links, paths and identifiers are skipped, as in the tokenizer.

Output: .cache/data/forms/<source>.tsv (word, count; resumable per source) and, with --export,
Resources/forms_en.tsv: a first line "#total <tokens>", then word<TAB>count for every word form
seen at least --min-count times.
"""
import argparse, json, os, re, sys, time, zlib
from collections import Counter, deque
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I drops the script directory
import build_corpus as bc
from tok import Vocab, UNK, NAME, SYM, LINK

HERE = Path(__file__).resolve().parent
OUT = HERE / ".cache" / "data" / "forms"
WORD = re.compile(r"(?<![A-Za-z0-9'])[A-Za-z]+(?:'[A-Za-z]+)*(?![A-Za-z0-9'])")
_vocab = None


def count(task):
    """(source, records) -> (word counts, training tokens) for the lines build_corpus.py trains on."""
    global _vocab
    if _vocab is None:
        _vocab = Vocab()
    source, records = task
    forms, tokens = Counter(), 0
    for line in bc.CLEAN[source](records):
        ids = _vocab.encode(line)
        if len(ids) < 2:
            continue
        junk = sum(1 for t in ids if t == UNK or t == NAME or t == SYM)
        if junk > 0.4 * len(ids):
            continue
        if zlib.crc32(line.encode("utf-8", "ignore")) % 1000 < 35:   # validation and simulated-typing lines
            continue
        tokens += len(ids) + 1
        for chunk in line.replace("’", "'").split():
            if any(ch in LINK for ch in chunk):
                continue
            for w in WORD.findall(chunk):
                if len(w) <= 24:
                    forms[w.lower()] += 1
    return forms, tokens


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    deadline = None
    if args.until:
        h, m = map(int, args.until.split(":"))
        deadline = time.mktime(time.localtime()[:3] + (h, m, 0) + time.localtime()[6:])
    with Pool(args.workers) as pool:
        for source in bc.READ:
            done = OUT / f"{source}.tsv"
            if done.exists() or not (bc.RAW / bc.PRESENT[source]).exists():
                continue
            t0, total, tokens, pending = time.time(), Counter(), 0, deque()

            def drain(block):
                nonlocal tokens
                while pending and (block or pending[0].ready()):
                    forms, n = pending.popleft().get()
                    total.update(forms); tokens += n
                    block = False

            for batch in bc.READ[source]():
                if tokens >= bc.CAPS[source]:
                    break
                if deadline and time.time() > deadline:
                    print(f"stopped at {args.until}: {source} unfinished, nothing written for it", flush=True)
                    pool.terminate()
                    return
                pending.append(pool.apply_async(count, ((source, batch),)))
                drain(block=len(pending) >= args.workers * 3)
            while pending and tokens < bc.CAPS[source]:
                drain(block=True)
            tmp = done.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(f"#total\t{tokens}\n")
                for w, c in total.most_common():
                    if c >= 2:
                        f.write(f"{w}\t{c}\n")
            tmp.rename(done)
            print(f"{time.strftime('%H:%M:%S')} {source}: {tokens / 1e6:.1f}M tokens, {len(total)} forms, {time.time() - t0:.0f}s", flush=True)


def export(args):
    vocab = Vocab()
    total, tokens = Counter(), 0
    for path in sorted(OUT.glob("*.tsv")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                w, c = line.rstrip("\n").split("\t")
                if w == "#total":
                    tokens += int(c)
                else:
                    total[w] += int(c)
    out = HERE.parent / "Resources" / "forms_en.tsv"
    kept = [(w, c) for w, c in total.most_common() if c >= args.min_count]
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"#total\t{tokens}\n")
        for w, c in kept:
            f.write(f"{w}\t{c}\n")
    outside = sum(1 for w, _ in kept if w not in vocab.index)
    print(f"{tokens / 1e6:.1f}M tokens, {len(total)} forms seen twice or more; wrote {len(kept)} seen "
          f"{args.min_count}+ times ({outside} of them outside the lexicon) -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--until", default="", help="HH:MM today: stop before this time (night window)")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--min-count", type=int, default=5)
    a = ap.parse_args()
    export(a) if a.export else run(a)
