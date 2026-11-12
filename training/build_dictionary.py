"""Build Resources/words_en.txt: `word<TAB>count` for the top-N English words, and
Resources/words_more_en.tsv: the rest of wordfreq's large list (about 240,000 rarer words and
names), which the context path uses as evidence that a word outside the lexicon is real.
The lexicon itself must not change: the context language model's vocabulary is its line order.

  python build_dictionary.py            words_en.txt (retrain the language model afterwards!)
  python build_dictionary.py --more     words_more_en.tsv only

Counts are derived from wordfreq's Zipf scale (10**zipf words per billion),
filtered to purely alphabetic words (apostrophes allowed, e.g. don't).
"""
import argparse, re, sys
from pathlib import Path
from wordfreq import top_n_list, zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
WORD_RE = re.compile(r"^[a-z]+(?:'[a-z]+)?$")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80000)
    ap.add_argument("--out", default=str(ROOT / "Resources" / "words_en.txt"))
    ap.add_argument("--more", action="store_true")
    args = ap.parse_args()
    if args.more:
        from wordfreq import iter_wordlist
        known = {line.split("\t")[0] for line in (ROOT / "Resources" / "words_en.txt").read_text().splitlines()}
        rows = []
        for w in iter_wordlist("en", "large"):
            if WORD_RE.match(w) and w not in known:
                z = zipf_frequency(w, "en", wordlist="large")
                if z > 0:
                    rows.append((w, int(round(10 ** z))))
        out = ROOT / "Resources" / "words_more_en.tsv"
        out.write_text("#total\t1000000000\n" + "".join(f"{w}\t{c}\n" for w, c in rows))
        print(f"wrote {len(rows)} words -> {out}", file=sys.stderr)
        return
    seen, rows = set(), []
    for w in top_n_list("en", args.n * 2, wordlist="large"):
        if not WORD_RE.match(w) or w in seen:
            continue
        z = zipf_frequency(w, "en", wordlist="large")
        if z <= 0:
            continue
        seen.add(w)
        rows.append((w, int(round(10 ** z))))  # occurrences per billion
        if len(rows) >= args.n:
            break
    rows.sort(key=lambda r: -r[1])
    Path(args.out).write_text("\n".join(f"{w}\t{c}" for w, c in rows) + "\n")
    print(f"wrote {len(rows)} words -> {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
