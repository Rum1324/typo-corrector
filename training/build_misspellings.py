"""Build Resources/misspellings_en.txt from Wikipedia's machine-readable list of common
misspellings (CC BY-SA). Keeps unambiguous entries whose fix is in our lexicon and whose
misspelling is rare enough that it is not a real word in common use.

  python build_misspellings.py path/to/wiki_misspellings.txt
"""
import re, sys
from pathlib import Path
from synth import load_lexicon

ROOT = Path(__file__).resolve().parents[1]
WORD = re.compile(r"^[a-z]+(?:'[a-z]+)?$")

def main(src):
    words, counts = load_lexicon(ROOT / "Resources" / "words_en.txt")
    freq = dict(zip(words, counts))
    out, dropped = {}, 0
    for line in Path(src).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "->" not in line:
            continue
        typo, fix = (s.strip().lower() for s in line.split("->", 1))
        if "," in fix or not WORD.match(typo) or not WORD.match(fix) or typo == fix:
            dropped += 1; continue
        if fix not in freq or freq.get(typo, 0) >= 5000:
            dropped += 1; continue
        out[typo] = fix
    path = ROOT / "Resources" / "misspellings_en.txt"
    path.write_text("".join(f"{t}\t{f}\n" for t, f in sorted(out.items())))
    print(f"wrote {len(out)} misspellings ({dropped} dropped) -> {path}", file=sys.stderr)

if __name__ == "__main__":
    main(sys.argv[1])
