"""Turns the held-out real-typing lines into accuracy-suite cases.

  .venv/bin/python -I build_keystroke_eval.py

Reads .cache/data/typos/keystroke_eval_lines.tsv (participant, typed line, meant line), made by
build_typos.py from the one sentence in ten that no training table saw. Writes
eval/cases/keystrokes.dev.tsv and keystrokes.test.tsv (split by sentence).

A typed word counts as a typo when its letters differ from the meant word's by more than case.
Its expected form is the meant word with the typed word's punctuation, in the typed word's case
or the meant word's own ("alocations" -> "allocations|Allocations"). A word that differs only in
case may stay or take the meant case ("THe" -> "THe|The"). Everything else (punctuation, words
typed as meant) is expected to stay exactly as typed.
"""
import os, re, sys, zlib
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / ".cache" / "data" / "typos" / "keystroke_eval_lines.tsv"
OUT = HERE.parent / "eval" / "cases"
PER_SENTENCE = 30
PARTS = re.compile(r"^([^A-Za-z']*)(.*?)([^A-Za-z']*)$")
SHIFT_SLIP = re.compile(r"^[A-Z]{2}[a-z']+$")
INNER = re.compile(r"^[A-Za-z']*$")
HEADER = """# keystrokes.{split}.tsv - real typing: first-attempt lines of transcription typists, sentences held out from all training tables
# source: Aalto 136M Keystrokes dataset (Dhakal et al., CHI 2018), non-commercial use with attribution; made by training/build_keystroke_eval.py
# columns: typed<TAB>expected ; a typo's expected form keeps the typed punctuation, in the typed case or the meant word's own case
"""


def recase(word, like):
    if like[:1].isupper() and not (len(like) > 1 and like.isupper()):
        return word[:1].upper() + word[1:]
    if like.islower():
        return word[:1].lower() + word[1:] if not word.isupper() else word
    return word


def expected_tokens(typed, meant):
    tt, mt = typed.split(" "), meant.split(" ")
    if len(tt) != len(mt) or not tt or any(not t for t in tt):
        return None
    out, typos = [], 0
    for t, m in zip(tt, mt):
        pt, ct, st = PARTS.match(t).groups()
        _, cm, _ = PARTS.match(m).groups()
        if not INNER.match(ct) or not INNER.match(cm):  # "four-city": the source keeps such tokens as typed,
            return None                                 # so what was meant is unknown; drop the line
        if ct and ct == cm and SHIFT_SLIP.match(ct):    # "THe": the source lines keep the typed case, so the meant
            out.append(t + "|" + pt + ct[0] + ct[1:].lower() + st)   # case is unknown; "The" is accepted as well
            continue
        if not ct or not cm or ct == cm:
            out.append(t)
            continue
        if ct.lower() == cm.lower():                    # "THe" for "The": fixing the case is fine, so is leaving it
            out.append(t + "|" + pt + cm + st)
            continue
        a, b = pt + recase(cm, ct) + st, pt + cm + st
        out.append(a if a == b else a + "|" + b)
        typos += 1
    return out, typos


def main():
    by_sentence = defaultdict(list)
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            _, typed, meant = line.rstrip("\n").split("\t")
            if any(ch in typed + meant for ch in "|␣\t<>") or "  " in typed or typed != typed.strip():
                continue
            res = expected_tokens(typed, meant)
            if res:
                by_sentence[meant].append((typed, " ".join(res[0]), res[1]))
    counts = {}
    for split in ("dev", "test"):
        lines = typos = words = 0
        with open(OUT / f"keystrokes.{split}.tsv", "w", encoding="utf-8") as f:
            f.write(HEADER.format(split=split))
            for meant in sorted(by_sentence):
                if (zlib.crc32(meant.encode("utf-8")) // 10) % 2 != (split == "test"):
                    continue
                rows = sorted(set(by_sentence[meant]), key=lambda r: zlib.crc32(r[0].encode("utf-8")))[:PER_SENTENCE]
                for typed, expected, n in rows:
                    f.write(f"{typed}\t{expected}\n")
                    lines += 1; typos += n; words += typed.count(" ") + 1
        counts[split] = (lines, words, typos)
    for split, (lines, words, typos) in counts.items():
        print(f"keystrokes.{split}.tsv: {lines} lines, {words} words, {typos} typos ({typos / max(words, 1):.1%})")


if __name__ == "__main__":
    main()
