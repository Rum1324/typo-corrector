"""Extracts real typing errors for the error model (and a real-typing test set).

  .venv/bin/python -I build_typos.py [--workers 12] [--limit N]

Sources (see fetch_data.sh):
  Aalto 136M keystrokes: 168k people each transcribing 15 sentences. Every key press is replayed;
      the word as it stood when the typist first pressed space is what an autocorrector would have
      seen, whether or not they went back and fixed it.
  Birkbeck corpora: misspellings collected from handwriting, spelling tests and Wikipedia edits.

Output, under .cache/data/typos/:
  keystroke_pairs.tsv        typed, intended, count, fixed_later   (lowercase, punctuation stripped)
  keystroke_clean.tsv        word, count                           (first attempts typed correctly)
  keystroke_merges.tsv       typed, intended ("in the"), count     (a dropped space)
  keystroke_eval_lines.tsv   participant, typed line, expected line (held-out sentences only)
  keystroke_sentences.txt    every prompt sentence (kept out of the language-model corpus)
  birkbeck.tsv               typo, correct, source, count

One sentence in ten (by hash) is held out: none of its typing goes into the pair counts, and its
lines become eval cases.
"""
import argparse, os, sys, time, zipfile, zlib
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I drops the script directory
from rapidfuzz.distance import OSA

HERE = Path(__file__).resolve().parent
RAW = HERE / ".cache" / "data" / "raw"
OUT = HERE / ".cache" / "data" / "typos"
ZIP = RAW / "keystrokes" / "Keystrokes.zip"
MODIFIERS = {"SHIFT", "CAPS_LOCK", "CTRL", "ALT", "META", "OS", "ALTGR", "CONTROL"}
EVAL_LINES_PER_SENTENCE = 40


def core(token):
    """The word inside a token: surrounding punctuation stripped."""
    i, j = 0, len(token)
    while i < j and not (token[i].isalnum() or token[i] == "'"):
        i += 1
    while j > i and not (token[j - 1].isalnum() or token[j - 1] == "'"):
        j -= 1
    return token[i:j].strip("'")


def is_word(s):
    return len(s) >= 1 and s.isascii() and s.replace("'", "").isalpha()


def held_out(sentence):
    return zlib.crc32(sentence.encode("utf-8")) % 10 == 0


def replay(keys, user_input):
    """Replays (press time, letter, key code) events. Returns {word index: word at its first space} or None
    when the result differs from what the test recorded (cursor keys, paste, selection)."""
    buf, first = [], {}

    def commit():
        k = len(buf)
        while k > 0 and buf[k - 1] != " ":
            k -= 1
        if k < len(buf):
            idx = len("".join(buf[:k]).split())
            if idx not in first:
                first[idx] = "".join(buf[k:])

    recovered = False
    for _, letter, code in sorted(keys):
        if letter == "":                      # the log lost the character: the key code still names it
            if 65 <= code <= 90:
                letter, recovered = chr(code + 32), True
            elif code == 32:
                letter = " "
            elif code == 8:
                letter = "BKSP"
            else:
                return None
        if letter == "BKSP":
            if buf:
                buf.pop()
        elif len(letter) == 1:
            if letter == " ":
                commit()
            buf.append(letter)
        elif letter not in MODIFIERS:
            return None
    text = "".join(buf)
    if text != user_input and not (recovered and text.lower() == user_input.lower()):
        return None
    commit()
    return first


def sections(text):
    """One participant file -> {section: (sentence, input, [(press time, letter, key code)])}."""
    out = {}
    for line in text.split("\n")[1:]:
        p = line.split("\t")
        if len(p) != 9:
            continue
        try:
            t, code = int(p[5]), int(p[8])
        except ValueError:
            continue
        s = out.get(p[1])
        if s is None:
            s = out[p[1]] = (p[2], p[3], [])
        s[2].append((t, p[7], code))
    return out


_zip = None


def work(names):
    global _zip
    if _zip is None:
        _zip = zipfile.ZipFile(ZIP)
    pairs, clean, merges, case = Counter(), Counter(), Counter(), Counter()
    fixed = Counter()
    eval_lines, sentences = [], set()
    stats = Counter()
    for name in names:
        pid = name.rsplit("/", 1)[-1].split("_")[0]
        try:
            text = _zip.read(name).decode("utf-8", "replace")
        except Exception:
            stats["unreadable"] += 1
            continue
        for sentence, user_input, keys in sections(text).values():
            stats["sections"] += 1
            sentences.add(sentence)
            first = replay(keys, user_input)
            if first is None:
                stats["replay_mismatch"] += 1
                continue
            target = sentence.split()
            final = user_input.split()
            test = held_out(sentence)
            offset, typos, aligned = 0, 0, True
            expected = []
            for i in range(len(first)):
                typed = first.get(i)
                j = i + offset
                if typed is None or j >= len(target):
                    aligned = False
                    break
                a, b = core(typed), core(target[j])
                if not is_word(b) or a == b:
                    expected.append(typed if not is_word(b) else target[j])
                    if is_word(b) and not test:
                        clean[b.lower()] += 1
                    continue
                al, bl = a.lower(), b.lower()
                if al == bl:                                   # capitalisation only
                    expected.append(typed)
                    if not test:
                        case[(a, b)] += 1
                    continue
                if j + 1 < len(target) and al == bl + core(target[j + 1]).lower():
                    if not test:
                        merges[(al, bl + " " + core(target[j + 1]).lower())] += 1
                    offset += 1
                    aligned = False                            # token counts differ: not an eval line
                    expected.append(typed)
                    continue
                cap = max(1, min(3, len(bl) // 2))
                if not is_word(a) or OSA.distance(al, bl, score_cutoff=cap) > cap:
                    aligned = False                            # skipped or split word: alignment is lost
                    stats["misaligned"] += 1
                    break
                typos += 1
                expected.append(target[j])
                if not test:
                    pairs[(al, bl)] += 1
                    if i < len(final) and core(final[i]) == b:
                        fixed[(al, bl)] += 1
            stats["typos"] += typos
            if test and aligned and len(first) == len(target):
                typed_line = " ".join(first[i] for i in range(len(first)))
                if typos or zlib.crc32((pid + sentence).encode()) % 20 == 0:   # all typo lines, 5% of clean ones
                    eval_lines.append((pid, typed_line, " ".join(expected), typos))
    return pairs, fixed, clean, merges, case, eval_lines, sentences, stats


def birkbeck():
    rows = Counter()
    for name in ("missp", "holbrook-missp", "aspell", "wikipedia"):
        path = RAW / "birkbeck" / f"{name}.dat"
        if not path.exists():
            continue
        correct = None
        for line in path.read_text(encoding="latin-1").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("$"):
                correct = line[1:].lower()
                continue
            parts = line.split()
            typo = parts[0].lower()
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            if correct and typo != correct and is_word(typo) and is_word(correct) and len(correct) >= 3:
                rows[(typo, correct, name)] += n
    with open(OUT / "birkbeck.tsv", "w", encoding="utf-8") as f:
        for (typo, correct, name), n in sorted(rows.items()):
            f.write(f"{typo}\t{correct}\t{name}\t{n}\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="participants to read (0 = all)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"birkbeck: {birkbeck()} pairs", flush=True)

    with zipfile.ZipFile(ZIP) as z:
        names = [n for n in z.namelist() if n.endswith("_keystrokes.txt")]
    if args.limit:
        names = names[:args.limit]
    chunks = [names[i:i + 200] for i in range(0, len(names), 200)]
    pairs, fixed, clean, merges, case = Counter(), Counter(), Counter(), Counter(), Counter()
    eval_lines, sentences, stats = [], set(), Counter()
    t0 = time.time()
    with Pool(args.workers) as pool:
        for n, (p, fx, c, m, cs, ev, sents, st) in enumerate(pool.imap_unordered(work, chunks), 1):
            pairs.update(p); fixed.update(fx); clean.update(c); merges.update(m); case.update(cs)
            eval_lines.extend(ev); sentences |= sents; stats.update(st)
            if n % 100 == 0:
                print(f"{time.strftime('%H:%M:%S')} {n}/{len(chunks)} chunks, {sum(pairs.values())} typos, {time.time() - t0:.0f}s", flush=True)

    with open(OUT / "keystroke_pairs.tsv", "w", encoding="utf-8") as f:
        for (a, b), n in sorted(pairs.items(), key=lambda kv: -kv[1]):
            f.write(f"{a}\t{b}\t{n}\t{fixed[(a, b)]}\n")
    with open(OUT / "keystroke_clean.tsv", "w", encoding="utf-8") as f:
        for w, n in clean.most_common():
            f.write(f"{w}\t{n}\n")
    with open(OUT / "keystroke_merges.tsv", "w", encoding="utf-8") as f:
        for (a, b), n in merges.most_common():
            f.write(f"{a}\t{b}\t{n}\n")
    with open(OUT / "keystroke_case.tsv", "w", encoding="utf-8") as f:
        for (a, b), n in case.most_common():
            f.write(f"{a}\t{b}\t{n}\n")
    (OUT / "keystroke_sentences.txt").write_text("\n".join(sorted(sentences)) + "\n", encoding="utf-8")
    # A bounded, reproducible sample of the held-out lines per sentence.
    per = Counter()
    eval_lines.sort(key=lambda r: zlib.crc32((r[0] + r[1]).encode()))
    with open(OUT / "keystroke_eval_lines.tsv", "w", encoding="utf-8") as f:
        for pid, typed, expected, typos in eval_lines:
            if per[expected] >= EVAL_LINES_PER_SENTENCE:
                continue
            per[expected] += 1
            f.write(f"{pid}\t{typed}\t{expected}\n")
    total_clean = sum(clean.values())
    print(f"sections {stats['sections']} replay mismatches {stats['replay_mismatch']} misaligned {stats['misaligned']}")
    print(f"typo commits {sum(pairs.values())} ({len(pairs)} distinct), clean commits {total_clean}, "
          f"error rate {sum(pairs.values()) / max(1, total_clean + sum(pairs.values())):.3%}, fixed later {sum(fixed.values())}")
    print(f"merges {sum(merges.values())}, case-only {sum(case.values())}, sentences {len(sentences)}, eval lines {sum(per.values())}")


if __name__ == "__main__":
    main()
