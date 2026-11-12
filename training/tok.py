"""Word tokenizer for the context language model.

Must stay in sync with Sources/TypoCore/ContextTokenizer.swift: the app tokenizes the line it has
tracked the same way before asking the model which word fits. The vocabulary is the lexicon
(Resources/words_en.txt, in file order) after 16 reserved ids.
"""
from pathlib import Path

PAD, BOS, CUT, UNK, NAME, NUM, DOT, BANG, QM, COMMA, COLON, SYM = range(12)
SPECIALS = ["<pad>", "<s>", "<cut>", "<unk>", "<name>", "<num>", "<.>", "<!>", "<?>", "<,>", "<:>", "<sym>"]
N_SPECIAL = 16
# A whitespace-delimited chunk with one of these is a link, path, identifier, markup or maths.
LINK = frozenset("/\\@_#=~<>`|{}[]$%^*+&")
_END = {".": DOT, "!": BANG, "?": QM}

LEXICON = Path(__file__).resolve().parents[1] / "Resources" / "words_en.txt"


class Vocab:
    def __init__(self, path=LEXICON):
        self.words = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                self.words.append(line.split("\t", 1)[0])
        self.index = {w: i + N_SPECIAL for i, w in enumerate(self.words)}
        assert len(self.index) == len(self.words), "duplicate lexicon entries would shift the Swift ids"
        self.size = N_SPECIAL + len(self.words)

    def token(self, i):
        return SPECIALS[i] if i < len(SPECIALS) else (self.words[i - N_SPECIAL] if i >= N_SPECIAL else "<reserved>")

    def word_id(self, word, capitalised=False):
        """Id of a finished word (letters and apostrophes): the lexicon id, else <name> / <unk>."""
        i = self.index.get(word.lower())
        return i if i is not None else (NAME if capitalised else UNK)

    def encode(self, text):
        """Token ids of one line of text (no <s>)."""
        out = []
        get = self.index.get
        for chunk in text.replace("’", "'").replace("‘", "'").split():
            if chunk.isascii() and chunk.isalpha():          # the common case
                i = get(chunk.lower())
                out.append(i if i is not None else (NAME if chunk[0].isupper() else UNK))
            else:
                self._chunk(chunk, out)
        return out

    def _chunk(self, chunk, out):
        n = len(chunk)
        for ch in chunk:
            if ch in LINK:
                out.append(SYM)
                return
        # A dot joining letters or a letter and a digit: "file.txt", "e.g.", "youtu.be" (not "3.5").
        for k in range(1, n - 1):
            if chunk[k] == "." and chunk[k - 1].isalnum() and chunk[k + 1].isalnum() \
                    and (chunk[k - 1].isalpha() or chunk[k + 1].isalpha()):
                out.append(SYM)
                return
        i = 0
        while i < n:
            c = chunk[i]
            if c.isalpha() or c == "'":
                j = i
                while j < n and (chunk[j].isalnum() or chunk[j] == "'"):
                    j += 1
                run = chunk[i:j].strip("'")
                i = j
                if not run:
                    continue
                if not run.replace("'", "").isalpha():       # "h264enc", "v2"
                    out.append(SYM)
                elif run.isascii():
                    low = run.lower()
                    w = self.index.get(low)
                    if w is None and low.endswith("'s"):     # "model's" counts as "model"
                        w = self.index.get(low[:-2])
                    out.append(w if w is not None else (NAME if run[0].isupper() else UNK))
                else:                                        # accented or non-Latin word
                    out.append(NAME if run[0].isupper() else UNK)
            elif c.isdigit():
                j = i
                while j < n and (chunk[j].isalnum() or (chunk[j] in ".,:" and j + 1 < n and chunk[j + 1].isdigit())):
                    j += 1
                out.append(NUM)                              # "3.5", "10,000", "12:30", "3rd"
                i = j
            elif c in _END:
                out.append(_END[c])
                while i < n and chunk[i] in _END:            # "...", "?!" count once
                    i += 1
            elif c == ",":
                out.append(COMMA)
                i += 1
            elif c == ";" or c == ":":
                out.append(COLON)
                i += 1
            else:                                            # quotes, brackets, dashes, emoji
                i += 1

    def decode(self, ids):
        return " ".join(self.token(int(i)) for i in ids)


if __name__ == "__main__":
    import sys
    v = Vocab()
    for line in sys.argv[1:] or ["Hey, are you free tonight? I'm at 3.5 GPA... see github.com/user/repo or file.txt (Dr. Haruto's h264enc)"]:
        print(v.decode(v.encode(line)))
