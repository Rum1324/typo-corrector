"""Builds the text corpus for the context language model from the raw downloads (fetch_data.sh).

  .venv/bin/python -I build_corpus.py [--only opensubtitles,enron] [--workers 10]

Per source it writes, under .cache/data/corpus/:
  <source>.train.u32   flat little-endian uint32 token ids (tok.py), <s> before every line
  <source>.val.u32     same, 0.5% of the lines, for perplexity
  <source>.sim.txt     3% of the lines as cleaned text, never shown to the language model: they are
                       typed through the Swift engine with slips to train and test the decision model
  meta.json            token and line counts per source

The raw files are untrusted input: they are only parsed as text (no XML entity expansion, no
archive extraction to disk), and this script is meant to run with python -I.
"""
import argparse, csv, hashlib, html, io, json, os, re, subprocess, sys, tarfile, time, zlib
from collections import deque
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I drops the script directory
import numpy as np
from tok import Vocab, BOS, UNK, NAME, SYM

HERE = Path(__file__).resolve().parent
RAW = HERE / ".cache" / "data" / "raw"
OUT = HERE / ".cache" / "data" / "corpus"
# Training-token budget per source: casual dialogue, formal prose, technical writing, work mail.
CAPS = {"opensubtitles": 170e6, "wikitext": 110e6, "stackexchange": 150e6, "enron": 60e6, "arxiv": 30e6, "simplewiki": 60e6}
SIM_MAX_CHARS = 300

MATH = re.compile(r"\$\$.*?\$\$|\$[^$\n]{1,200}\$", re.S)
WS = re.compile(r"\s+")

# ---------------------------------------------------------------- cleaning (runs in workers)

SUB_CUES = re.compile(r"\[[^\]]*\]|\([^)]*\)|<[^>]+>|\{[^}]*\}")
SUB_DASH = re.compile(r"^\s*-+\s*")


def clean_opensubtitles(records):
    out = []
    for raw in records:
        s = raw.strip()
        if not s or "♪" in s or "#" in s:
            continue
        s = SUB_DASH.sub("", SUB_CUES.sub(" ", s)).strip()
        if len(s) < 3:
            continue
        letters = [c for c in s if c.isalpha()]
        if len(letters) > 8 and sum(c.isupper() for c in letters) > 0.6 * len(letters):
            continue                                          # captions in capitals
        # A subtitle that continues the previous one ("...that maybe" / "we could go.") is one line.
        if out and s[0].islower() and out[-1][-1] not in ".!?\"…" and len(out[-1]) < 200:
            out[-1] += " " + s
        else:
            out.append(s)
    return out


WIKI_CLITIC = re.compile(r" '(s|t|re|ve|ll|d|m)\b")


def clean_wikitext(records):
    out = []
    for raw in records:
        s = raw.strip()
        if not s or s.startswith("="):
            continue
        s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
        out.append(WIKI_CLITIC.sub(r"'\1", s))
    return out


SE_ATTR = re.compile(r' (?:Body|Text)="([^"]*)"')
SE_PRE = re.compile(r"<pre.*?</pre>|<blockquote>.*?</blockquote>", re.S)
SE_CODE = re.compile(r"<code>.*?</code>", re.S)
SE_BREAK = re.compile(r"</p>|<br\s*/?>|</li>|</h\d>", re.I)
SE_TAG = re.compile(r"<[^>]+>")


def clean_stackexchange(records):
    out = []
    for raw in records:
        m = SE_ATTR.search(raw)
        if not m:
            continue
        h = html.unescape(m.group(1))                         # XML attribute -> HTML (posts) or text (comments)
        h = SE_PRE.sub("\n", h)
        h = SE_CODE.sub(" `code` ", h)
        h = SE_TAG.sub("", SE_BREAK.sub("\n", h))
        h = MATH.sub(" $math$ ", html.unescape(h))
        for p in h.split("\n"):
            p = WS.sub(" ", p).strip()
            if p.count(" ") >= 2:
                out.append(p)
    return out


ENRON_CUT = re.compile(r"^\s*-{2,}\s*(?:Original Message|Forwarded by)|^\s*-{5,}|^(?:From|To|Sent|Subject|cc|Cc):\s|^>|^\s*_{5,}|^\s*\*{5,}", re.M)
PARA = re.compile(r"\n\s*\n")
WORDS = re.compile(r"[a-z']+")
_banned = None   # 6-grams of the Aalto typing-test sentences (they come from Enron mail)


def banned_ngrams():
    global _banned
    if _banned is None:
        _banned = set()
        path = HERE / ".cache" / "data" / "typos" / "keystroke_sentences.txt"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                w = WORDS.findall(line.lower())
                for i in range(len(w) - 5):
                    _banned.add(" ".join(w[i:i + 6]))
    return _banned


def clean_enron(records):
    banned = banned_ngrams()
    out = []
    for raw in records:
        body = raw.partition("\n\n")[2].replace("=\n", "").replace("=20", " ").replace("=09", " ")
        m = ENRON_CUT.search(body)
        if m:
            body = body[:m.start()]
        for p in PARA.split(body):
            s = " ".join(p.split())
            if s.count(" ") < 3 or sum(c.isalpha() or c == " " for c in s) < 0.8 * len(s):
                continue                                      # signatures, tables, addresses
            if banned:
                w = WORDS.findall(s.lower())
                if any(" ".join(w[i:i + 6]) in banned for i in range(len(w) - 5)):
                    continue
            out.append(s)
    return out


def clean_arxiv(records):
    out = []
    for title, abstract in records:
        for s in (title, abstract):
            s = MATH.sub(" $math$ ", " ".join(s.split()))
            if s.count(" ") >= 2:
                out.append(s)
    return out


WIKI_MARKUP = [
    (re.compile(r"\{\{[^{}]*\}\}|\{\|.*?\|\}|<ref[^>]*/>|<ref.*?</ref>|<!--.*?-->|<[^>]+>", re.S), " "),
    (re.compile(r"\[\[(?:File|Image|Category|Datei)[^\]]*\]\]", re.I), " "),
    (re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]"), r"\1"),
    (re.compile(r"\[https?://\S+ ?([^\]]*)\]"), r"\1"),
    (re.compile(r"'{2,}"), ""),
]


def clean_simplewiki(records):
    out = []
    for raw in records:
        text = html.unescape(raw)
        for _ in range(3):                                    # nested templates
            for rx, rep in WIKI_MARKUP[:1]:
                text = rx.sub(rep, text)
        for rx, rep in WIKI_MARKUP[1:]:
            text = rx.sub(rep, text)
        for p in text.split("\n"):
            p = WS.sub(" ", p).strip()
            if p.count(" ") >= 5 and p[0].isalpha() and "|" not in p and "{" not in p:
                out.append(p)
    return out


CLEAN = {"opensubtitles": clean_opensubtitles, "wikitext": clean_wikitext, "stackexchange": clean_stackexchange,
         "enron": clean_enron, "arxiv": clean_arxiv, "simplewiki": clean_simplewiki}
_vocab = None


def sim_line(line):
    """A held-out line short enough to type: long paragraphs are cut after a sentence (or a word)."""
    if len(line) <= SIM_MAX_CHARS:
        return line
    head = line[:SIM_MAX_CHARS]
    cut = max(head.rfind(". "), head.rfind("? "), head.rfind("! "))
    return head[:cut + 1] if cut >= 60 else head[:head.rfind(" ")]


def process(task):
    """(source, records) -> (train ids, val ids, sim lines, kept lines)."""
    global _vocab
    if _vocab is None:
        _vocab = Vocab()
    source, records = task
    train, val, sim, kept = [], [], [], 0
    for line in CLEAN[source](records):
        ids = _vocab.encode(line)
        if len(ids) < 2:
            continue
        junk = sum(1 for t in ids if t == UNK or t == NAME or t == SYM)
        if junk > 0.4 * len(ids):                             # not English prose
            continue
        kept += 1
        h = zlib.crc32(line.encode("utf-8", "ignore")) % 1000
        if h < 5:
            val.append(BOS); val.extend(ids)
        elif h < 35:
            sim.append(sim_line(line))
        else:
            train.append(BOS); train.extend(ids)
    return np.asarray(train, dtype="<u4"), np.asarray(val, dtype="<u4"), sim, kept


# ---------------------------------------------------------------- readers (run in the parent)

def batched(it, n):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def read_opensubtitles(keep_percent=40):
    """Decompresses the gzip prefix and keeps a spread of 20k-line blocks (the file is ordered by film)."""
    path = RAW / "opensubtitles" / "en.txt.gz.part"
    d = zlib.decompressobj(31)
    tail = b""

    def lines():
        nonlocal tail
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 22)
                if not chunk:
                    break
                try:
                    data = tail + d.decompress(chunk)
                except zlib.error:
                    break
                parts = data.split(b"\n")
                tail = parts.pop()
                for p in parts:
                    yield p.decode("utf-8", "ignore")

    for i, batch in enumerate(batched(lines(), 20_000)):
        if zlib.crc32(str(i).encode()) % 100 < keep_percent:
            yield batch


def read_wikitext():
    import pyarrow.parquet as pq
    for path in sorted((RAW / "wikitext").glob("*.parquet")):
        for rb in pq.ParquetFile(path).iter_batches(batch_size=20_000, columns=["text"]):
            yield rb.column("text").to_pylist()


def read_stackexchange():
    for archive in sorted((RAW / "stackexchange").glob("*.7z")):
        for member in ("Comments.xml", "Posts.xml"):
            proc = subprocess.Popen(["7zz", "e", "-so", str(archive), member], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            rows = (line for line in io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="ignore") if line.lstrip().startswith("<row"))
            yield from batched(rows, 4_000)
            proc.wait()


def read_enron():
    seen = set()

    def mails():
        with tarfile.open(RAW / "enron" / "enron_mail_20150507.tar.gz", mode="r|gz") as tar:
            for member in tar:
                if not member.isfile() or member.size > 200_000:
                    continue
                f = tar.extractfile(member)                   # read in memory, nothing is written to disk
                if f is None:
                    continue
                text = f.read().decode("latin-1")
                key = hashlib.md5(text.partition("\n\n")[2][:400].encode("latin-1", "ignore")).digest()
                if key in seen:
                    continue                                  # the same mail sits in several folders
                seen.add(key)
                yield text

    yield from batched(mails(), 2_000)


def read_arxiv():
    csv.field_size_limit(1 << 24)
    with open(RAW / "arxiv" / "ML-Arxiv-Papers.csv", newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        ti, ai = header.index("title"), header.index("abstract")
        rows = ((r[ti], r[ai]) for r in reader if len(r) > max(ti, ai))
        yield from batched(rows, 2_000)


def read_simplewiki():
    import bz2
    text_open = re.compile(r"<text[^>]*>")

    def pages():
        buf, inside = [], False
        with bz2.open(RAW / "simplewiki" / "simplewiki-latest-pages-articles.xml.bz2", "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not inside:
                    m = text_open.search(line)
                    if not m:
                        continue
                    line, inside = line[m.end():], True
                if "</text>" in line:
                    buf.append(line.split("</text>")[0])
                    yield "".join(buf)
                    buf, inside = [], False
                else:
                    buf.append(line)

    yield from batched(pages(), 500)


READ = {"opensubtitles": read_opensubtitles, "wikitext": read_wikitext, "stackexchange": read_stackexchange,
        "enron": read_enron, "arxiv": read_arxiv, "simplewiki": read_simplewiki}
PRESENT = {"opensubtitles": "opensubtitles/en.txt.gz.part.done", "wikitext": "wikitext/train-00001-of-00002.parquet.done",
           "stackexchange": "stackexchange/ai.7z.done", "enron": "enron/enron_mail_20150507.tar.gz.done",
           "arxiv": "arxiv/ML-Arxiv-Papers.csv.done", "simplewiki": "simplewiki/simplewiki-latest-pages-articles.xml.bz2.done"}


def build(source, pool, workers):
    cap = CAPS[source]
    t0 = time.time()
    counts = {"train": 0, "val": 0, "sim_lines": 0, "lines": 0}
    pending = deque()
    with open(OUT / f"{source}.train.u32", "wb") as ftrain, open(OUT / f"{source}.val.u32", "wb") as fval, \
            open(OUT / f"{source}.sim.txt", "w", encoding="utf-8") as fsim:

        def drain(block):
            while pending and (block or pending[0].ready()):
                train, val, sim, kept = pending.popleft().get()
                train.tofile(ftrain); val.tofile(fval)
                for line in sim:
                    fsim.write(line + "\n")
                counts["train"] += len(train); counts["val"] += len(val)
                counts["sim_lines"] += len(sim); counts["lines"] += kept
                block = False

        for batch in READ[source]():
            if counts["train"] >= cap:
                break
            pending.append(pool.apply_async(process, ((source, batch),)))
            drain(block=len(pending) >= workers * 3)          # bounded queue: readers outrun the workers
        while pending and counts["train"] < cap:
            drain(block=True)
    print(f"{time.strftime('%H:%M:%S')} {source}: {counts['train'] / 1e6:.1f}M train tokens, {counts['val'] / 1e6:.2f}M val, "
          f"{counts['sim_lines']} sim lines, {counts['lines']} lines, {time.time() - t0:.0f}s", flush=True)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta_path = OUT / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    wanted = [s for s in args.only.split(",") if s] or list(READ)
    with Pool(args.workers) as pool:
        for source in wanted:
            if not (RAW / PRESENT[source]).exists():
                print(f"skip {source}: not downloaded", flush=True)
                continue
            meta[source] = build(source, pool, args.workers)
            meta_path.write_text(json.dumps(meta, indent=1))
    print("total train tokens: %.1fM" % (sum(m["train"] for m in meta.values()) / 1e6))


if __name__ == "__main__":
    main()
