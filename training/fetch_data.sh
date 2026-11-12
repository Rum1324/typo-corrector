#!/bin/sh
# Downloads the raw corpora for the v2 models into training/.cache/data/raw/<name>/ (gitignored).
# Re-runnable: finished files are skipped, partial ones resumed. Nothing here is executed, only read.
#   text:   OpenSubtitles (casual dialogue), WikiText-103 (formal), StackExchange dumps (technical,
#           informal written), Enron mail (work messages), ML arXiv abstracts
#   typos:  Aalto 136M keystrokes (real keyboard slips), Birkbeck misspelling corpora
set -u
cd "$(dirname "$0")"
RAW=.cache/data/raw
UA="curl/8.7"

get() {  # get <dir> <file> <url> [extra curl args]
    dir="$RAW/$1"; file="$2"; url="$3"; shift 3
    mkdir -p "$dir"
    [ -f "$dir/$file.done" ] && { echo "skip $file"; return 0; }
    echo "$(date +%H:%M:%S) fetch $file"
    if curl -fL -sS -A "$UA" -C - --retry 4 --retry-delay 20 "$@" -o "$dir/$file" "$url"; then
        touch "$dir/$file.done"; echo "$(date +%H:%M:%S) ok $file $(du -h "$dir/$file" | cut -f1)"
    else
        echo "$(date +%H:%M:%S) FAILED $file ($url)"; return 1
    fi
}

# Hugging Face rate-limits shared IPs: wait and retry a few times.
get_hf() {
    for attempt in 1 2 3 4 5 6; do
        get "$@" && return 0
        echo "  hf attempt $attempt failed, sleeping 200 s"; sleep 200
    done
    return 1
}

B=https://www.dcs.bbk.ac.uk/~ROGER
for f in missp.dat holbrook-missp.dat aspell.dat wikipedia.dat; do get birkbeck "$f" "$B/$f"; done

SE=https://archive.org/download/stackexchange
for s in datascience ai cs biology stats softwareengineering; do
    get stackexchange "$s.7z" "$SE/$s.stackexchange.com.7z"
done

# First 700 MB of the 3.6 GB gzip stream (about 2.3 GB of text); a gzip prefix decompresses fine.
get opensubtitles en.txt.gz.part "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/en.txt.gz" -r 0-734003199

get keystrokes Keystrokes.zip "https://userinterfaces.aalto.fi/136Mkeystrokes/data/Keystrokes.zip"
get enron enron_mail_20150507.tar.gz "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"

HF=https://huggingface.co/datasets
get_hf wikitext train-00000-of-00002.parquet "$HF/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/train-00000-of-00002.parquet"
get_hf wikitext train-00001-of-00002.parquet "$HF/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/train-00001-of-00002.parquet"
get_hf arxiv ML-Arxiv-Papers.csv "$HF/CShorten/ML-ArXiv-Papers/resolve/main/ML-Arxiv-Papers.csv"
if [ ! -f "$RAW/wikitext/train-00000-of-00002.parquet.done" ]; then
    # Fallback for formal prose when Hugging Face stays closed: Simple English Wikipedia dump.
    get simplewiki simplewiki-latest-pages-articles.xml.bz2 "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
fi
echo "$(date +%H:%M:%S) fetch finished"; du -sh "$RAW"/*
