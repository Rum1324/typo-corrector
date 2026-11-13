# Typo

A macOS menu-bar app that fixes spelling slips as you type, in every app, and learns your
own typos from the way you fix them. Everything runs and stays on your Mac.

```
you type:  "I recieve teh message␣"
you see:   "I receive the message "
delete a fix and type your own word again → it stays, and Typo holds that fix back next time
```

## How it works

| Layer | What it does | Where |
|---|---|---|
| Event tap | Sees every keystroke and mouse click system-wide; swallows the space/return that ends a misspelled word and re-posts it after the fix | `Sources/Typo/KeyTap.swift`, `Engine.swift` |
| Policy | Which finished words are analysed at all (trigger keys, first word after a click, links/paths/identifiers), shared by the app and the accuracy suite | `AutocorrectPolicy.swift` |
| Text tracker | Rebuilds the current line from keystrokes, notices when you delete a word and retype it differently (a manual fix) | `Sources/TypoCore/TextTracker.swift` |
| Personal model | Counts your (typo → fix) pairs, undone corrections, and non-dictionary words you type on purpose. Two identical fixes become an automatic correction. Each pair keeps a running tally of fixes you kept and fixes you undid (newer ones weigh more): undos hold a fix back in proportion to their share, and block a word only while they outnumber the kept fixes | `PersonalModel.swift`, `Store.swift` (SQLite) |
| Candidate generation | SymSpell delete-index over 80k English words (wordfreq), distance ≤ 2 | `SymSpell.swift` |
| Ranking | `log P(word)` + keyboard-aware edit cost + neural pair score + personal boost, softmax over the candidates **and** "keep the word as typed", confidence threshold. Also offers splits of merged words (`inthe` → `in the`) and fixes shift slips (`THe` → `The`) | `Corrector.swift`, `EditDistance.swift` |
| Context language model | 7.8M-parameter word transformer over the lexicon, trained from scratch on 450M words (subtitles, Wikipedia, Stack Exchange, email, arXiv). Says how well each candidate fits the words before it on the line. Inference is hand-written Swift, about 2.7 ms per finished word | `ContextLM.swift`, `ContextTokenizer.swift`, `training/lm.py` |
| Decider | Three small networks (averaged) that weigh the language model against the slip's likelihood for every option, "keep the word" included. Trained on 1.27M simulated typed words whose numbers were written by the Swift engine itself, so training and the app see the same inputs | `ContextModel.swift`, `training/decider.py` |
| Error model | Edit rates by position and word length fitted on 402,912 real typos (Aalto keystroke study) plus real misspellings (Birkbeck); used to simulate typing and to train the context path's pair ranker | `training/noise.py` |
| Word-form evidence | Counts of 266,870 word forms in the training text and 227,583 rarer words from wordfreq: `backprop`, `timesteps` and names the lexicon lacks are known to exist | `WordForms.swift`, `training/build_forms.py` |
| Curated word list | macOS's `/usr/share/dict/web2` tells rare real words (`mien`, `tarn`) from web misspellings that are in the frequency list (`befor`, `somthing`) | `CuratedWords.swift` |
| Neural ranker | 72k-parameter character transformer scoring (typo, candidate). Trained on synthetic slips + Wikipedia's curated misspellings; fine-tuned nightly on your pairs. Inference is hand-written Swift (Accelerate), ~20 µs per candidate | `NeuralRanker.swift`, `training/` |
| Injector | Backspaces the typo, types the fix as a Unicode string, re-posts your boundary key. All injected events are tagged and counted until they pass the tap; a real key that arrives meanwhile is held and re-posted behind them, so fast typing never lands inside a fix | `Injector.swift`, `InjectionGate.swift` |
| Trainer | When idle ≥ 10 min, on AC power, ≤ once a day and with ≥ 10 new examples, runs `train.py --personal` at nice 19 and hot-reloads the weights | `Trainer.swift` |

Corrections fire on space and return. After `. , ! ? ; :` the fix waits for the next key and
is applied only if that key is space or return, so `youtu.be`, `file.txt` and `e.g.` are left
alone. The word must be 3–24 ASCII letters (1–24 with the context path; tokens with digits like
`h264enc` are never touched), not ALL CAPS, not camelCase, and (unless learned) not Capitalised
mid-sentence.

### Two paths
With the model files in `Resources/` (`lm.*`, `ranker2.*`, `context.json`, `forms_en.tsv`,
`words_more_en.tsv`) and the text before the word known, the **context path** decides: every near
word is an option, dictionary words included, and the decider needs to be sure. Without the files,
for merged words (`inthe`), and as a second opinion, the **rules** below decide as before.
The context path's own rails (thresholds in `CorrectorConfig`, tuned on dev):
- an unknown word is replaced at ≥ 0.90; a dictionary word (`form` → `from`) only at ≥ 0.97;
  the first word after a click at ≥ 0.95; a capitalised unknown word at ≥ 0.95, and at ≥ 0.995 when
  it reads as romaji (`Yuu`, `Haruto`);
- when the rules would make the same replacement on their own, 0.70 is enough (two independent
  judgements agree);
- a word the curated dictionary lists, or a regular inflection of a known word, needs ≥ 0.98;
- the decider never adds an apostrophe (`its` stays as typed), a lone `i` is never changed, and
  stretched words (`sooo`) are left to the decider, which has seen them as intended.

Two settings in the menu change this (both are off in `CorrectorConfig`, which the tests and the
tables below use):
- **How eagerly to correct** lowers every bar above except the romaji one by a number of
  log-odds (`contextLooseness`; the name bar moves half as far). Test split, 2026-11-12, with
  apostrophes on; "correct words changed" is over all 49,228 correct words and leaves apostrophe
  fixes out:

  | Level | Looseness | Real typing fixed | Classic sets fixed | Real-word slips fixed | Wrong fixes (5,756 typos) | Correct words changed |
  |---|---|---|---|---|---|---|
  | Careful | 0 | 55.7% | 84.8% | 18.6% | 50 | 16 |
  | Balanced | 0.5 | 58.9% | 87.1% | 23.4% | 63 | 20 |
  | Eager (the app's default) | 1.0 | 61.5% | 88.9% | 29.0% | 83 | 31 |
  | Very eager | 1.5 | 64.2% | 89.6% | 35.2% | 92 | 44 |
  | Maximum | 2.0 | 66.2% | 90.4% | 39.3% | 116 | 74 |

  The levels were picked on dev and only read off on test. Any value can be set with
  `defaults write <bundle id> looseness -float 0.75`.
- **Add missing apostrophes** (on in the app) fixes a fixed list of 47 contractions whose bare
  form is not a word of its own (`doesnt` → `doesn't`, `im` → `I'm`; `Corrector.contractions`),
  without asking the models. `its`, `were`, `well`, `ill` and the like are not on the list. The
  chat test sets count these fixes as errors, because they treat `dont` as a style.

### Safety rails against wrong corrections
- Dictionary words are only "typos" if they are on the curated misspelling list (3,921 entries
  from Wikipedia, each replaced only by its listed fix), or rarer than 0.5 per million with a
  distance-1 neighbour 500× more common reachable by a mechanical slip (adjacent key,
  transposition, doubled letter). `comet`, `sears`, `untied`, `tome` stay.
- A second edit costs 2.5 nats extra; two non-adjacent substitutions are never applied.
- Winner needs ≥ 60% of the candidate probability mass.
- Password fields (`AXSecureTextField`, secure input mode) are checked *before* anything is
  learned or counted; blocklisted apps (terminals, IDEs, password managers, remote desktops and
  VMs by default) are never touched. Option+Delete and any caret movement reset the tracker.
- After a reset (click, arrow key, app switch, shortcut) the caret may have been inside a longer
  word, and "fixing" its tail (`eople` → `people`) would leave `ppeople`. So the first word is
  corrected only if no common word ends with it (`teh`, `waht` are fixed; `ple`, `ing`, `nally`
  are left alone). A capitalised first word there may be a name: keeping it gets a bonus.
  Held-key repeats and the accent menu also reset the tracker.
- Words inside URLs, paths, emails, hashtags, mentions and identifiers (anything after `/ @ _ #
  . = ~` etc. since the last space) are left alone.
- Unknown words compete with "you meant it" (a name, jargon, romaji, slang), so `gyoza`,
  `refactor`, `Haruto`, `tldr` stay; words that read as romaji get extra protection. Correct
  inflections of known words (`embeddings`, `annotating`, `rerunning`) count as intended;
  misspelled ones (`forgeting`) and fake prefixes (`prefrom`) do not.
- A capitalised dictionary word that is not in the curated list is taken for a name
  (`Persson`, `Yuu`) and left alone, at the start of a sentence too.
- Inflected real words (`sways`, `culls`, `doling`) count as curated through their base form.
- First-letter slips are fixed when clear (`slso` → `also`), but a dictionary word is never
  changed to one with another first letter (`vars` stays).
- Shift slips are fixed only when the word is common (`THe` → `The`); acronyms with a suffix
  (`DMed`, `TAing`, `IDs`, `BEd`) are left alone.
- Merged words are split only when the joined form is unknown or ends in "the" (`ofthe`), never
  for capitalised words, code words (`redis`, `goto`, `init`) or dropped g's (`headin`).
- Before applying a fix that waited on punctuation, Typo reads the text before the caret over
  Accessibility; if the app's own autocorrect already changed the word, Typo leaves it alone.
- Tracking pauses while a composing input method (Japanese, Chinese, Korean) is active; the
  Roman/ABC mode of those IMEs is treated as Latin.
- Rejecting a fix: delete it and type something else, or type your own word back (that word is
  then left alone, not fixed again). "Backspace undoes the last fix" in the menu is off by
  default; when on, any backspace within 15 s of a correction puts the typed word back, and it
  counts as a rejection only if the next key types a character (another backspace means you were
  rewriting, and nothing is learned). It is off because a backspace meant to go back and fix the
  word by hand looks the same. Return is never undoable (in chat apps the message has already
  been sent).
- What a rejection does (`PersonalModel.swift`): each (typed word → fix) pair has a tally of kept
  and undone fixes in which every new outcome shrinks the older ones by 0.85. An undo or a fix
  made by hand counts 1; an automatic fix that was not undone counts 0.5, because a rejection
  after a click or a shortcut goes unseen. The fix loses 3 × undone² / (undone + kept + 1) from
  its score: 1.5 for one undo with no history (it then needs about 98% confidence instead of
  90%), about 0.6 for one undo after many kept fixes. A word is left alone only while its undone
  tally is at least 1.5 (two undos in a row) and at least its kept tally: a fix undone a third of
  the time ends up blocked, one undone a fifth of the time does not. Fixing the word by hand
  brings it back. A learned fix of a real word (form → from) stops being applied without looking
  at the sentence after one undo, until about five more fixes have been kept.

## Accuracy

### Context path (2026-11-11)

Rules alone vs. rules + context path, bundled weights, no personal data. `dev` was tuned on,
`test` was checked a few times (thresholds were tightened once after the first check), `holdout`
was run once at the end; its failures were never read. (A re-run after a later clean-up of
the case files gave identical counts.)

| split | set | typos | fixed: rules → context | wrong fixes | false fixes on correct words |
|---|---|---|---|---|---|
| holdout | classic sets | 900 | 79.2% → **87.1%** | 9 → 1 | 10 → 8 of 16,336 |
| holdout | real-word slips | 137 | 0% → 21.9% | 0 → 0 | 0 → 1 of 2,603 |
| test | classic sets | 1,298 | 79.4% → 84.8% | 18 → 13 | 14 → 17 of 19,955 |
| test | real typing | 4,313 | 35.4% → **54.6%** | 72 → 32 | 0 → 0 of 26,274 |
| test | real-word slips | 145 | 0% → 18.6% | 0 → 0 | 0 → 0 of 2,999 |
| dev | classic sets | 1,291 | 82.6% → 86.7% | 6 → 7 | 18 → 12 of 18,752 |
| dev | real typing | 4,723 | 35.2% → 53.2% | 86 → 42 | 1 → 2 of 29,265 |
| dev | real-word slips | 141 | 0% → 26.2% | 0 → 0 | 0 → 0 of 2,813 |

New sets:
- **Real typing** (`keystrokes.*`): first-attempt lines of transcription typists from the Aalto
  136M Keystrokes dataset (Dhakal et al., CHI 2018), from sentences held out of every training
  table. Non-commercial licence with attribution: leave these files out of a commercial release.
  Most real typos are short words (`th`, `yo`, `t`) and real-word slips, which the rules skip.
- **Real-word slips** (`realword.*`) and their correct counterparts (`clean-realword.*`): written
  by three independent agents (one per split, different domains), each checked by a second one.
  Two authors briefly saw each other's list of slip pairs (not sentences) through a shared
  scratch file.

Honest caveats:
- First-letter slips got worse: `firstletter.test` 78.0% → 61.3% fixed. Real typists rarely slip
  on the first letter (they drop it in 0.06% of 5-6 letter words, the last letter in 0.95%), so the
  model learned to doubt such fixes; the set over-represents them.
- Real-word slips are fixed only when the words *before* them give them away and the decider is
  very sure: about one in five. Nothing wrong was put in their place.
- In these tables `cant`, `dont`, `im` are not given apostrophes (143 of the dev real-typing
  misses): in chat they are often typed on purpose. The app adds them from a fixed list unless
  "Add missing apostrophes" is turned off.
- False fixes are at the old level, not below it (test: 17 vs 14).
- On a private set of 29 typos from the author's own typing (`real-user.test`, counted in the
  test rows above but not included in this repository): 51.7% → 58.6% fixed, wrong 13.8% → 10.3%.
  A clone of this repository therefore measures 1,269 classic test typos: 80.1% → 85.4% fixed,
  wrong fixes 14 → 10, false fixes 14 → 17 of 19,864.
- The context ranker is not fine-tuned on your pairs yet; your learned pairs, undos and accepted
  words still apply to both paths.

### Rules alone (2026-11-09)

`eval/cases/` holds about 16,000 typed lines (7,100 before the real-typing and real-word sets). The original 12 use cases are casual chat,
Obsidian-style technical notes, formal writing, dropped letters, spelling mistakes,
capitalisation and shift slips, the first word after a click, and five clean sets: slang,
code/URLs, names and romaji, British/Australian spellings and rare real words. A private set built
from the author's own fixes and undos (`real-user.test.tsv`) is scored when present; it is not in
this repository. A code review added more: names that start a message,
word fragments after a click, plurals and inflections, abbreviations with suffixes, first-letter
slips, and after-click variants. Each line is typed key by key through the same tracker and policy
as the app, and every token is scored: fixed / wrong fix / missed for typos, false fix for correct
words.

The original cases were written by independent agents and checked by a second one, in three
splits by different authors and topics. Settings are tuned on `dev` only (objective: fixes minus
3 × (wrong fixes + false fixes)) and checked on `test`; `holdout` was generated after the first
tuning and never tuned on. Holdout, 2026-11-09, bundled ranker, no personal data:

| minProbability | typos fixed | wrong fix | false fixes on 16,336 correct words |
|---|---|---|---|
| before (`--legacy`) | 78.1% | 3.4% | 134 (0.82%) |
| **0.8 (default)** | 79.2% | 1.0% | 10 (0.06%) |
| 0.7 | 83.0% | 1.8% | 14 |
| 0.6 | 84.3% | 2.2% | 19 |

What changed: wrong fixes −70% and false fixes on names, jargon, romaji, British spellings and
rare words −90%; first word after a click 19% → 65% fixed; shift slips (`WHat`) 68% → 88%;
merged words (`inthe`) 0% → 7 of 15. Honest caveats:

- On typo words that never appear in `dev`, the fix rate did **not** improve (holdout 79.0% →
  77.6%, test 83.1% → 77.3%); the gain there is precision (wrong fixes 4.1% → 1.2%). The default
  trades some fixes for far fewer bad ones; `defaults write io.github.rum1324.typo minProbability 0.7`
  fixes more at a small cost (table above).
- On the private set of 29 real typos, 0.8 fixes 52% (legacy 66%),
  0.6 fixes 66%; wrong fixes 14% vs legacy 17%.
- The rules do not fix real-word errors (`form` for `from`); the context path does, cautiously.
- On the old synthetic benchmark (`typo-cli eval`, random slips) fewer typos are fixed (80% → 62%)
  and wrong fixes drop (7.8% → 2.4%): the default is cautious with garbled rare words.

```sh
swift build -c release
.build/release/typo-cli suite --split dev          # per-use-case table + eval/reports/failures-dev.tsv
.build/release/typo-cli suite --split dev --no-context --threads 4   # the rules alone
.build/release/typo-cli suite --split holdout --legacy
.build/release/typo-cli correct form --before "I got an email"       # a decision with context
.build/release/typo-cli suite --cfg minprob=0.7     # try settings (keys: see main.swift)
swift test -c release -Xswiftc -enable-testing --filter AccuracyTests   # includes a regression guard
```

## Build and run

```sh
swift build -c release
.build/release/typo-cli correct teh recieve Seperate form      # engine without permissions
.build/release/typo-cli eval --n 3000                           # synthetic accuracy + false-fix rate
scripts/make_app.sh && open dist/Typo.app                       # menu-bar app
```

First launch asks for **Accessibility** (to see and swallow keys) and **Input Monitoring**.
`make_app.sh` signs with your Apple Development identity so those grants survive rebuilds. The
bundle identifier is `io.github.rum1324.typo`; put `BUNDLE_ID=...` in `scripts/local.env` (not in
git) to use your own. `scripts/make_release.sh` builds `dist/release/Typo.zip` for sharing: no
local paths in the binary and an ad-hoc signature that names nobody.

## Training

```sh
training/setup.sh                                    # uv venv with torch, wordfreq, rapidfuzz
cd training
.venv/bin/python build_dictionary.py                 # Resources/words_en.txt
.venv/bin/python build_misspellings.py wiki.txt      # Resources/misspellings_en.txt (see file header)
.venv/bin/python train.py --base                     # base ranker → Resources/ranker.{json,bin}
.venv/bin/python train.py --personal --db ~/Library/Application\ Support/Typo/typo.sqlite \
                          --out ~/Library/Application\ Support/Typo
```

Context path (each step is resumable; the two marked *night* load the GPU for hours and must not
run while the Mac is in use: `run_batches.sh` trains only between the hours given, never on battery):

```sh
cd training
./fetch_data.sh                                      # 4.3 GB of public text and typing data → .cache
.venv/bin/python -I build_corpus.py                  # tokenised training text (450M words)
.venv/bin/python -I build_typos.py                   # real typo tables from the keystroke study
.venv/bin/python -I noise.py --fit                   # error model → noise.json
./run_batches.sh 80000 1 7 0 ".venv/bin/python -I lm_hidden.py"   # night: language model, 3.5 h
.venv/bin/python -I train.py --base --realistic --d 96 --layers 3 --ff 192 --epochs 6 \
        --n 600000 --out runs/pair_v2_big            # night: context pair ranker, 40 min
.venv/bin/python -I build_forms.py && .venv/bin/python -I build_forms.py --export
.venv/bin/python -I build_dictionary.py --more       # Resources/words_more_en.tsv
.venv/bin/python -I build_instances.py && .venv/bin/python -I lm_hidden.py
.venv/bin/python -I build_typed.py --n 1272275 --out .cache/data/instances/typed_all.tsv
../.build/release/typo-cli features --in .cache/data/instances/typed_all.tsv \
        --hidden .cache/data/instances/hidden.npy --out .cache/data/decider_all --threads 8
.venv/bin/python -I decider.py --data .cache/data/decider_all --typed .cache/data/instances/typed_all.tsv \
        --hidden 128 64 --epochs 20 --members 3      # decider → runs/decider/context.json (minutes, CPU)
.venv/bin/python -I parity.py                        # then: typo-cli tok-parity / lm-parity (Swift = PyTorch)
```
Copy `lm.json`, `lm.bin`, the ranker as `ranker2.json` / `ranker2.bin` and `context.json` into
`Resources/`. The lexicon's line order is the language model's vocabulary: do not rebuild
`words_en.txt` without retraining it.

The app runs the `--personal` command for you when the Mac is idle. Personal weights in
`~/Library/Application Support/Typo/` take precedence over the bundled base weights.
Training log: `~/Library/Application Support/Typo/train.log`.

## What is stored

`~/Library/Application Support/Typo/typo.sqlite`
- `fixes`: time, app bundle id, typo, fix, up to 3 preceding words, source (`manual` = you
  fixed it, `auto` = we fixed it, `undo` = you rejected our fix)
- `unknown_words`: counts of non-dictionary words you typed and left alone (names, jargon),
  so they stop being candidates for correction after 3 sightings
- no keystroke log, no text of what you typed

## Layout

```
Sources/TypoCore   engine (pure Swift, testable, no AppKit)
Sources/Typo       menu-bar app
Sources/typo-cli   correct / eval / simulate / bench
Tests/             XCTest for tracker, detector, engine, store, accuracy suite
eval/cases/        accuracy suite cases (<use case>.<dev|test|holdout>.tsv); eval/reports/ failure dumps
Resources/         words_en.txt, misspellings_en.txt, ranker.json + ranker.bin (rules);
                   lm.json + lm.bin, ranker2.json + ranker2.bin, context.json, forms_en.tsv,
                   words_more_en.tsv (context path, 38 MB)
training/          PyTorch trainer and data builders
scripts/           make_app.sh, make_release.sh
```

## Data sources

Nothing in this repository comes from anyone's private typing. The models and word lists were
built from public data, and each source has its own terms:

- Word frequencies: [wordfreq](https://github.com/rspeer/wordfreq) (`words_en.txt`, `words_more_en.tsv`).
- Curated misspellings: Wikipedia's list of common misspellings (`misspellings_en.txt`).
- Language-model text and word-form counts (`lm.bin`, `forms_en.tsv`): OpenSubtitles 2018 (OPUS),
  WikiText-103, Simple English Wikipedia, the Stack Exchange data dump, the Enron email corpus and
  the ML-ArXiv-Papers abstracts.
- Real typos: the Aalto 136M Keystrokes dataset (Dhakal, Feit, Kristensson and Oulasvirta,
  "Observations on Typing from 136 Million Keystrokes", CHI 2018), which is free for
  non-commercial use with attribution. It is the source of `eval/cases/keystrokes.*.tsv`,
  `training/noise.json` and the typos `ranker2` and the decider were trained on.
- Real misspellings: the Birkbeck spelling error corpora (Roger Mitton).

The evaluation sentences in `eval/cases` other than `keystrokes.*` were written for this project;
the people in them are invented.

## Known limits
- Chromium apps (Slack, VS Code, Dia) accept the injected keystrokes but expose nothing via
  Accessibility, so secure-field detection there relies on macOS secure-input mode only.
- Moving the caret with the mouse or arrow keys resets the tracker; fixes made by selecting
  text and retyping are not learned.
- Sentence context only looks backwards: the words after a slip are not used, so `their`/`there`
  and most real-word slips are still missed.
- A mouse click in the few milliseconds while a fix is being typed is not held back like keys
  are, and can move the caret before the fix lands.
- The first word typed after a click or app switch is not corrected when a common word ends
  with it (it may be the rest of a word you clicked into).
- Truncations with several readings (`contr` → control or contract?) are often missed or fixed to
  the commoner word.
- Dictionary-listed merged words with a one-letter part (`alot`) are not split.

## In plain words

### How the system works

Typo sits in the menu bar and watches the keys you press. It keeps track of the line you are
typing. Each time you finish a word with space or return, it asks one question: was that a slip,
and if so, what did you mean?

It answers in four steps.

1. It looks up nearby words. From a list of 80,000 English words it picks the ones that are one
   or two keystrokes away from what you typed. For `teh` that gives `the`, `tea`, `ten` and a few
   more. "Leave the word as it is" is always one of the options.
2. It reads the sentence so far. A small language model looks at the words before the one you
   just typed and says how well each option would fit there. After "I went to", `the` fits and
   `tea` does not.
3. It judges the slip itself. How likely is it that someone who meant `the` would type `teh`?
   Swapping two letters happens all the time. Hitting a key on the far side of the keyboard does
   not. These odds were counted from real typing mistakes.
4. It decides, and it is cautious. A last small model puts the three things together and gives
   every option a probability. Typo changes the word only when one option is far ahead: 90% sure
   for a word that is not in the dictionary, 97% for a real word such as `form` that you may have
   meant. Otherwise it leaves your typing alone. A typo left in costs little. A word changed
   against your will costs a lot. How sure it has to be is yours to set: "How eagerly to
   correct" in the menu has five levels, from careful to maximum.

When it does change a word, it deletes what you typed and types the fix, faster than you can
see. If you did mean what you typed, delete the fix and type your word again: it stays. Typo
weighs the fixes you keep against the ones you take back, so one stray rejection of a fix you
usually want changes little, and a fix you reject every time stops after two.

It also learns from you. When you delete a word and retype it, Typo remembers the pair, and
after two identical fixes it makes that fix for you. Words you keep typing that are not in the
dictionary, such as names and jargon, are taken as meant.

It stays out of places where a fix would do harm: password fields, terminals, code editors,
links, file names, and text typed with a Japanese, Chinese or Korean input method. Nothing
leaves your Mac. The only things saved are your typo and fix pairs with up to three words
before them.

### How the model was trained

There is not one model but four small ones, built one after the other. All of it was done on
one Apple-silicon laptop. No cloud service was used, and no personal typing
went into the models that ship with the app.

**The language model** is the part that reads the sentence. It was trained from scratch, not
downloaded, on about 450 million words of public English text: film and TV subtitles, Wikipedia,
Stack Exchange, the Enron emails and abstracts of machine-learning papers from arXiv. The mix
is deliberate. Subtitles are close to chat,
Stack Exchange is close to technical notes, and emails are close to emails. The training task is
the usual one: read the start of a line, guess the next word, see how wrong the guess was,
adjust, repeat. That ran for 80,000 rounds, about three and a half hours on the laptop's GPU,
overnight, in 20-minute batches that can stop and pick up again. The result holds 7.8 million
numbers and takes 31 MB on disk. That is tiny next to a chatbot, and it has to be: it must
answer within a few thousandths of a second every time you press space.

**The error model** describes how people really mistype. It is plain counting, with no neural
network. A public typing study from Aalto University recorded volunteers copying sentences. In
11.5 million of their typed words there were 402,912 typos. For each one the code works out what
went wrong and tallies it by position in the word and by word length. Leaving a letter out is
the most common slip (42%), then a wrong key (32%), an extra letter (18%) and two letters
swapped (8%). With these counts the computer can make up typos that look like real ones, as many
as training needs.

**The pair ranker** answers "does this typed word look like a slip of that word?". It reads both
words letter by letter and gives a score. It has 232,000 numbers and trained for about 40
minutes on roughly 800,000 examples. Many of the typos in them are ones people really made in
the study, some are well-known misspellings, and the rest are made up, by the error model or as
random slips. On
realistic typos it puts the right word first 86.7% of the time. The older ranker, trained on
random slips, managed 73.8%.

**The decider** makes the final call. It is three small networks whose answers are averaged. To
train them, 1.27 million words were taken from text the language model had never seen, and half
of them were mistyped using the error model. Then the app's own code, not a Python copy of it,
ran over every one of those words and wrote down the 37 numbers it would see in real use: how
well each option fits the sentence, how likely the slip is, how common the word is, and so on.
The networks learned from those numbers which option was right, "keep it" included. Because the
numbers came from the app itself, what the model saw in training is what it sees on your Mac.
This step takes a few minutes and needs no GPU.

**Checking the result.** The test sentences are split into three piles. The first was used to
choose the confidence thresholds. The second was used to check them. The third stayed shut until
the very end and was run once. No test sentence was used for training. The Swift code that runs
the language model inside the app was compared with the original PyTorch version: the way text
is split into words on 20,000 lines, and the model's output on 400 sentences. They agree to
within 0.0002. The numbers, and the things that got worse, are under "Accuracy" above.
