import Foundation

public struct Candidate {
    public let word: String
    public let distance: Int
    public let editCost: Double
    public let logPrior: Double
    public let neural: Double
    public let boost: Double
    public let score: Double
    public var probability: Double
}

public struct Correction {
    public let original: String
    public let replacement: String
    public let confidence: Double
    public let reason: String
}

public struct Analysis {
    public let word: String
    public let candidates: [Candidate]
    public let decision: Correction?
    public let note: String
    /// The option rows of the context path (see `ContextModel`); nil on the path without context.
    public var rows: ContextRows? = nil
}

/// Defaults were tuned on eval/cases (*.dev.tsv) with `typo-cli suite` and checked on the test and
/// holdout splits; `CorrectorConfig.legacy` reproduces the engine before that tuning (2026-11-09).
public struct CorrectorConfig {
    /// Softmax probability the winner needs over the other candidates (and over keeping the word).
    /// Used only when `minShareAmongFixes` is nil.
    public var minProbability = 0.8
    /// Two-part rule: correct only if "keep the word" has at most `maxKeepShare` of the probability,
    /// and the winner has at least `minShareAmongFixes` of what is left (no close second fix).
    public var minShareAmongFixes: Double? = nil
    public var maxKeepShare = 0.3
    /// Reject edits that are implausible as slips.
    public var maxEditCost = 4.5
    /// Extra cost per edit, on top of the per-edit plausibility: a second slip is much rarer.
    public var perEditPenalty = 1.5
    /// Candidates rarer than this (per billion words) are ignored unless personally learned.
    public var minCandidateFrequency = 100.0
    /// A dictionary word not on the misspelling list is treated as a typo only when it is
    /// rarer than this, a distance-1 neighbour is `knownWordRatio` times more common, and the
    /// slip is mechanical (edit cost <= `knownWordMaxEditCost`).
    public var knownWordMaxFrequency = 500.0
    public var knownWordRatio = 500.0
    public var knownWordMaxEditCost = 1.0
    /// Misspellings on the curated list are correctable up to this frequency.
    public var listedMaxFrequency = 5000.0
    public var maxCandidates = 12
    public var weightPrior = 1.0
    public var weightEdit = 1.0
    public var weightNeural = 0.25
    public var minLength = 3
    public var maxLength = 24
    /// After a click or app switch, correct the first word unless it could be the tail of a longer
    /// word at least `tailWordMinFrequency` common (then the caret may have been inside that word).
    public var correctFirstWordAfterReset = true
    public var tailWordMinFrequency = 1000.0
    /// Leave words alone when the text since the last space marks a URL, path or identifier.
    public var skipLinksAndCode = true
    /// "THe" / "WHat": shift held one letter too long. Fixed to "The" when the word is common.
    public var fixShiftSlips = true
    public var shiftSlipMinFrequency = 10_000.0
    /// Extra ranking cost when the first letter differs, except a swap of the first two letters
    /// ("hte"). Off: with the keep option and name rules it no longer prevents false fixes, and it
    /// cost real first-letter slips ("slso", "vrom").
    public var firstLetterPenalty = 0.0
    /// Dictionary words may be "fixed" to a word with another first letter (off: "vars" stays).
    public var knownWordFirstLetterFixes = false
    /// Keyboard edit costs. Omissions are cheaper than in `EditModel()`: people drop letters a lot.
    public var edit: EditModel = {
        var m = EditModel()
        m.del = 1.5; m.delDouble = 0.5
        m.subAdjacent = 1.2; m.subVowel = 1.2
        m.transpose = 1.2
        m.insAdjacent = 1.2; m.ins = 2.6
        return m
    }()
    /// Known (dictionary) words: a distance-1 neighbour replaces the word only when
    /// ln(f_neighbour / f_word) >= editCost + knownWordMargin. nil keeps the fixed `knownWordRatio` rule.
    public var knownWordMargin: Double? = nil
    /// Unknown words compete with "the word is intended" (a name, jargon, slang). Its log-score is
    /// `keepLogScore` (+ `keepCapitalBonus` when capitalised); for a regular inflection of a known
    /// word ("embeddings", "aliased", "rerunning") it is ln(f_stem) + `inflectionKeepOffset`.
    /// nil disables the keep option (the best candidate only has to beat the other candidates).
    public var keepLogScore: Double? = -15
    public var keepCapitalBonus = 1.0
    /// Extra keep score for a capitalised word right after a click or app switch (name or sentence start?).
    public var unknownStartCapitalKeepBonus = 2.0
    /// Added to the keep score per letter beyond 6 (negative: long near-words are typos).
    public var keepPerLetter = -1.5
    public var inflectionKeepOffset = -1.0
    public var inflectionStemMinFrequency = 100.0
    /// Dictionary words missing from the curated list (`CuratedWords`) are usually web misspellings
    /// ("befor", "completly"): they get their own, looser known-word limits.
    public var uncuratedKnownMaxFrequency = 1000.0
    public var uncuratedKnownMaxEditCost = 1.8
    public var uncuratedKnownMargin: Double? = nil
    /// Keep-score bonus for an unknown word that is in the curated list (a rare real word).
    public var curatedKeepBonus = 4.0
    /// A capitalised dictionary word missing from the curated list is treated as a name and left alone.
    public var capitalizedUncuratedIsName = true
    /// Keep-score bonus for an unknown word that reads as romaji (Japanese names and words).
    public var romajiKeepBonus = 1.0
    /// Cap on the stem frequency a prefixed form ("re" + "factor") may claim for its keep score.
    public var prefixStemMaxFrequency = 10_000.0
    /// "sooo", "yesss", "okkk": a letter typed three times in a row is deliberate. Off: the
    /// lexicon already knows the common elongations, and tripled letters are also real slips.
    public var skipElongations = false
    /// Curated misspellings are fixed even when capitalised mid-sentence ("Wensday", "Febuary").
    public var listedIgnoresCapitalization = true
    /// "inthe" -> "in the": offer a split when one part is a function word and the other common.
    public var splitMergedWords = true
    public var splitPartMinFrequency = 3000.0
    public var splitCost = 3.0
    /// Keep score offset for an uncurated dictionary word competing with its split.
    public var uncuratedKeepOffset = 0.0
    /// Context path (language model + decider, see `ContextModel`), used when the model is loaded and
    /// the text before the word is known. Off, or without the model files, the rules above decide alone.
    public var useContext = true
    /// Probability the best replacement needs against keeping the word and against the other
    /// candidates: for a word that is not in the dictionary, for a dictionary word (a real-word slip
    /// such as "form" for "from": only when very sure), and for a capitalised word inside a sentence.
    /// Tuned on eval/cases dev (2026-11-11) so that false fixes stay at the level of the rules alone.
    public var contextMinProbability = 0.9
    public var contextKnownMinProbability = 0.97
    public var contextNameMinProbability = 0.95
    /// One- and two-letter words ("t", "th", "yo") are the most common real slips; context makes them fixable.
    public var contextMinLength = 1
    /// Near words beyond the `maxCandidates` most common that the language model may add.
    public var contextExtraCandidates = 3
    /// "dont" -> "don't", "its" -> "it's": off, apostrophe-less contractions are typed on purpose in chat.
    public var contextAddsApostrophes = false
    /// Threshold for the first word after a click or app switch, which may be a fragment of a word.
    public var contextUnknownStartMinProbability = 0.95
    /// A capitalised word outside the lexicon that reads as romaji ("Yuu", "Haruto") is a Japanese name
    /// far more often than a slip.
    public var contextRomajiNameMinProbability = 0.995
    /// A typed word outside the lexicon that the curated dictionary lists, or that is a regular
    /// inflection of a known word, is probably a real word: it needs this probability, and the rules'
    /// agreement does not lower it. nil: no extra bar.
    public var contextAttestedMinProbability: Double? = 0.98
    /// When the rules (which do not see the sentence) would make the same replacement on their own,
    /// this lower probability is enough: two independent judgements agree. nil turns the rule off.
    public var contextAgreeMinProbability: Double? = 0.7
    public init() {}

    /// The engine as it was before the accuracy suite (for comparisons).
    public static var legacy: CorrectorConfig {
        var c = CorrectorConfig()
        c.minProbability = 0.6; c.maxEditCost = 4.0; c.perEditPenalty = 2.5; c.minCandidateFrequency = 300
        c.weightNeural = 0.5; c.correctFirstWordAfterReset = false; c.skipLinksAndCode = false; c.fixShiftSlips = false
        c.firstLetterPenalty = 0; c.edit = EditModel(); c.keepLogScore = nil; c.keepPerLetter = 0; c.inflectionKeepOffset = -3
        c.uncuratedKnownMaxEditCost = 1.0; c.uncuratedKnownMaxFrequency = 500; c.curatedKeepBonus = 0; c.listedIgnoresCapitalization = false
        c.splitMergedWords = false; c.splitCost = 3.0; c.capitalizedUncuratedIsName = false; c.knownWordFirstLetterFixes = true
        return c
    }
}

/// Decides whether a just-finished word is a typo and what to replace it with.
public final class Corrector {
    public let lexicon: Lexicon
    public let symSpell: SymSpell
    public var ranker: NeuralRanker?
    public let personal: PersonalModel
    public var misspellings: [String: String]
    public var config: CorrectorConfig
    /// Curated real-word list; nil treats every dictionary word as real (the old behaviour).
    public var curated: CuratedWords?
    /// Language model + decider for the context path; nil decides by the rules alone.
    public var context: ContextModel?

    public init(lexicon: Lexicon, symSpell: SymSpell, ranker: NeuralRanker?, personal: PersonalModel,
                misspellings: [String: String] = [:], config: CorrectorConfig = CorrectorConfig()) {
        self.lexicon = lexicon
        self.symSpell = symSpell
        self.ranker = ranker
        self.personal = personal
        self.misspellings = misspellings
        self.config = config
    }

    public func correct(_ word: String, sentenceStart: Bool = true) -> Correction? {
        analyze(word, sentenceStart: sentenceStart).decision
    }

    public static func isAsciiWord(_ bytes: [UInt8]) -> Bool {
        bytes.allSatisfy { ($0 >= 97 && $0 <= 122) || $0 == 39 }
    }

    /// `positionUnknown`: the word starts where tracking began after a click or app switch, so it may
    /// or may not start a sentence. A capitalised word there is corrected, but "keep it" (a name) gets
    /// `unknownStartCapitalKeepBonus`.
    /// `line`: the text before the word, for the context path. `state`: a ready-made language-model
    /// state instead (used when collecting training rows from stored states).
    public func analyze(_ word: String, sentenceStart: Bool = true, positionUnknown: Bool = false,
                        line: LineContext? = nil, state: LMState? = nil) -> Analysis {
        analyze(word, sentenceStart: sentenceStart, positionUnknown: positionUnknown, line: line, state: state, rulesOnly: false)
    }

    /// `rulesOnly`: decide without the context path even when it is available (its second opinion).
    func analyze(_ word: String, sentenceStart: Bool, positionUnknown: Bool, line: LineContext?, state: LMState?, rulesOnly: Bool) -> Analysis {
        func skip(_ note: String) -> Analysis { Analysis(word: word, candidates: [], decision: nil, note: note) }

        let model = config.useContext && !rulesOnly ? context : nil
        let contextual = model != nil && (state != nil || (line != nil && model!.decider != nil))
        let chars = Array(word)
        guard chars.count >= (contextual ? config.contextMinLength : config.minLength), chars.count <= config.maxLength else { return skip("length") }
        let lower = word.lowercased()
        let bytes = Array(lower.utf8)
        guard Self.isAsciiWord(bytes) else { return skip("non-ascii") }
        guard bytes.first != 39, bytes.last != 39 else { return skip("apostrophe edge") }

        // "THe", "WHat", "DOn't": shift released one letter late. Only when the word itself is common,
        // and not an acronym with a suffix ("IDs", "PRs", "DMed", "TAing", "BEd", "LAer").
        if config.fixShiftSlips, chars.count >= 3, chars[0].isUppercase, chars[1].isUppercase,
           chars.dropFirst(2).allSatisfy({ $0.isLowercase || $0 == "'" }),
           !Self.acronymSuffixes.contains(String(chars.dropFirst(2))),
           lexicon.frequency(lower) >= config.shiftSlipMinFrequency {
            if personal.neverCorrect(lower) { return skip("user accepted word") }
            let normalized = String(chars[0]) + String(chars.dropFirst()).lowercased()
            return Analysis(word: word, candidates: [], decision: Correction(original: word, replacement: normalized, confidence: 1, reason: "shift slip"), note: "shift slip")
        }

        let upper = chars.filter { $0.isUppercase }.count
        if upper == chars.count { return skip("all caps") }
        if upper > 1 || (upper == 1 && !chars[0].isUppercase) { return skip("mixed case") }
        let capitalized = chars[0].isUppercase

        if personal.neverCorrect(lower) { return skip("user accepted word") }
        let freq = lexicon.frequency(lower)
        let known = freq > 0

        if let fix = personal.autoFix(for: lower, isDictionaryWord: known) {
            return Analysis(word: word, candidates: [], decision: Correction(original: word, replacement: recase(fix, like: word), confidence: 1, reason: "learned"), note: "learned pair")
        }
        if config.skipElongations, Self.hasTripleLetter(bytes) { return skip("elongation") }
        let listedNameSafe = config.listedIgnoresCapitalization && misspellings[lower] != nil
        let capitalAtUnknownStart = capitalized && positionUnknown
        // With context, a capitalised word inside a sentence is not skipped outright: the decider knows
        // it is capitalised there and needs `contextNameMinProbability` to change it.
        if capitalized && !sentenceStart && !listedNameSafe && !capitalAtUnknownStart && !contextual { return skip("capitalised mid-sentence (name?)") }

        // A curated misspelling is authoritative: it is replaced by its listed fix or left alone,
        // never by a third word. (Same rails as above still apply: accepted words, undo, names.)
        if let fix = misspellings[lower] {
            guard freq < config.listedMaxFrequency else { return skip("common word (listed)") }
            guard lexicon.contains(fix) else { return skip("listed fix not in lexicon") }
            return Analysis(word: word, candidates: [], decision: Correction(original: word, replacement: recase(fix, like: word), confidence: 1, reason: "listed misspelling"), note: "listed")
        }

        let maxDist = bytes.count <= 4 ? 1 : 2
        var hits = symSpell.lookup(bytes, maxDistance: maxDist).filter { lexicon.strings[Int($0.id)] != lower }

        let real = known && isCuratedWord(lower)
        // A capitalised dictionary word missing from the curated list is most likely a name
        // ("Persson", "Sajid", "Tyree"): leave it, whatever the neighbours.
        if config.capitalizedUncuratedIsName, capitalized && known && !real { return skip("capitalised, likely a name") }
        // Context path. Merged words ("inthe") stay with the rules below, which know how to split.
        if contextual, let model,
           !(config.splitMergedWords && !real && !capitalized && mergedWordSplit(lower, dictionaryWord: known) != nil) {
            return contextAnalysis(word: word, lower: lower, bytes: bytes, capitalized: capitalized, sentenceStart: sentenceStart,
                                   positionUnknown: positionUnknown, freq: freq, known: known, real: real, hits: hits, model: model,
                                   state: { state ?? model.state(for: line!) })
        }
        guard chars.count >= config.minLength else { return skip("length") }
        // Merged words ("inthe", "ofthe") are offered a split unless they are curated real words.
        let split = (config.splitMergedWords && !real && !capitalized) ? mergedWordSplit(lower, dictionaryWord: known) : nil
        if known {
            let maxFreq = real ? config.knownWordMaxFrequency : config.uncuratedKnownMaxFrequency
            let maxCost = real ? config.knownWordMaxEditCost : config.uncuratedKnownMaxEditCost
            let margin = real ? config.knownWordMargin : (config.uncuratedKnownMargin ?? config.knownWordMargin)
            guard freq < maxFreq || split != nil else { return skip("common word") }
            hits = hits.filter {
                guard $0.distance == 1, freq < maxFreq else { return false }
                let f = lexicon.frequency(id: $0.id), cand = lexicon.bytes[Int($0.id)]
                guard config.edit.cost(typo: bytes, intended: cand) <= maxCost else { return false }
                // A dictionary word with another first letter is usually another word ("vars", "cars").
                if !config.knownWordFirstLetterFixes, Self.firstLetterDiffers(bytes, cand) { return false }
                let cost = editCost(bytes, cand)
                if let margin { return log(f / freq) >= cost + margin }
                return f >= freq * config.knownWordRatio
            }
            if hits.isEmpty && split == nil { return skip("known word") }
        }
        hits = hits.filter {
            let w = lexicon.strings[Int($0.id)]
            return lexicon.frequency(id: $0.id) >= config.minCandidateFrequency || personal.manualCount(typo: lower, fix: w) > 0
        }
        guard !hits.isEmpty || split != nil else { return skip("no candidates") }

        hits.sort {
            let a = lexicon.frequency(id: $0.id), b = lexicon.frequency(id: $1.id)
            return a != b ? a > b : $0.id < $1.id   // deterministic when frequencies tie
        }
        if hits.count > config.maxCandidates { hits.removeLast(hits.count - config.maxCandidates) }

        let candBytes = hits.map { lexicon.bytes[Int($0.id)] }
        let useNeural = bytes.count >= 4   // the ranker is trained on words of 4+ letters
        let neural: [Double] = (useNeural ? ranker : nil).map { r in r.scores(typo: bytes, candidates: candBytes).map { $0.isFinite ? Double($0) : 0 } } ?? Array(repeating: 0, count: hits.count)

        var cands: [Candidate] = []
        for (i, h) in hits.enumerated() {
            let w = lexicon.strings[Int(h.id)]
            // The first-letter penalty ranks candidates; the plausibility cap below sees the keyboard cost.
            let edit = config.edit.cost(typo: bytes, intended: candBytes[i])
            let penalty = editCost(bytes, candBytes[i]) - edit
            let prior = lexicon.logProb(id: h.id)
            let boost = personal.boost(typo: lower, candidate: w)
            let score = config.weightPrior * prior
                - config.weightEdit * (edit + penalty + config.perEditPenalty * Double(h.distance))
                + config.weightNeural * neural[i] + boost
            cands.append(Candidate(word: w, distance: h.distance, editCost: edit, logPrior: prior, neural: neural[i], boost: boost, score: score, probability: 0))
        }
        if let s = split {
            let boost = personal.boost(typo: lower, candidate: s.text)
            let score = config.weightPrior * s.logPrior - config.weightEdit * (config.splitCost + config.perEditPenalty) + boost
            cands.append(Candidate(word: s.text, distance: 1, editCost: config.splitCost, logPrior: s.logPrior, neural: 0, boost: boost, score: score, probability: 0))
        }
        // "The word is intended" competes too (distance 0): names, jargon, slang, rare inflections.
        if known, split != nil {
            // A dictionary word competes with its split as itself ("inbox" stays, "ofthe" splits).
            let keep = log(freq / lexicon.total) + config.uncuratedKeepOffset
            cands.append(Candidate(word: lower, distance: 0, editCost: 0, logPrior: keep, neural: 0, boost: 0, score: keep, probability: 0))
        }
        if !known, var keep = keepScore(lower, capitalized: capitalized) {
            if capitalAtUnknownStart { keep += config.unknownStartCapitalKeepBonus }
            cands.append(Candidate(word: lower, distance: 0, editCost: 0, logPrior: keep, neural: 0, boost: 0, score: keep, probability: 0))
        }
        let mx = cands.map(\.score).max() ?? 0
        let z = cands.reduce(0.0) { $0 + exp($1.score - mx) }
        for i in cands.indices { cands[i].probability = exp(cands[i].score - mx) / z }
        cands.sort { $0.score > $1.score }

        let best = cands[0]
        var decision: Correction? = nil
        var note = "ok"
        // Two questions: is this a typo at all (the keep option's share), and which word was meant
        // (the winner's share among the corrections). One threshold on the raw share mixes them.
        let keepShare = cands.first(where: { $0.distance == 0 })?.probability ?? 0
        let amongFixes = best.probability / max(1e-9, 1 - keepShare)
        let twoPart = config.minShareAmongFixes != nil
        if best.distance == 0 { note = "kept (likely intended)" }
        else if twoPart, keepShare > config.maxKeepShare { note = "may be intended" }
        else if twoPart, amongFixes < config.minShareAmongFixes! { note = "low confidence" }
        else if !twoPart, best.probability < config.minProbability { note = "low confidence" }
        else if best.editCost > config.maxEditCost && best.boost <= 0 { note = "implausible edit" }
        else if capitalized && best.distance > 1 && best.boost <= 0 { note = "capitalised, distance 2" }
        else {
            let reason = best.word.contains(" ") ? "merged words" : (known ? "rare word, common neighbour" : "unknown word")
            decision = Correction(original: word, replacement: recase(best.word, like: word), confidence: best.probability, reason: reason)
        }
        return Analysis(word: word, candidates: cands, decision: decision, note: note)
    }

    func recase(_ fix: String, like original: String) -> String {
        guard let f = original.first, f.isUppercase, let c = fix.first else { return fix }
        return String(c).uppercased() + fix.dropFirst()
    }

    static func hasTripleLetter(_ b: [UInt8]) -> Bool {
        guard b.count >= 3 else { return false }
        for i in 2..<b.count where b[i] == b[i - 1] && b[i] == b[i - 2] { return true }
        return false
    }

    /// Log-score of "this unknown word is what the user meant" (see `CorrectorConfig.keepLogScore`).
    func keepScore(_ lower: String, capitalized: Bool) -> Double? {
        guard let base = config.keepLogScore else { return nil }
        if let stem = inflectionStemFrequency(lower) {
            return max(log(stem / lexicon.total) + config.inflectionKeepOffset, base)
        }
        // Long strings one or two edits from a real word are rarely deliberate new words.
        let lengthAdjust = config.keepPerLetter * Double(max(lower.count - 6, 0))
        let real = (curated?.contains(Array(lower.utf8)) ?? false) ? config.curatedKeepBonus : 0
        // Japanese names and words in romaji ("Haruto", "gyoza", "natto", "Yuu") are typed on purpose.
        let romaji = Self.isRomaji(lower) ? config.romajiKeepBonus : 0
        return base + lengthAdjust + real + romaji + (capitalized ? config.keepCapitalBonus : 0)
    }

    /// Frequency of the known word that `w` regularly inflects or prefixes ("embeddings" -> embedding,
    /// "annotating" -> annotate, "rerunning" -> running), or nil.
    func inflectionStemFrequency(_ w: String) -> Double? {
        let minF = config.inflectionStemMinFrequency
        if let v = inflectionStem(w, accept: { s in
            let v = self.lexicon.frequency(s)
            return v >= minF ? v : nil
        }) { return v }
        // Prefixed forms of a curated, non-function word: "refactor", "rebase", "rerunning". Not "prefrom"
        // (pre + from); the stem frequency is capped so a prefix on a very common word cannot outscore a fix.
        for prefix in ["re", "un", "pre", "non", "over", "under", "sub", "multi", "auto", "inter", "post", "anti", "micro", "mini", "self"]
        where w.hasPrefix(prefix) && w.count - prefix.count >= 4 {
            let rest = String(w.dropFirst(prefix.count))
            guard !Self.functionWords.contains(rest), isCuratedWord(rest) else { continue }
            if let v = inflectionStem(rest, accept: { s in
                let v = self.lexicon.frequency(s)
                return v >= minF ? v : nil
            }) { return min(v, config.prefixStemMaxFrequency) }
        }
        return nil
    }

    /// A curated real word, or a regular inflection of one (web2 lists base forms only, so "sways",
    /// "culls", "doling" count through "sway", "cull", "dole"). Without a curated list, every word counts.
    func isCuratedWord(_ lower: String) -> Bool {
        guard let curated else { return true }
        return inflectionStem(lower, accept: { curated.contains(Array($0.utf8)) ? 1 : nil }) != nil
    }

    /// Runs the regular-inflection rules on `w` and returns `accept(stem)` for the first stem accepted.
    func inflectionStem(_ w: String, accept: (String) -> Double?) -> Double? {
        func f(_ s: String) -> Double? {
            guard s.count >= 3 else { return nil }
            return accept(s)
        }
        let vowels = Set("aeiou")
        /// Ends consonant-vowel-consonant ("run", "forget"): -ing/-ed may double the final consonant.
        func endsCVC(_ s: String) -> Bool {
            let c = Array(s.suffix(3))
            guard c.count == 3, let last = c.last, !"wxy".contains(last) else { return false }
            return !vowels.contains(c[0]) && vowels.contains(c[1]) && !vowels.contains(c[2])
        }
        /// `s` is a correctly spelled regular inflection of a known word: plural / 3rd person / possessive,
        /// -ed / -ing / -er with the silent-e and consonant-doubling rules ("hikeing", "forgeting" are not).
        func inflected(_ s: String) -> Double? {
            if let v = f(s) { return v }
            if s.hasSuffix("'s"), let v = f(String(s.dropLast(2))) { return v }
            for suffix in ["ies", "ied", "ier", "iers"] where s.hasSuffix(suffix) {
                let stem = String(s.dropLast(suffix.count))
                if let l = stem.last, !vowels.contains(l), let v = f(stem + "y") { return v }
            }
            if s.hasSuffix("es"), let v = f(String(s.dropLast(2))), ["s", "x", "z", "ch", "sh", "o"].contains(where: { s.dropLast(2).hasSuffix($0) }) { return v }
            if s.hasSuffix("s"), !s.hasSuffix("ss"), !s.hasSuffix("'s"), let v = f(String(s.dropLast())) { return v }
            for suffix in ["ing", "ed", "er", "ers"] where s.hasSuffix(suffix) && s.count > suffix.count + 2 {
                let stem = String(s.dropLast(suffix.count))
                // silent e dropped before the vowel suffix: annotate -> annotating / annotated
                if let v = f(stem + "e"), !stem.hasSuffix("e") { return v }
                // doubled consonant: run -> running, forget -> forgetting
                if let l = stem.last, stem.dropLast().last == l {
                    let base = String(stem.dropLast())
                    if endsCVC(base), let v = f(base) { return v }
                }
                // plain: lint -> linter, alias -> aliased (not after silent e, not where doubling is the spelling)
                if !stem.hasSuffix("e") || stem.hasSuffix("ee"), let v = f(stem) {
                    if endsCVC(stem), let l = stem.last, lexicon.frequency(stem + String(l) + suffix) >= 100 { return nil }
                    return v
                }
            }
            return nil
        }
        return inflected(w)
    }

    static let acronymSuffixes: Set<String> = ["s", "es", "d", "ed", "ing", "er", "ers", "'s", "'d"]

    /// Hepburn-style romaji: (doubled consonant) + syllable, syllabic n, "t" before "ch".
    static let romaji = try! NSRegularExpression(pattern:
        "^(?:(?:([kstpgdbzmhrfjc])(?=\\1))?(?:(?:ky|gy|ny|hy|my|ry|by|py)[auo]|(?:sh|ch|j)[aiueo]|tsu|f[aiueo]|w[ao]|y[auo]|[kgsztdnhbpmr][aiueo]|[aiueo])|t(?=ch)|n(?![aiueoy]))+$")

    static func isRomaji(_ lower: String) -> Bool {
        guard lower.count >= 3 else { return false }
        return romaji.firstMatch(in: lower, range: NSRange(lower.startIndex..., in: lower)) != nil
    }

    static let functionWords: Set<String> = ["a", "i", "an", "the", "to", "of", "in", "on", "at", "for", "with", "and", "or", "but",
        "is", "are", "was", "were", "be", "been", "it", "that", "this", "you", "we", "he", "she", "they", "my", "your", "our",
        "me", "so", "if", "as", "by", "from", "not", "no", "do", "have", "has", "had", "will", "can", "would", "should",
        "could", "just", "all", "up", "out", "about", "into", "than", "then", "what", "when", "there", "their", "its"]

    /// "inthe" -> ("in the", ln P(in) + ln P(the)) when one part is a function word and the other common.
    /// A dictionary word is only split as "<word> the" ("ofthe", "inthe"): "init", "redis", "whois",
    /// "goto", "matcha" are words. "headin" is never "head in" (a dropped g).
    func mergedWordSplit(_ w: String, dictionaryWord: Bool = false) -> (text: String, logPrior: Double)? {
        let chars = Array(w)
        guard chars.count >= 4 else { return nil }
        var best: (String, Double)?
        for i in 1..<chars.count {
            let a = String(chars[..<i]), b = String(chars[i...])
            guard a != "'" && b.first != "'" else { continue }
            let fa = lexicon.frequency(a), fb = lexicon.frequency(b)
            let fwA = Self.functionWords.contains(a), fwB = Self.functionWords.contains(b)
            guard (fwA && fb >= config.splitPartMinFrequency) || (fwB && fa >= config.splitPartMinFrequency) else { continue }
            guard fa > 0, fb > 0, a.count > 1 || a == "a" || a == "i", b.count > 1 || b == "a" else { continue }
            if dictionaryWord, b != "the" || a.count < 2 { continue }   // "ofthe" yes; "init", "redis", "goto", "matcha" no
            if b == "in", lexicon.frequency(a + "ing") >= 100 { continue }       // "headin" is "heading" with the g dropped
            let lp = log(fa / lexicon.total) + log(fb / lexicon.total)
            if best == nil || lp > best!.1 { best = ("\(a) \(b)", lp) }
        }
        return best
    }

    /// The first letters differ, other than by swapping the first two ("hte").
    static func firstLetterDiffers(_ typo: [UInt8], _ intended: [UInt8]) -> Bool {
        guard let t0 = typo.first, let i0 = intended.first, t0 != i0 else { return false }
        return !(typo.count > 1 && intended.count > 1 && typo[0] == intended[1] && typo[1] == intended[0])
    }

    /// Keyboard edit cost plus the first-letter penalty (see `CorrectorConfig.firstLetterPenalty`).
    func editCost(_ typo: [UInt8], _ intended: [UInt8]) -> Double {
        var cost = config.edit.cost(typo: typo, intended: intended)
        if config.firstLetterPenalty > 0, let t0 = typo.first, let i0 = intended.first, t0 != i0 {
            let swapped = typo.count > 1 && intended.count > 1 && typo[0] == intended[1] && typo[1] == intended[0]
            if !swapped { cost += config.firstLetterPenalty }
        }
        return cost
    }

    // MARK: construction helpers

    public static func loadMisspellings() -> [String: String] {
        guard let url = Paths.resource("misspellings_en.txt"), let text = try? String(contentsOf: url, encoding: .utf8) else { return [:] }
        var out: [String: String] = [:]
        for line in text.split(separator: "\n") {
            let p = line.split(separator: "\t")
            if p.count == 2 { out[String(p[0])] = String(p[1]) }
        }
        return out
    }

    public static func loadDefault(store: Store? = nil, config: CorrectorConfig = CorrectorConfig()) throws -> Corrector {
        guard let url = Paths.resource("words_en.txt") else { throw TypoError.missingResource("words_en.txt") }
        let lexicon = try Lexicon(url: url)
        let sym = SymSpell(lexicon: lexicon)
        let personal = store.map { PersonalModel(store: $0) } ?? PersonalModel()
        let corrector = Corrector(lexicon: lexicon, symSpell: sym, ranker: NeuralRanker.load(), personal: personal,
                                  misspellings: loadMisspellings(), config: config)
        corrector.curated = CuratedWords()
        corrector.context = ContextModel.load(lexicon: lexicon)
        // Build the word-tail index now, not on the first keystroke after a click (the event tap
        // has a few milliseconds per key).
        _ = lexicon.endsLongerWord(Array("e".utf8), minFrequency: config.tailWordMinFrequency)
        return corrector
    }
}
