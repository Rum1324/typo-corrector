import Foundation

/// The text before the word being finished, as the tracker saw it.
public struct LineContext {
    public let before: String
    /// False when the line began after a click or app switch: what precedes `before` is unknown.
    public let startKnown: Bool
    public init(before: String, startKnown: Bool) { self.before = before; self.startKnown = startKnown }
}

/// One row of numbers per option: keep the word as typed (first row, word "="), or replace it
/// with a candidate. `ContextModel.featureNames` names the columns.
public struct ContextRows {
    public var words: [String] = []
    public var features: [[Float]] = []
}

/// Small network that scores one option row. Trained in training/decider.py on simulated typing
/// whose rows were written by this same code (`typo-cli features`), exported as context.json.
public struct Decider: Decodable {
    public struct Dense: Decodable { let w: [[Float]]; let b: [Float] }
    public let version: Int
    public let features: [String]
    public let mean: [Float]
    public let scale: [Float]
    public let layers: [Dense]
    /// Further networks trained from other random starts; the score is the mean over all of them.
    public let ensemble: [[Dense]]?

    public func score(_ f: [Float]) -> Float {
        var x = [Float](repeating: 0, count: f.count)
        for i in 0..<f.count { x[i] = (f[i] - mean[i]) / scale[i] }
        guard let more = ensemble, !more.isEmpty else { return Self.run(layers, x) }
        return more.reduce(Self.run(layers, x)) { $0 + Self.run($1, x) } / Float(more.count + 1)
    }

    private static func run(_ layers: [Dense], _ input: [Float]) -> Float {
        var x = input
        for (n, layer) in layers.enumerated() {
            var y = layer.b
            for o in 0..<y.count {
                let w = layer.w[o]
                var s: Float = 0
                for j in 0..<x.count { s += w[j] * x[j] }
                y[o] += s
                if n < layers.count - 1 && y[o] < 0 { y[o] = 0 }          // ReLU between layers
            }
            x = y
        }
        return x[0]
    }

    var isUsable: Bool {
        features == ContextModel.featureNames && mean.count == features.count && scale.count == features.count
            && !layers.isEmpty && layers[0].w.allSatisfy { $0.count == features.count } && layers.last!.b.count == 1
    }
}

/// How English a string of letters looks: a letter-trigram model counted from the lexicon's words
/// when the app starts. Slips often produce shapes no word has ("xharged", "wvil", "snakcs"); rare
/// real words and names mostly do not ("tradie", "copters").
struct WordShape {
    private var logp: [Float]
    private static let n = 28                      // word boundary, a-z, apostrophe

    @inline(__always) private static func symbol(_ c: UInt8) -> Int { c >= 97 && c <= 122 ? Int(c) - 96 : (c == 39 ? 27 : 0) }

    init(lexicon: Lexicon) {
        let n = Self.n
        var counts = [Float](repeating: 0.05, count: n * n * n)
        for bytes in lexicon.bytes {
            var a = 0, b = 0
            for c in bytes + [0] {
                let k = Self.symbol(c)
                counts[(a * n + b) * n + k] += 1
                a = b; b = k
            }
        }
        for ctx in 0..<(n * n) {
            var total: Float = 0
            for k in 0..<n { total += counts[ctx * n + k] }
            for k in 0..<n { counts[ctx * n + k] = log(counts[ctx * n + k] / total) }
        }
        logp = counts
    }

    /// Mean and lowest log-probability of the word's letter transitions (the word's end included).
    func score(_ bytes: [UInt8]) -> (mean: Float, worst: Float) {
        var a = 0, b = 0, sum: Float = 0, worst: Float = 0
        for c in bytes + [0] {
            let k = Self.symbol(c)
            let lp = logp[(a * Self.n + b) * Self.n + k]
            sum += lp; worst = min(worst, lp)
            a = b; b = k
        }
        return (sum / Float(bytes.count + 1), worst)
    }
}

/// Context language model + decision network: the context-aware correction path of `Corrector`.
public final class ContextModel {
    /// Columns of a `ContextRows` row. The decider file lists the names it was trained on; a
    /// mismatch disables it (the engine then decides without context, as before).
    public static let featureNames = [
        "keep", "lm", "lmTyped", "lmGain", "prior", "priorTyped", "priorGain", "typedKnown", "typedCurated", "typedRareReal",
        "distance2", "editCost", "neural", "neuralShare", "firstLetterDiffers", "length", "lengthChange", "capitalised",
        "sentenceStart", "positionUnknown", "contextLength", "contextCut", "romaji", "hasStem", "stemPrior", "tripleLetter",
        "candidates", "lmBestGain", "lmBehindBest", "typedOutOfVocabulary",
        "corpus", "corpusTyped", "corpusGain", "elongation", "lmTop", "shape", "shapeWorst",
    ]

    public let tokenizer: ContextTokenizer
    public let lm: ContextLM
    /// Word-form counts of the language model's training text.
    public let forms: WordForms
    /// wordfreq's words beyond the lexicon (per billion words); unseen forms count as 0.01, like `Lexicon.logProb`.
    public let moreWords: WordForms
    let shape: WordShape
    /// The pair ranker the decider was trained with (ranker2.json + ranker2.bin, trained on realistic
    /// typos). The rules keep their own ranker (`Corrector.ranker`), which their weights were tuned for.
    public var ranker: NeuralRanker?
    /// nil while collecting rows for training: `Corrector.analyze` then returns rows and no decision.
    public var decider: Decider?

    public init?(lexicon: Lexicon, lm: ContextLM, forms: WordForms, moreWords: WordForms, decider: Decider?) {
        // Token ids are lexicon ids + 16: a different word list would silently scramble the model.
        guard lm.config.vocab == lexicon.count + Int(ContextTokenizer.reserved) else { return nil }
        tokenizer = ContextTokenizer(lexicon: lexicon)
        self.lm = lm
        self.forms = forms
        self.moreWords = moreWords
        shape = WordShape(lexicon: lexicon)
        self.decider = decider.flatMap { $0.isUsable ? $0 : nil }
    }

    /// lm.json, lm.bin, forms_en.tsv, words_more_en.tsv and context.json from `directory`, or the bundled
    /// ones. nil without a language model or the word tables; without a usable context.json the model
    /// loads with `decider == nil`.
    public static func load(lexicon: Lexicon, directory: URL? = nil, deciderFile: URL? = nil) -> ContextModel? {
        guard let lm = ContextLM.load(directory: directory), let forms = WordForms.load("forms_en.tsv", directory: directory),
              let more = WordForms.load("words_more_en.tsv", directory: directory, unseenCount: 0.01) else { return nil }
        func file(_ name: String) -> URL? {
            if let u = directory?.appendingPathComponent(name), FileManager.default.fileExists(atPath: u.path) { return u }
            return Paths.resource(name)
        }
        guard let rm = file("ranker2.json"), let rw = file("ranker2.bin"), let ranker = try? NeuralRanker(manifest: rm, weights: rw) else { return nil }
        let url = deciderFile ?? directory?.appendingPathComponent("context.json") ?? Paths.resource("context.json")
        let decider = url.flatMap { try? Data(contentsOf: $0) }.flatMap { try? JSONDecoder().decode(Decider.self, from: $0) }
        let model = ContextModel(lexicon: lexicon, lm: lm, forms: forms, moreWords: more, decider: decider)
        model?.ranker = ranker
        return model
    }

    public func state(for line: LineContext) -> LMState {
        LMState(lm: lm, tokens: tokenizer.context(before: line.before, lineStartKnown: line.startKnown, maxLen: lm.config.maxLen))
    }
}

extension Corrector {
    /// Context-aware decision for a word that passed the fixed rails (length, case, accepted and
    /// learned words, listed misspellings). Every near word is an option, dictionary words included:
    /// the language model says which option fits the line, the decider weighs that against how likely
    /// the slip is, and "keep" must lose clearly before anything is changed.
    func contextAnalysis(word: String, lower: String, bytes: [UInt8], capitalized: Bool, sentenceStart: Bool, positionUnknown: Bool,
                         freq: Double, known: Bool, real: Bool, hits allHits: [SymSpell.Hit], model: ContextModel, state makeState: () -> LMState) -> Analysis {
        var hits = allHits.filter {
            lexicon.frequency(id: $0.id) >= config.minCandidateFrequency || personal.manualCount(typo: lower, fix: lexicon.strings[Int($0.id)]) > 0
        }
        // A contraction typed without its apostrophe ("dont", "im") is a spelling people choose: the
        // word is left alone. Where the bare form is the commoner word ("were", "its"), only the
        // apostrophe is not offered.
        if !config.contextAddsApostrophes {
            if hits.contains(where: { Self.sameLetters(bytes, lexicon.bytes[Int($0.id)]) && lexicon.frequency(id: $0.id) > freq }) {
                return Analysis(word: word, candidates: [], decision: nil, note: "contraction without apostrophe")
            }
            hits = hits.filter { !Self.sameLetters(bytes, lexicon.bytes[Int($0.id)]) }
        }
        guard !hits.isEmpty else { return Analysis(word: word, candidates: [], decision: nil, note: "no candidates") }
        let state = makeState()                       // the language model runs only when there is a choice to make
        hits.sort {
            let a = lexicon.frequency(id: $0.id), b = lexicon.frequency(id: $1.id)
            return a != b ? a > b : $0.id < $1.id
        }
        // The most common near words, plus the few rarer ones the context asks for.
        if hits.count > config.maxCandidates {
            let rest = hits[config.maxCandidates...].sorted {
                let a = state.logProb($0.id + ContextTokenizer.reserved), b = state.logProb($1.id + ContextTokenizer.reserved)
                return a != b ? a > b : $0.id < $1.id
            }
            hits = Array(hits[..<config.maxCandidates]) + rest.prefix(config.contextExtraCandidates)
        }

        let candBytes = hits.map { lexicon.bytes[Int($0.id)] }
        let pairRanker = model.ranker ?? ranker
        let neural: [Float] = pairRanker.map { r in r.scores(typo: bytes, candidates: candBytes).map { $0.isFinite ? $0 : 0 } }
            ?? Array(repeating: 0, count: hits.count)
        let nmax = neural.max() ?? 0
        let nlse = nmax + log(neural.reduce(Float(0)) { $0 + exp($1 - nmax) })
        let lmCand = hits.map { state.logProb($0.id + ContextTokenizer.reserved) }
        let typedToken = model.tokenizer.wordId(lower, capitalised: capitalized)
        let lmTyped = state.logProb(typedToken)
        let lmBest = lmCand.max() ?? lmTyped
        let priorTyped = known ? Float(lexicon.logProb(lower)) : model.moreWords.logProbability(bytes)
        let stem = known ? nil : inflectionStemFrequency(lower)

        func flag(_ b: Bool) -> Float { b ? 1 : 0 }
        var shared = [Float](repeating: 0, count: ContextModel.featureNames.count)
        shared[2] = lmTyped
        shared[5] = priorTyped
        shared[7] = flag(known)
        shared[8] = flag(real)
        shared[9] = flag(!known && (curated?.contains(bytes) ?? false))
        shared[15] = Float(min(bytes.count, 24)) / 10
        shared[17] = flag(capitalized)
        shared[18] = flag(sentenceStart)
        shared[19] = flag(positionUnknown)
        shared[20] = Float(state.contextLength) / Float(model.lm.config.maxLen)
        shared[21] = flag(state.startsWithCut)
        shared[22] = flag(!known && Self.isRomaji(lower))
        shared[23] = flag(stem != nil)
        shared[24] = stem.map { Float(log($0 / lexicon.total)) } ?? -25
        shared[25] = flag(Self.hasTripleLetter(bytes))
        shared[26] = Float(hits.count) / 12
        shared[27] = lmBest - lmTyped
        shared[29] = flag(typedToken < ContextTokenizer.reserved)
        let corpusTyped = model.forms.logProbability(bytes)
        shared[31] = corpusTyped
        shared[34] = state.topLogProb
        let shape = model.shape.score(bytes)
        shared[35] = shape.mean; shared[36] = shape.worst

        var rows = ContextRows()
        var keep = shared
        keep[0] = 1; keep[1] = lmTyped; keep[4] = priorTyped; keep[28] = lmTyped - lmBest; keep[30] = corpusTyped
        rows.words.append("="); rows.features.append(keep)
        var edits: [Double] = []
        for (i, h) in hits.enumerated() {
            let cand = candBytes[i]
            let edit = config.edit.cost(typo: bytes, intended: cand)
            let prior = Float(lexicon.logProb(id: h.id))
            var f = shared
            f[1] = lmCand[i]; f[3] = lmCand[i] - lmTyped
            f[4] = prior; f[6] = prior - priorTyped
            f[10] = flag(h.distance >= 2)
            f[11] = Float(edit)
            f[12] = neural[i]; f[13] = pairRanker == nil ? 0 : neural[i] - nlse
            f[14] = flag(Self.firstLetterDiffers(bytes, cand))
            f[16] = Float(cand.count - bytes.count)
            f[28] = lmCand[i] - lmBest
            let corpus = model.forms.logProbability(cand)
            f[30] = corpus; f[32] = corpus - corpusTyped
            f[33] = Float(min(Self.elongation(bytes, of: cand), 4)) / 4
            rows.words.append(lexicon.strings[Int(h.id)]); rows.features.append(f)
            edits.append(edit)
        }
        guard let decider = model.decider else {
            return Analysis(word: word, candidates: [], decision: nil, note: "rows", rows: rows)
        }

        var scores = rows.features.map { Double(decider.score($0)) }
        var cands: [Candidate] = []
        cands.append(Candidate(word: lower, distance: 0, editCost: 0, logPrior: Double(priorTyped), neural: 0, boost: 0, score: scores[0], probability: 0))
        for (i, h) in hits.enumerated() {
            let w = lexicon.strings[Int(h.id)]
            let boost = personal.boost(typo: lower, candidate: w)
            scores[i + 1] += boost
            cands.append(Candidate(word: w, distance: h.distance, editCost: edits[i], logPrior: lexicon.logProb(id: h.id),
                                   neural: Double(neural[i]), boost: boost, score: scores[i + 1], probability: 0))
        }
        let mx = scores.max() ?? 0
        let z = scores.reduce(0.0) { $0 + exp($1 - mx) }
        for i in cands.indices { cands[i].probability = exp(cands[i].score - mx) / z }
        cands.sort { $0.score > $1.score }

        let best = cands[0]
        var decision: Correction? = nil
        var note = "ok"
        var needed = known ? config.contextKnownMinProbability : config.contextMinProbability
        if positionUnknown { needed = max(needed, config.contextUnknownStartMinProbability) }
        // A capitalised word that is not in the lexicon, or any capitalised word inside a sentence, may be a name.
        if capitalized && (!known || (!sentenceStart && !positionUnknown)) { needed = max(needed, config.contextNameMinProbability) }
        // The counts (larger word list, corpus) also hold common misspellings, so they stay evidence for
        // the decider; only the curated dictionary and regular inflections count as "this word exists".
        let attested = !known && (shared[9] > 0 || stem != nil)
        if attested, let bar = config.contextAttestedMinProbability { needed = max(needed, bar) }
        let romajiName = capitalized && shared[22] > 0
        if romajiName { needed = max(needed, config.contextRomajiNameMinProbability) }
        let fix = recase(best.word, like: word)
        if best.distance == 0 { note = "kept (likely intended)" }
        else if best.probability >= needed {
            decision = Correction(original: word, replacement: fix, confidence: best.probability,
                                  reason: known ? "another word fits the sentence" : "unknown word")
        } else if !(attested && config.contextAttestedMinProbability != nil), !romajiName,
                  let agree = config.contextAgreeMinProbability, best.probability >= agree,
                  let rules = analyze(word, sentenceStart: sentenceStart, positionUnknown: positionUnknown, line: nil, state: nil, rulesOnly: true).decision,
                  rules.replacement == fix {
            decision = Correction(original: word, replacement: fix, confidence: best.probability, reason: rules.reason)
            note = "ok (rules agree)"
        } else {
            note = attested ? "low confidence (the typed word exists)" : "low confidence"
        }
        return Analysis(word: word, candidates: cands, decision: decision, note: note, rows: rows)
    }

    /// The two words are the same once apostrophes are dropped ("dont" / "don't", "its" / "it's").
    static func sameLetters(_ a: [UInt8], _ b: [UInt8]) -> Bool {
        a.count != b.count && a.filter { $0 != 39 } == b.filter { $0 != 39 }
    }

    /// How many extra letters `typed` has if it is `word` with letters held down ("sooo" of "so": 2,
    /// "crazyyy" of "crazy": 2, "gratefull" of "grateful": 1); 0 when it is not such a stretch.
    /// One extra letter is usually a slip, several are usually meant.
    static func elongation(_ typed: [UInt8], of word: [UInt8]) -> Int {
        guard typed.count > word.count else { return 0 }
        func runs(_ b: [UInt8]) -> [(UInt8, Int)] {
            var out: [(UInt8, Int)] = []
            for c in b { if let l = out.last, l.0 == c { out[out.count - 1].1 += 1 } else { out.append((c, 1)) } }
            return out
        }
        let t = runs(typed), w = runs(word)
        guard t.count == w.count else { return 0 }
        for i in t.indices where t[i].0 != w[i].0 || t[i].1 < w[i].1 { return 0 }
        return typed.count - word.count
    }
}
