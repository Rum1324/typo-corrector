import Foundation
import TypoCore

// typo-cli: exercise the correction engine without any system permissions.
//   typo-cli correct <word>...         show candidates and the decision
//   typo-cli eval [--n N] [--seed S] [--no-neural] [--min-prob P]
//   typo-cli simulate                  read key scripts from stdin ("teh<bs><bs><bs>the ")
//   typo-cli suite [--split dev|test|all] [--only CAT] [--ranker base|personal|none] [--dump FILE]
//                                      replay eval/cases through the app's policy and score every token
//   typo-cli bench                     time 2000 corrections
//   typo-cli tok-parity FILE           check the Swift tokenizer against training/parity.py's reference
//   typo-cli lm-parity FILE            check the Swift language model against PyTorch, and time it
//   typo-cli features --in TSV --hidden NPY --out DIR
//                                      option rows for simulated typing (training data of the decider)
// Model switches: --no-context, --context-dir DIR (lm.json, lm.bin, ranker2.json, ranker2.bin, context.json;
//                 files missing there come from Resources), --decider FILE,
//                 --ranker-dir DIR (ranker.json, ranker.bin: the rules' ranker)

let args = Array(CommandLine.arguments.dropFirst())
guard let command = args.first else {
    print("usage: typo-cli correct|eval|simulate|bench"); exit(1)
}

func flag(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}

var config = args.contains("--legacy") ? CorrectorConfig.legacy : CorrectorConfig()
if let p = flag("--min-prob").flatMap(Double.init) { config.minProbability = p }
if let w = flag("--w-neural").flatMap(Double.init) { config.weightNeural = w }
if let w = flag("--w-edit").flatMap(Double.init) { config.weightEdit = w }
if let w = flag("--w-prior").flatMap(Double.init) { config.weightPrior = w }
if args.contains("--no-first-word") { config.correctFirstWordAfterReset = false }
if let v = flag("--tail-min").flatMap(Double.init) { config.tailWordMinFrequency = v }
// Experiment switches: --cfg links=1,shift=1,first=1.5
for kv in (flag("--cfg") ?? "").split(separator: ",") {
    let p = kv.split(separator: "=", omittingEmptySubsequences: false).map(String.init)
    let k = p[0]
    guard !k.isEmpty, p.count <= 2, p.count == 1 || Double(p[1]) != nil else {
        FileHandle.standardError.write("bad --cfg entry '\(kv)': use key=number\n".data(using: .utf8)!); exit(1)
    }
    let v = p.count == 2 ? Double(p[1])! : 1
    switch k {
    case "links": config.skipLinksAndCode = v != 0
    case "shift": config.fixShiftSlips = v != 0
    case "first": config.firstLetterPenalty = v
    case "firstword": config.correctFirstWordAfterReset = v != 0
    case "minprob": config.minProbability = v
    case "edit": config.maxEditCost = v
    case "peredit": config.perEditPenalty = v
    case "minfreq": config.minCandidateFrequency = v
    case "knownmargin": config.knownWordMargin = v
    case "knowncost": config.knownWordMaxEditCost = v
    case "knownmax": config.knownWordMaxFrequency = v
    case "keep": config.keepLogScore = v
    case "keepcap": config.keepCapitalBonus = v
    case "keepletter": config.keepPerLetter = v
    case "startcap": config.unknownStartCapitalKeepBonus = v
    case "uckeep": config.uncuratedKeepOffset = v
    case "ucmax": config.uncuratedKnownMaxFrequency = v
    case "uccost": config.uncuratedKnownMaxEditCost = v
    case "ucmargin": config.uncuratedKnownMargin = v
    case "curkeep": config.curatedKeepBonus = v
    case "romaji": config.romajiKeepBonus = v
    case "share": config.minShareAmongFixes = v
    case "keepmax": config.maxKeepShare = v
    case "prefixcap": config.prefixStemMaxFrequency = v
    case "capname": config.capitalizedUncuratedIsName = v != 0
    case "inflectmin": config.inflectionStemMinFrequency = v
    case "inflect": config.inflectionKeepOffset = v
    case "elong": config.skipElongations = v != 0
    case "listedcaps": config.listedIgnoresCapitalization = v != 0
    case "split": config.splitMergedWords = v != 0
    case "splitcost": config.splitCost = v
    case "splitmin": config.splitPartMinFrequency = v
    case "del": config.edit.del = v
    case "deldouble": config.edit.delDouble = v
    case "subadj": config.edit.subAdjacent = v
    case "subvowel": config.edit.subVowel = v
    case "sub": config.edit.sub = v
    case "insadj": config.edit.insAdjacent = v
    case "ins": config.edit.ins = v
    case "trans": config.edit.transpose = v
    case "neural": config.weightNeural = v
    case "ctx": config.useContext = v != 0
    case "ctxprob": config.contextMinProbability = v
    case "ctxknown": config.contextKnownMinProbability = v
    case "ctxname": config.contextNameMinProbability = v
    case "ctxextra": config.contextExtraCandidates = Int(v)
    case "ctxminlen": config.contextMinLength = Int(v)
    case "maxcand": config.maxCandidates = Int(v)
    case "ctxstart": config.contextUnknownStartMinProbability = v
    case "ctxagree": config.contextAgreeMinProbability = v > 0 ? v : nil
    case "ctxattested": config.contextAttestedMinProbability = v > 0 ? v : nil
    case "ctxromaji": config.contextRomajiNameMinProbability = v
    case "ctxapos": config.contextAddsApostrophes = v != 0
    case "loose": config.contextLooseness = v
    case "contr": config.fixContractions = v != 0
    default: FileHandle.standardError.write("unknown --cfg key \(k)\n".data(using: .utf8)!); exit(1)
    }
}

let t0 = Date()
let corrector = try Corrector.loadDefault(config: config)
if args.contains("--no-neural") { corrector.ranker = nil }
func fail(_ message: String) -> Never { FileHandle.standardError.write((message + "\n").data(using: .utf8)!); exit(1) }
if args.contains("--no-context") {
    corrector.context = nil
} else if flag("--context-dir") != nil || flag("--decider") != nil {
    corrector.context = ContextModel.load(lexicon: corrector.lexicon, directory: flag("--context-dir").map { URL(fileURLWithPath: $0) },
                                          deciderFile: flag("--decider").map { URL(fileURLWithPath: $0) })
    if corrector.context == nil { fail("no usable language model in \(flag("--context-dir") ?? "Resources")") }
}
let rankerDir = flag("--ranker-dir").map { URL(fileURLWithPath: $0) }
if let rankerDir {
    guard let r = try? NeuralRanker(manifest: rankerDir.appendingPathComponent("ranker.json"), weights: rankerDir.appendingPathComponent("ranker.bin"))
    else { fail("no usable ranker in \(rankerDir.path)") }
    corrector.ranker = r
}
let loadMs = Int(Date().timeIntervalSince(t0) * 1000)
let contextNote = corrector.context.map { $0.decider == nil ? "language model only" : "on" } ?? "off"
FileHandle.standardError.write("loaded \(corrector.lexicon.count) words, ranker: \(corrector.ranker?.source.path ?? "none"), context: \(contextNote), \(loadMs) ms\n".data(using: .utf8)!)

struct RNG: RandomNumberGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// Synthetic typo generator mirroring training/synth.py.
func makeTypo(_ word: String, rng: inout RNG) -> String {
    let adj = EditModel.shared.adjacency
    var w = Array(word.utf8)
    let edits = Double.random(in: 0..<1, using: &rng) < 0.82 ? 1 : 2
    for _ in 0..<edits {
        guard w.count >= 2 else { break }
        let r = Double.random(in: 0..<1, using: &rng)
        let i = Int.random(in: 0..<w.count, using: &rng)
        if r < 0.32, let n = adj[w[i]]?.randomElement(using: &rng) { w[i] = n }
        else if r < 0.50 { if w.count > 3 { w.remove(at: i) } }
        else if r < 0.64, let n = adj[w[i]]?.randomElement(using: &rng) { w.insert(n, at: i) }
        else if r < 0.82 { if i + 1 < w.count && w[i] != w[i + 1] { w.swapAt(i, i + 1) } }
        else if r < 0.88 { w.insert(w[i], at: i) }
        else if r < 0.94 {
            if let j = (0..<w.count - 1).first(where: { w[$0] == w[$0 + 1] }) { w.remove(at: j) }
        } else {
            let vowels = Array("aeiou".utf8)
            if vowels.contains(w[i]) { w[i] = vowels.randomElement(using: &rng)! }
        }
    }
    let s = String(decoding: w, as: UTF8.self)
    return s == word ? makeTypo(word, rng: &rng) : s
}

switch command {
case "correct":
    // --before "text typed so far on the line" turns on the context path.
    let valued: Set<String> = ["--min-prob", "--before", "--cfg", "--context-dir", "--decider", "--ranker-dir", "--w-neural", "--w-edit", "--w-prior", "--tail-min"]
    let skipped = Set(valued.compactMap { flag($0) })
    let line = flag("--before").map { LineContext(before: $0, startKnown: true) }
    for word in args.dropFirst() where !word.hasPrefix("--") && !skipped.contains(word) {
        let a = corrector.analyze(word, sentenceStart: line.map { $0.before.trimmingCharacters(in: .whitespaces).isEmpty || ".!?:".contains($0.before.trimmingCharacters(in: .whitespaces).last!) } ?? true, line: line)
        print("\(word): " + (a.decision.map { "→ \($0.replacement) (p=\(String(format: "%.2f", $0.confidence)), \($0.reason))" } ?? "no change (\(a.note))"))
        for c in a.candidates.prefix(6) {
            print(String(format: "    %-16@ d=%d edit=%.2f prior=%.2f neural=%+.2f boost=%.1f  p=%.3f",
                         c.word as NSString, c.distance, c.editCost, c.logPrior, c.neural, c.boost, c.probability))
        }
    }

case "eval":
    let n = flag("--n").flatMap(Int.init) ?? 2000
    var rng = RNG(state: UInt64(flag("--seed").flatMap(Int.init) ?? 42))
    let lex = corrector.lexicon
    // Sample intended words with zipf >= 3.3 (at least ~2 per million), length >= 4, by sqrt frequency.
    let pool = (0..<lex.count).filter { lex.counts[$0] >= 2000 && lex.bytes[$0].count >= 4 && Corrector.isAsciiWord(lex.bytes[$0]) }
    let weights = pool.map { sqrt(lex.counts[$0]) }
    let total = weights.reduce(0, +)
    func sample() -> String {
        var r = Double.random(in: 0..<total, using: &rng)
        for (i, w) in weights.enumerated() { r -= w; if r <= 0 { return lex.strings[pool[i]] } }
        return lex.strings[pool.last!]
    }
    var fixed = 0, wrong = 0, missed = 0, falsePos = 0, untouched = 0
    var wrongExamples: [String] = [], missedExamples: [String] = [], fpExamples: [String] = []
    let start = Date()
    for _ in 0..<n {
        let word = sample()
        let typo = makeTypo(word, rng: &rng)
        if lex.frequency(typo) > lex.frequency(word) { continue }   // typo is itself a commoner word; not a fair test
        if let d = corrector.correct(typo) {
            if d.replacement == word { fixed += 1 } else { wrong += 1; if wrongExamples.count < 12 { wrongExamples.append("\(typo)→\(d.replacement) (wanted \(word))") } }
        } else { missed += 1; if missedExamples.count < 12 { missedExamples.append("\(typo) (wanted \(word))") } }
    }
    // False positives: correctly spelled words that get "fixed".
    let rarePool = (0..<lex.count).filter { lex.counts[$0] >= 300 && lex.counts[$0] < 20000 && lex.bytes[$0].count >= 4 && Corrector.isAsciiWord(lex.bytes[$0]) }
    for _ in 0..<n {
        let w = lex.strings[rarePool.randomElement(using: &rng)!]
        if let d = corrector.correct(w) { falsePos += 1; if fpExamples.count < 12 { fpExamples.append("\(w)→\(d.replacement)") } } else { untouched += 1 }
    }
    let ms = Date().timeIntervalSince(start) * 1000 / Double(2 * n)
    let tested = fixed + wrong + missed
    print(String(format: "typos: %d  fixed %.1f%%  wrong %.1f%%  missed %.1f%%", tested, 100.0 * Double(fixed) / Double(tested), 100.0 * Double(wrong) / Double(tested), 100.0 * Double(missed) / Double(tested)))
    print(String(format: "correct rare words: %d  false corrections %.2f%%", falsePos + untouched, 100.0 * Double(falsePos) / Double(falsePos + untouched)))
    print(String(format: "avg %.3f ms per decision", ms))
    print("wrong:  " + wrongExamples.joined(separator: ", "))
    print("missed: " + missedExamples.joined(separator: ", "))
    print("false+: " + fpExamples.joined(separator: ", "))

case "simulate":
    let tracker = TextTracker()
    while let line = readLine() {
        tracker.reset()
        var rest = Substring(line)
        while !rest.isEmpty {
            let input: KeyInput
            if rest.hasPrefix("<bs>") { input = .backspace; rest = rest.dropFirst(4) }
            else if rest.hasPrefix("<reset>") { input = .reset; rest = rest.dropFirst(7) }
            else if rest.hasPrefix("<ret>") { input = .char("\n"); rest = rest.dropFirst(5) }
            else { input = .char(rest.first!); rest = rest.dropFirst() }
            if let c = tracker.feed(input) {
                var out = "commit '\(c.word)' by '\(c.boundary == "\n" ? "\\n" : String(c.boundary))' ctx=\(c.context) sentenceStart=\(c.sentenceStart)"
                if let p = c.pair {
                    let ok = PairValidator.isPlausible(p, lexicon: corrector.lexicon, personal: corrector.personal)
                    out += "  LEARN \(p.typo)→\(p.fix) \(ok ? "✓" : "✗ rejected")"
                }
                if let d = corrector.correct(c.word, sentenceStart: c.sentenceStart) { out += "  FIX → \(d.replacement)" }
                print(out)
            }
        }
        print("line: \"\(tracker.text)\"")
    }

case "suite":
    // Bundled base ranker by default so results are reproducible (the app prefers personal weights).
    switch rankerDir != nil ? "given" : (flag("--ranker") ?? "base") {
    case "none": corrector.ranker = nil
    case "personal", "given": break
    default:
        if let m = Paths.resource("ranker.json"), let w = Paths.resource("ranker.bin") {
            corrector.ranker = try? NeuralRanker(manifest: m, weights: w)
        }
    }
    let dir = flag("--dir").map { URL(fileURLWithPath: $0) } ?? Paths.repoRoot.appendingPathComponent("eval/cases")
    let split = flag("--split") ?? "dev"
    let (cases, problems) = SuiteRunner.load(directory: dir, split: split, only: flag("--only"))
    for p in problems { FileHandle.standardError.write("skipped \(p)\n".data(using: .utf8)!) }
    let runner = SuiteRunner(corrector: corrector)
    let (stats, failures) = runner.run(cases, threads: flag("--threads").flatMap(Int.init) ?? 1)
    print("\(cases.count) cases (split: \(split), ranker: \(corrector.ranker?.source.lastPathComponent ?? "none"))\n")
    printSuiteReport(stats)
    // Partial, legacy or experimental runs get their own default file, so the full report is not overwritten.
    let tag = (flag("--only").map { "-" + $0 } ?? "") + (args.contains("--legacy") ? "-legacy" : "") + (flag("--cfg") != nil ? "-cfg" : "")
    let dump = flag("--dump").map { URL(fileURLWithPath: $0) } ?? Paths.repoRoot.appendingPathComponent("eval/reports/failures-\(split)\(tag).tsv")
    writeFailures(failures, to: dump)
    print("\n\(failures.count) failures written to \(dump.path)")

case "bench":
    var rng = RNG(state: 7)
    // 3+ letters: makeTypo cannot change a one-letter word and would retry forever.
    let common = (0..<5000).map { corrector.lexicon.strings[$0] }.filter { $0.count >= 3 && Corrector.isAsciiWord(Array($0.utf8)) }
    let words = (0..<2000).map { _ in common[Int.random(in: 0..<common.count, using: &rng)] }
    let typos = words.map { makeTypo($0, rng: &rng) }
    let s = Date()
    var n = 0
    for t in typos where corrector.correct(t) != nil { n += 1 }
    let ms = Date().timeIntervalSince(s) * 1000
    print(String(format: "%d corrections, %.3f ms each (%d applied)", typos.count, ms / Double(typos.count), n))

case "tok-parity":
    // Lines of: text <TAB> token ids (space-separated), written by training/parity.py.
    guard args.count > 1, let text = try? String(contentsOfFile: args[1], encoding: .utf8) else { fail("usage: typo-cli tok-parity FILE") }
    let tokenizer = ContextTokenizer(lexicon: corrector.lexicon)
    var lines = 0, bad = 0
    for row in text.split(separator: "\n", omittingEmptySubsequences: true) {
        let cols = row.split(separator: "\t", maxSplits: 1, omittingEmptySubsequences: false)
        guard cols.count == 2 else { continue }
        let want = cols[1].split(separator: " ").compactMap { Int32($0) }
        let got = tokenizer.encode(String(cols[0]))
        lines += 1
        if got != want {
            bad += 1
            if bad <= 5 { print("MISMATCH: \(cols[0])\n  python \(want)\n  swift  \(got)") }
        }
    }
    print("\(lines) lines, \(bad) tokenized differently")
    exit(bad == 0 && lines > 0 ? 0 : 1)

case "lm-parity":
    // Lines of: context ids <TAB> word ids <TAB> log-probabilities (PyTorch), written by training/parity.py.
    guard args.count > 1, let text = try? String(contentsOfFile: args[1], encoding: .utf8) else { fail("usage: typo-cli lm-parity FILE") }
    guard let lm = corrector.context?.lm else { fail("no language model loaded") }
    var worst: Float = 0, sum: Float = 0, count = 0, contexts = 0
    var seconds = 0.0, forward = 0.0
    for row in text.split(separator: "\n", omittingEmptySubsequences: true) {
        let cols = row.split(separator: "\t", omittingEmptySubsequences: false)
        guard cols.count == 3 else { continue }
        let ctx = cols[0].split(separator: " ").compactMap { Int32($0) }
        let words = cols[1].split(separator: " ").compactMap { Int32($0) }
        let want = cols[2].split(separator: " ").compactMap { Float($0) }
        let s = Date()
        let state = LMState(lm: lm, tokens: ctx)
        forward += Date().timeIntervalSince(s)
        let got = words.map { state.logProb($0) }
        seconds += Date().timeIntervalSince(s)
        contexts += 1
        for (g, w) in zip(got, want) { let e = abs(g - w); worst = max(worst, e); sum += e; count += 1 }
    }
    print(String(format: "%d contexts, %d log-probabilities: largest difference %.5f, mean %.6f; %.2f ms per context (%.2f ms of it the forward pass)",
                 contexts, count, worst, sum / Float(max(count, 1)), 1000 * seconds / Double(max(contexts, 1)), 1000 * forward / Double(max(contexts, 1))))
    exit(worst < 0.01 && count > 0 ? 0 : 1)

case "features":
    // --in:     TSV with a header: id, typed, sentence_start, start_known, n_ctx, cut
    // --hidden: .npy of float16 [N, d], the language-model state per id (training/lm_hidden.py)
    // --out:    DIR/rows.tsv (id, note, option words; "=" is keeping the word) and DIR/features.f32
    //           (float32, one row of ContextModel.featureNames per option, in rows.tsv order), DIR/columns.txt
    guard let input = flag("--in"), let hiddenPath = flag("--hidden"), let outDir = flag("--out") else { fail("usage: typo-cli features --in TSV --hidden NPY --out DIR") }
    guard let model = corrector.context else { fail("no language model loaded") }
    model.decider = nil
    guard let table = try? String(contentsOfFile: input, encoding: .utf8),
          let npy = try? Data(contentsOf: URL(fileURLWithPath: hiddenPath), options: .alwaysMapped) else { fail("cannot read the inputs") }
    guard npy.count > 12, npy[0] == 0x93, npy[6] == 1 else { fail("\(hiddenPath) is not a version 1 .npy file") }
    let headerLength = Int(npy[8]) | (Int(npy[9]) << 8)
    let header = String(decoding: npy[10..<(10 + headerLength)], as: UTF8.self)
    guard header.contains("'<f2'"), header.contains("'fortran_order': False") else { fail("expected C-ordered float16 states: \(header)") }
    let d = model.lm.config.d, dataStart = 10 + headerLength
    let stored = (npy.count - dataStart) / (2 * d)
    try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
    let rowsURL = URL(fileURLWithPath: outDir).appendingPathComponent("rows.tsv")
    let featURL = URL(fileURLWithPath: outDir).appendingPathComponent("features.f32")
    FileManager.default.createFile(atPath: rowsURL.path, contents: nil)
    FileManager.default.createFile(atPath: featURL.path, contents: nil)
    guard let rowsOut = try? FileHandle(forWritingTo: rowsURL), let featOut = try? FileHandle(forWritingTo: featURL) else { fail("cannot write to \(outDir)") }
    try? ContextModel.featureNames.joined(separator: "\n").write(to: URL(fileURLWithPath: outDir).appendingPathComponent("columns.txt"), atomically: true, encoding: .utf8)
    var done = 0, withRows = 0
    let lines = Array(table.split(separator: "\n", omittingEmptySubsequences: true).dropFirst())
    let workers = max(1, flag("--threads").flatMap(Int.init) ?? 1)
    /// One word: its rows.tsv line and its option rows.
    func rowsFor(_ row: Substring, _ n: Int) -> (String, [Float], Bool) {
        var rowsText = "", floats: [Float] = []
        var withRows = 0
        let c = row.split(separator: "\t", omittingEmptySubsequences: false)
        guard c.count >= 6, let id = Int(c[0]), id >= 0, id < stored, let length = Int(c[4]) else { fail("bad row \(n + 2) in \(input)") }
        let typed = String(c[1])
        var hidden = [Float](repeating: 0, count: d)
        npy.withUnsafeBytes { raw in
            let base = dataStart + id * 2 * d
            for j in 0..<d { hidden[j] = Float(Float16(bitPattern: raw.loadUnaligned(fromByteOffset: base + 2 * j, as: UInt16.self))) }
        }
        let state = LMState(lm: model.lm, hidden: hidden, contextLength: length, startsWithCut: c[5] == "1")
        let decision = AutocorrectPolicy.decide(word: typed, boundary: " ", sentenceStart: c[2] == "1", startKnown: c[3] == "1",
                                                prefix: "", line: nil, state: state, with: corrector)
        let rows = decision.analysis?.rows
        let note = decision.analysis?.decision.map { "fixed:" + $0.replacement } ?? decision.note
        rowsText += "\(id)\t\(note)\t\(rows?.words.joined(separator: " ") ?? "")\n"
        if let rows { for f in rows.features { floats.append(contentsOf: f) }; withRows += 1 }
        return (rowsText, floats, withRows > 0)
    }
    let batch = 20_000
    for start in stride(from: 0, to: lines.count, by: batch) {
        let end = min(start + batch, lines.count)
        var results = [(String, [Float], Bool)?](repeating: nil, count: end - start)
        let lock = NSLock()
        DispatchQueue.concurrentPerform(iterations: min(workers, end - start)) { w in
            var i = start + w
            while i < end {
                let r = rowsFor(lines[i], i)
                lock.lock(); results[i - start] = r; lock.unlock()
                i += workers
            }
        }
        var text = "", floats: [Float] = []
        for r in results { text += r!.0; floats.append(contentsOf: r!.1); if r!.2 { withRows += 1 } }
        rowsOut.write(text.data(using: .utf8)!)
        floats.withUnsafeBufferPointer { featOut.write(Data(buffer: $0)) }
        done = end
        FileHandle.standardError.write("\(done) words\n".data(using: .utf8)!)
    }
    print("\(done) words, \(withRows) with option rows, \(ContextModel.featureNames.count) columns: \(ContextModel.featureNames.joined(separator: " "))")

default:
    print("unknown command \(command)"); exit(1)
}
