import Foundation
import TypoCore

// Accuracy suite (`typo-cli suite`, AccuracySuiteTests): replays the accuracy case files in eval/cases through the same tracker and
// correction policy the app uses, and scores every token.
//
// Case file format (one case per line, TAB-separated):  typed <TAB> expected [<TAB> note]
//   - typed is fed key by key, then Return. A leading "<reset>" means the line starts after a click
//     or app switch (text before the caret unknown); otherwise it starts on a fresh line.
//   - expected has the same space-separated tokens as typed; "a|b" lists acceptable alternatives,
//     "␣" inside a token is a space the corrector should insert.
// File names are <category>.<split>.tsv; categories starting with "clean" contain no typos.

public struct SuiteCase {
    public let category: String
    public let split: String
    public let line: Int
    public let reset: Bool
    public let typed: String
    public let expected: [[String]]
    public let note: String
}

public enum TokenOutcome: String {
    case untouched, falseFix = "false-fix", acceptableChange = "ok-change"
    case fixed, wrong, missed
}

public struct CommitTrace {
    public let token: Int
    public let word: String
    public let note: String
    public let candidates: String
    public let applied: String?
}

public struct SuiteStats {
    public var typos = 0, fixed = 0, wrong = 0, missed = 0
    public var realWordTypos = 0, realWordFixed = 0, realWordWrong = 0
    public var correctTokens = 0, falseFixes = 0, acceptableChanges = 0
    public init() {}
    public mutating func add(_ o: SuiteStats) {
        typos += o.typos; fixed += o.fixed; wrong += o.wrong; missed += o.missed
        realWordTypos += o.realWordTypos; realWordFixed += o.realWordFixed; realWordWrong += o.realWordWrong
        correctTokens += o.correctTokens; falseFixes += o.falseFixes; acceptableChanges += o.acceptableChanges
    }
}

public final class SuiteRunner {
    public let corrector: Corrector
    public init(corrector: Corrector) { self.corrector = corrector }

    static let spaceMark: Character = "␣"

    public static func load(directory: URL, split: String?, only: String?) -> (cases: [SuiteCase], problems: [String]) {
        var cases: [SuiteCase] = [], problems: [String] = []
        let files = ((try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "tsv" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        for file in files {
            let stem = file.deletingPathExtension().lastPathComponent
            guard let dot = stem.lastIndex(of: ".") else { problems.append("\(file.lastPathComponent): name is not <category>.<split>.tsv"); continue }
            let category = String(stem[..<dot]), fileSplit = String(stem[stem.index(after: dot)...])
            if let split, split != "all", split != fileSplit { continue }
            if let only, !category.hasPrefix(only) { continue }
            guard let text = try? String(contentsOf: file, encoding: .utf8) else { problems.append("\(file.lastPathComponent): unreadable"); continue }
            for (n, raw) in text.components(separatedBy: "\n").enumerated() {
                let line = raw.trimmingCharacters(in: CharacterSet(charactersIn: "\r"))
                if line.trimmingCharacters(in: .whitespaces).isEmpty || line.hasPrefix("#") { continue }
                let cols = line.components(separatedBy: "\t")
                guard cols.count >= 2 else { problems.append("\(file.lastPathComponent):\(n + 1): no TAB"); continue }
                var typed = cols[0]
                var reset = false
                if typed.hasPrefix("<reset>") {
                    reset = true
                    typed = String(typed.dropFirst("<reset>".count).drop(while: { $0 == " " }))
                }
                let tt = typed.split(separator: " ")
                let et = cols[1].split(separator: " ")
                guard tt.count == et.count, !tt.isEmpty else {
                    problems.append("\(file.lastPathComponent):\(n + 1): \(tt.count) typed vs \(et.count) expected tokens")
                    continue
                }
                if typed.contains(spaceMark) { problems.append("\(file.lastPathComponent):\(n + 1): ␣ in typed"); continue }
                let expected = et.map { $0.split(separator: "|").map(String.init) }
                cases.append(SuiteCase(category: category, split: fileSplit, line: n + 1, reset: reset, typed: typed,
                                       expected: expected, note: cols.count > 2 ? cols[2] : ""))
            }
        }
        return (cases, problems)
    }

    /// Types `typed` (then Return) the way the app sees it and returns the resulting text plus a trace.
    public func simulate(_ c: SuiteCase) -> (output: String, trace: [CommitTrace]) {
        let tracker = TextTracker()
        if !c.reset { tracker.feed(.char("\n")) }   // a fresh line: the start of the first word is known
        var out: [Character] = []
        var trace: [CommitTrace] = []
        var pending: (word: String, fix: String, punct: Character, traceIndex: Int)?
        var token = 0
        var prevSpace = true
        for ch in Array(c.typed) + ["\n"] {
            if let p = pending {
                pending = nil
                let held = Array(p.word) + [p.punct]
                if AutocorrectPolicy.confirmsDeferred(ch), out.count >= held.count, Array(out.suffix(held.count)) == held,
                   tracker.endsWith(p.word + String(p.punct)) {
                    out.removeLast(held.count)
                    out.append(contentsOf: Self.mark(p.fix)); out.append(p.punct)
                    tracker.feed(.char(ch))
                    if ch != "\n" { tracker.replaceWord(p.word, endingAt: 2, with: p.fix, original: p.word) }
                    let t = trace[p.traceIndex]
                    trace[p.traceIndex] = CommitTrace(token: t.token, word: t.word, note: t.note, candidates: t.candidates, applied: p.fix)
                    out.append(ch)
                    if ch == " " { if !prevSpace { token += 1 }; prevSpace = true } else { prevSpace = false }
                    continue
                }
            }
            out.append(ch)
            if let commit = tracker.feed(.char(ch)) {
                let d = AutocorrectPolicy.decide(commit, with: corrector)
                let cands = d.analysis?.candidates.prefix(3).map { String(format: "%@ %.2f", $0.word, $0.probability) }.joined(separator: ", ") ?? ""
                trace.append(CommitTrace(token: token, word: commit.word, note: d.note, candidates: cands, applied: nil))
                switch d.action {
                case .none: break
                case .now(let fix):
                    let boundary = out.removeLast()
                    out.removeLast(commit.word.count)
                    out.append(contentsOf: Self.mark(fix.replacement)); out.append(boundary)
                    tracker.replaceLastWord(with: fix.replacement, original: commit.word)
                    let t = trace[trace.count - 1]
                    trace[trace.count - 1] = CommitTrace(token: t.token, word: t.word, note: t.note, candidates: t.candidates, applied: fix.replacement)
                case .deferred(let fix):
                    pending = (commit.word, fix.replacement, ch, trace.count - 1)
                }
            }
            if ch == " " { if !prevSpace { token += 1 }; prevSpace = true } else { prevSpace = false }
        }
        if out.last == "\n" { out.removeLast() }
        return (String(out), trace)
    }

    /// Spaces inside a replacement are written as ␣ so output tokens stay aligned with typed tokens.
    private static func mark(_ s: String) -> [Character] { s.map { $0 == " " ? spaceMark : $0 } }

    public func isRealWord(_ token: String) -> Bool {
        let core = token.trimmingCharacters(in: CharacterSet.letters.inverted).lowercased()
        return core.count >= 2 && corrector.lexicon.frequency(core) >= 1000 && corrector.misspellings[core] == nil
    }

    public func score(_ c: SuiteCase, output: String) -> [(index: Int, outcome: TokenOutcome, typed: String, out: String)] {
        let tt = c.typed.split(separator: " ").map(String.init)
        let ot = output.split(separator: " ").map(String.init)
        var res: [(Int, TokenOutcome, String, String)] = []
        for i in tt.indices {
            let t = tt[i], o = i < ot.count ? ot[i] : "∅", alts = c.expected[i]
            let outcome: TokenOutcome
            if alts.contains(t) {
                outcome = o == t ? .untouched : (alts.contains(o) ? .acceptableChange : .falseFix)
            } else {
                outcome = alts.contains(o) ? .fixed : (o == t ? .missed : .wrong)
            }
            res.append((i, outcome, t, o))
        }
        return res
    }

    public struct Failure { public let c: SuiteCase; public let kind: TokenOutcome; public let typed: String; public let out: String; public let expected: String; public let realWord: Bool; public let trace: String }

    /// `threads` > 1 replays cases in parallel (each case has its own tracker; the corrector is only read).
    public func run(_ cases: [SuiteCase], threads: Int = 1) -> (stats: [String: SuiteStats], failures: [Failure]) {
        var stats: [String: SuiteStats] = [:]
        var failures: [Failure] = []
        var simulated = [(output: String, trace: [CommitTrace])?](repeating: nil, count: cases.count)
        let workers = max(1, min(threads, cases.count))
        if workers > 1 {
            let lock = NSLock()
            DispatchQueue.concurrentPerform(iterations: workers) { w in
                var i = w
                while i < cases.count {
                    let r = simulate(cases[i])
                    lock.lock(); simulated[i] = r; lock.unlock()
                    i += workers
                }
            }
        }
        for (n, c) in cases.enumerated() {
            let (output, trace) = simulated[n] ?? simulate(c)
            var s = SuiteStats()
            for r in score(c, output: output) {
                let rw = isRealWord(r.typed)
                switch r.outcome {
                case .untouched: s.correctTokens += 1
                case .acceptableChange: s.correctTokens += 1; s.acceptableChanges += 1
                case .falseFix: s.correctTokens += 1; s.falseFixes += 1
                case .fixed: s.typos += 1; s.fixed += 1; if rw { s.realWordTypos += 1; s.realWordFixed += 1 }
                case .wrong: s.typos += 1; s.wrong += 1; if rw { s.realWordTypos += 1; s.realWordWrong += 1 }
                case .missed: s.typos += 1; s.missed += 1; if rw { s.realWordTypos += 1 }
                }
                if r.outcome == .falseFix || r.outcome == .wrong || r.outcome == .missed {
                    let tr = trace.filter { $0.token == r.index }
                        .map { "\($0.word): \($0.applied.map { "→\($0)" } ?? $0.note)\($0.candidates.isEmpty ? "" : " [\($0.candidates)]")" }
                        .joined(separator: "; ")
                    failures.append(Failure(c: c, kind: r.outcome, typed: r.typed, out: r.out,
                                            expected: c.expected[r.index].joined(separator: "|"), realWord: rw, trace: tr))
                }
            }
            stats["\(c.category).\(c.split)", default: SuiteStats()].add(s)
        }
        return (stats, failures)
    }
}

