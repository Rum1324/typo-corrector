import XCTest
@testable import TypoCore

/// Behaviour added with the accuracy suite (eval/cases), plus a regression guard on the suite itself.
final class AccuracyTests: XCTestCase {
    static let corrector: Corrector = {
        let c = try! Corrector.loadDefault()
        // Bundled weights, not this Mac's personal fine-tune, so results match on every machine.
        if let m = Paths.resource("ranker.json"), let w = Paths.resource("ranker.bin") {
            c.ranker = try? NeuralRanker(manifest: m, weights: w)
        }
        return c
    }()

    /// Types `text` after a click (`reset`) or on a fresh line and returns the policy's fixes.
    private func fixes(_ text: String, reset: Bool) -> [String] {
        let t = TextTracker()
        if !reset { t.feed(.char("\n")) }
        var out: [String] = []
        for ch in text {
            guard let commit = t.feed(.char(ch)) else { continue }
            switch AutocorrectPolicy.decide(commit, with: Self.corrector).action {
            case .now(let c), .deferred(let c): out.append("\(commit.word)>\(c.replacement)")
            case .none: break
            }
        }
        return out
    }

    func testFirstWordAfterClickIsFixedUnlessItCouldBeATail() {
        XCTAssertEqual(fixes("teh cat ", reset: true), ["teh>the"])
        XCTAssertEqual(fixes("Coudl you ", reset: true), ["Coudl>Could"])
        XCTAssertEqual(fixes("THe meeting ", reset: true), ["THe>The"], "a capital starts a word: not a tail")
        // Without the tail check some word tails get "fixed"; with it, none do.
        let tails = ["acked up ", "ared it ", "ands on ", "ople came ", "ernal links ", "ure thing ", "tion of it ", "ically so "]
        let saved = Self.corrector.config.tailWordMinFrequency
        Self.corrector.config.tailWordMinFrequency = 1e18   // no word that common: check disabled
        let unguarded = tails.flatMap { fixes($0, reset: true) }
        Self.corrector.config.tailWordMinFrequency = saved
        XCTAssertFalse(unguarded.isEmpty, "at least one tail must be correctable for this test to mean anything")
        for line in tails { XCTAssertEqual(fixes(line, reset: true), [], "\(line) may be the tail of a longer word") }
        XCTAssertEqual(fixes("Haruto said hi ", reset: true), [], "a name, not a typo")
    }

    func testNamesAtTheStartOfAMessageStay() throws {
        try XCTSkipIf(Self.corrector.curated == nil, "no /usr/share/dict/web2 on this machine")
        for line in ["Yuu, can you check this ", "Mayu said it was fine ", "Timo, are you coming ", "Persson will join us "] {
            XCTAssertEqual(fixes(line, reset: true), [], line)
            XCTAssertEqual(fixes(line, reset: false), [], line)
        }
    }

    func testFirstLetterSlipsAreFixed() {
        let c = Self.corrector
        XCTAssertEqual(c.correct("slso", sentenceStart: false)?.replacement, "also")
        XCTAssertEqual(c.correct("vrom", sentenceStart: false)?.replacement, "from")
        XCTAssertEqual(c.correct("eith", sentenceStart: false)?.replacement, "with")
        XCTAssertNil(c.correct("vars", sentenceStart: false), "first-letter change on a code word")
    }

    /// Fixes held at punctuation are applied only when the next key ends the clause.
    func testDeferredPunctuation() {
        let runner = SuiteRunner(corrector: Self.corrector)
        func type(_ s: String) -> String {
            runner.simulate(SuiteCase(category: "t", split: "t", line: 1, reset: false, typed: s, expected: [], note: "")).output
        }
        XCTAssertEqual(type("see teh.js now"), "see teh.js now", "a file name")
        XCTAssertEqual(type("see youtu.be now"), "see youtu.be now", "a short link")
        XCTAssertEqual(type("i saw teh. then"), "i saw the. then", "a sentence end")
        XCTAssertEqual(type("ok hte, fine"), "ok the, fine")
    }

    func testLinksPathsAndIdentifiersAreLeftAlone() {
        XCTAssertEqual(fixes("see github.com/someuser/repo-nmae now ", reset: false), [])
        XCTAssertEqual(fixes("set my_vairable to two ", reset: false), [])
        XCTAssertEqual(fixes("ping @recieve or #recieve ", reset: false), [])
        XCTAssertEqual(fixes("i recieve it ", reset: false), ["recieve>receive"])
    }

    func testShiftSlips() {
        let c = Self.corrector
        XCTAssertEqual(c.correct("THe")?.replacement, "The")
        XCTAssertEqual(c.correct("WHat")?.replacement, "What")
        XCTAssertEqual(c.correct("DOn't")?.replacement, "Don't")
        XCTAssertNil(c.correct("IDs"), "plural acronym")
        XCTAssertNil(c.correct("PRs"), "plural acronym")
        for w in ["DMed", "IDed", "TAing", "BEd", "KOd", "LAer", "BEng"] {
            XCTAssertNil(c.correct(w, sentenceStart: false), "\(w): acronym with a suffix")
        }
    }

    func testJargonSlangAndInflectionsAreKept() {
        let c = Self.corrector
        for w in ["gyoza", "karaage", "refactor", "rebase", "embeddings", "aliased", "annotating", "rerunning",
                  "mockups", "tldr", "linter", "webhook", "codegen", "conda"] {
            XCTAssertNil(c.correct(w, sentenceStart: false), w)
        }
        XCTAssertNil(c.correct("Haruto", sentenceStart: true))
        XCTAssertNil(c.correct("Figma", sentenceStart: true))
    }

    func testMisspelledInflectionsAreStillFixed() {
        let c = Self.corrector
        XCTAssertEqual(c.correct("forgeting", sentenceStart: false)?.replacement, "forgetting")
        XCTAssertEqual(c.correct("prefrom", sentenceStart: false)?.replacement, "perform", "pre + from is not a word")
        XCTAssertEqual(c.correct("presure", sentenceStart: false)?.replacement, "pressure")
    }

    func testInflectedRealWordsAreLeftAlone() throws {
        try XCTSkipIf(Self.corrector.curated == nil, "no /usr/share/dict/web2 on this machine")
        let c = Self.corrector
        for w in ["sways", "culls", "doling", "chanced", "homed", "crated", "convolutions", "aubergines", "adjudicators"] {
            XCTAssertNil(c.correct(w, sentenceStart: false), w)
        }
    }

    func testMergedWordsAreSplit() {
        let c = Self.corrector
        XCTAssertEqual(c.correct("inthe", sentenceStart: false)?.replacement, "in the")
        XCTAssertEqual(c.correct("ofthe", sentenceStart: false)?.replacement, "of the")
        XCTAssertNil(c.correct("hotfix", sentenceStart: false), "compound jargon is not split")
        XCTAssertNil(c.correct("inbox", sentenceStart: false), "a real compound stays")
        XCTAssertNil(c.correct("online", sentenceStart: false), "a real compound stays")
        XCTAssertNil(c.correct("init", sentenceStart: false), "code word, not \"in it\"")
        for w in ["headin", "seein", "showin", "redis", "whois", "goto", "onclick", "matcha", "forgoing", "meany"] {
            XCTAssertFalse(c.correct(w, sentenceStart: false)?.replacement.contains(" ") ?? false, "\(w) must not be split")
        }
        XCTAssertNil(c.correct("Inman", sentenceStart: true), "a capitalised word is not split")
    }

    func testCuratedListSeparatesWebMisspellingsFromRareWords() throws {
        try XCTSkipIf(Self.corrector.curated == nil, "no /usr/share/dict/web2 on this machine")
        let c = Self.corrector
        XCTAssertEqual(c.correct("befor", sentenceStart: false)?.replacement, "before")
        XCTAssertEqual(c.correct("somthing", sentenceStart: false)?.replacement, "something")
        for w in ["mien", "tarn", "sate", "cleat", "tome", "comet"] {
            XCTAssertNil(c.correct(w, sentenceStart: false), w)
        }
    }

    func testTailIndex() {
        let lex = Self.corrector.lexicon
        XCTAssertTrue(lex.endsLongerWord(Array("ple".utf8), minFrequency: 1000))
        XCTAssertFalse(lex.endsLongerWord(Array("teh".utf8), minFrequency: 1000))
        XCTAssertFalse(lex.endsLongerWord(Array("people".utf8), minFrequency: 1_000_000_000), "no word that common")
    }

    /// The dev split of the accuracy suite must not get worse than when the defaults were tuned.
    /// 2026-11-11, context path (language model + decider): classic sets 1,119 of 1,291 typos fixed (86.7%),
    /// 7 wrong; real typing (keystrokes) 2,523 of 4,723 (53.4%); 14 false fixes on correct words.
    /// 2026-11-09, rules alone: classic sets 1,066 of 1,291 (82.6%), 6 wrong; real typing 35.2%; 19 false fixes.
    func testAccuracySuiteDevSplitDoesNotRegress() throws {
        #if DEBUG
        // ~60k analyses take minutes unoptimised. Run: swift test -c release -Xswiftc -enable-testing --filter AccuracyTests
        throw XCTSkip("release builds only")
        #endif
        let dir = Paths.repoRoot.appendingPathComponent("eval/cases")
        let (cases, problems) = SuiteRunner.load(directory: dir, split: "dev", only: nil)
        try XCTSkipIf(cases.isEmpty, "eval/cases not found")
        XCTAssertTrue(problems.isEmpty, problems.joined(separator: "\n"))
        var classic = SuiteStats(), keys = SuiteStats(), all = SuiteStats()
        for (name, s) in SuiteRunner(corrector: Self.corrector).run(cases, threads: 4).stats {
            all.add(s)
            if name.hasPrefix("keystrokes.") { keys.add(s) } else if !name.hasPrefix("realword.") { classic.add(s) }
        }
        let context = Self.corrector.context?.decider != nil      // without the model files the rules decide alone
        XCTAssertGreaterThanOrEqual(Double(classic.fixed) / Double(classic.typos), context ? 0.85 : 0.82, "fixed rate, classic sets")
        XCTAssertLessThanOrEqual(Double(classic.wrong) / Double(classic.typos), 0.008, "wrong-fix rate, classic sets")
        if keys.typos > 0 {
            XCTAssertGreaterThanOrEqual(Double(keys.fixed) / Double(keys.typos), context ? 0.50 : 0.33, "fixed rate, real typing")
        }
        XCTAssertLessThanOrEqual(all.falseFixes, 21, "false fixes on correct words")
    }
}
