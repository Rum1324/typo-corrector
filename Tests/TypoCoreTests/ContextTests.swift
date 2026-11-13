import XCTest
@testable import TypoCore

/// The context path: tokenizer, language model, and decisions that need the sentence.
/// Exact agreement with PyTorch is checked by `typo-cli tok-parity` / `lm-parity` (training/parity.py).
final class ContextTests: XCTestCase {
    static let corrector: Corrector = {
        let c = try! Corrector.loadDefault()
        if let m = Paths.resource("ranker.json"), let w = Paths.resource("ranker.bin") {
            c.ranker = try? NeuralRanker(manifest: m, weights: w)
        }
        return c
    }()

    private func fixes(_ text: String) -> [String] {
        let t = TextTracker()
        t.feed(.char("\n"))
        var out: [String] = []
        for ch in text {
            guard let commit = t.feed(.char(ch)) else { continue }
            switch AutocorrectPolicy.decide(commit, with: Self.corrector).action {
            case .now(let c), .deferred(let c):
                out.append("\(commit.word)>\(c.replacement)")
                t.replaceLastWord(with: c.replacement, original: commit.word)
            case .none: break
            }
        }
        return out
    }

    func testTokenizerSpecialTokens() {
        let tok = ContextTokenizer(lexicon: Self.corrector.lexicon)
        let the = tok.wordId("the", capitalised: false)
        XCTAssertGreaterThanOrEqual(the, ContextTokenizer.reserved)
        XCTAssertEqual(tok.encode("The the THE"), [the, the, the])
        XCTAssertEqual(tok.encode("the, the. the?! the; 3.5 10,000 12:30"),
                       [the, ContextTokenizer.comma, the, ContextTokenizer.dot, the, ContextTokenizer.question, the, ContextTokenizer.colon,
                        ContextTokenizer.num, ContextTokenizer.num, ContextTokenizer.num])
        XCTAssertEqual(tok.encode("github.com/user/repo file.txt my_var h264enc"), Array(repeating: ContextTokenizer.sym, count: 4))
        XCTAssertEqual(tok.encode("Zxqvwk zxqvwk"), [ContextTokenizer.name, ContextTokenizer.unk])
        XCTAssertEqual(tok.encode("don\u{2019}t"), tok.encode("don't"))
        XCTAssertEqual(tok.encode("(the) \"the\" the's"), [the, the, the])
    }

    func testContextStartsWithCutWhenTheLineStartIsUnknownOrTooLong() {
        let tok = ContextTokenizer(lexicon: Self.corrector.lexicon)
        XCTAssertEqual(tok.context(before: "", lineStartKnown: true, maxLen: 32), [ContextTokenizer.bos])
        XCTAssertEqual(tok.context(before: "", lineStartKnown: false, maxLen: 32), [ContextTokenizer.cut])
        let long = tok.context(before: String(repeating: "the ", count: 40), lineStartKnown: true, maxLen: 32)
        XCTAssertEqual(long.count, 32)
        XCTAssertEqual(long.first, ContextTokenizer.cut)
    }

    func testTrackerHandsTheLineToThePolicy() {
        let t = TextTracker()
        t.feed(.char("\n"))
        var last: Commit?
        for ch in "I went to th " { if let c = t.feed(.char(ch)) { last = c } }
        XCTAssertEqual(last?.word, "th")
        XCTAssertEqual(last?.lineBefore, "I went to ")
        XCTAssertEqual(last?.lineStartKnown, true)
        t.feed(.reset)
        for ch in "ok th " { if let c = t.feed(.char(ch)) { last = c } }
        XCTAssertEqual(last?.lineBefore, "ok ")
        XCTAssertEqual(last?.lineStartKnown, false)
    }

    func testLanguageModelPrefersTheWordThatFits() throws {
        guard let model = Self.corrector.context else { throw XCTSkip("no language model in Resources") }
        let state = model.state(for: LineContext(before: "Thank you very ", startKnown: true))
        let much = state.logProb(model.tokenizer.wordId("much", capitalised: false))
        let mulch = state.logProb(model.tokenizer.wordId("mulch", capitalised: false))
        XCTAssertGreaterThan(much, mulch + 5)
        XCTAssertGreaterThan(much, -3, "\"much\" should be likely after \"Thank you very\"")
        XCTAssertLessThanOrEqual(state.topLogProb, 0)
    }

    func testShortAndRealWordSlipsAreFixedFromContext() throws {
        try XCTSkipIf(Self.corrector.context?.decider == nil, "no context model in Resources")
        XCTAssertEqual(fixes("I went to th "), ["th>the"], "two-letter slips are fixable with context")
        XCTAssertEqual(fixes("Let me know if you have ay "), ["ay>any"])
    }

    func testCorrectWordsThatLookLikeSlipsStay() throws {
        try XCTSkipIf(Self.corrector.context?.decider == nil, "no context model in Resources")
        XCTAssertEqual(fixes("Please fill out the form before Friday "), [])
        XCTAssertEqual(fixes("dont worry, im on my way "), [], "contractions typed without apostrophes are left alone")
        XCTAssertEqual(fixes("that was sooo good "), [], "stretched words are typed on purpose")
        XCTAssertEqual(fixes("we use backprop and timesteps here "), [], "words the lexicon lacks but the corpus knows")
    }

    func testContractionsAreFixedFromTheListOnlyWhenTurnedOn() {
        XCTAssertEqual(fixes("this doesnt work "), [])
        let saved = Self.corrector.config.fixContractions
        Self.corrector.config.fixContractions = true
        defer { Self.corrector.config.fixContractions = saved }
        XCTAssertEqual(fixes("this doesnt work and im sure thats why "), ["doesnt>doesn't", "im>I'm", "thats>that's"])
        XCTAssertEqual(fixes("Dont worry "), ["Dont>Don't"])
        XCTAssertEqual(fixes("its fine "), [], "a bare form that is a word of its own never gets an apostrophe")
        let c = Self.corrector
        func decide(_ word: String, start: Bool, before: String) -> Correction? {
            c.analyze(word, sentenceStart: start, positionUnknown: false, line: LineContext(before: before, startKnown: true), state: nil).decision
        }
        XCTAssertEqual(decide("Im", start: true, before: "")?.replacement, "I'm")
        XCTAssertNotEqual(decide("Ive", start: false, before: "I met Jony ")?.reason, "contraction", "a capitalised word inside a sentence may be a name")
        XCTAssertEqual(decide("DOnt", start: true, before: "")?.replacement, "Don't", "shift slip and apostrophe in one fix")
        XCTAssertEqual(decide("mustve", start: false, before: "you ")?.replacement, "must've")
        c.personal.addApplied(typo: "mustve", fix: "must've"); c.personal.addUndo(typo: "mustve", fix: "must've")
        XCTAssertNil(decide("mustve", start: false, before: "you "), "taken back once and never kept: left alone")
        for word in ["its", "were", "well", "ill", "hell", "shell", "wed", "shed", "id", "lets"] {
            XCTAssertNil(Corrector.contractions[word], "\(word) is a word")
        }
        for (bare, full) in Corrector.contractions {
            XCTAssertEqual(full.lowercased().replacingOccurrences(of: "'", with: ""), bare)
        }
    }

    func testLoosenessLowersTheBars() {
        var c = CorrectorConfig()
        XCTAssertEqual(c.loosened(0.9), 0.9)
        c.contextLooseness = 1
        XCTAssertEqual(c.loosened(0.9), 0.768, accuracy: 0.001)
        XCTAssertEqual(c.loosened(0.97), 0.922, accuracy: 0.001)
        XCTAssertEqual(c.loosened(0.95, by: 0.5), 0.920, accuracy: 0.001)
    }

    func testALooserSettingFixesWhatTheCarefulOneLeaves() throws {
        try XCTSkipIf(Self.corrector.context?.decider == nil, "no context model in Resources")
        XCTAssertEqual(fixes("I want ot go "), [], "\"ot\" is in the dictionary: 96% is under the careful bar of 97%")
        let saved = Self.corrector.config.contextLooseness
        Self.corrector.config.contextLooseness = 1
        defer { Self.corrector.config.contextLooseness = saved }
        XCTAssertEqual(fixes("I want ot go "), ["ot>to"])
    }

    func testALoneIIsNeverChanged() throws {
        try XCTSkipIf(Self.corrector.context?.decider == nil, "no context model in Resources")
        let saved = Self.corrector.config.contextLooseness
        Self.corrector.config.contextLooseness = 2
        defer { Self.corrector.config.contextLooseness = saved }
        let a = Self.corrector.analyze("i", sentenceStart: false, positionUnknown: false,
                                       line: LineContext(before: "so ", startKnown: true), state: nil)
        XCTAssertNil(a.decision)
        XCTAssertEqual(a.note, "kept (pronoun)")
    }

    func testRulesDecideWithoutTheContextModel() {
        let saved = Self.corrector.config.useContext
        Self.corrector.config.useContext = false
        defer { Self.corrector.config.useContext = saved }
        XCTAssertEqual(fixes("I went to th "), [], "two-letter words are skipped by the rules")
        XCTAssertEqual(fixes("this is definately right "), ["definately>definitely"])
    }

    func testElongationCount() {
        func e(_ a: String, _ b: String) -> Int { Corrector.elongation(Array(a.utf8), of: Array(b.utf8)) }
        XCTAssertEqual(e("sooo", "so"), 2)
        XCTAssertEqual(e("crazyyy", "crazy"), 2)
        XCTAssertEqual(e("gratefull", "grateful"), 1)
        XCTAssertEqual(e("tthe", "the"), 1)
        XCTAssertEqual(e("teh", "the"), 0)
        XCTAssertTrue(Corrector.sameLetters(Array("dont".utf8), Array("don't".utf8)))
        XCTAssertFalse(Corrector.sameLetters(Array("dont".utf8), Array("done".utf8)))
    }
}
