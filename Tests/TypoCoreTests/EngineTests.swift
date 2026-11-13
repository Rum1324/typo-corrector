import XCTest
@testable import TypoCore

final class EngineTests: XCTestCase {
    static var corrector: Corrector = {
        try! Corrector.loadDefault()
    }()

    func testOSA() {
        XCTAssertEqual(EditDistance.osa("teh", "the", max: 2), 1)
        XCTAssertEqual(EditDistance.osa("recieve", "receive", max: 2), 1)
        XCTAssertEqual(EditDistance.osa("abc", "abc", max: 2), 0)
        XCTAssertEqual(EditDistance.osa("abcd", "xyzw", max: 2), -1)
    }

    func testEditModelPrefersAdjacentKeys() {
        let m = EditModel.shared
        XCTAssertLessThan(m.cost(typo: Array("hwllo".utf8), intended: Array("hello".utf8)),
                          m.cost(typo: Array("hzllo".utf8), intended: Array("hello".utf8)))
        XCTAssertLessThan(m.cost(typo: Array("teh".utf8), intended: Array("the".utf8)),
                          m.cost(typo: Array("tex".utf8), intended: Array("the".utf8)))
    }

    func testSymSpellFindsNeighbours() {
        let c = Self.corrector
        let hits = c.symSpell.lookup("teh", maxDistance: 1).map { c.lexicon.strings[Int($0.id)] }
        XCTAssertTrue(hits.contains("the"))
        let hits2 = c.symSpell.lookup("definately", maxDistance: 2).map { c.lexicon.strings[Int($0.id)] }
        XCTAssertTrue(hits2.contains("definitely"))
    }

    func testClassicTypos() {
        let c = Self.corrector
        XCTAssertEqual(c.correct("teh")?.replacement, "the")
        XCTAssertEqual(c.correct("recieve")?.replacement, "receive")
        XCTAssertEqual(c.correct("definately")?.replacement, "definitely")
        XCTAssertEqual(c.correct("Seperate")?.replacement, "Separate")
        XCTAssertEqual(c.correct("accomodate")?.replacement, "accommodate")
        XCTAssertEqual(c.correct("occured")?.replacement, "occurred")
        // "sprus" is ambiguous (spurs / sprue / spruce); the engine may decline, but never pick a wrong word
        XCTAssertTrue([nil, "spurs"].contains(c.correct("sprus")?.replacement))
        XCTAssertEqual(c.correct("hrere")?.replacement, "here")
        // curated list entries must map to their listed fix, never a third word
        XCTAssertEqual(c.correct("payed")?.replacement, "paid")
        XCTAssertEqual(c.correct("theif")?.replacement, "thief")
        XCTAssertEqual(c.correct("toke")?.replacement, "took")
        XCTAssertEqual(c.correct("seeked")?.replacement, "sought")
    }

    func testLeavesRealWordsAlone() {
        let c = Self.corrector
        XCTAssertNil(c.correct("form"))
        XCTAssertNil(c.correct("their"))
        XCTAssertNil(c.correct("NASA"))
        XCTAssertNil(c.correct("iPhone"))
        XCTAssertNil(c.correct("Vexley", sentenceStart: false))
        XCTAssertNil(c.correct("comet"))
        XCTAssertNil(c.correct("sears"))
        XCTAssertNil(c.correct("untied"))
        XCTAssertNil(c.correct("tome"))
    }

    func testPersonalLearning() {
        let p = PersonalModel()
        p.addManual(typo: "hte", fix: "the")
        XCTAssertNil(p.autoFix(for: "hte", isDictionaryWord: false))
        p.addManual(typo: "hte", fix: "the")
        XCTAssertEqual(p.autoFix(for: "hte", isDictionaryWord: false), "the")
        p.addUndo(typo: "hte", fix: "the")
        XCTAssertNil(p.autoFix(for: "hte", isDictionaryWord: false))
        XCTAssertFalse(p.neverCorrect("hte"))          // one rejection may be a slip of the finger
        p.addUndo(typo: "hte", fix: "the")
        XCTAssertTrue(p.neverCorrect("hte"))
    }

    func testOneUndoWithNoHistoryHoldsTheFixBackButDoesNotBlock() {
        let p = PersonalModel()
        p.addApplied(typo: "yuu", fix: "you"); p.addUndo(typo: "yuu", fix: "you")
        XCTAssertEqual(p.evidence(typo: "yuu", fix: "you"), .init(kept: 0, undone: 1), "an undone fix is not also counted as kept")
        XCTAssertEqual(p.undoPenalty(typo: "yuu", candidate: "you"), 1.5, accuracy: 1e-9)
        XCTAssertEqual(p.undoPenalty(typo: "yuu", candidate: "your"), 0)
        XCTAssertFalse(p.neverCorrect("yuu"))
    }

    func testTwoUndosWithNothingKeptBlockUntilTheUserMakesTheFixByHand() {
        let p = PersonalModel()
        for _ in 0..<2 { p.addApplied(typo: "yuu", fix: "you"); p.addUndo(typo: "yuu", fix: "you") }
        XCTAssertTrue(p.neverCorrect("yuu"))
        p.addManual(typo: "yuu", fix: "you")
        XCTAssertTrue(p.neverCorrect("yuu"), "one hand-made fix against two undos is not enough")
        p.addManual(typo: "yuu", fix: "you")
        XCTAssertFalse(p.neverCorrect("yuu"))
    }

    func testAFixThatIsUsuallyKeptSurvivesStrayUndos() {
        let p = PersonalModel()
        for _ in 0..<10 { p.addApplied(typo: "teh", fix: "the") }
        p.addApplied(typo: "teh", fix: "the"); p.addUndo(typo: "teh", fix: "the")
        XCTAssertFalse(p.neverCorrect("teh"))
        XCTAssertLessThan(p.undoPenalty(typo: "teh", candidate: "the"), 0.8)
        for _ in 0..<5 { p.addApplied(typo: "teh", fix: "the") }
        p.addApplied(typo: "teh", fix: "the"); p.addUndo(typo: "teh", fix: "the")
        XCTAssertFalse(p.neverCorrect("teh"), "two undos among many kept fixes do not block the word")
        XCTAssertLessThan(p.undoPenalty(typo: "teh", candidate: "the"), 1.5)
    }

    func testUndoingEveryTimeOutweighsAnOldHabit() {
        let p = PersonalModel()
        for _ in 0..<10 { p.addApplied(typo: "teh", fix: "the") }
        var penalties: [Double] = []
        for _ in 0..<2 {
            p.addApplied(typo: "teh", fix: "the"); p.addUndo(typo: "teh", fix: "the")
            penalties.append(p.undoPenalty(typo: "teh", candidate: "the"))
        }
        XCTAssertFalse(p.neverCorrect("teh"))
        p.addApplied(typo: "teh", fix: "the"); p.addUndo(typo: "teh", fix: "the")
        penalties.append(p.undoPenalty(typo: "teh", candidate: "the"))
        XCTAssertEqual(penalties, penalties.sorted(), "each further undo costs the fix more")
        XCTAssertGreaterThan(penalties[2], 2.5)
        XCTAssertTrue(p.neverCorrect("teh"), "three undos in a row outweigh ten kept fixes")
    }

    func testAFixUndoneOneTimeInThreeEndsUpBlockedButOneInFiveDoesNot() {
        let third = PersonalModel(), fifth = PersonalModel()
        for cycle in 1...2 {
            for _ in 0..<2 { third.addApplied(typo: "teh", fix: "the") }
            third.addApplied(typo: "teh", fix: "the"); third.addUndo(typo: "teh", fix: "the")
            XCTAssertEqual(third.neverCorrect("teh"), cycle == 2)
        }
        for _ in 1...8 {
            for _ in 0..<4 { fifth.addApplied(typo: "teh", fix: "the") }
            fifth.addApplied(typo: "teh", fix: "the"); fifth.addUndo(typo: "teh", fix: "the")
            XCTAssertFalse(fifth.neverCorrect("teh"))
        }
        XCTAssertLessThan(fifth.undoPenalty(typo: "teh", candidate: "the"), 2.0)
    }

    func testALearnedFixOfARealWordGoesBackToTheContextAfterOneUndo() {
        let p = PersonalModel()
        for _ in 0..<4 { p.addManual(typo: "form", fix: "from") }
        XCTAssertEqual(p.autoFix(for: "form", isDictionaryWord: true), "from")
        p.addApplied(typo: "form", fix: "from"); p.addUndo(typo: "form", fix: "from")
        XCTAssertNil(p.autoFix(for: "form", isDictionaryWord: true), "\"form\" was meant once: stop replacing it without looking")
        for _ in 0..<4 { p.addApplied(typo: "form", fix: "from") }
        XCTAssertNil(p.autoFix(for: "form", isDictionaryWord: true))
        p.addApplied(typo: "form", fix: "from")
        XCTAssertEqual(p.autoFix(for: "form", isDictionaryWord: true), "from", "five kept fixes later the habit is trusted again")
    }

    func testUndosOfOneFixDoNotBlockAWordWhoseOtherFixIsKept() {
        let p = PersonalModel()
        for _ in 0..<8 { p.addApplied(typo: "wnat", fix: "want") }
        for _ in 0..<2 { p.addApplied(typo: "wnat", fix: "what"); p.addUndo(typo: "wnat", fix: "what") }
        XCTAssertFalse(p.neverCorrect("wnat"))
        XCTAssertGreaterThan(p.undoPenalty(typo: "wnat", candidate: "what"), 3)
        XCTAssertEqual(p.undoPenalty(typo: "wnat", candidate: "want"), 0)
    }

    func testTypingOverAnEarlierFixTakesItsKeptCountBack() {
        let p = PersonalModel()
        p.addApplied(typo: "form", fix: "from")
        p.addApplied(typo: "teh", fix: "the")          // another fix in between: the undo below is not of the latest one
        p.addUndo(typo: "form", fix: "from")
        XCTAssertEqual(p.evidence(typo: "form", fix: "from"), .init(kept: 0, undone: 1))
        XCTAssertEqual(p.evidence(typo: "teh", fix: "the"), .init(kept: p.appliedWeight, undone: 0))
    }

    func testEvidenceIsRebuiltFromTheStoreInOrder() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("typo-test-\(UUID()).sqlite")
        defer { try? FileManager.default.removeItem(at: url) }
        let s = try Store(url: url)
        let live = PersonalModel()
        for _ in 0..<3 {
            try s.recordFix(typo: "Teh", fix: "The", app: "test", before: [], after: [], source: .auto)   // stored as typed
            live.addApplied(typo: "teh", fix: "the")
        }
        try s.recordFix(typo: "teh", fix: "the", app: "test", before: [], after: [], source: .undo)
        live.addUndo(typo: "teh", fix: "the")
        try s.recordFix(typo: "teh", fix: "the", app: "test", before: [], after: [], source: .manual)
        live.addManual(typo: "teh", fix: "the")
        let p = PersonalModel(store: s)
        XCTAssertEqual(p.evidence(typo: "teh", fix: "the"), live.evidence(typo: "teh", fix: "the"))
        XCTAssertEqual(p.evidence(typo: "teh", fix: "the").undone, 0.85, accuracy: 1e-9)
        XCTAssertEqual(p.manualCount(typo: "teh", fix: "the"), 1)
    }

    func testAutoFixTieIsAmbiguous() {
        let p = PersonalModel()
        p.addManual(typo: "wnat", fix: "want"); p.addManual(typo: "wnat", fix: "want")
        p.addManual(typo: "wnat", fix: "what"); p.addManual(typo: "wnat", fix: "what")
        XCTAssertNil(p.autoFix(for: "wnat", isDictionaryWord: false))
        p.addManual(typo: "wnat", fix: "want")
        XCTAssertEqual(p.autoFix(for: "wnat", isDictionaryWord: false), "want")
    }

    func testDigitTokensAreNeverCorrected() {
        XCTAssertNil(Self.corrector.correct("h264enc"))
        XCTAssertNil(Self.corrector.correct("2nd"))
    }

    func testStoreRoundTrip() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("typo-test-\(UUID()).sqlite")
        defer { try? FileManager.default.removeItem(at: url) }
        let s = try Store(url: url)
        try s.recordFix(typo: "teh", fix: "the", app: "test", before: ["a", "b"], after: [], source: .manual)
        try s.recordFix(typo: "teh", fix: "the", app: "test", before: [], after: [], source: .manual)
        s.bumpUnknown("vexley"); s.bumpUnknown("vexley")
        let pairs = try s.pairCounts()
        XCTAssertEqual(pairs.first?.count, 2)
        XCTAssertEqual(try s.unknownCounts()["vexley"], 2)
        let p = PersonalModel(store: s)
        XCTAssertEqual(p.autoFix(for: "teh", isDictionaryWord: true), nil)   // known-word bar is 4
        XCTAssertEqual(p.manualCount(typo: "teh", fix: "the"), 2)
        s.setMeta("k", "v")
        XCTAssertEqual(s.meta("k"), "v")
    }
}
