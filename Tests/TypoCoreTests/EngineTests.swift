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
