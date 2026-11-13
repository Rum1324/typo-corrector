import XCTest
@testable import TypoCore

final class TextTrackerTests: XCTestCase {
    func type(_ t: TextTracker, _ s: String) -> [Commit] {
        var out: [Commit] = []
        for c in s { if let k = t.feed(.char(c)) { out.append(k) } }
        return out
    }

    func testCommitOnSpaceWithContext() {
        let t = TextTracker()
        let commits = type(t, "\nhello big world ")
        XCTAssertEqual(commits.map(\.word), ["hello", "big", "world"])
        XCTAssertEqual(commits.last?.context, ["hello", "big"])
        XCTAssertTrue(commits.first!.sentenceStart)
        XCTAssertFalse(commits.last!.sentenceStart)
    }

    func testEroseCommittedWordAndRetype() {
        let t = TextTracker()
        _ = type(t, "teh ")
        for _ in 0..<4 { t.feed(.backspace) }          // space + teh
        let c = type(t, "the ").first!
        XCTAssertEqual(c.pair, LearnedPair(typo: "teh", fix: "the"))
        XCTAssertEqual(t.text, "the ")
    }

    func testPartialErosion() {
        let t = TextTracker()
        _ = type(t, "I recieve ")
        for _ in 0..<5 { t.feed(.backspace) }          // " ieve"
        let c = type(t, "eive ").first!
        XCTAssertEqual(c.word, "receive")
        XCTAssertEqual(c.pair, LearnedPair(typo: "recieve", fix: "receive"))
    }

    func testFixInsideCurrentWord() {
        let t = TextTracker()
        _ = type(t, "thw")
        t.feed(.backspace)
        let c = type(t, "e ").first!
        XCTAssertEqual(c.pair, LearnedPair(typo: "thw", fix: "the"))
    }

    func testTruncationIsNotAFix() {
        let t = TextTracker()
        _ = type(t, "there")
        t.feed(.backspace); t.feed(.backspace)
        let c = type(t, " ").first!
        XCTAssertEqual(c.word, "the")
        XCTAssertNil(c.pair)
    }

    func testPrefixContinuationIsNotAFix() {
        let t = TextTracker()
        _ = type(t, "th")
        t.feed(.backspace)
        let c = type(t, "he ").first!
        XCTAssertEqual(c.word, "the")
        XCTAssertNil(c.pair)
    }

    func testSameWordRetypedIsNotAFix() {
        let t = TextTracker()
        _ = type(t, "word ")
        for _ in 0..<5 { t.feed(.backspace) }
        let c = type(t, "word ").first!
        XCTAssertNil(c.pair)
    }

    func testResetClearsState() {
        let t = TextTracker()
        _ = type(t, "teh ")
        t.feed(.reset)
        XCTAssertEqual(t.text, "")
        let c = type(t, "the ").first!
        XCTAssertNil(c.pair)
    }

    func testReplaceLastWordRemembersTypedWord() {
        let t = TextTracker()
        _ = type(t, "I wrod ")
        t.replaceLastWord(with: "word", original: "wrod")
        XCTAssertEqual(t.text, "I word ")
        for _ in 0..<5 { t.feed(.backspace) }
        let c = type(t, "world ").first!
        XCTAssertEqual(c.pair, LearnedPair(typo: "wrod", fix: "world"))
        XCTAssertEqual(c.rejectedFix, "word")
    }

    func testTypingTheOriginalWordBackOverAFixIsARejectionWithNothingToLearn() {
        let t = TextTracker()
        _ = type(t, "I teh ")
        t.replaceLastWord(with: "the", original: "teh")
        for _ in 0..<4 { t.feed(.backspace) }
        let c = type(t, "teh ").first!
        XCTAssertEqual(c.word, "teh")
        XCTAssertNil(c.pair)
        XCTAssertEqual(c.rejectedFix, "the", "the engine must leave the word alone, not fix it again")
    }

    func testTypingTheOriginalBackIsSeenAfterALaterWordWasErasedToo() {
        let t = TextTracker()
        _ = type(t, "yuu ")
        t.replaceLastWord(with: "you", original: "yuu")
        _ = type(t, "are ")                             // noticed the fix one word late
        for _ in 0..<8 { t.feed(.backspace) }
        let c = type(t, "yuu ").first!
        XCTAssertNil(c.pair)
        XCTAssertEqual(c.rejectedFix, "you")
    }

    func testAFixReplacedAfterALaterWordWasErasedIsLearned() {
        let t = TextTracker()
        _ = type(t, "I wrod ")
        t.replaceLastWord(with: "word", original: "wrod")
        _ = type(t, "is ")
        for _ in 0..<8 { t.feed(.backspace) }
        let c = type(t, "world ").first!
        XCTAssertEqual(c.pair, LearnedPair(typo: "wrod", fix: "world"))
        XCTAssertEqual(c.rejectedFix, "word")
    }

    func testAFixStaysKnownAfterItsSpaceIsDeletedAndTypedAgain() {
        let t = TextTracker()
        _ = type(t, "I teh ")
        t.replaceLastWord(with: "the", original: "teh")
        t.feed(.backspace)
        XCTAssertEqual(type(t, " ").first?.context, ["I"], "the word is not listed twice")
        for _ in 0..<4 { t.feed(.backspace) }
        let c = type(t, "teh ").first!
        XCTAssertNil(c.pair)
        XCTAssertEqual(c.rejectedFix, "the")
    }

    func testRetypingTheFixOrOnlyItsSpaceIsNotARejection() {
        let t = TextTracker()
        _ = type(t, "I teh ")
        t.replaceLastWord(with: "the", original: "teh")
        t.feed(.backspace)                              // reached back, saw it was already fixed
        var c = type(t, " ").first!
        XCTAssertEqual(c.word, "the")
        XCTAssertNil(c.pair)
        XCTAssertNil(c.rejectedFix)
        for _ in 0..<4 { t.feed(.backspace) }
        c = type(t, "the ").first!
        XCTAssertNil(c.pair)
        XCTAssertNil(c.rejectedFix)
    }

    func testUndoRestoreDoesNotReportRejection() {
        let t = TextTracker()
        _ = type(t, "I teh ")
        t.replaceLastWord(with: "the", original: "teh")
        t.replaceLastWord(with: "teh", original: nil)
        for _ in 0..<4 { t.feed(.backspace) }
        let c = type(t, "tea ").first!
        XCTAssertEqual(c.pair, LearnedPair(typo: "teh", fix: "tea"))
        XCTAssertNil(c.rejectedFix)
    }

    func testOvershootErosionStillLearns() {
        let t = TextTracker()
        _ = type(t, "I teh ")
        for _ in 0..<5 { t.feed(.backspace) }          // one too many: line is "I"
        let commits = type(t, " the ")
        XCTAssertEqual(commits.map(\.word), ["I", "the"])
        XCTAssertNil(commits[0].pair)
        XCTAssertEqual(commits[1].pair, LearnedPair(typo: "teh", fix: "the"))
    }

    func testDeletingPastCurrentWordClearsSnapshot() {
        let t = TextTracker()
        _ = type(t, "to tht")
        for _ in 0..<4 { t.feed(.backspace) }          // "to"
        let commits = type(t, " the ")
        XCTAssertNil(commits[0].pair)                    // "to" is not a fix of "tht"
    }

    func testReturnIsAHardReset() {
        let t = TextTracker()
        _ = type(t, "see you later\n")
        t.feed(.backspace); t.feed(.backspace)
        let c = type(t, "ok ").first!
        XCTAssertEqual(c.word, "ok")
        XCTAssertEqual(c.context, [])
    }

    func testDigitsStayInsideTokens() {
        let t = TextTracker()
        let commits = type(t, "h264enc 2nd ")
        XCTAssertEqual(commits.map(\.word), ["h264enc", "2nd"])
    }

    func testSentenceStartAfterPeriod() {
        let t = TextTracker()
        let commits = type(t, "Done. Next ")
        XCTAssertTrue(commits[1].sentenceStart)
    }

    func testReplaceWordBeforePunctuationAndBoundary() {
        let t = TextTracker()
        _ = type(t, "I recieve. ")
        XCTAssertTrue(t.endsWith("recieve. "))
        XCTAssertTrue(t.replaceWord("recieve", endingAt: 2, with: "receive", original: "recieve"))
        XCTAssertEqual(t.text, "I receive. ")
        // The fixed word is now the committed slot: erasing it and typing something else
        // reports the automatic fix as rejected.
        for _ in 0..<9 { t.feed(.backspace) }
        let c = type(t, "reprieve ").last
        XCTAssertEqual(c?.pair, LearnedPair(typo: "recieve", fix: "reprieve"))
        XCTAssertEqual(c?.rejectedFix, "receive")
    }

    func testUnknownLineStartIsNotASentenceStart() {
        let t = TextTracker()
        let c = type(t, " Haruto ").last   // after a click: what precedes the caret is unknown
        XCTAssertEqual(c?.startKnown, true)
        XCTAssertEqual(c?.sentenceStart, false)
    }

    func testWordStartUnknownUntilABoundaryIsSeen() {
        let t = TextTracker()
        XCTAssertEqual(type(t, "eople ").last?.startKnown, false, "may be the tail of a word")
        XCTAssertEqual(type(t, "teh ").last?.startKnown, true)
        _ = type(t, "x\n")
        XCTAssertEqual(type(t, "teh ").last?.startKnown, true, "after Return the line start is known")
        t.reset()
        XCTAssertEqual(type(t, "teh ").last?.startKnown, false)
        let u = TextTracker()
        _ = type(u, "a\n")
        for _ in 0..<2 { u.feed(.backspace) }   // the second one deletes text we never saw
        XCTAssertEqual(type(u, "teh ").last?.startKnown, false)
    }

    func testUndoOfASplitFixRestoresTheLine() {
        let t = TextTracker()
        _ = type(t, "\nok inthe ")
        t.replaceLastWord(with: "in the", original: "inthe")
        XCTAssertEqual(t.text, "ok in the ")
        t.replaceLastWord(with: "inthe", original: nil, replacing: "in the")
        XCTAssertEqual(t.text, "ok inthe ")
    }

    func testReplaceWordRefusesWhenLineChanged() {
        let t = TextTracker()
        _ = type(t, "see youtu.")
        XCTAssertFalse(t.replaceWord("youtu", endingAt: 2, with: "youth", original: "youtu"))
        XCTAssertFalse(t.replaceWord("", endingAt: 0, with: "x", original: nil))
        XCTAssertEqual(t.text, "see youtu.")
    }
}
