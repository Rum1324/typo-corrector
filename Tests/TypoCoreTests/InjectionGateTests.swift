import XCTest
@testable import TypoCore

final class InjectionGateTests: XCTestCase {
    func testBusyUntilEveryPostedEventReturns() {
        var g = InjectionGate()
        XCTAssertFalse(g.isBusy)
        g.posted(at: 0)
        g.posted(3, at: 0)
        XCTAssertEqual(g.inFlight, 4)
        for i in 0..<3 {
            g.returned(at: 0.01 * Double(i))
            XCTAssertTrue(g.isBusy)
        }
        g.returned(at: 0.05)
        XCTAssertFalse(g.isBusy)
    }

    func testExtraReturnsNeverGoNegative() {
        var g = InjectionGate()
        g.returned(at: 0)
        XCTAssertEqual(g.inFlight, 0)
        g.posted(at: 1)
        XCTAssertTrue(g.isBusy)
    }

    func testStaleCountIsWrittenOff() {
        var g = InjectionGate(staleAfter: 0.5)
        g.posted(5, at: 10)
        g.returned(at: 10.2)
        XCTAssertEqual(g.expireIfStale(at: 10.6), 0, "events still trickling back")
        XCTAssertTrue(g.isBusy)
        XCTAssertEqual(g.expireIfStale(at: 10.8), 4)
        XCTAssertFalse(g.isBusy)
    }

    func testPostingRefreshesTheStaleClock() {
        var g = InjectionGate(staleAfter: 0.5)
        g.posted(at: 0)
        g.posted(at: 0.4)
        XCTAssertEqual(g.expireIfStale(at: 0.8), 0)
        XCTAssertEqual(g.inFlight, 2)
    }

    func testReset() {
        var g = InjectionGate()
        g.posted(7, at: 0)
        g.reset()
        XCTAssertFalse(g.isBusy)
    }
}
