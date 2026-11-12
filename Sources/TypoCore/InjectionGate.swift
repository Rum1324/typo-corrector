import Foundation

/// Counts the synthetic key events that are still on their way through the event tap.
///
/// While any are in flight, a real key that reaches the tap is ahead of them in the WindowServer
/// queue. Letting it pass would land it in the middle of a correction (and it would absorb one of
/// the correction's backspaces), so the app holds such keys and re-posts them behind its own.
public struct InjectionGate {
    /// After this long without one of our events coming back, assume the rest were dropped
    /// (secure input switched on, tap restarted) instead of holding the user's keys forever.
    public let staleAfter: TimeInterval
    public private(set) var inFlight = 0
    private var lastActivity: TimeInterval = 0

    public init(staleAfter: TimeInterval = 0.5) {
        self.staleAfter = staleAfter
    }

    public var isBusy: Bool { inFlight > 0 }

    /// We posted `count` events that the tap will see.
    public mutating func posted(_ count: Int = 1, at now: TimeInterval) {
        inFlight += max(count, 0)
        lastActivity = now
    }

    /// One of our events passed the tap.
    public mutating func returned(at now: TimeInterval) {
        if inFlight > 0 { inFlight -= 1 }
        lastActivity = now
    }

    /// Forgets a count that has not moved for `staleAfter`. Returns how many events were written off.
    @discardableResult
    public mutating func expireIfStale(at now: TimeInterval) -> Int {
        guard inFlight > 0, now - lastActivity > staleAfter else { return 0 }
        let dropped = inFlight
        inFlight = 0
        return dropped
    }

    public mutating func reset() {
        inFlight = 0
    }
}
