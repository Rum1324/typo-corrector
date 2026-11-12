import CoreGraphics
import Foundation
import TypoCore

/// Posts synthetic keystrokes. Every event we post is tagged so the tap passes it through, and
/// counted until it comes back through the tap so real keys never overtake a correction.
/// Main thread only (the tap runs on the main run loop).
final class Injector {
    static let marker: Int64 = 0x5459_504F  // "TYPO"
    private let source = CGEventSource(stateID: .hidSystemState)
    /// Generous: a leaked count only means real keys get re-posted (still in order) for a while,
    /// while giving up too early on a slow, loaded system would let a key overtake a correction.
    private var gate = InjectionGate(staleAfter: 2.0)

    private var now: TimeInterval { ProcessInfo.processInfo.systemUptime }

    func isInjected(_ event: CGEvent) -> Bool {
        event.getIntegerValueField(.eventSourceUserData) == Self.marker
    }

    /// One of our events came back through the tap on its way to the app.
    func sawInjected() {
        gate.returned(at: now)
    }

    /// True while events we posted have not passed the tap yet. A real key arriving now is ahead
    /// of them in the queue and must be re-posted behind them (`holdBehindInFlight`).
    var hasEventsInFlight: Bool {
        let dropped = gate.expireIfStale(at: now)
        if dropped > 0 {
            Log.write("injector: \(dropped) posted events never came back through the tap; no longer holding keys")
        }
        return gate.isBusy
    }

    func reset() {
        gate.reset()
    }

    private func post(_ event: CGEvent) {
        event.setIntegerValueField(.eventSourceUserData, value: Self.marker)
        gate.posted(at: now)
        event.post(tap: .cghidEventTap)
    }

    func pressKey(_ keyCode: CGKeyCode, flags: CGEventFlags = []) {
        guard let down = CGEvent(keyboardEventSource: source, virtualKey: keyCode, keyDown: true),
              let up = CGEvent(keyboardEventSource: source, virtualKey: keyCode, keyDown: false) else { return }
        down.flags = flags
        up.flags = flags
        post(down)
        post(up)
    }

    func backspace(_ count: Int) {
        for _ in 0..<max(count, 0) { pressKey(51) }
    }

    func type(_ text: String) {
        let units = Array(text.utf16)
        var i = 0
        while i < units.count {
            let chunk = Array(units[i..<min(i + 20, units.count)])
            guard let down = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: true),
                  let up = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: false) else { return }
            chunk.withUnsafeBufferPointer { p in
                down.keyboardSetUnicodeString(stringLength: chunk.count, unicodeString: p.baseAddress)
                up.keyboardSetUnicodeString(stringLength: chunk.count, unicodeString: p.baseAddress)
            }
            // Never inherit modifiers the user is holding (Shift for "?", or Cmd about to send).
            down.flags = []
            up.flags = []
            post(down)
            post(up)
            i += 20
        }
    }

    /// Re-posts a copy of one of the user's own key events (the space or return that ended the
    /// word, or a key that arrived while a correction was still in flight).
    /// The copy keeps the original timestamp: event time is in mach ticks, not nanoseconds, and a
    /// slightly old stamp is harmless where a wrongly converted one lands far in the future.
    func repost(_ original: CGEvent, as type: CGEventType? = nil) {
        guard let copy = original.copy() else { return }
        if let type { copy.type = type }
        post(copy)
    }

    /// Replaces the trailing `deleteCount` characters with `text`, then re-posts `boundary`.
    /// Called from inside the tap callback that swallows `boundary`, so these events queue up
    /// right where the swallowed key would have gone. `withKeyUp` also sends its release, for a
    /// boundary whose real release went to the app long ago (undo).
    func replaceTrailing(deleteCount: Int, with text: String, thenRepost boundary: CGEvent) {
        backspace(deleteCount)
        type(text)
        repost(boundary)
    }

    /// Replaces the trailing `deleteCount` characters with `text` (which may include the boundary).
    func replaceTrailing(deleteCount: Int, with text: String) {
        backspace(deleteCount)
        type(text)
    }

    /// Sends a real key again behind the events still in flight, keeping the order the user typed.
    func holdBehindInFlight(_ event: CGEvent) {
        repost(event)
    }
}
