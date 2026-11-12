import AppKit
import ApplicationServices
import Foundation

/// Tracks whether the focused UI element is a password field, without ever blocking the event tap.
///
/// Accessibility queries go to the focused app over Mach IPC and can stall for the full messaging
/// timeout (or longer when an app is hung). The tap used to run them inline, which made the
/// callback slow enough for the WindowServer to deliver the key before we answered. Now the queries
/// run on a background queue, the tap reads a cached answer, and anything stale counts as unknown.
final class FocusMonitor {
    enum State { case unknown, normal, secure }

    private let queue = DispatchQueue(label: "typo.focus", qos: .userInteractive)
    private let lock = NSLock()
    private var state: State = .unknown
    private var updated: Date = .distantPast
    private var requested: Date = .distantPast
    private var inFlight = false
    private var observer: AXObserver?
    private var observedPID: pid_t = 0

    private let systemWide: AXUIElement = {
        let el = AXUIElementCreateSystemWide()
        AXUIElementSetMessagingTimeout(el, 0.05)   // also the process-wide default for other elements
        return el
    }()

    /// Refresh at most this often while keys arrive.
    static let refreshInterval: TimeInterval = 0.25
    /// A cached answer older than this is treated as unknown (no learning, no corrections).
    static let maxAge: TimeInterval = 2.0

    /// True only when a recent query said the focused element is not a secure text field.
    var isKnownNotSecure: Bool {
        lock.lock(); defer { lock.unlock() }
        return state == .normal && Date().timeIntervalSince(updated) < Self.maxAge
    }

    var isSecure: Bool {
        lock.lock(); defer { lock.unlock() }
        return state == .secure
    }

    /// Cheap; call on every key. Starts a background query if the cache is getting old.
    func refreshIfStale() {
        lock.lock()
        let due = Date().timeIntervalSince(requested) >= Self.refreshInterval && !inFlight
        if due { inFlight = true; requested = Date() }
        lock.unlock()
        if due { queue.async { [weak self] in self?.query() } }
    }

    /// Forces a query (focus or app changed); the cached answer is invalidated first.
    func invalidate() {
        lock.lock()
        state = .unknown
        inFlight = true
        requested = Date()
        lock.unlock()
        queue.async { [weak self] in self?.query() }
    }

    private func query() {
        let result = Self.focusedElementIsSecure(systemWide)
        lock.lock()
        state = result ? .secure : .normal
        updated = Date()
        inFlight = false
        lock.unlock()
    }

    private static func focusedElementIsSecure(_ systemWide: AXUIElement) -> Bool {
        var focused: CFTypeRef?
        guard AXUIElementCopyAttributeValue(systemWide, kAXFocusedUIElementAttribute as CFString, &focused) == .success,
              let f = focused else { return false }
        let element = f as! AXUIElement
        AXUIElementSetMessagingTimeout(element, 0.05)
        var role: CFTypeRef?
        AXUIElementCopyAttributeValue(element, kAXRoleAttribute as CFString, &role)
        var subrole: CFTypeRef?
        AXUIElementCopyAttributeValue(element, kAXSubroleAttribute as CFString, &subrole)
        return (role as? String) == "AXSecureTextField" || (subrole as? String) == "AXSecureTextField"
    }

    // MARK: text before the caret

    enum CaretText {
        case text(String)
        /// The app does not expose its text through Accessibility (many Electron apps).
        case unsupported
        /// The app did not answer in time, or the caret/selection is not where the tracker thinks.
        case failed
    }

    /// Reads the `length` UTF-16 units before the caret in the focused element, off the tap thread,
    /// after `delay`. `completion` runs on the focus queue.
    func readTextBeforeCaret(length: Int, after delay: TimeInterval, completion: @escaping (CaretText) -> Void) {
        queue.asyncAfter(deadline: .now() + delay) { [systemWide] in
            completion(Self.textBeforeCaret(systemWide, length: length))
        }
    }

    private static func classify(_ error: AXError) -> CaretText {
        switch error {
        case .attributeUnsupported, .parameterizedAttributeUnsupported, .noValue, .notImplemented: return .unsupported
        default: return .failed  // cannotComplete (timeout), failure, illegalArgument, ...
        }
    }

    private static func textBeforeCaret(_ systemWide: AXUIElement, length: Int) -> CaretText {
        guard length > 0 else { return .failed }
        var focused: CFTypeRef?
        let e1 = AXUIElementCopyAttributeValue(systemWide, kAXFocusedUIElementAttribute as CFString, &focused)
        guard e1 == .success else { return classify(e1) }
        guard let f = focused, CFGetTypeID(f) == AXUIElementGetTypeID() else { return .unsupported }
        let element = f as! AXUIElement
        AXUIElementSetMessagingTimeout(element, 0.05)
        var rangeRef: CFTypeRef?
        let e2 = AXUIElementCopyAttributeValue(element, kAXSelectedTextRangeAttribute as CFString, &rangeRef)
        guard e2 == .success else { return classify(e2) }
        guard let rv = rangeRef, CFGetTypeID(rv) == AXValueGetTypeID() else { return .unsupported }
        var selection = CFRange()
        guard AXValueGetValue(rv as! AXValue, .cfRange, &selection) else { return .unsupported }
        // A selection (inline autocomplete) or a caret too close to the start: the text near the caret is
        // not what we typed, and blind backspaces would hit the wrong characters.
        guard selection.length == 0, selection.location >= length else { return .failed }
        var wanted = CFRange(location: selection.location - length, length: length)
        guard let param = AXValueCreate(.cfRange, &wanted) else { return .failed }
        var out: CFTypeRef?
        let e3 = AXUIElementCopyParameterizedAttributeValue(element, kAXStringForRangeParameterizedAttribute as CFString,
                                                            param, &out)
        guard e3 == .success else { return classify(e3) }
        guard let s = out as? String else { return .unsupported }
        return .text(s)
    }

    // MARK: focus-change notifications

    /// Watches the front app for focus changes so password fields are recognised before the first word ends.
    func track(pid: pid_t) {
        guard pid != observedPID else { return }
        if let old = observer {
            CFRunLoopRemoveSource(CFRunLoopGetMain(), AXObserverGetRunLoopSource(old), .commonModes)
            observer = nil
        }
        observedPID = pid
        invalidate()
        guard pid > 0 else { return }
        var obs: AXObserver?
        let refcon = Unmanaged.passUnretained(self).toOpaque()
        guard AXObserverCreate(pid, { _, _, _, refcon in
            guard let refcon else { return }
            Unmanaged<FocusMonitor>.fromOpaque(refcon).takeUnretainedValue().invalidate()
        }, &obs) == .success, let obs else { return }
        let app = AXUIElementCreateApplication(pid)
        AXUIElementSetMessagingTimeout(app, 0.05)
        AXObserverAddNotification(obs, app, kAXFocusedUIElementChangedNotification as CFString, refcon)
        AXObserverAddNotification(obs, app, kAXFocusedWindowChangedNotification as CFString, refcon)
        CFRunLoopAddSource(CFRunLoopGetMain(), AXObserverGetRunLoopSource(obs), .commonModes)
        observer = obs
    }
}
