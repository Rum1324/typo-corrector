import ApplicationServices
import CoreGraphics
import Foundation

enum Permissions {
    static var accessibility: Bool { AXIsProcessTrusted() }
    static var inputMonitoring: Bool { CGPreflightListenEventAccess() }

    /// `AXIsProcessTrusted` is an XPC round trip to tccd; the event tap reads this cached copy instead.
    /// Written on the main thread only.
    private(set) static var accessibilityCached: Bool = AXIsProcessTrusted()
    private static var pollTimer: DispatchSourceTimer?

    static func startPolling(interval: TimeInterval = 3) {
        guard pollTimer == nil else { return }
        let t = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
        t.schedule(deadline: .now() + interval, repeating: interval)
        t.setEventHandler {
            let v = AXIsProcessTrusted()
            DispatchQueue.main.async { accessibilityCached = v }
        }
        t.resume()
        pollTimer = t
    }

    static func requestAccessibility() {
        let opts = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
        _ = AXIsProcessTrustedWithOptions(opts)
    }

    static func requestInputMonitoring() { _ = CGRequestListenEventAccess() }

    static var allGranted: Bool { accessibility && inputMonitoring }
}
