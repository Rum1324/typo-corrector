import Carbon
import Foundation

/// Tracks whether the current input source types Latin text directly. While a
/// composing input method (Japanese, Chinese, Korean…) is active we stop tracking entirely.
final class InputSourceMonitor {
    private(set) var isLatin = true
    private(set) var sourceID = ""
    var onChange: (() -> Void)?

    init() {
        refresh()
        DistributedNotificationCenter.default().addObserver(
            forName: NSNotification.Name(kTISNotifySelectedKeyboardInputSourceChanged as String),
            object: nil, queue: .main) { [weak self] _ in
                self?.refresh()
                self?.onChange?()
            }
    }

    func refresh() {
        guard let src = TISCopyCurrentKeyboardInputSource()?.takeRetainedValue() else { return }
        var id = ""
        if let p = TISGetInputSourceProperty(src, kTISPropertyInputSourceID) {
            id = Unmanaged<CFString>.fromOpaque(p).takeUnretainedValue() as String
        }
        var type = ""
        if let p = TISGetInputSourceProperty(src, kTISPropertyInputSourceType) {
            type = Unmanaged<CFString>.fromOpaque(p).takeUnretainedValue() as String
        }
        sourceID = id
        let layout = type == (kTISTypeKeyboardLayout as String)
        let romanMode = id.hasSuffix(".Roman") || id.hasSuffix(".Latin") || id.hasSuffix(".ABC")
        isLatin = layout || romanMode
    }
}
