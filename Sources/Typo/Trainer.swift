import CoreGraphics
import Foundation
import IOKit.ps
import TypoCore

/// Runs training/train.py --personal when the Mac is idle and on power, at most once a day.
final class Trainer {
    let store: Store
    var onFinished: ((Int32) -> Void)?
    private var process: Process?
    private var startedAt: Date?
    private(set) var lastError: String?

    var isRunning: Bool { process?.isRunning ?? false }
    var python: URL { Paths.trainingDir.appendingPathComponent(".venv/bin/python") }
    var logFile: URL { Paths.appSupport.appendingPathComponent("train.log") }
    var available: Bool { FileManager.default.fileExists(atPath: python.path) }

    var lastTrained: Date? {
        store.meta("last_train_ts").flatMap(Double.init).map { Date(timeIntervalSince1970: $0) }
    }

    init(store: Store) { self.store = store }

    static func idleSeconds() -> Double {
        let types: [CGEventType] = [.keyDown, .mouseMoved, .leftMouseDown, .scrollWheel]
        return types.map { CGEventSource.secondsSinceLastEventType(.combinedSessionState, eventType: $0) }.min() ?? 0
    }

    static func onACPower() -> Bool {
        guard let info = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
              let type = IOPSGetProvidingPowerSourceType(info)?.takeUnretainedValue() else { return true }
        return (type as String) == kIOPMACPowerKey
    }

    /// Returns a reason string when training is skipped, nil when it was started.
    @discardableResult
    func maybeTrain(force: Bool = false) -> String? {
        if isRunning { return "already running" }
        guard available else { return "training/.venv missing (run training/setup.sh)" }
        let newExamples = store.countFixes(source: .manual, since: lastTrained) + store.countFixes(source: .undo, since: lastTrained)
        if !force {
            if Self.idleSeconds() < 600 { return "not idle" }
            if !Self.onACPower() { return "on battery" }
            if let t = lastTrained, Date().timeIntervalSince(t) < 20 * 3600 { return "trained recently" }
            if let f = store.meta("last_train_fail_ts").flatMap(Double.init), Date().timeIntervalSince1970 - f < 6 * 3600 { return "failed recently, backing off" }
            let staleWeek = lastTrained.map { Date().timeIntervalSince($0) > 7 * 86400 } ?? true
            if newExamples < 10 && !(staleWeek && newExamples > 0) { return "too few new examples (\(newExamples))" }
        } else if newExamples == 0 && lastTrained != nil {
            return "no new examples since last training"
        }
        start()
        return nil
    }

    func terminate() {
        if let p = process, p.isRunning { p.terminate() }
    }

    private func start() {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/nice")
        p.arguments = ["-n", "19", python.path, "train.py", "--personal",
                       "--db", store.url.path, "--out", Paths.appSupport.path]
        p.currentDirectoryURL = Paths.trainingDir
        FileManager.default.createFile(atPath: logFile.path, contents: nil)
        if let h = try? FileHandle(forWritingTo: logFile) {
            h.seekToEndOfFile()
            p.standardOutput = h
            p.standardError = h
        }
        p.terminationHandler = { [weak self] proc in
            DispatchQueue.main.async {
                guard let self else { return }
                if proc.terminationStatus == 0 {
                    self.store.setMeta("last_train_ts", String(self.startedAt?.timeIntervalSince1970 ?? Date().timeIntervalSince1970))
                    self.lastError = nil
                } else {
                    self.lastError = "exit \(proc.terminationStatus), see train.log"
                    self.store.setMeta("last_train_fail_ts", String(Date().timeIntervalSince1970))
                }
                self.process = nil
                self.onFinished?(proc.terminationStatus)
            }
        }
        do {
            startedAt = Date()
            try p.run()
            process = p
        } catch {
            lastError = error.localizedDescription
        }
    }
}
