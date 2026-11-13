import AppKit
import TypoCore

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var engine: Engine?
    private var trainer: Trainer?
    private var statusMenu: StatusMenu?
    private let tap = KeyTap()
    private var permissionTimer: Timer?
    private var trainTimer: Timer?

    func applicationDidFinishLaunching(_ notification: Notification) {
        Log.write("launch \(Bundle.main.bundleURL.path) accessibility=\(Permissions.accessibility) inputMonitoring=\(Permissions.inputMonitoring)")
        if !Permissions.accessibility { Permissions.requestAccessibility() }
        if !Permissions.inputMonitoring { Permissions.requestInputMonitoring() }
        Permissions.startPolling()

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let store = try Store(url: Paths.database)
                var config = CorrectorConfig()
                config.minProbability = Settings.shared.minProbability
                config.contextLooseness = Settings.shared.looseness
                config.fixContractions = Settings.shared.fixContractions
                let corrector = try Corrector.loadDefault(store: store, config: config)
                DispatchQueue.main.async { self.finishSetup(store: store, corrector: corrector) }
            } catch {
                DispatchQueue.main.async {
                    let a = NSAlert()
                    a.messageText = "Typo could not start"
                    a.informativeText = "\(error)"
                    a.runModal()
                    NSApp.terminate(nil)
                }
            }
        }
    }

    private func finishSetup(store: Store, corrector: Corrector) {
        let engine = Engine(store: store, corrector: corrector)
        let trainer = Trainer(store: store)
        let menu = StatusMenu(engine: engine, trainer: trainer)
        Log.write("engine ready: \(corrector.lexicon.count) words, ranker=\(corrector.ranker?.source.path ?? "none"), misspellings=\(corrector.misspellings.count), context=\(corrector.context.map { $0.decider != nil ? "on" : "language model only" } ?? "none")")
        engine.onChange = { [weak menu] in menu?.refreshIcon() }
        trainer.onFinished = { [weak self] status in
            guard status == 0, let self else { return }
            self.engine?.corrector.ranker = NeuralRanker.load()
            Log.write("reloaded ranker from \(self.engine?.corrector.ranker?.source.path ?? "nil")")
        }
        self.engine = engine
        self.trainer = trainer
        self.statusMenu = menu

        // Not `engine?.handle(...) ?? pass`: handle returns nil to swallow a key, and `??` would
        // turn that nil back into "pass", delivering the very key a correction replaces.
        tap.handler = { [weak engine] event, type in
            guard let engine else { return Unmanaged.passUnretained(event) }
            return engine.handle(event, type: type)
        }
        tap.onDisabled = { [weak engine] type in
            engine?.tapWasDisabled(type)
        }
        startTapWhenPermitted()

        trainTimer = Timer.scheduledTimer(withTimeInterval: 300, repeats: true) { [weak trainer] _ in
            if let reason = trainer?.maybeTrain() { Log.write("training skipped: \(reason)") } else { Log.write("training started") }
        }
    }

    private func startTapWhenPermitted() {
        if Permissions.accessibility, tap.start() {
            Log.write("event tap running; status=\(engine?.status ?? .disabled) front=\(engine?.frontBundleID ?? "-") inputSource=\(engine?.inputSource.sourceID ?? "-") latin=\(engine?.inputSource.isLatin ?? false)")
            statusMenu?.refreshIcon()
            permissionTimer?.invalidate()
            permissionTimer = nil
            return
        }
        statusMenu?.refreshIcon()
        if permissionTimer == nil {
            Log.write("waiting for Accessibility (trusted=\(Permissions.accessibility)); tap start \(Permissions.accessibility ? "failed" : "deferred")")
            permissionTimer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in
                self?.startTapWhenPermitted()
            }
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        tap.stop()
        trainer?.terminate()
    }
}
