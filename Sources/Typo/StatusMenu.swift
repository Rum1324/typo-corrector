import AppKit
import ServiceManagement
import TypoCore

final class StatusMenu: NSObject, NSMenuDelegate {
    private let item: NSStatusItem
    private let menu = NSMenu()
    private let engine: Engine
    private let trainer: Trainer
    var onReloadRanker: (() -> Void)?

    init(engine: Engine, trainer: Trainer) {
        self.engine = engine
        self.trainer = trainer
        item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        super.init()
        menu.delegate = self
        item.menu = menu
        refreshIcon()
    }

    func refreshIcon() {
        guard let button = item.button else { return }
        let (symbol, tip): (String, String)
        switch engine.status {
        case .active: (symbol, tip) = ("textformat.abc.dottedunderline", "Typo: correcting")
        case .disabled: (symbol, tip) = ("pause.circle", "Typo: disabled")
        case .blockedApp: (symbol, tip) = ("nosign", "Typo: off in this app")
        case .composing: (symbol, tip) = ("character.ja", "Typo: paused while composing")
        case .noPermission: (symbol, tip) = ("exclamationmark.triangle", "Typo: needs permissions")
        }
        button.image = NSImage(systemSymbolName: symbol, accessibilityDescription: tip)
        button.toolTip = tip
    }

    func menuNeedsUpdate(_ menu: NSMenu) {
        menu.removeAllItems()
        let status: String
        switch engine.status {
        case .active: status = "Correcting in \(engine.frontAppName ?? "current app")"
        case .disabled: status = "Disabled"
        case .blockedApp: status = "Off in \(engine.frontAppName ?? "this app") (blocklist)"
        case .composing: status = "Paused: \(engine.inputSource.sourceID.split(separator: ".").last.map(String.init) ?? "input method") is composing"
        case .noPermission: status = "Needs Accessibility + Input Monitoring"
        }
        menu.addItem(disabled(status))
        menu.addItem(disabled("Today: \(engine.autoToday) fixed · \(engine.undoneToday) undone · \(engine.learnedToday) learned"))
        let model = engine.corrector.ranker.map { $0.source.path.contains("Application Support") ? "personal ranker" : "base ranker" } ?? "no neural ranker"
        menu.addItem(disabled("Engine: \(engine.corrector.lexicon.count) words · \(model)"))
        menu.addItem(.separator())

        let toggle = NSMenuItem(title: engine.settings.enabled ? "Disable corrections" : "Enable corrections", action: #selector(toggleEnabled), keyEquivalent: "")
        toggle.target = self
        menu.addItem(toggle)

        if !engine.recent.isEmpty {
            let sub = NSMenu()
            for (t, f) in engine.recent { sub.addItem(disabled("\(t) → \(f)")) }
            let recentItem = NSMenuItem(title: "Recent corrections", action: nil, keyEquivalent: "")
            recentItem.submenu = sub
            menu.addItem(recentItem)
        }
        menu.addItem(.separator())

        let trainTitle = trainer.isRunning ? "Training… (see train.log)" : "Train personal model now"
        let train = NSMenuItem(title: trainTitle, action: #selector(trainNow), keyEquivalent: "")
        train.target = self
        train.isEnabled = !trainer.isRunning && trainer.available
        menu.addItem(train)
        let last = trainer.lastTrained.map { "Last trained \(Self.relative($0))" } ?? "Never trained (base model)"
        menu.addItem(disabled(trainer.lastError.map { "\(last) · error: \($0)" } ?? last))
        menu.addItem(.separator())

        for (title, sel) in [("Edit blocklist…", #selector(editBlocklist)), ("Open data folder", #selector(openData)), ("Permissions…", #selector(openPermissions))] {
            let mi = NSMenuItem(title: title, action: sel, keyEquivalent: "")
            mi.target = self
            menu.addItem(mi)
        }
        if Bundle.main.bundleIdentifier != nil {
            let login = NSMenuItem(title: "Launch at login", action: #selector(toggleLogin), keyEquivalent: "")
            login.target = self
            login.state = SMAppService.mainApp.status == .enabled ? .on : .off
            menu.addItem(login)
        }
        menu.addItem(.separator())
        let quit = NSMenuItem(title: "Quit Typo", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        menu.addItem(quit)
    }

    private func disabled(_ title: String) -> NSMenuItem {
        let mi = NSMenuItem(title: title, action: nil, keyEquivalent: "")
        mi.isEnabled = false
        return mi
    }

    private static func relative(_ d: Date) -> String {
        let f = RelativeDateTimeFormatter()
        f.unitsStyle = .short
        return f.localizedString(for: d, relativeTo: Date())
    }

    @objc private func toggleEnabled() {
        engine.settings.enabled.toggle()
        engine.tracker.reset()
        refreshIcon()
    }

    @objc private func trainNow() {
        if let reason = trainer.maybeTrain(force: true) {
            let a = NSAlert()
            a.messageText = "Training not started"
            a.informativeText = reason
            a.runModal()
        }
    }

    @objc private func editBlocklist() {
        NSWorkspace.shared.open(engine.settings.blocklistFile)
    }

    @objc private func openData() {
        NSWorkspace.shared.open(Paths.appSupport)
    }

    @objc private func openPermissions() {
        Permissions.requestAccessibility()
        Permissions.requestInputMonitoring()
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility") {
            NSWorkspace.shared.open(url)
        }
    }

    @objc private func toggleLogin() {
        do {
            if SMAppService.mainApp.status == .enabled { try SMAppService.mainApp.unregister() }
            else { try SMAppService.mainApp.register() }
        } catch {
            NSLog("launch at login: \(error)")
        }
    }
}
