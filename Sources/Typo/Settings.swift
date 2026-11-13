import Foundation
import TypoCore

final class Settings {
    static let shared = Settings()
    private let defaults = UserDefaults.standard

    static let defaultBlocklist = """
    # One bundle identifier per line. A trailing * matches a prefix. Lines starting with # are ignored.
    com.apple.Terminal
    com.googlecode.iterm2
    dev.warp.Warp-Stable
    com.github.wez.wezterm
    net.kovidgoyal.kitty
    com.mitchellh.ghostty
    com.apple.dt.Xcode
    com.microsoft.VSCode
    com.todesktop.230313mzl4w4u92
    dev.zed.Zed
    com.jetbrains.*
    com.sublimetext.*
    com.apple.Passwords
    com.1password.*
    com.agilebits.*
    com.bitwarden.desktop
    # remote desktops and VMs relay keystrokes to another machine, where password fields are invisible to us
    com.apple.ScreenSharing
    com.microsoft.rdc.macos
    com.realvnc.vncviewer
    com.teamviewer.TeamViewer
    com.anydesk.anydesk
    com.p5sys.jump.mac.viewer
    com.parallels.desktop.console
    com.vmware.fusion
    org.virtualbox.app.VirtualBox
    com.utmapp.UTM
    """

    var enabled: Bool {
        get { defaults.object(forKey: "enabled") as? Bool ?? true }
        set { defaults.set(newValue, forKey: "enabled") }
    }

    var minProbability: Double {
        // Unset: the engine's tuned default (CorrectorConfig), not a copy of it that can go stale.
        get { defaults.object(forKey: "minProbability") as? Double ?? CorrectorConfig().minProbability }
        set { defaults.set(newValue, forKey: "minProbability") }
    }

    /// How far the context path's confidence bars are lowered (`CorrectorConfig.contextLooseness`).
    /// The menu offers `eagernessLevels`; any number can be set with
    /// `defaults write <bundle id> looseness -float 0.75`.
    static let defaultLooseness = 1.0
    /// The menu's levels ("How eagerly to correct"). The details are test-split results of
    /// 2026-11-12 with apostrophes on: the share of real-typing typos fixed, and how often a
    /// correct word was changed over all sets (apostrophe fixes not counted).
    static let eagernessLevels: [(title: String, value: Double, detail: String)] = [
        ("Careful", 0.0, "fixes 56% of typos, changes 1 correct word in 3,100"),
        ("Balanced", 0.5, "fixes 59%, changes 1 in 2,500"),
        ("Eager", 1.0, "fixes 62%, changes 1 in 1,600"),
        ("Very eager", 1.5, "fixes 64%, changes 1 in 1,100"),
        ("Maximum", 2.0, "fixes 66%, changes 1 in 670"),
    ]
    var looseness: Double {
        get { defaults.object(forKey: "looseness") as? Double ?? Settings.defaultLooseness }
        set { defaults.set(newValue, forKey: "looseness") }
    }

    /// "doesnt" -> "doesn't", "im" -> "I'm" (`Corrector.contractions`).
    var fixContractions: Bool {
        get { defaults.object(forKey: "fixContractions") as? Bool ?? true }
        set { defaults.set(newValue, forKey: "fixContractions") }
    }

    /// Backspace right after a fix puts the typed word back. Off by default: a backspace meant to
    /// go back and fix the word by hand looks the same, and would undo a fix the user wanted.
    var backspaceUndo: Bool {
        get { defaults.bool(forKey: "backspaceUndo") }
        set { defaults.set(newValue, forKey: "backspaceUndo") }
    }

    /// `defaults write <bundle id> debugInjection -bool YES` logs every injected event.
    var debugInjection: Bool { defaults.bool(forKey: "debugInjection") }

    var blocklistFile: URL { Paths.appSupport.appendingPathComponent("blocklist.txt") }

    private(set) var blocklist: [String] = []
    private var blocklistLoaded: Date?

    init() { loadBlocklist() }

    func loadBlocklist() {
        if !FileManager.default.fileExists(atPath: blocklistFile.path) {
            try? Self.defaultBlocklist.write(to: blocklistFile, atomically: true, encoding: .utf8)
        }
        let text = (try? String(contentsOf: blocklistFile, encoding: .utf8)) ?? ""
        blocklist = text.split(separator: "\n").map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }
        blocklistLoaded = Date()
    }

    /// Re-reads the blocklist file if it is more than 30 s old. Called on app switches, never from the tap.
    func reloadBlocklistIfStale() {
        if let t = blocklistLoaded, Date().timeIntervalSince(t) > 30 { loadBlocklist() }
    }

    func isBlocked(bundleID: String?) -> Bool {
        guard let id = bundleID else { return false }
        for entry in blocklist {
            if entry.hasSuffix("*") { if id.hasPrefix(entry.dropLast()) { return true } }
            else if entry == id { return true }
        }
        return false
    }
}
