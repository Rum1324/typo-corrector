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
