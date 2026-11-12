import Foundation
import TypoCore

/// Appends timestamped lines to ~/Library/Application Support/Typo/typo.log only (never the
/// system log, since lines contain words you typed). Rotates once at 2 MB.
enum Log {
    static let url = Paths.appSupport.appendingPathComponent("typo.log")
    private static let queue = DispatchQueue(label: "typo.log")
    private static let formatter: DateFormatter = {
        let f = DateFormatter(); f.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"; return f
    }()

    static func write(_ message: String) {
        let line = "\(formatter.string(from: Date())) \(message)\n"
        queue.async {
            if let size = (try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? Int, size > 2_000_000 {
                let old = url.deletingPathExtension().appendingPathExtension("log.1")
                try? FileManager.default.removeItem(at: old)
                try? FileManager.default.moveItem(at: url, to: old)
            }
            if !FileManager.default.fileExists(atPath: url.path) {
                FileManager.default.createFile(atPath: url.path, contents: nil)
            }
            if let h = try? FileHandle(forWritingTo: url) {
                h.seekToEndOfFile()
                h.write(line.data(using: .utf8)!)
                try? h.close()
            }
        }
    }
}
