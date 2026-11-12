import Foundation

public enum Paths {
    /// Repository root, found at run time: the nearest folder above the executable, a loaded
    /// bundle (the test bundle) or the current directory that holds Package.swift and
    /// Sources/TypoCore. It is not taken from `#filePath`, which would bake the build machine's
    /// path into every binary. A copy of the app outside the repository gets a folder that does
    /// not exist, so the repository's Resources and training folders are simply not found.
    public static let repoRoot: URL = {
        let fm = FileManager.default
        var starts = [URL(fileURLWithPath: CommandLine.arguments[0])]
        if let exe = Bundle.main.executableURL { starts.append(exe) }
        starts += Bundle.allBundles.map(\.bundleURL)
        starts.append(URL(fileURLWithPath: fm.currentDirectoryPath, isDirectory: true).appendingPathComponent("_"))
        for start in starts {
            var dir = start.resolvingSymlinksInPath().deletingLastPathComponent()
            for _ in 0..<8 {
                if fm.fileExists(atPath: dir.appendingPathComponent("Package.swift").path),
                   fm.fileExists(atPath: dir.appendingPathComponent("Sources/TypoCore").path) {
                    return dir
                }
                let up = dir.deletingLastPathComponent()
                if up.path == dir.path { break }
                dir = up
            }
        }
        return URL(fileURLWithPath: "/nonexistent/typo-corrector", isDirectory: true)
    }()

    public static var appSupport: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let dir = base.appendingPathComponent("Typo", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    public static var database: URL { appSupport.appendingPathComponent("typo.sqlite") }
    public static var trainingDir: URL { repoRoot.appendingPathComponent("training", isDirectory: true) }

    /// Finds a bundled resource: app bundle first, then the repo's Resources folder (dev builds).
    public static func resource(_ name: String) -> URL? {
        var candidates: [URL] = []
        if let r = Bundle.main.resourceURL { candidates.append(r.appendingPathComponent(name)) }
        candidates.append(repoRoot.appendingPathComponent("Resources").appendingPathComponent(name))
        let exe = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
        candidates.append(exe.appendingPathComponent("Resources").appendingPathComponent(name))
        return candidates.first { FileManager.default.fileExists(atPath: $0.path) }
    }

    /// Personal (fine-tuned) weights win over the bundled base weights.
    public static func rankerFiles() -> (manifest: URL, weights: URL)? {
        let personal = (appSupport.appendingPathComponent("ranker.json"), appSupport.appendingPathComponent("ranker.bin"))
        if FileManager.default.fileExists(atPath: personal.0.path), FileManager.default.fileExists(atPath: personal.1.path) {
            return personal
        }
        if let m = resource("ranker.json"), let w = resource("ranker.bin") { return (m, w) }
        return nil
    }
}

public enum TypoError: Error, CustomStringConvertible {
    case missingResource(String)
    case badResource(String)
    case sqlite(String)

    public var description: String {
        switch self {
        case .missingResource(let s): return "missing resource: \(s)"
        case .badResource(let s): return "bad resource: \(s)"
        case .sqlite(let s): return "sqlite: \(s)"
        }
    }
}
