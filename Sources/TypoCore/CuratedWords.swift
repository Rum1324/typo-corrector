import Foundation

/// Membership test against a curated English word list (macOS ships Webster's 2nd at
/// /usr/share/dict/web2). The frequency lexicon comes from web text, so it also contains common
/// misspellings ("befor", "somthing"); a curated list tells those apart from rare real words
/// ("mien", "tarn", "cleat") that sit just as close to common words.
public final class CuratedWords {
    private var hashes = Set<UInt64>()

    public static let systemPath = "/usr/share/dict/web2"

    public init?(path: String = CuratedWords.systemPath) {
        guard let data = FileManager.default.contents(atPath: path) else { return nil }
        var set = Set<UInt64>()
        set.reserveCapacity(240_000)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let b = raw.bindMemory(to: UInt8.self)
            var start = 0
            for i in 0...b.count where i == b.count || b[i] == 10 {
                // Lowercase entries only: capitalised ones are proper nouns.
                if i > start, (start..<i).allSatisfy({ b[$0] >= 97 && b[$0] <= 122 }) {
                    set.insert(Self.hash(b, start, i))
                }
                start = i + 1
            }
        }
        guard !set.isEmpty else { return nil }
        hashes = set
    }

    public var count: Int { hashes.count }

    public func contains(_ lowercased: [UInt8]) -> Bool {
        lowercased.withUnsafeBufferPointer { hashes.contains(Self.hash($0, 0, $0.count)) }
    }

    /// FNV-1a, 64-bit.
    private static func hash(_ b: UnsafeBufferPointer<UInt8>, _ from: Int, _ to: Int) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for k in from..<to { h = (h ^ UInt64(b[k])) &* 0x100_0000_01b3 }
        return h
    }
}
