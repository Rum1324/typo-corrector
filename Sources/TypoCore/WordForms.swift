import Foundation

/// How often each word form occurs in the language model's training text (training/build_forms.py),
/// including the many real words the 80,000-word lexicon lacks ("backprop", "timesteps", names).
/// The decider uses it as evidence that a typed word outside the lexicon is meant: a form the corpus
/// has seen often is probably deliberate, one it has never seen next to a common word probably a slip.
public final class WordForms {
    private var table: [UInt64: Float] = [:]
    /// Log-probability given to a form that is not in the table.
    public let floor: Float
    public var count: Int { table.count }

    /// File: a first line "#total<TAB>tokens", then "word<TAB>count" per line (lowercase words).
    /// A form missing from the file counts as seen `unseenCount` times.
    public init?(url: URL, unseenCount: Double = 1) {
        guard let data = try? Data(contentsOf: url, options: .alwaysMapped), !data.isEmpty else { return nil }
        var total = 0.0
        var entries: [(UInt64, Double)] = []
        entries.reserveCapacity(700_000)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let b = raw.bindMemory(to: UInt8.self)
            var start = 0
            for i in 0...b.count where i == b.count || b[i] == 10 {
                defer { start = i + 1 }
                guard i > start, let tab = (start..<i).first(where: { b[$0] == 9 }) else { continue }
                var n = 0.0
                for k in (tab + 1)..<i where b[k] >= 48 && b[k] <= 57 { n = n * 10 + Double(b[k] - 48) }
                if b[start] == 35 { total = n; continue }                       // "#total"
                entries.append((Self.hash(b, start, tab), n))
            }
        }
        guard total > 0, !entries.isEmpty else { return nil }
        floor = Float(log(unseenCount / total))
        table.reserveCapacity(entries.count)
        for (h, n) in entries { table[h] = Float(log(max(n, 1) / total)) }
    }

    /// The bundled table `name` (Resources), or the one in `directory`.
    public static func load(_ name: String, directory: URL? = nil, unseenCount: Double = 1) -> WordForms? {
        let inDirectory = directory?.appendingPathComponent(name)
        if let u = inDirectory, FileManager.default.fileExists(atPath: u.path) { return WordForms(url: u, unseenCount: unseenCount) }
        return Paths.resource(name).flatMap { WordForms(url: $0, unseenCount: unseenCount) }
    }

    /// Natural-log share of the corpus that is this (lowercased) word form.
    public func logProbability(_ lowercased: [UInt8]) -> Float {
        lowercased.withUnsafeBufferPointer { table[Self.hash($0, 0, $0.count)] } ?? floor
    }

    /// FNV-1a, 64-bit.
    private static func hash(_ b: UnsafeBufferPointer<UInt8>, _ from: Int, _ to: Int) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for k in from..<to { h = (h ^ UInt64(b[k])) &* 0x100_0000_01b3 }
        return h
    }
}
