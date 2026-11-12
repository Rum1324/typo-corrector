import Foundation

/// Word-frequency list. Counts are occurrences per billion words (from wordfreq).
public final class Lexicon {
    public let strings: [String]
    public let bytes: [[UInt8]]
    public let counts: [Double]
    public let total: Double = 1_000_000_000
    private let index: [String: Int32]

    public init(url: URL) throws {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else {
            throw TypoError.badResource(url.lastPathComponent)
        }
        var strings: [String] = [], bytes: [[UInt8]] = [], counts: [Double] = []
        var index: [String: Int32] = [:]
        strings.reserveCapacity(90_000); bytes.reserveCapacity(90_000)
        counts.reserveCapacity(90_000); index.reserveCapacity(90_000)
        for line in text.split(separator: "\n") {
            let parts = line.split(separator: "\t", maxSplits: 1)
            guard parts.count == 2, let c = Double(parts[1]) else { continue }
            let w = String(parts[0])
            guard index[w] == nil else { continue }
            index[w] = Int32(strings.count)
            strings.append(w); bytes.append(Array(w.utf8)); counts.append(c)
        }
        self.strings = strings; self.bytes = bytes; self.counts = counts; self.index = index
    }

    public var count: Int { strings.count }
    public func id(_ word: String) -> Int32? { index[word] }
    public func contains(_ word: String) -> Bool { index[word] != nil }
    public func frequency(_ word: String) -> Double { id(word).map { counts[Int($0)] } ?? 0 }
    public func frequency(id: Int32) -> Double { counts[Int(id)] }
    public func logProb(id: Int32) -> Double { log(counts[Int(id)] / total) }
    public func logProb(_ word: String) -> Double {
        let f = frequency(word)
        return log(max(f, 0.01) / total)
    }
    /// Words with frequency >= `reversedMin`, each reversed, sorted bytewise: a word's tails become prefixes.
    private var reversedWords: [[UInt8]] = []
    private var reversedMin: Double = -1

    /// Whether `tail` is the end of a longer word at least `minFrequency` common ("ple" of "people").
    /// After a click the caret may sit inside a word, so a typed word may be only such a tail.
    public func endsLongerWord(_ tail: [UInt8], minFrequency: Double) -> Bool {
        guard !tail.isEmpty else { return false }
        if reversedMin != minFrequency {
            reversedWords = (0..<count).filter { counts[$0] >= minFrequency }.map { Array(bytes[$0].reversed()) }
                .sorted { $0.lexicographicallyPrecedes($1) }
            reversedMin = minFrequency
        }
        let key = Array(tail.reversed())
        var lo = 0, hi = reversedWords.count
        while lo < hi {
            let mid = (lo + hi) / 2
            if reversedWords[mid].lexicographicallyPrecedes(key) { lo = mid + 1 } else { hi = mid }
        }
        // Entries starting with `key` are contiguous from `lo`; the first may be the word itself.
        var i = lo
        while i < reversedWords.count, reversedWords[i].starts(with: key) {
            if reversedWords[i].count > key.count { return true }
            i += 1
        }
        return false
    }

    /// Zipf scale: log10(occurrences per billion). 7 = "the", ~3 = once per million words.
    public func zipf(_ word: String) -> Double {
        let f = frequency(word)
        return f > 0 ? log10(f) : 0
    }
}
