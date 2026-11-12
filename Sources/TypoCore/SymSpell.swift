import Foundation

/// Delete-based candidate generation (SymSpell). The index is two sorted parallel arrays
/// (hash of a deletion variant -> word id), so it is compact and fast to build.
public final class SymSpell {
    public let lexicon: Lexicon
    public let prefixLength: Int
    public let maxIndexDistance: Int
    private var keys: [UInt64] = []
    private var ids: [Int32] = []

    public init(lexicon: Lexicon, prefixLength: Int = 7, maxDistance: Int = 2) {
        self.lexicon = lexicon
        self.prefixLength = prefixLength
        self.maxIndexDistance = maxDistance
        build()
    }

    private func build() {
        var pairs: [(UInt64, Int32)] = []
        pairs.reserveCapacity(lexicon.count * 24)
        var set = Set<UInt64>()
        for (i, w) in lexicon.bytes.enumerated() {
            let prefix = Array(w.prefix(prefixLength))
            set.removeAll(keepingCapacity: true)
            SymSpell.deletes(of: prefix, maxDistance: maxIndexDistance, into: &set)
            let id = Int32(i)
            for h in set { pairs.append((h, id)) }
        }
        pairs.sort { $0.0 < $1.0 }
        keys = pairs.map { $0.0 }
        ids = pairs.map { $0.1 }
    }

    @inline(__always) static func fnv(_ bytes: [UInt8]) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for b in bytes { h ^= UInt64(b); h = h &* 0x100_0000_01b3 }
        return h
    }

    static func deletes(of bytes: [UInt8], maxDistance: Int, into set: inout Set<UInt64>) {
        set.insert(fnv(bytes))
        guard maxDistance > 0, bytes.count > 1 else { return }
        for i in 0..<bytes.count {
            var b = bytes
            b.remove(at: i)
            if set.insert(fnv(b)).inserted {
                deletes(of: b, maxDistance: maxDistance - 1, into: &set)
            }
        }
    }

    private func lowerBound(_ h: UInt64) -> Int {
        var lo = 0, hi = keys.count
        while lo < hi {
            let mid = (lo + hi) >> 1
            if keys[mid] < h { lo = mid + 1 } else { hi = mid }
        }
        return lo
    }

    public struct Hit { public let id: Int32; public let distance: Int }

    /// All dictionary words within `maxDistance` (OSA) of `word`.
    public func lookup(_ word: [UInt8], maxDistance: Int) -> [Hit] {
        let md = Swift.min(maxDistance, maxIndexDistance)
        let prefix = Array(word.prefix(prefixLength))
        var set = Set<UInt64>()
        SymSpell.deletes(of: prefix, maxDistance: md, into: &set)
        var seen = Set<Int32>()
        var out: [Hit] = []
        for h in set {
            var i = lowerBound(h)
            while i < keys.count && keys[i] == h {
                let id = ids[i]; i += 1
                guard seen.insert(id).inserted else { continue }
                let cand = lexicon.bytes[Int(id)]
                if abs(cand.count - word.count) > md { continue }
                let d = EditDistance.osa(word, cand, max: md)
                if d >= 0 { out.append(Hit(id: id, distance: d)) }
            }
        }
        return out
    }

    public func lookup(_ word: String, maxDistance: Int) -> [Hit] {
        lookup(Array(word.utf8), maxDistance: maxDistance)
    }
}
