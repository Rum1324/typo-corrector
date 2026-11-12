import Foundation

public enum EditDistance {
    /// Optimal string alignment (restricted Damerau-Levenshtein) with a cutoff.
    /// Returns -1 when the distance exceeds `max`.
    public static func osa(_ a: [UInt8], _ b: [UInt8], max: Int) -> Int {
        let n = a.count, m = b.count
        if abs(n - m) > max { return -1 }
        if n == 0 { return m <= max ? m : -1 }
        if m == 0 { return n <= max ? n : -1 }
        var prev2 = [Int](repeating: 0, count: m + 1)
        var prev = Array(0...m)
        var cur = [Int](repeating: 0, count: m + 1)
        for i in 1...n {
            cur[0] = i
            var rowMin = i
            for j in 1...m {
                let cost = a[i - 1] == b[j - 1] ? 0 : 1
                var v = Swift.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
                if i > 1, j > 1, a[i - 1] == b[j - 2], a[i - 2] == b[j - 1] {
                    v = Swift.min(v, prev2[j - 2] + 1)
                }
                cur[j] = v
                if v < rowMin { rowMin = v }
            }
            if rowMin > max { return -1 }
            swap(&prev2, &prev)
            swap(&prev, &cur)
        }
        let d = prev[m]
        return d > max ? -1 : d
    }

    public static func osa(_ a: String, _ b: String, max: Int) -> Int {
        osa(Array(a.utf8), Array(b.utf8), max: max)
    }
}

/// Keyboard-aware edit cost: how surprising is it to type `typo` when you meant `intended`?
/// Lower is more plausible. Roughly in nats, so it can be added to log-probabilities.
public struct EditModel {
    public static let shared = EditModel()

    public let adjacency: [UInt8: Set<UInt8>]
    private let vowels: Set<UInt8> = Set("aeiou".utf8)

    public var subAdjacent = 1.0
    public var subVowel = 1.6
    public var sub = 2.6
    public var insAdjacent = 1.2
    public var ins = 2.2
    public var delDouble = 0.9
    public var del = 1.8
    public var transpose = 0.9

    public init() {
        let rows: [[UInt8]] = ["qwertyuiop", "asdfghjkl", "zxcvbnm"].map { Array($0.utf8) }
        var adj: [UInt8: Set<UInt8>] = [:]
        for (r, row) in rows.enumerated() {
            for (i, c) in row.enumerated() {
                var s = Set<UInt8>()
                let neighbours = [(r, i - 1), (r, i + 1), (r - 1, i), (r - 1, i + 1), (r + 1, i), (r + 1, i - 1)]
                for (rr, ii) in neighbours where rr >= 0 && rr < rows.count && ii >= 0 && ii < rows[rr].count {
                    s.insert(rows[rr][ii])
                }
                adj[c] = s
            }
        }
        adjacency = adj
    }

    public func adjacent(_ a: UInt8, _ b: UInt8) -> Bool { adjacency[a]?.contains(b) ?? false }

    private func subCost(_ typed: UInt8, _ meant: UInt8) -> Double {
        if adjacent(typed, meant) { return subAdjacent }
        if vowels.contains(typed) && vowels.contains(meant) { return subVowel }
        return sub
    }

    private func insCost(_ extra: UInt8, _ intended: [UInt8], at j: Int) -> Double {
        for k in [j - 1, j] where k >= 0 && k < intended.count {
            let nb = intended[k]
            if nb == extra || adjacent(extra, nb) { return insAdjacent }
        }
        return ins
    }

    private func delCost(_ intended: [UInt8], _ j: Int) -> Double {
        let c = intended[j]
        if (j > 0 && intended[j - 1] == c) || (j + 1 < intended.count && intended[j + 1] == c) { return delDouble }
        return del
    }

    public func cost(typo: [UInt8], intended: [UInt8]) -> Double {
        let n = typo.count, m = intended.count
        if n == 0 && m == 0 { return 0 }
        var d = [[Double]](repeating: [Double](repeating: 0, count: m + 1), count: n + 1)
        if n > 0 { for i in 1...n { d[i][0] = d[i - 1][0] + insCost(typo[i - 1], intended, at: 0) } }
        if m > 0 { for j in 1...m { d[0][j] = d[0][j - 1] + delCost(intended, j - 1) } }
        if n == 0 || m == 0 { return d[n][m] }
        for i in 1...n {
            for j in 1...m {
                let s = typo[i - 1] == intended[j - 1] ? 0 : subCost(typo[i - 1], intended[j - 1])
                var v = d[i - 1][j - 1] + s
                v = Swift.min(v, d[i - 1][j] + insCost(typo[i - 1], intended, at: j))
                v = Swift.min(v, d[i][j - 1] + delCost(intended, j - 1))
                if i > 1, j > 1, typo[i - 1] == intended[j - 2], typo[i - 2] == intended[j - 1] {
                    v = Swift.min(v, d[i - 2][j - 2] + transpose)
                }
                d[i][j] = v
            }
        }
        return d[n][m]
    }
}
