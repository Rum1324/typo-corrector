import Foundation

/// In-memory view of what the user has taught us: manual fixes, rejected corrections,
/// and unknown words they type and leave alone. Backed by `Store`, updated incrementally.
public final class PersonalModel {
    public private(set) var manual: [String: [String: Int]] = [:]
    public private(set) var undone: [String: [String: Int]] = [:]
    public private(set) var unknownSeen: [String: Int] = [:]

    /// Manual fixes needed before a pair is auto-applied.
    public var autoThreshold = 2
    /// Higher bar when the "typo" is itself a dictionary word (form -> from).
    public var autoThresholdKnownWord = 4
    /// Times an unknown word must be typed and left alone before we stop touching it.
    public var acceptThreshold = 3

    public init() {}

    public convenience init(store: Store) {
        self.init()
        if let pairs = try? store.pairCounts() {
            for p in pairs {
                switch p.source {
                case .manual: manual[p.typo, default: [:]][p.fix, default: 0] += p.count
                case .undo: undone[p.typo, default: [:]][p.fix, default: 0] += p.count
                case .auto: break
                }
            }
        }
        unknownSeen = (try? store.unknownCounts()) ?? [:]
    }

    public func addManual(typo: String, fix: String) {
        manual[typo, default: [:]][fix, default: 0] += 1
    }

    public func addUndo(typo: String, fix: String) {
        undone[typo, default: [:]][fix, default: 0] += 1
    }

    public func sawUnknown(_ word: String) { unknownSeen[word, default: 0] += 1 }

    public func forgetUnknown(_ word: String) {
        if let c = unknownSeen[word] { unknownSeen[word] = max(c - 1, 0) }
    }

    public func manualCount(typo: String, fix: String) -> Int { manual[typo]?[fix] ?? 0 }
    public func undoCount(typo: String, fix: String) -> Int { undone[typo]?[fix] ?? 0 }

    /// Rejections (backspace right after a fix) before a word is left alone for good.
    public var rejectThreshold = 2

    /// The user has rejected corrections of this word, or types it regularly on purpose.
    public func neverCorrect(_ word: String) -> Bool {
        if let u = undone[word], u.values.reduce(0, +) >= rejectThreshold { return true }
        let manualTotal = manual[word]?.values.reduce(0, +) ?? 0
        return (unknownSeen[word] ?? 0) - manualTotal >= acceptThreshold
    }

    /// A fix the user has made often enough that we apply it without asking the ranker.
    public func autoFix(for word: String, isDictionaryWord: Bool) -> String? {
        guard let fixes = manual[word] else { return nil }
        let need = isDictionaryWord ? autoThresholdKnownWord : autoThreshold
        let eligible = fixes.filter { $0.value >= need && undoCount(typo: word, fix: $0.key) == 0 }
            .sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }
        guard let best = eligible.first else { return nil }
        if eligible.count > 1, eligible[1].value == best.value { return nil }   // ambiguous: let the ranker decide
        return best.key
    }

    /// Additive score bonus for candidates the user has chosen before.
    public func boost(typo: String, candidate: String) -> Double {
        let m = manualCount(typo: typo, fix: candidate)
        let u = undoCount(typo: typo, fix: candidate)
        return 2.0 * log(1 + Double(m)) - 3.0 * Double(u)
    }
}
