import Foundation

/// In-memory view of what the user has taught us: manual fixes, rejected corrections,
/// and unknown words they type and leave alone. Backed by `Store`, updated incrementally.
public final class PersonalModel {
    /// What happened the last few times one (typed word -> fix) pair came up. Every new outcome
    /// shrinks the older ones by `memory`, so the counts follow what the user does now.
    public struct Evidence: Equatable {
        /// Fixes that stayed: automatic fixes the user did not undo, and fixes made by hand.
        public var kept = 0.0
        /// Automatic fixes the user undid or typed over.
        public var undone = 0.0
    }

    public private(set) var manual: [String: [String: Int]] = [:]
    public private(set) var evidence: [String: [String: Evidence]] = [:]
    public private(set) var unknownSeen: [String: Int] = [:]
    /// The latest automatic fix and the pair's evidence before it: an undo of that fix takes it back.
    private var lastApplied: (typo: String, fix: String, before: Evidence)?

    /// Manual fixes needed before a pair is auto-applied.
    public var autoThreshold = 2
    /// Higher bar when the "typo" is itself a dictionary word (form -> from).
    public var autoThresholdKnownWord = 4
    /// Times an unknown word must be typed and left alone before we stop touching it.
    public var acceptThreshold = 3
    /// Weight an outcome keeps each time a newer one arrives for the same pair (about the last 7 count).
    public var memory = 0.85
    /// What an automatic fix that was not undone counts for, next to 1 for an undo or a fix made by
    /// hand. Not undoing is weak evidence: a rejection after a click or a shortcut goes unseen. At
    /// 0.5 a fix that is undone a third of the time or more ends up blocked.
    public var appliedWeight = 0.5
    /// Score taken from a fix per unit of undo evidence when undos are all there is.
    public var undoWeight = 3.0
    /// Undo evidence at which a word is left alone, when it is also at least the kept evidence.
    /// Two undos in a row weigh 1.85: one undo never blocks a word.
    public var rejectThreshold = 1.5

    public init() {}

    public convenience init(store: Store) {
        self.init()
        // Replayed in the order they happened: the evidence depends on what came last.
        if let events = try? store.pairEvents() {
            for e in events {
                switch e.source {
                case .manual: addManual(typo: e.typo, fix: e.fix)
                case .auto: addApplied(typo: e.typo, fix: e.fix)
                case .undo: addUndo(typo: e.typo, fix: e.fix)
                }
            }
        }
        lastApplied = nil
        unknownSeen = (try? store.unknownCounts()) ?? [:]
    }

    private func record(typo: String, fix: String, kept: Double = 0, undone: Double = 0) {
        var e = evidence[typo]?[fix] ?? Evidence()
        e.kept = e.kept * memory + kept
        e.undone = e.undone * memory + undone
        evidence[typo, default: [:]][fix] = e
    }

    public func addManual(typo: String, fix: String) {
        manual[typo, default: [:]][fix, default: 0] += 1
        if let last = lastApplied, last.typo == typo, last.fix == fix { lastApplied = nil }
        record(typo: typo, fix: fix, kept: 1)
    }

    /// An automatic fix was made. It counts as kept unless `addUndo` follows for the same pair.
    public func addApplied(typo: String, fix: String) {
        lastApplied = (typo, fix, evidence[typo]?[fix] ?? Evidence())
        record(typo: typo, fix: fix, kept: appliedWeight)
    }

    /// The user took an automatic fix back (backspace right after it, or typed something else over it).
    public func addUndo(typo: String, fix: String) {
        if let last = lastApplied, last.typo == typo, last.fix == fix {
            evidence[typo, default: [:]][fix] = last.before          // that fix was not kept after all
            lastApplied = nil
        } else if var e = evidence[typo]?[fix] {
            e.kept = max(e.kept - appliedWeight, 0)                  // an earlier fix, already counted as kept
            evidence[typo]?[fix] = e
        }
        record(typo: typo, fix: fix, undone: 1)
    }

    public func sawUnknown(_ word: String) { unknownSeen[word, default: 0] += 1 }

    public func forgetUnknown(_ word: String) {
        if let c = unknownSeen[word] { unknownSeen[word] = max(c - 1, 0) }
    }

    public func manualCount(typo: String, fix: String) -> Int { manual[typo]?[fix] ?? 0 }
    public func evidence(typo: String, fix: String) -> Evidence { evidence[typo]?[fix] ?? Evidence() }

    /// The user keeps rejecting corrections of this word, or types it regularly on purpose.
    /// Undos block a word only while they are at least as many as the fixes that stayed: a fix
    /// that is usually kept survives the odd undo, and fixing the word by hand brings it back.
    public func neverCorrect(_ word: String) -> Bool {
        if let pairs = evidence[word] {
            let undone = pairs.values.reduce(0) { $0 + $1.undone }
            let kept = pairs.values.reduce(0) { $0 + $1.kept }
            if undone >= rejectThreshold, undone >= kept { return true }
        }
        let manualTotal = manual[word]?.values.reduce(0, +) ?? 0
        return (unknownSeen[word] ?? 0) - manualTotal >= acceptThreshold
    }

    /// A fix the user has made often enough that we apply it without asking the ranker.
    /// Not while undos are half the kept evidence or more. A dictionary word (form -> from) is
    /// replaced without looking at the sentence, so there one undo sends the pair back to the
    /// context model until about five more fixes have been kept.
    public func autoFix(for word: String, isDictionaryWord: Bool) -> String? {
        guard let fixes = manual[word] else { return nil }
        let need = isDictionaryWord ? autoThresholdKnownWord : autoThreshold
        let eligible = fixes.filter {
            let e = evidence(typo: word, fix: $0.key)
            return $0.value >= need && e.undone * 2 < e.kept && (!isDictionaryWord || e.undone < 0.5)
        }.sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }
        guard let best = eligible.first else { return nil }
        if eligible.count > 1, eligible[1].value == best.value { return nil }   // ambiguous: let the ranker decide
        return best.key
    }

    /// Score taken from a candidate because the user undid it: the undo evidence times the share
    /// of the pair's recent outcomes that were undos. One undo with no history costs 1.5 (the fix
    /// then needs about 98% confidence instead of 90%); one undo after many kept fixes about 0.6.
    public func undoPenalty(typo: String, candidate: String) -> Double {
        let e = evidence(typo: typo, fix: candidate)
        guard e.undone > 0 else { return 0 }
        return undoWeight * e.undone * e.undone / (e.undone + e.kept + 1)
    }

    /// Additive score bonus for candidates the user has chosen before, less the undo penalty.
    public func boost(typo: String, candidate: String) -> Double {
        2.0 * log(1 + Double(manualCount(typo: typo, fix: candidate))) - undoPenalty(typo: typo, candidate: candidate)
    }
}
