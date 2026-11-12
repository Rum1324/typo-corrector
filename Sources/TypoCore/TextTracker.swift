import Foundation

public enum KeyInput: Equatable {
    case char(Character)
    case backspace
    case reset
}

public struct LearnedPair: Equatable {
    public let typo: String
    public let fix: String
    public init(typo: String, fix: String) { self.typo = typo; self.fix = fix }
}

public struct Commit: Equatable {
    public let word: String
    public let boundary: Character
    public let start: Int
    /// (what the user originally typed → what they retyped), when a delete-and-retype was seen.
    public let pair: LearnedPair?
    /// An automatic correction the user replaced by retyping (evidence that correction was wrong).
    public let rejectedFix: String?
    public let context: [String]
    public let sentenceStart: Bool
    /// False when the word begins where tracking began after a reset (click, app switch, shortcut):
    /// the caret may have been inside a longer word, so this may be only its tail.
    public var startKnown: Bool = true
    /// The characters between the last space (or line start) and the word: "github.com/user/" for
    /// "github.com/user/repo". Non-empty prefixes like "/", "@" or "_" mark links, paths and code.
    public var prefix: String = ""
    /// The line's text before the word (context for the language model) and whether that line's
    /// start was seen (false after a click or app switch).
    public var lineBefore: String = ""
    public var lineStartKnown: Bool = false
}

/// Rebuilds the text of the current line from keystrokes and notices when a word is
/// deleted and retyped differently (a manual typo fix). Pure logic, no system calls.
public final class TextTracker {
    public private(set) var line: [Character] = []
    private struct Slot { var text: String; var start: Int; var original: String? }
    private var committed: [Slot] = []
    private var eroded: Slot?
    private var erodedCurrent: String?
    private let maxLine = 600
    /// Whether the character before `line[0]` is known to be a non-word character (true right after
    /// Return; false after a reset, when the caret could be anywhere, including mid-word).
    public private(set) var lineStartKnown = false

    public init() {}

    /// Letters, digits and apostrophes form tokens; tokens containing digits are never corrected.
    public static func isWordChar(_ c: Character) -> Bool {
        c.isLetter || c.isNumber || c == "'" || c == "\u{2019}"
    }

    public var text: String { String(line) }

    private func trailingWord() -> (String, Int) {
        var i = line.count
        while i > 0 && Self.isWordChar(line[i - 1]) { i -= 1 }
        return (String(line[i...]), i)
    }

    private func isSentenceStart(before start: Int) -> Bool {
        var i = start - 1
        while i >= 0 && line[i] == " " { i -= 1 }
        // Nothing seen before: only a Return tells us a new sentence (or message) starts here.
        // Guessing "yes" would let capitalised names mid-sentence be "corrected".
        if i < 0 { return lineStartKnown }
        return ".!?\n\r:".contains(line[i])
    }

    @discardableResult
    public func feed(_ input: KeyInput) -> Commit? {
        switch input {
        case .reset:
            reset()
            return nil

        case .char(let c):
            if Self.isWordChar(c) {
                line.append(c)
                trim()
                return nil
            }
            let (word, start) = trailingWord()
            let sentenceStart = isSentenceStart(before: start)
            let before = String(line[..<start]), lineKnown = lineStartKnown
            line.append(c)
            var pair: LearnedPair?
            var rejected: String?
            if !word.isEmpty {
                if let e = eroded {
                    if e.start == start {
                        if e.text != word, !isTruncation(e.original ?? e.text, word) {
                            pair = LearnedPair(typo: e.original ?? e.text, fix: word)
                            if e.original != nil { rejected = e.text }
                        }
                        eroded = nil
                    } else if start > e.start {
                        eroded = nil
                    }
                } else if let ec = erodedCurrent, ec != word, !isTruncation(ec, word) {
                    pair = LearnedPair(typo: ec, fix: word)
                }
                committed.append(Slot(text: word, start: start, original: nil))
            }
            erodedCurrent = nil
            let ctx = committed.dropLast().suffix(3).map { $0.text }
            let startKnown = start > 0 || lineStartKnown
            var p = start
            while p > 0 && !line[p - 1].isWhitespace { p -= 1 }
            let prefix = String(line[p..<start])
            if c == "\n" {
                // Return sends the message in chat apps; never let later backspaces walk into it.
                reset()
                lineStartKnown = true
            } else {
                trim()
            }
            guard !word.isEmpty else { return nil }
            return Commit(word: word, boundary: c, start: start, pair: pair, rejectedFix: rejected, context: Array(ctx),
                          sentenceStart: sentenceStart, startKnown: startKnown, prefix: prefix,
                          lineBefore: before, lineStartKnown: lineKnown)

        case .backspace:
            guard let last = line.last else {
                lineStartKnown = false  // deleting text from before what we saw
                return nil
            }
            if let slot = committed.last, slot.start + slot.text.count == line.count {
                if eroded == nil { eroded = slot }
                committed.removeLast()
            } else if !Self.isWordChar(last) {
                erodedCurrent = nil            // leaving the current word: its snapshot is stale
            } else if eroded == nil && erodedCurrent == nil {
                let (cur, _) = trailingWord()
                if cur.count >= 3 { erodedCurrent = cur }
            }
            line.removeLast()
            if let e = eroded, line.count < e.start - 12 { eroded = nil }
            return nil
        }
    }

    /// One word is a prefix of the other: the user shortened or extended it, not a typo fix.
    private func isTruncation(_ a: String, _ b: String) -> Bool {
        a.lowercased().hasPrefix(b.lowercased()) || b.lowercased().hasPrefix(a.lowercased())
    }

    /// After injecting `fix` in place of the trailing word (the boundary char is already in `line`).
    /// `original` is the word the user actually typed (nil when restoring it on undo).
    /// `replacing`: the text currently before the boundary when it is not a single word (undoing a
    /// split fix "in the" -> "inthe"); by default the trailing word is replaced.
    public func replaceLastWord(with fix: String, original: String?, replacing old: String? = nil) {
        guard let boundary = line.popLast() else { return }
        var start = trailingWord().1
        if let old, !old.isEmpty, endsWith(old) { start = line.count - old.count }
        line.removeSubrange(start...)
        line.append(contentsOf: fix)
        line.append(boundary)
        if let slot = committed.last, slot.start == start {
            committed[committed.count - 1] = Slot(text: fix, start: start, original: original)
        }
    }

    public func endsWith(_ suffix: String) -> Bool {
        let s = Array(suffix)
        return s.count <= line.count && Array(line.suffix(s.count)) == s
    }

    /// Replaces `old`, whose last character sits `suffixLength` characters before the end of the
    /// line, with `fix` (a fix applied after the word's punctuation and boundary were typed).
    /// Returns false, changing nothing, when the line no longer reads that way.
    @discardableResult
    public func replaceWord(_ old: String, endingAt suffixLength: Int, with fix: String, original: String?) -> Bool {
        let o = Array(old)
        let end = line.count - suffixLength
        let start = end - o.count
        guard suffixLength >= 0, !o.isEmpty, start >= 0, Array(line[start..<end]) == o else { return false }
        line.replaceSubrange(start..<end, with: Array(fix))
        if let i = committed.lastIndex(where: { $0.start == start }) {
            committed[i] = Slot(text: fix, start: start, original: original)
        }
        trim()
        return true
    }

    public func reset() {
        line.removeAll(keepingCapacity: true)
        committed.removeAll(keepingCapacity: true)
        eroded = nil
        erodedCurrent = nil
        lineStartKnown = false
    }

    private func trim() {
        guard line.count > maxLine else { return }
        let drop = maxLine / 2
        lineStartKnown = !Self.isWordChar(line[drop - 1])
        line.removeFirst(drop)
        committed = committed.compactMap { $0.start >= drop ? Slot(text: $0.text, start: $0.start - drop, original: $0.original) : nil }
        if let e = eroded { eroded = e.start >= drop ? Slot(text: e.text, start: e.start - drop, original: e.original) : nil }
    }
}

/// Validates a detected (typo, fix) pair before it is stored as a learning example.
public enum PairValidator {
    public static func isPlausible(_ pair: LearnedPair, lexicon: Lexicon, personal: PersonalModel) -> Bool {
        let t = pair.typo.lowercased(), f = pair.fix.lowercased()
        guard t != f else { return false }
        let tb = Array(t.utf8), fb = Array(f.utf8)
        guard Corrector.isAsciiWord(tb), Corrector.isAsciiWord(fb) else { return false }
        guard tb.count >= 3, fb.count >= 2 else { return false }
        guard !t.hasPrefix(f), !f.hasPrefix(t) else { return false }
        let maxD = tb.count >= 8 ? 3 : 2
        let d = EditDistance.osa(tb, fb, max: maxD)
        guard d >= 1 else { return false }
        return lexicon.frequency(f) >= 100 || personal.manualCount(typo: t, fix: f) > 0
    }
}
