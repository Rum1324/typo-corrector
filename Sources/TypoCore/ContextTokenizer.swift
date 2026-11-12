import Foundation

/// Word tokenizer of the context language model. Mirrors training/tok.py line for line: the model
/// only knows text the way that file cut it up. Token ids are the lexicon ids after 16 reserved ones.
public struct ContextTokenizer {
    public static let pad: Int32 = 0, bos: Int32 = 1, cut: Int32 = 2, unk: Int32 = 3, name: Int32 = 4, num: Int32 = 5
    public static let dot: Int32 = 6, bang: Int32 = 7, question: Int32 = 8, comma: Int32 = 9, colon: Int32 = 10, sym: Int32 = 11
    public static let reserved: Int32 = 16

    public let lexicon: Lexicon
    public init(lexicon: Lexicon) { self.lexicon = lexicon }

    /// A whitespace-delimited chunk with one of these is a link, path, identifier, markup or maths.
    static let link = Set("/\\@_#=~<>`|{}[]$%^*+&".unicodeScalars)
    static let apostrophe: Unicode.Scalar = "'"

    // Python's str.isalpha / isdigit / isalnum / isupper / isspace, per code point.
    @inline(__always) static func isAlpha(_ s: Unicode.Scalar) -> Bool {
        if s.isASCII { return (s.value >= 65 && s.value <= 90) || (s.value >= 97 && s.value <= 122) }
        switch s.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter: return true
        default: return false
        }
    }
    @inline(__always) static func isDigit(_ s: Unicode.Scalar) -> Bool {
        if s.isASCII { return s.value >= 48 && s.value <= 57 }
        let t = s.properties.numericType
        return t == .decimal || t == .digit
    }
    @inline(__always) static func isAlnum(_ s: Unicode.Scalar) -> Bool {
        if s.isASCII { return isAlpha(s) || isDigit(s) }
        return isAlpha(s) || s.properties.numericType != nil
    }
    @inline(__always) static func isUpper(_ s: Unicode.Scalar) -> Bool {
        s.isASCII ? (s.value >= 65 && s.value <= 90) : s.properties.isUppercase
    }
    @inline(__always) static func isSpace(_ s: Unicode.Scalar) -> Bool {
        s.properties.isWhitespace || (s.value >= 0x1c && s.value <= 0x1f)
    }

    /// Id of a finished word (letters and apostrophes): the lexicon id, else <name> / <unk>.
    public func wordId(_ word: String, capitalised: Bool) -> Int32 {
        if let i = lexicon.id(word.lowercased()) { return i + Self.reserved }
        return capitalised ? Self.name : Self.unk
    }

    private func lookup(_ run: ArraySlice<Unicode.Scalar>) -> Int32? {
        var s = String.UnicodeScalarView()
        for u in run { s.append(u.value >= 65 && u.value <= 90 ? Unicode.Scalar(u.value + 32)! : u) }
        return lexicon.id(String(s)).map { $0 + Self.reserved }
    }

    /// Token ids of one line of text (no <s>).
    public func encode(_ text: String) -> [Int32] {
        var out: [Int32] = []
        var chunk: [Unicode.Scalar] = []
        func flush() {
            guard !chunk.isEmpty else { return }
            if chunk.allSatisfy({ $0.isASCII && Self.isAlpha($0) }) {                // the common case
                out.append(lookup(chunk[...]) ?? (Self.isUpper(chunk[0]) ? Self.name : Self.unk))
            } else {
                encodeChunk(chunk, into: &out)
            }
            chunk.removeAll(keepingCapacity: true)
        }
        for var u in text.unicodeScalars {
            if u == "\u{2019}" || u == "\u{2018}" { u = Self.apostrophe }
            if Self.isSpace(u) { flush() } else { chunk.append(u) }
        }
        flush()
        return out
    }

    private func encodeChunk(_ c: [Unicode.Scalar], into out: inout [Int32]) {
        let n = c.count
        if c.contains(where: { Self.link.contains($0) }) { out.append(Self.sym); return }
        // A dot joining letters or a letter and a digit: "file.txt", "e.g.", "youtu.be" (not "3.5").
        if n >= 3 {
            for k in 1..<(n - 1) where c[k] == "." && Self.isAlnum(c[k - 1]) && Self.isAlnum(c[k + 1])
                && (Self.isAlpha(c[k - 1]) || Self.isAlpha(c[k + 1])) {
                out.append(Self.sym); return
            }
        }
        var i = 0
        while i < n {
            let ch = c[i]
            if Self.isAlpha(ch) || ch == Self.apostrophe {
                var j = i
                while j < n && (Self.isAlnum(c[j]) || c[j] == Self.apostrophe) { j += 1 }
                var a = i, b = j
                while a < b && c[a] == Self.apostrophe { a += 1 }
                while b > a && c[b - 1] == Self.apostrophe { b -= 1 }
                i = j
                if a == b { continue }
                let run = c[a..<b]
                if !run.allSatisfy({ $0 == Self.apostrophe || Self.isAlpha($0) }) {  // "h264enc", "v2"
                    out.append(Self.sym)
                } else if run.allSatisfy({ $0.isASCII }) {
                    var w = lookup(run)
                    if w == nil, run.count >= 2, run[b - 2] == Self.apostrophe, run[b - 1] == "s" || run[b - 1] == "S" {
                        w = lookup(c[a..<(b - 2)])                                   // "model's" counts as "model"
                    }
                    out.append(w ?? (Self.isUpper(run[a]) ? Self.name : Self.unk))
                } else {                                                             // accented or non-Latin word
                    out.append(Self.isUpper(run[a]) ? Self.name : Self.unk)
                }
            } else if Self.isDigit(ch) {
                var j = i
                while j < n && (Self.isAlnum(c[j]) || ((c[j] == "." || c[j] == "," || c[j] == ":") && j + 1 < n && Self.isDigit(c[j + 1]))) { j += 1 }
                out.append(Self.num)                                                 // "3.5", "10,000", "12:30", "3rd"
                i = j
            } else if ch == "." || ch == "!" || ch == "?" {
                out.append(ch == "." ? Self.dot : (ch == "!" ? Self.bang : Self.question))
                while i < n && (c[i] == "." || c[i] == "!" || c[i] == "?") { i += 1 }   // "...", "?!" count once
            } else if ch == "," {
                out.append(Self.comma); i += 1
            } else if ch == ";" || ch == ":" {
                out.append(Self.colon); i += 1
            } else {                                                                 // quotes, brackets, dashes, emoji
                i += 1
            }
        }
    }

    /// What the model reads before a word: <s> when the line's start was seen and everything fits,
    /// <cut> otherwise, then the last `maxLen - 1` tokens of the text before the word.
    public func context(before: String, lineStartKnown: Bool, maxLen: Int) -> [Int32] {
        let full = encode(before)
        let truncated = full.count > maxLen - 1
        return [(lineStartKnown && !truncated) ? Self.bos : Self.cut] + full.suffix(maxLen - 1)
    }
}
