import Foundation

/// When a finished word gets corrected. Shared by the app's event tap and `typo-cli suite`, so the
/// accuracy suite measures exactly the decisions the app makes (minus timing and Accessibility checks).
public enum AutocorrectPolicy {
    /// Keys that end a word and may trigger a fix.
    public static let triggers: Set<Character> = [" ", "\n", ".", ",", "!", "?", ";", ":"]
    /// Triggers whose fix waits for the next key ("youtu.be" must not become "youth.be").
    public static let deferredTriggers: Set<Character> = [".", ",", "!", "?", ";", ":"]

    public enum Action {
        case none
        /// Replace the word now, before the boundary key reaches the app.
        case now(Correction)
        /// Replace the word only if the next key confirms the punctuation ended a clause.
        case deferred(Correction)
    }

    public struct Decision {
        public let action: Action
        /// nil when the word was not analysed at all (not a trigger, or its start is unknown).
        public let analysis: Analysis?
        public let note: String
    }

    public static func decide(_ commit: Commit, with corrector: Corrector) -> Decision {
        decide(word: commit.word, boundary: commit.boundary, sentenceStart: commit.sentenceStart, startKnown: commit.startKnown,
               prefix: commit.prefix, line: LineContext(before: commit.lineBefore, startKnown: commit.lineStartKnown), with: corrector)
    }

    /// The decision for a finished word given what the tracker knows about it. `state` replaces
    /// `line` when a stored language-model state is used (collecting training rows).
    public static func decide(word: String, boundary: Character, sentenceStart: Bool, startKnown: Bool, prefix: String,
                              line: LineContext?, state: LMState? = nil, with corrector: Corrector) -> Decision {
        let commit = (word: word, boundary: boundary, sentenceStart: sentenceStart, startKnown: startKnown, prefix: prefix)
        guard triggers.contains(commit.boundary) else { return Decision(action: .none, analysis: nil, note: "not a trigger") }
        // A word at the very start of what we have seen since a click, app switch or shortcut may be
        // the tail of a longer word ("eople" of "people"): only fix it if no common word ends that way.
        if !commit.startKnown {
            guard corrector.config.correctFirstWordAfterReset else { return Decision(action: .none, analysis: nil, note: "start unknown") }
            // A word typed with a capital first letter starts there: it cannot be the rest of a word
            // the caret landed inside ("THe", "Thier" are not tails of "breathe", "healthier").
            let tail = Array(commit.word.lowercased().utf8)
            if commit.word.first?.isUppercase != true,
               corrector.lexicon.endsLongerWord(tail, minFrequency: corrector.config.tailWordMinFrequency) {
                return Decision(action: .none, analysis: nil, note: "start unknown, may be a word's tail")
            }
        }
        if corrector.config.skipLinksAndCode, looksLikeLinkOrCode(prefix: commit.prefix) {
            return Decision(action: .none, analysis: nil, note: "inside a link, path or identifier")
        }
        let analysis = corrector.analyze(commit.word, sentenceStart: commit.sentenceStart, positionUnknown: !commit.startKnown,
                                         line: line, state: state)
        guard let correction = analysis.decision else { return Decision(action: .none, analysis: analysis, note: analysis.note) }
        let action: Action = deferredTriggers.contains(commit.boundary) ? .deferred(correction) : .now(correction)
        return Decision(action: action, analysis: analysis, note: analysis.note)
    }

    /// Characters that, between the last space and a word, mean the word is part of a URL, path,
    /// email, hashtag, mention or code identifier ("github.com/user/", "my_", "@", "#").
    static let linkChars: Set<Character> = ["/", "\\", "@", "_", "#", "=", "~", "<", ">", "`", "|", "{", "}", "[", "$", "%", "^", "*", "+", "&", "."]

    public static func looksLikeLinkOrCode(prefix: String) -> Bool {
        prefix.contains { linkChars.contains($0) }
    }

    /// The key typed after "word<punct>" confirms a held fix only when it ends the clause.
    public static func confirmsDeferred(_ next: Character) -> Bool { next == " " || next == "\n" }
}
