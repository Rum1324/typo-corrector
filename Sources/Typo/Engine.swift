import AppKit
import ApplicationServices
import Carbon
import TypoCore

/// Glue between the event tap, the tracker, the corrector and the injector.
final class Engine {
    let store: Store
    let corrector: Corrector
    let personal: PersonalModel
    let tracker = TextTracker()
    let injector = Injector()
    let inputSource = InputSourceMonitor()
    let settings = Settings.shared
    let focus = FocusMonitor()
    /// Database writes happen here so the event tap never waits on SQLite.
    private let storeQueue = DispatchQueue(label: "typo.store", qos: .utility)

    private(set) var frontBundleID: String? = NSWorkspace.shared.frontmostApplication?.bundleIdentifier
    private(set) var frontAppName: String? = NSWorkspace.shared.frontmostApplication?.localizedName
    private(set) var frontPID: pid_t = NSWorkspace.shared.frontmostApplication?.processIdentifier ?? 0

    /// Callback latency bookkeeping. A filtering tap that answers late has already lost: the
    /// WindowServer delivers the key anyway, so a correction injected after that corrupts the text.
    private(set) var slowCallbacks = 0
    private(set) var tapDisables = 0
    private(set) var worstCallbackMs: Double = 0
    /// Do not inject if the callback has already taken this long (ms); the key may be through.
    static let maxSwallowMillis: Double = 100
    static let slowCallbackMillis: Double = 20

    struct LastCorrection {
        let typo: String
        let fix: String
        /// Punctuation typed between the word and the boundary ("." in "teh. "), usually empty.
        let suffix: String
        let boundary: CGEvent
        let time: Date
    }
    private var lastCorrection: LastCorrection?
    /// A backspace-undo that has not been counted as a rejection yet. It counts once the next key
    /// types a character (the restored word was meant); another Delete, a click or a shortcut means
    /// the user was rewriting, and nothing is learned.
    private var unconfirmedUndo: (typo: String, fix: String)?
    /// Set when undo swallowed a Delete press: its release is swallowed too, so the app never sees
    /// a key-up without a key-down.
    private var swallowKeyUp: Int64?

    /// A fix decided at "word<punct>" but held back until the next key shows the punctuation ended a
    /// sentence or clause (space/return) rather than being part of a URL, file name or number.
    struct PendingFix {
        let word: String
        let correction: Correction
        let punct: Character
        let context: [String]
        let time: Date
        /// What the app shows before the caret, read once the punctuation has gone through: the
        /// system's own autocorrect may already have fixed the word, and backspacing blindly would
        /// then eat the wrong characters.
        let check: AppTextCheck
    }
    private var pendingFix: PendingFix?

    /// Result of an asynchronous Accessibility read, filled on the focus queue.
    final class AppTextCheck {
        enum Result { case pending, done(FocusMonitor.CaretText) }
        private let lock = NSLock()
        private var value: Result = .pending
        var result: Result { lock.lock(); defer { lock.unlock() }; return value }
        func finish(_ caret: FocusMonitor.CaretText) { lock.lock(); value = .done(caret); lock.unlock() }
    }

    /// Set after a held letter repeated: the app may be showing the accent menu, where Space or
    /// Return picks an accent instead of typing a boundary.
    private var accentMenuMayBeOpen = false

    private(set) var autoToday = 0
    private(set) var undoneToday = 0
    private(set) var learnedToday = 0
    private(set) var recent: [(String, String)] = []
    var onChange: (() -> Void)?

    // Which keys trigger a fix, and which fixes wait for the next key: see AutocorrectPolicy.
    static let pendingFixLifetime: TimeInterval = 10
    /// Backspace within this window after a fix restores the original word.
    static let undoWindow: TimeInterval = 15

    init(store: Store, corrector: Corrector) {
        self.store = store
        self.corrector = corrector
        self.personal = corrector.personal
        let start = Calendar.current.startOfDay(for: Date())
        autoToday = store.countFixes(source: .auto, since: start)
        undoneToday = store.countFixes(source: .undo, since: start)
        learnedToday = store.countFixes(source: .manual, since: start)
        recent = ((try? store.recent(8, source: .auto)) ?? []).map { ($0.typo, $0.fix) }
        inputSource.onChange = { [weak self] in
            guard let self else { return }
            self.tracker.reset()
            self.lastCorrection = nil
            self.unconfirmedUndo = nil
            self.pendingFix = nil
            Log.write("input source → \(self.inputSource.sourceID) latin=\(self.inputSource.isLatin)")
            self.onChange?()
        }
        NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification, object: nil, queue: .main) { [weak self] note in
                guard let self else { return }
                let app = note.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication
                // The tap may already have switched to this app from the first key typed into it;
                // resetting again now would wipe the letters typed since.
                if let app, app.processIdentifier == self.frontPID { return }
                self.noteFrontApp(app)
            }
        focus.track(pid: frontPID)
    }

    private func noteFrontApp(_ app: NSRunningApplication?) {
        frontBundleID = app?.bundleIdentifier
        frontAppName = app?.localizedName
        frontPID = app?.processIdentifier ?? 0
        tracker.reset()
        lastCorrection = nil
        unconfirmedUndo = nil
        pendingFix = nil
        focus.track(pid: frontPID)
        settings.reloadBlocklistIfStale()
        Log.write("front app → \(frontBundleID ?? "?") status=\(status)")
        notifyChange()
    }

    private func notifyChange() {
        DispatchQueue.main.async { [weak self] in self?.onChange?() }
    }

    func tapWasDisabled(_ type: CGEventType) {
        tapDisables += 1
        Log.write("event tap was disabled by the system (\(type == .tapDisabledByTimeout ? "timeout" : "user input")) and re-enabled; the pending key went through unmodified")
        tracker.reset()
        lastCorrection = nil
        unconfirmedUndo = nil
        pendingFix = nil
        injector.reset()
    }

    private func noteLatency(since start: DispatchTime, stage: String) {
        let ms = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000
        if ms > worstCallbackMs { worstCallbackMs = ms }
        if ms > Self.slowCallbackMillis {
            slowCallbacks += 1
            Log.write(String(format: "slow tap callback: %.1f ms at %@", ms, stage))
        }
    }

    // MARK: status

    enum Status { case active, disabled, blockedApp, composing, noPermission }

    var status: Status {
        if !settings.enabled { return .disabled }
        if !Permissions.accessibilityCached { return .noPermission }
        if !inputSource.isLatin { return .composing }
        if settings.isBlocked(bundleID: frontBundleID) { return .blockedApp }
        return .active
    }

    // MARK: event handling

    func handle(_ event: CGEvent, type: CGEventType) -> Unmanaged<CGEvent>? {
        if injector.isInjected(event) {
            injector.sawInjected()
            if settings.debugInjection {
                Log.write("passthrough injected key=\(event.getIntegerValueField(.keyboardEventKeycode)) type=\(type.rawValue)")
            }
            return Unmanaged.passUnretained(event)
        }
        let result = process(event, type: type)
        // A real key that passes while our own events are still queued would overtake them: it would
        // land inside the correction and absorb one of its backspaces. Send it again behind them.
        if result != nil, type == .keyDown || type == .keyUp, injector.hasEventsInFlight {
            injector.holdBehindInFlight(event)
            return nil
        }
        return result
    }

    private func process(_ event: CGEvent, type: CGEventType) -> Unmanaged<CGEvent>? {
        let started = DispatchTime.now()
        var stage = "entry"
        defer { noteLatency(since: started, stage: stage) }
        let pass = Unmanaged.passUnretained(event)
        let keyCode = event.getIntegerValueField(.keyboardEventKeycode)
        if type == .keyUp {
            if let k = swallowKeyUp, k == keyCode {
                swallowKeyUp = nil
                return nil
            }
            return pass
        }
        guard type == .keyDown else {
            tracker.reset(); lastCorrection = nil; unconfirmedUndo = nil; pendingFix = nil
            return pass
        }
        let undone = unconfirmedUndo      // counted below only if this key types a character
        unconfirmedUndo = nil

        // The WindowServer stamps each event with the pid it is headed for. Trust that over the
        // activation notification, which can arrive late or not at all (Spotlight-style panels).
        // Checked before the status guard so Typo also notices leaving a blocklisted panel.
        let targetPID = pid_t(event.getIntegerValueField(.eventTargetUnixProcessID))
        if targetPID != 0, targetPID != frontPID {
            stage = "front app change"
            noteFrontApp(NSRunningApplication(processIdentifier: targetPID))
        }
        guard status == .active else {
            // Whatever is typed now is invisible to us: forget the line and any pending undo.
            tracker.reset(); lastCorrection = nil; pendingFix = nil
            return pass
        }
        focus.refreshIfStale()
        let pending = pendingFix
        pendingFix = nil
        if accentMenuMayBeOpen {
            if (123...126).contains(keyCode) { tracker.reset(); return pass }  // moving within the menu
            accentMenuMayBeOpen = false
            if keyCode == 49 || keyCode == 36 || keyCode == 76 { tracker.reset(); return pass }  // picks an accent
        }

        let flags = event.flags
        if flags.contains(.maskCommand) || flags.contains(.maskControl) || (flags.contains(.maskAlternate) && keyCode == 51) {
            // shortcuts, and Option+Delete (word delete) which the tracker cannot model
            tracker.reset(); lastCorrection = nil
            return pass
        }
        switch keyCode {
        case 51: // delete
            if settings.backspaceUndo, let lc = lastCorrection, Date().timeIntervalSince(lc.time) < Self.undoWindow {
                stage = "undo"
                undo(lc)
                swallowKeyUp = 51
                return nil
            }
            lastCorrection = nil
            swallowKeyUp = nil
            tracker.feed(.backspace)
            return pass
        case 123, 124, 125, 126, 115, 119, 116, 121, 117, 53, 48: // arrows, home/end/page, fwd delete, esc, tab
            tracker.reset(); lastCorrection = nil
            return pass
        default:
            break
        }
        lastCorrection = nil

        guard let s = unicodeString(of: event) else { return pass }
        guard s.count == 1, let c = s.first else {
            tracker.reset()
            return pass
        }
        let ch: Character = (keyCode == 36 || keyCode == 76 || c == "\r") ? "\n" : c
        if event.getIntegerValueField(.keyboardEventAutorepeat) != 0, TextTracker.isWordChar(ch) {
            // Press-and-hold apps drop letter repeats (accent menu), others insert them: we cannot
            // know which, so forget the word instead of guessing its length.
            tracker.reset()
            accentMenuMayBeOpen = true
            return pass
        }
        if let u = undone {
            // Typing went on after the undo: the restored word was meant, so the fix was wrong.
            recordRejection(typo: u.typo, fix: u.fix)
        }
        if let p = pending, AutocorrectPolicy.confirmsDeferred(ch) {
            stage = "deferred fix"
            if applyDeferred(p, boundaryEvent: event, ch: ch, started: started) { return nil }
        }
        stage = "tracker"
        guard let commit = tracker.feed(.char(ch)) else { return pass }

        // Safety rails first: nothing typed into a password field is learned, counted or touched.
        // The AX answer is cached by FocusMonitor; an unknown or stale answer means hands off.
        stage = "secure check"
        if IsSecureEventInputEnabled() || focus.isSecure {
            tracker.reset()
            return pass
        }
        guard focus.isKnownNotSecure else { return pass }

        stage = "learn"
        learn(from: commit)
        if commit.pair == nil, let rejected = commit.rejectedFix {
            // The user deleted our fix and typed their own word back: it is meant. Fixing it again
            // would trap them in a loop, so leave it and count the rejection.
            recordRejection(typo: commit.word, fix: rejected)
            noteUnknown(commit.word)
            Log.write("kept \(commit.word): typed back over the fix \(rejected)")
            return pass
        }

        stage = "analyze"
        let decision = AutocorrectPolicy.decide(commit, with: corrector)
        let correction: Correction
        switch decision.action {
        case .none:
            if let a = decision.analysis, a.note != "user accepted word" { noteUnknown(commit.word) }
            return pass
        case .now(let c):
            correction = c
        case .deferred(let c):
            correction = c
            let check = AppTextCheck()
            let expected = commit.word + String(ch)
            focus.readTextBeforeCaret(length: expected.utf16.count, after: 0.02) { check.finish($0) }
            pendingFix = PendingFix(word: commit.word, correction: correction, punct: ch, context: commit.context,
                                    time: Date(), check: check)
            return pass
        }
        guard !isLate(started, word: commit.word, fix: correction.replacement) else { return pass }

        stage = "inject"
        let boundary = event.copy() ?? event
        injector.replaceTrailing(deleteCount: commit.word.count, with: correction.replacement, thenRepost: boundary)
        tracker.replaceLastWord(with: correction.replacement, original: commit.word)
        // Return may have sent a message or submitted a form: there is nothing safe to undo into.
        lastCorrection = ch == "\n" ? nil : LastCorrection(typo: commit.word, fix: correction.replacement, suffix: "", boundary: boundary, time: Date())
        recordAuto(word: commit.word, correction: correction, context: commit.context)
        return nil
    }

    /// The user typed space or return right after "word<punct>", so the punctuation ended a sentence
    /// or clause and the fix held back at <punct> is safe now. Returns false (nothing done) when the
    /// line no longer matches; the key is then handled normally.
    private func applyDeferred(_ p: PendingFix, boundaryEvent event: CGEvent, ch: Character, started: DispatchTime) -> Bool {
        let suffix = String(p.punct)
        guard Date().timeIntervalSince(p.time) < Self.pendingFixLifetime, tracker.endsWith(p.word + suffix) else { return false }
        guard !IsSecureEventInputEnabled(), !focus.isSecure, focus.isKnownNotSecure else { return false }
        // When the app's text may differ from the tracker's, drop the fix and forget the line: the
        // tracker's offsets would be wrong for the next fix too.
        switch p.check.result {
        case .pending:
            Log.write("deferred fix \(p.word) dropped: app text not read yet")
            tracker.reset()
            return false
        case .done(.failed):
            Log.write("deferred fix \(p.word) dropped: could not read the app's text near the caret")
            tracker.reset()
            return false
        case .done(.text(let shown)) where shown != p.word + suffix:
            Log.write("deferred fix \(p.word) dropped: the app changed the word (its own autocorrect?)")
            tracker.reset()
            return false
        case .done(.text), .done(.unsupported):
            break  // matches, or the app never exposes its text (no system autocorrect there either)
        }
        let fix = p.correction.replacement
        guard !isLate(started, word: p.word, fix: fix) else { return false }
        tracker.feed(.char(ch))  // commits nothing (the line ends in punctuation); Return clears the line
        if ch != "\n" { tracker.replaceWord(p.word, endingAt: 2, with: fix, original: p.word) }
        let boundary = event.copy() ?? event
        injector.replaceTrailing(deleteCount: p.word.count + suffix.count, with: fix + suffix, thenRepost: boundary)
        lastCorrection = ch == "\n" ? nil : LastCorrection(typo: p.word, fix: fix, suffix: suffix, boundary: boundary, time: Date())
        recordAuto(word: p.word, correction: p.correction, context: p.context)
        return true
    }

    /// If we are already late, the WindowServer may have given up waiting and delivered the key;
    /// injecting now would leave a stray letter. Skipping one fix is the safer failure.
    private func isLate(_ started: DispatchTime, word: String, fix: String) -> Bool {
        let elapsedMs = Double(DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds) / 1_000_000
        guard elapsedMs > Self.maxSwallowMillis else { return false }
        Log.write(String(format: "skipped fix %@ → %@: callback already %.0f ms", word, fix, elapsedMs))
        return true
    }

    private func recordAuto(word: String, correction: Correction, context: [String]) {
        let app = frontBundleID ?? ""
        let fix = correction.replacement
        personal.addApplied(typo: word.lowercased(), fix: fix.lowercased())   // kept, unless an undo follows
        storeQueue.async { [store] in
            try? store.recordFix(typo: word, fix: fix, app: app, before: context, after: [], source: .auto)
        }
        autoToday += 1
        Log.write("fix \(word) → \(fix) p=\(String(format: "%.2f", correction.confidence)) \(correction.reason) in \(frontBundleID ?? "?")")
        recent.insert((word, fix), at: 0)
        if recent.count > 8 { recent.removeLast() }
        notifyChange()
    }

    private func undo(_ lc: LastCorrection) {
        // One text event including the boundary: re-sending the space as a key would end the word
        // again and let the app's own autocorrect undo the undo.
        let boundaryText = unicodeString(of: lc.boundary) ?? " "
        injector.replaceTrailing(deleteCount: lc.fix.count + lc.suffix.count + 1, with: lc.typo + lc.suffix + boundaryText)
        if lc.suffix.isEmpty {
            tracker.replaceLastWord(with: lc.typo, original: nil, replacing: lc.fix)   // fix may be "in the"
        } else if !tracker.replaceWord(lc.fix, endingAt: lc.suffix.count + 1, with: lc.typo, original: nil) {
            tracker.reset()
        }
        lastCorrection = nil
        unconfirmedUndo = (lc.typo, lc.fix)
        Log.write("undo \(lc.fix) → \(lc.typo)")
    }

    private func recordRejection(typo: String, fix: String) {
        let t = typo.lowercased(), f = fix.lowercased()
        personal.addUndo(typo: t, fix: f)
        let app = frontBundleID ?? ""
        storeQueue.async { [store] in
            try? store.recordFix(typo: t, fix: f, app: app, before: [], after: [], source: .undo)
        }
        undoneToday += 1
        notifyChange()
    }

    private func learn(from commit: Commit) {
        guard let pair = commit.pair else { return }
        guard PairValidator.isPlausible(pair, lexicon: corrector.lexicon, personal: personal) else { return }
        let t = pair.typo.lowercased(), f = pair.fix.lowercased()
        if let rejected = commit.rejectedFix, rejected.lowercased() != f {
            // The user replaced our automatic fix with something else: that fix was wrong for this typo.
            recordRejection(typo: t, fix: rejected)
            Log.write("rejected auto-fix \(t) → \(rejected) (user wanted \(f))")
        }
        personal.addManual(typo: t, fix: f)
        personal.forgetUnknown(t)
        let app = frontBundleID ?? "", context = commit.context
        storeQueue.async { [store] in
            store.decrementUnknown(t)
            try? store.recordFix(typo: t, fix: f, app: app, before: context, after: [], source: .manual)
        }
        learnedToday += 1
        Log.write("learned \(t) → \(f) (\(personal.manualCount(typo: t, fix: f))x)")
        notifyChange()
    }

    private func noteUnknown(_ word: String) {
        let lower = word.lowercased()
        let bytes = Array(lower.utf8)
        guard bytes.count >= 3, Corrector.isAsciiWord(bytes), corrector.lexicon.frequency(lower) == 0 else { return }
        personal.sawUnknown(lower)
        storeQueue.async { [store] in store.bumpUnknown(lower) }
    }

    // MARK: helpers

    private func unicodeString(of event: CGEvent) -> String? {
        var length = 0
        var buffer = [UniChar](repeating: 0, count: 8)
        event.keyboardGetUnicodeString(maxStringLength: 8, actualStringLength: &length, unicodeString: &buffer)
        guard length > 0 else { return nil }
        return String(utf16CodeUnits: buffer, count: length)
    }
}
