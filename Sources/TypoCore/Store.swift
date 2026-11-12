import Foundation
import SQLite3

private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

public enum FixSource: String { case manual, auto, undo }

public struct FixRecord {
    public let id: Int64
    public let time: Date
    public let app: String
    public let typo: String
    public let fix: String
    public let before: String
    public let after: String
    public let source: FixSource
}

/// SQLite store. Keeps only what the learner needs: (typo -> fix) events with a few words
/// of context, undo events, and counts of non-dictionary words you type and leave alone.
public final class Store {
    public let url: URL
    private var db: OpaquePointer?
    private let lock = NSLock()

    public init(url: URL) throws {
        self.url = url
        if sqlite3_open_v2(url.path, &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX, nil) != SQLITE_OK {
            throw TypoError.sqlite(String(cString: sqlite3_errmsg(db)))
        }
        try exec("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS fixes(
            id INTEGER PRIMARY KEY, ts REAL NOT NULL, app TEXT NOT NULL DEFAULT '',
            typo TEXT NOT NULL, fix TEXT NOT NULL, before_ctx TEXT NOT NULL DEFAULT '',
            after_ctx TEXT NOT NULL DEFAULT '', source TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_fixes_typo ON fixes(typo);
        CREATE INDEX IF NOT EXISTS idx_fixes_ts ON fixes(ts);
        CREATE TABLE IF NOT EXISTS unknown_words(word TEXT PRIMARY KEY, count INTEGER NOT NULL, last_ts REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
    }

    deinit { sqlite3_close(db) }

    private func exec(_ sql: String) throws {
        var err: UnsafeMutablePointer<CChar>?
        if sqlite3_exec(db, sql, nil, nil, &err) != SQLITE_OK {
            let msg = err.map { String(cString: $0) } ?? "unknown"
            sqlite3_free(err)
            throw TypoError.sqlite(msg)
        }
    }

    private func run(_ sql: String, _ args: [Any] = [], row: ((OpaquePointer) -> Void)? = nil) throws {
        lock.lock(); defer { lock.unlock() }
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw TypoError.sqlite(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(stmt) }
        for (i, a) in args.enumerated() {
            let idx = Int32(i + 1)
            switch a {
            case let s as String: sqlite3_bind_text(stmt, idx, s, -1, SQLITE_TRANSIENT)
            case let d as Double: sqlite3_bind_double(stmt, idx, d)
            case let n as Int: sqlite3_bind_int64(stmt, idx, Int64(n))
            case let n as Int64: sqlite3_bind_int64(stmt, idx, n)
            default: sqlite3_bind_null(stmt, idx)
            }
        }
        while true {
            let rc = sqlite3_step(stmt)
            if rc == SQLITE_ROW { row?(stmt!) ; continue }
            if rc == SQLITE_DONE { break }
            throw TypoError.sqlite(String(cString: sqlite3_errmsg(db)))
        }
    }

    private func text(_ stmt: OpaquePointer, _ col: Int32) -> String {
        sqlite3_column_text(stmt, col).map { String(cString: $0) } ?? ""
    }

    // MARK: fixes

    public func recordFix(typo: String, fix: String, app: String, before: [String], after: [String], source: FixSource) throws {
        try run("INSERT INTO fixes(ts, app, typo, fix, before_ctx, after_ctx, source) VALUES(?,?,?,?,?,?,?)",
                [Date().timeIntervalSince1970, app, typo, fix, before.joined(separator: " "), after.joined(separator: " "), source.rawValue])
    }

    public struct PairCount { public let typo: String; public let fix: String; public let source: FixSource; public let count: Int }

    public func pairCounts() throws -> [PairCount] {
        var out: [PairCount] = []
        try run("SELECT typo, fix, source, COUNT(*) FROM fixes GROUP BY typo, fix, source") { s in
            if let src = FixSource(rawValue: self.text(s, 2)) {
                out.append(PairCount(typo: self.text(s, 0), fix: self.text(s, 1), source: src, count: Int(sqlite3_column_int64(s, 3))))
            }
        }
        return out
    }

    public func recent(_ limit: Int, source: FixSource? = nil) throws -> [FixRecord] {
        var out: [FixRecord] = []
        let sql = "SELECT id, ts, app, typo, fix, before_ctx, after_ctx, source FROM fixes " +
            (source != nil ? "WHERE source = ? " : "") + "ORDER BY ts DESC LIMIT ?"
        let args: [Any] = source != nil ? [source!.rawValue, limit] : [limit]
        try run(sql, args) { s in
            out.append(FixRecord(id: sqlite3_column_int64(s, 0), time: Date(timeIntervalSince1970: sqlite3_column_double(s, 1)),
                                 app: self.text(s, 2), typo: self.text(s, 3), fix: self.text(s, 4),
                                 before: self.text(s, 5), after: self.text(s, 6),
                                 source: FixSource(rawValue: self.text(s, 7)) ?? .manual))
        }
        return out
    }

    public func countFixes(source: FixSource? = nil, since: Date? = nil) -> Int {
        var n = 0
        var clauses: [String] = [], args: [Any] = []
        if let source { clauses.append("source = ?"); args.append(source.rawValue) }
        if let since { clauses.append("ts >= ?"); args.append(since.timeIntervalSince1970) }
        let sql = "SELECT COUNT(*) FROM fixes" + (clauses.isEmpty ? "" : " WHERE " + clauses.joined(separator: " AND "))
        try? run(sql, args) { s in n = Int(sqlite3_column_int64(s, 0)) }
        return n
    }

    // MARK: unknown words

    public func bumpUnknown(_ word: String) {
        try? run("INSERT INTO unknown_words(word, count, last_ts) VALUES(?, 1, ?) ON CONFLICT(word) DO UPDATE SET count = count + 1, last_ts = excluded.last_ts",
                 [word, Date().timeIntervalSince1970])
    }

    public func decrementUnknown(_ word: String) {
        try? run("UPDATE unknown_words SET count = MAX(count - 1, 0) WHERE word = ?", [word])
    }

    public func unknownCounts() throws -> [String: Int] {
        var out: [String: Int] = [:]
        try run("SELECT word, count FROM unknown_words WHERE count > 0") { s in
            out[self.text(s, 0)] = Int(sqlite3_column_int64(s, 1))
        }
        return out
    }

    // MARK: meta

    public func meta(_ key: String) -> String? {
        var v: String?
        try? run("SELECT value FROM meta WHERE key = ?", [key]) { s in v = self.text(s, 0) }
        return v
    }

    public func setMeta(_ key: String, _ value: String) {
        try? run("INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", [key, value])
    }
}
