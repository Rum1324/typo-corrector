import Foundation
import TypoCore

// Report printing for `typo-cli suite`; the runner itself is TypoCore/AccuracySuite.swift.

func pct(_ a: Int, _ b: Int) -> String { b == 0 ? "   -  " : String(format: "%5.1f%%", 100.0 * Double(a) / Double(b)) }

func printSuiteReport(_ stats: [String: SuiteStats]) {
    let keys = stats.keys.sorted()
    let typoKeys = keys.filter { !$0.hasPrefix("clean") }, cleanKeys = keys.filter { $0.hasPrefix("clean") }
    print("TYPO SETS                       typos  fixed  wrong missed | non-realword: fixed  wrong | correct words  false-fix")
    var tTot = SuiteStats()
    for k in typoKeys {
        let s = stats[k]!; tTot.add(s)
        let nr = s.typos - s.realWordTypos
        print(String(format: "  %-28@ %6d %@ %@ %@ |        %@ %@ | %8d  %@", k as NSString, s.typos, pct(s.fixed, s.typos), pct(s.wrong, s.typos), pct(s.missed, s.typos),
                     pct(s.fixed - s.realWordFixed, nr), pct(s.wrong - s.realWordWrong, nr), s.correctTokens, pct(s.falseFixes, s.correctTokens)))
    }
    let nr = tTot.typos - tTot.realWordTypos
    print(String(format: "  %-28@ %6d %@ %@ %@ |        %@ %@ | %8d  %@", "ALL TYPO SETS" as NSString, tTot.typos, pct(tTot.fixed, tTot.typos), pct(tTot.wrong, tTot.typos), pct(tTot.missed, tTot.typos),
                 pct(tTot.fixed - tTot.realWordFixed, nr), pct(tTot.wrong - tTot.realWordWrong, nr), tTot.correctTokens, pct(tTot.falseFixes, tTot.correctTokens)))
    print("\nCLEAN SETS                      words  false-fix")
    var cTot = SuiteStats()
    for k in cleanKeys {
        let s = stats[k]!; cTot.add(s)
        print(String(format: "  %-28@ %6d  %@  (%d)", k as NSString, s.correctTokens, pct(s.falseFixes, s.correctTokens), s.falseFixes))
    }
    print(String(format: "  %-28@ %6d  %@  (%d)", "ALL CLEAN SETS" as NSString, cTot.correctTokens, pct(cTot.falseFixes, cTot.correctTokens), cTot.falseFixes))
    var all = tTot; all.add(cTot)
    print("SUMMARY typos=\(tTot.typos) fixed=\(tTot.fixed) wrong=\(tTot.wrong) missed=\(tTot.missed) correct=\(all.correctTokens) falsefix=\(all.falseFixes) cleanfalsefix=\(cTot.falseFixes)")
}

func writeFailures(_ failures: [SuiteRunner.Failure], to url: URL) {
    var s = "category\tsplit\tline\tkind\ttyped\toutput\texpected\trealword\ttrace\tsentence\tnote\n"
    for f in failures {
        let sentence = (f.c.reset ? "<reset>" : "") + f.c.typed
        s += [f.c.category, f.c.split, String(f.c.line), f.kind.rawValue, f.typed, f.out, f.expected, f.realWord ? "realword" : "",
              f.trace, sentence, f.c.note].map { $0.replacingOccurrences(of: "\t", with: " ") }.joined(separator: "\t") + "\n"
    }
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try? s.write(to: url, atomically: true, encoding: .utf8)
}
