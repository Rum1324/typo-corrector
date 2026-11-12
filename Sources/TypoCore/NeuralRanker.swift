import Foundation
import Accelerate

/// Tiny character-level transformer that scores (typo, candidate) pairs.
/// Weights are trained in training/train.py and exported as ranker.json + ranker.bin,
/// so inference here has no Core ML / coremltools dependency.
public final class NeuralRanker {
    public struct Config: Decodable {
        public let version: Int
        public let d: Int
        public let heads: Int
        public let layers: Int
        public let ff: Int
        public let maxLen: Int
        public let maxWord: Int
        public let vocab: String
    }
    struct TensorMeta: Decodable { let name: String; let shape: [Int]; let offset: Int }
    struct Manifest: Decodable { let config: Config; let tensors: [TensorMeta] }

    struct Layer {
        var ln1g, ln1b: [Float]
        var wq, bq, wk, bk, wv, bv, wo, bo: [Float]   // weights pre-transposed to [in, out]
        var ln2g, ln2b: [Float]
        var w1, b1, w2, b2: [Float]
    }

    public let config: Config
    public let source: URL
    private var charToId: [UInt8: Int32] = [:]
    private let tokEmb, posEmb, segEmb: [Float]
    private let layers: [Layer]
    private let lnfG, lnfB, headW: [Float]
    private let headB: Float

    static let pad: Int32 = 0, cls: Int32 = 1, sep: Int32 = 2, unk: Int32 = 3

    public init(manifest manifestURL: URL, weights weightsURL: URL) throws {
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
        let data = try Data(contentsOf: weightsURL)
        let all: [Float] = data.withUnsafeBytes { raw in
            let n = raw.count / MemoryLayout<Float>.size
            var out = [Float](repeating: 0, count: n)
            _ = out.withUnsafeMutableBytes { raw.copyBytes(to: $0, count: n * MemoryLayout<Float>.size) }
            return out
        }
        config = manifest.config
        source = manifestURL
        var metas: [String: TensorMeta] = [:]
        for t in manifest.tensors { metas[t.name] = t }

        func tensor(_ name: String) throws -> [Float] {
            guard let m = metas[name] else { throw TypoError.badResource("ranker tensor \(name) missing") }
            let n = m.shape.reduce(1, *)
            guard m.offset + n <= all.count else { throw TypoError.badResource("ranker tensor \(name) out of range") }
            return Array(all[m.offset..<(m.offset + n)])
        }
        func linearT(_ name: String, out: Int, inp: Int) throws -> [Float] {
            // PyTorch Linear weight is [out, in]; store as [in, out] for x·W.
            let w = try tensor(name)
            var t = [Float](repeating: 0, count: w.count)
            for o in 0..<out { for i in 0..<inp { t[i * out + o] = w[o * inp + i] } }
            return t
        }

        let d = config.d, ff = config.ff
        tokEmb = try tensor("tok_emb.weight")
        posEmb = try tensor("pos_emb.weight")
        segEmb = try tensor("seg_emb.weight")
        var ls: [Layer] = []
        for i in 0..<config.layers {
            let p = "layers.\(i)."
            ls.append(Layer(
                ln1g: try tensor(p + "ln1.weight"), ln1b: try tensor(p + "ln1.bias"),
                wq: try linearT(p + "q.weight", out: d, inp: d), bq: try tensor(p + "q.bias"),
                wk: try linearT(p + "k.weight", out: d, inp: d), bk: try tensor(p + "k.bias"),
                wv: try linearT(p + "v.weight", out: d, inp: d), bv: try tensor(p + "v.bias"),
                wo: try linearT(p + "o.weight", out: d, inp: d), bo: try tensor(p + "o.bias"),
                ln2g: try tensor(p + "ln2.weight"), ln2b: try tensor(p + "ln2.bias"),
                w1: try linearT(p + "ff1.weight", out: ff, inp: d), b1: try tensor(p + "ff1.bias"),
                w2: try linearT(p + "ff2.weight", out: d, inp: ff), b2: try tensor(p + "ff2.bias")))
        }
        layers = ls
        lnfG = try tensor("lnf.weight"); lnfB = try tensor("lnf.bias")
        headW = try tensor("head.weight"); headB = try tensor("head.bias")[0]
        for (i, c) in config.vocab.utf8.enumerated() { charToId[c] = Int32(4 + i) }
    }

    /// Personal (fine-tuned) weights win over the bundled ones, unless they were tuned from an older
    /// generation of the base ranker: the rest of the engine is fitted to the bundled generation.
    public static func load() -> NeuralRanker? {
        var bundled: NeuralRanker?
        if let m = Paths.resource("ranker.json"), let w = Paths.resource("ranker.bin") { bundled = try? NeuralRanker(manifest: m, weights: w) }
        guard let files = Paths.rankerFiles(), files.manifest != bundled?.source,
              let personal = try? NeuralRanker(manifest: files.manifest, weights: files.weights) else { return bundled }
        return bundled == nil || personal.config.version == bundled!.config.version ? personal : bundled
    }

    // MARK: encoding

    func encode(typo: [UInt8], candidate: [UInt8]) -> (tokens: [Int32], segs: [Int32]) {
        let L = config.maxLen, W = config.maxWord
        var tokens = [Int32](repeating: Self.pad, count: L)
        var segs = [Int32](repeating: 0, count: L)
        var p = 0
        tokens[p] = Self.cls; p += 1
        for c in typo.prefix(W) { tokens[p] = charToId[c] ?? Self.unk; p += 1 }
        tokens[p] = Self.sep; segs[p] = 1; p += 1
        for c in candidate.prefix(W) { tokens[p] = charToId[c] ?? Self.unk; segs[p] = 1; p += 1 }
        return (tokens, segs)
    }

    // MARK: math helpers

    @inline(__always) private func matmul(_ a: [Float], _ m: Int, _ k: Int, _ b: [Float], _ n: Int) -> [Float] {
        var c = [Float](repeating: 0, count: m * n)
        a.withUnsafeBufferPointer { pa in
            b.withUnsafeBufferPointer { pb in
                c.withUnsafeMutableBufferPointer { pc in
                    vDSP_mmul(pa.baseAddress!, 1, pb.baseAddress!, 1, pc.baseAddress!, 1,
                              vDSP_Length(m), vDSP_Length(n), vDSP_Length(k))
                }
            }
        }
        return c
    }

    @inline(__always) private func addBias(_ x: inout [Float], _ b: [Float], rows: Int) {
        let n = b.count
        for r in 0..<rows { for j in 0..<n { x[r * n + j] += b[j] } }
    }

    private func layerNorm(_ x: [Float], _ g: [Float], _ b: [Float], rows: Int, d: Int) -> [Float] {
        var out = [Float](repeating: 0, count: x.count)
        for r in 0..<rows {
            let base = r * d
            var mean: Float = 0
            for j in 0..<d { mean += x[base + j] }
            mean /= Float(d)
            var vs: Float = 0
            for j in 0..<d { let t = x[base + j] - mean; vs += t * t }
            let inv = 1 / sqrt(vs / Float(d) + 1e-5)
            for j in 0..<d { out[base + j] = (x[base + j] - mean) * inv * g[j] + b[j] }
        }
        return out
    }

    /// Logit for one (typo, candidate) pair.
    public func score(typo: [UInt8], candidate: [UInt8]) -> Float {
        let (tokens, segs) = encode(typo: typo, candidate: candidate)
        // Padding sits after the real tokens and is masked out of attention, so the rows before
        // it come out the same when the padding is not computed at all.
        let L = tokens.firstIndex(of: Self.pad) ?? config.maxLen
        let d = config.d, H = config.heads, dh = d / H
        var x = [Float](repeating: 0, count: L * d)
        for t in 0..<L {
            let tok = Int(tokens[t]), seg = Int(segs[t])
            for j in 0..<d { x[t * d + j] = tokEmb[tok * d + j] + posEmb[t * d + j] + segEmb[seg * d + j] }
        }
        let valid = tokens.prefix(L).map { $0 != Self.pad }
        let scale = 1 / sqrt(Float(dh))
        var scores = [Float](repeating: 0, count: L)
        var ctx = [Float](repeating: 0, count: L * d)

        for layer in layers {
            let h = layerNorm(x, layer.ln1g, layer.ln1b, rows: L, d: d)
            var q = matmul(h, L, d, layer.wq, d); addBias(&q, layer.bq, rows: L)
            var k = matmul(h, L, d, layer.wk, d); addBias(&k, layer.bk, rows: L)
            var v = matmul(h, L, d, layer.wv, d); addBias(&v, layer.bv, rows: L)
            for hd in 0..<H {
                let off = hd * dh
                for i in 0..<L {
                    var mx: Float = -Float.greatestFiniteMagnitude
                    for j in 0..<L {
                        if !valid[j] { scores[j] = -Float.greatestFiniteMagnitude; continue }
                        var s: Float = 0
                        for e in 0..<dh { s += q[i * d + off + e] * k[j * d + off + e] }
                        s *= scale
                        scores[j] = s
                        if s > mx { mx = s }
                    }
                    var sum: Float = 0
                    for j in 0..<L where valid[j] { scores[j] = expf(scores[j] - mx); sum += scores[j] }
                    for e in 0..<dh {
                        var acc: Float = 0
                        for j in 0..<L where valid[j] { acc += scores[j] * v[j * d + off + e] }
                        ctx[i * d + off + e] = acc / sum
                    }
                }
            }
            var o = matmul(ctx, L, d, layer.wo, d); addBias(&o, layer.bo, rows: L)
            for i in 0..<x.count { x[i] += o[i] }
            let h2 = layerNorm(x, layer.ln2g, layer.ln2b, rows: L, d: d)
            var f1 = matmul(h2, L, d, layer.w1, config.ff); addBias(&f1, layer.b1, rows: L)
            VectorMath.gelu(&f1)
            var f2 = matmul(f1, L, config.ff, layer.w2, d); addBias(&f2, layer.b2, rows: L)
            for i in 0..<x.count { x[i] += f2[i] }
        }
        let fin = layerNorm(Array(x[0..<d]), lnfG, lnfB, rows: 1, d: d)
        var logit = headB
        for j in 0..<d { logit += fin[j] * headW[j] }
        return logit
    }

    public func scores(typo: [UInt8], candidates: [[UInt8]]) -> [Float] {
        candidates.map { score(typo: typo, candidate: $0) }
    }
}
