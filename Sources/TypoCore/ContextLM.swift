import Foundation
import Accelerate

/// Context language model: which word fits after the words typed so far on this line?
/// A small causal word transformer over the lexicon, trained in training/lm.py and exported as
/// lm.json + lm.bin. Inference is written by hand here (Accelerate), like `NeuralRanker`.
public final class ContextLM {
    public struct Config: Decodable {
        public let version: Int, d: Int, heads: Int, layers: Int, ff: Int, maxLen: Int, vocab: Int
        /// Vocabulary ids below bands[0] use dims[0]-wide embeddings, below bands[1] dims[1], …
        public let bands: [Int], dims: [Int]
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
    let bounds: [Int]              // [0, bands…, vocab]
    private let emb: [[Float]]     // band i: [size_i, dims_i]
    private let inProj: [[Float]]  // band i: [dims_i, d]
    private let outProj: [[Float]] // band i: [d, dims_i]
    private let outBias: [[Float]]
    private let headW: [Float]     // [size_0 + rare bands, dims_0]: frequent words, then one row per rare band
    private let headB: [Float]
    private let posEmb: [Float]
    private let layers: [Layer]
    private let lnfG, lnfB: [Float]

    public init(manifest manifestURL: URL, weights weightsURL: URL) throws {
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
        let data = try Data(contentsOf: weightsURL, options: .alwaysMapped)
        config = manifest.config
        source = manifestURL
        var metas: [String: TensorMeta] = [:]
        for t in manifest.tensors { metas[t.name] = t }
        let total = data.count / MemoryLayout<Float>.size

        func tensor(_ name: String) throws -> [Float] {
            guard let m = metas[name] else { throw TypoError.badResource("lm tensor \(name) missing") }
            let n = m.shape.reduce(1, *)
            guard m.offset >= 0, m.offset + n <= total else { throw TypoError.badResource("lm tensor \(name) out of range") }
            var out = [Float](repeating: 0, count: n)
            out.withUnsafeMutableBytes { dst in
                data.withUnsafeBytes { src in
                    dst.copyMemory(from: UnsafeRawBufferPointer(rebasing: src[(m.offset * 4)..<((m.offset + n) * 4)]))
                }
            }
            return out
        }
        /// PyTorch Linear weight is [out, in]; stored as [in, out] for x·W.
        func linearT(_ name: String, out: Int, inp: Int) throws -> [Float] {
            let w = try tensor(name)
            guard w.count == out * inp else { throw TypoError.badResource("lm tensor \(name) has the wrong shape") }
            var t = [Float](repeating: 0, count: w.count)
            w.withUnsafeBufferPointer { src in
                t.withUnsafeMutableBufferPointer { dst in
                    vDSP_mtrans(src.baseAddress!, 1, dst.baseAddress!, 1, vDSP_Length(inp), vDSP_Length(out))
                }
            }
            return t
        }

        let d = config.d, ff = config.ff, nb = config.dims.count
        guard nb >= 1, config.bands.count == nb - 1, d % config.heads == 0 else { throw TypoError.badResource("lm config") }
        bounds = [0] + config.bands + [config.vocab]
        var e: [[Float]] = [], ip: [[Float]] = [], op: [[Float]] = [], ob: [[Float]] = []
        for i in 0..<nb {
            e.append(try tensor("emb.\(i).weight"))
            guard e[i].count == (bounds[i + 1] - bounds[i]) * config.dims[i] else { throw TypoError.badResource("lm embedding \(i)") }
            ip.append(try linearT("in_proj.\(i).weight", out: d, inp: config.dims[i]))
            op.append(try linearT("out_proj.\(i).weight", out: config.dims[i], inp: d))
            ob.append(try tensor("out_bias.\(i)"))
        }
        emb = e; inProj = ip; outProj = op; outBias = ob
        headW = e[0] + (nb > 1 ? try tensor("cluster") : [])
        headB = ob[0] + (nb > 1 ? try tensor("cluster_bias") : [])
        posEmb = try tensor("pos_emb.weight")
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
    }

    /// lm.json + lm.bin from `directory`, or the bundled ones.
    public static func load(directory: URL? = nil) -> ContextLM? {
        let m = directory?.appendingPathComponent("lm.json") ?? Paths.resource("lm.json")
        let w = directory?.appendingPathComponent("lm.bin") ?? Paths.resource("lm.bin")
        guard let m, let w, FileManager.default.fileExists(atPath: m.path), FileManager.default.fileExists(atPath: w.path) else { return nil }
        return try? ContextLM(manifest: m, weights: w)
    }

    // MARK: math helpers

    /// x [m, k] · w [k, n] + b [n] per row.
    private func affine(_ x: [Float], _ m: Int, _ k: Int, _ w: [Float], _ b: [Float]?, _ n: Int) -> [Float] {
        var c = [Float](repeating: 0, count: m * n)
        x.withUnsafeBufferPointer { px in
            w.withUnsafeBufferPointer { pw in
                c.withUnsafeMutableBufferPointer { pc in
                    if n == 1 {                      // matrix · vector: every word's score against one state
                        cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(m), Int32(k), 1, px.baseAddress!, Int32(k),
                                    pw.baseAddress!, 1, 0, pc.baseAddress!, 1)
                    } else {
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(m), Int32(n), Int32(k), 1,
                                    px.baseAddress!, Int32(k), pw.baseAddress!, Int32(n), 0, pc.baseAddress!, Int32(n))
                    }
                    if let b {
                        b.withUnsafeBufferPointer { pb in
                            for r in 0..<m { vDSP_vadd(pc.baseAddress! + r * n, 1, pb.baseAddress!, 1, pc.baseAddress! + r * n, 1, vDSP_Length(n)) }
                        }
                    }
                }
            }
        }
        return c
    }

    private func layerNorm(_ x: [Float], _ g: [Float], _ b: [Float], rows: Int) -> [Float] {
        let d = config.d
        var out = [Float](repeating: 0, count: rows * d)
        x.withUnsafeBufferPointer { px in
            out.withUnsafeMutableBufferPointer { po in
                for r in 0..<rows {
                    let src = px.baseAddress! + r * d, dst = po.baseAddress! + r * d
                    var mean: Float = 0
                    vDSP_meanv(src, 1, &mean, vDSP_Length(d))
                    var neg = -mean
                    vDSP_vsadd(src, 1, &neg, dst, 1, vDSP_Length(d))
                    var ms: Float = 0
                    vDSP_measqv(dst, 1, &ms, vDSP_Length(d))
                    var inv = 1 / (ms + 1e-5).squareRoot()
                    vDSP_vsmul(dst, 1, &inv, dst, 1, vDSP_Length(d))
                    vDSP_vma(dst, 1, g, 1, b, 1, dst, 1, vDSP_Length(d))
                }
            }
        }
        return out
    }

    private func add(_ a: inout [Float], _ b: [Float]) {
        let n = vDSP_Length(a.count)
        a.withUnsafeMutableBufferPointer { pa in vDSP_vadd(pa.baseAddress!, 1, b, 1, pa.baseAddress!, 1, n) }
    }

    private func logSoftmax(_ x: inout [Float]) {
        let n = vDSP_Length(x.count)
        var count = Int32(x.count)
        var e = [Float](repeating: 0, count: x.count)
        x.withUnsafeMutableBufferPointer { px in
            e.withUnsafeMutableBufferPointer { pe in
                var mx: Float = 0
                vDSP_maxv(px.baseAddress!, 1, &mx, n)
                var neg = -mx
                vDSP_vsadd(px.baseAddress!, 1, &neg, pe.baseAddress!, 1, n)
                vvexpf(pe.baseAddress!, pe.baseAddress!, &count)
                var sum: Float = 0
                vDSP_sve(pe.baseAddress!, 1, &sum, n)
                var shift = -(mx + logf(sum))
                vDSP_vsadd(px.baseAddress!, 1, &shift, px.baseAddress!, 1, n)
            }
        }
    }

    private func band(of token: Int) -> Int {
        var b = 0
        while b + 1 < bounds.count - 1 && token >= bounds[b + 1] { b += 1 }
        return b
    }

    // MARK: forward

    /// The model's state after `tokens` (at most `maxLen`; the first is <s> or <cut>): what it uses
    /// to predict the next word.
    public func hidden(_ tokens: [Int32]) -> [Float] {
        let n = tokens.count, d = config.d, H = config.heads, dh = d / H, ff = config.ff
        precondition(n >= 1 && n <= config.maxLen, "context length")
        var x = [Float](repeating: 0, count: n * d)
        for t in 0..<n {
            let tok = min(max(Int(tokens[t]), 0), config.vocab - 1)
            let b = band(of: tok), e = config.dims[b]
            let row = affine(Array(emb[b][((tok - bounds[b]) * e)..<((tok - bounds[b] + 1) * e)]), 1, e, inProj[b], nil, d)
            for j in 0..<d { x[t * d + j] = row[j] + posEmb[t * d + j] }
        }
        let scale = 1 / Float(dh).squareRoot()
        var weights = [Float](repeating: 0, count: n)
        for (li, layer) in layers.enumerated() {
            // Only the last position is read after the last layer, so that layer computes one row.
            let r0 = li == layers.count - 1 ? n - 1 : 0
            let r = n - r0
            let h = layerNorm(x, layer.ln1g, layer.ln1b, rows: n)
            let k = affine(h, n, d, layer.wk, layer.bk, d)
            let v = affine(h, n, d, layer.wv, layer.bv, d)
            let q = affine(Array(h[(r0 * d)...]), r, d, layer.wq, layer.bq, d)
            var ctx = [Float](repeating: 0, count: r * d)
            q.withUnsafeBufferPointer { pq in
                k.withUnsafeBufferPointer { pk in
                    v.withUnsafeBufferPointer { pv in
                        ctx.withUnsafeMutableBufferPointer { pc in
                            for i in r0..<n {
                                for hd in 0..<H {
                                    let off = hd * dh
                                    let qrow = pq.baseAddress! + (i - r0) * d + off
                                    var mx = -Float.greatestFiniteMagnitude
                                    for j in 0...i {                               // causal: earlier tokens only
                                        var s: Float = 0
                                        vDSP_dotpr(qrow, 1, pk.baseAddress! + j * d + off, 1, &s, vDSP_Length(dh))
                                        s *= scale
                                        weights[j] = s
                                        if s > mx { mx = s }
                                    }
                                    var sum: Float = 0
                                    for j in 0...i { weights[j] = expf(weights[j] - mx); sum += weights[j] }
                                    let dst = pc.baseAddress! + (i - r0) * d + off
                                    for j in 0...i {
                                        var w = weights[j] / sum
                                        vDSP_vsma(pv.baseAddress! + j * d + off, 1, &w, dst, 1, dst, 1, vDSP_Length(dh))
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let o = affine(ctx, r, d, layer.wo, layer.bo, d)
            var xr = r0 == 0 ? x : Array(x[(r0 * d)...])
            add(&xr, o)
            let h2 = layerNorm(xr, layer.ln2g, layer.ln2b, rows: r)
            var f1 = affine(h2, r, d, layer.w1, layer.b1, ff)
            VectorMath.gelu(&f1)
            let f2 = affine(f1, r, ff, layer.w2, layer.b2, d)
            add(&xr, f2)
            x = xr
        }
        return layerNorm(Array(x.suffix(d)), lnfG, lnfB, rows: 1)
    }

    /// log P(frequent word) for ids below bands[0], then log P(the next word is in rare band i).
    func headLogProbs(_ h: [Float]) -> [Float] {
        let e = config.dims[0]
        let p = affine(h, 1, config.d, outProj[0], nil, e)
        var logits = affine(headW, headB.count, e, p, nil, 1)
        add(&logits, headB)
        logSoftmax(&logits)
        return logits
    }

    /// log P(word | the next word is in rare band `i`), for every word of the band.
    func tailLogProbs(_ i: Int, _ h: [Float]) -> [Float] {
        let e = config.dims[i]
        let p = affine(h, 1, config.d, outProj[i], nil, e)
        var logits = affine(emb[i], outBias[i].count, e, p, nil, 1)
        add(&logits, outBias[i])
        logSoftmax(&logits)
        return logits
    }
}

/// The model's view of one moment in a line. Word probabilities are computed on demand: the
/// frequent-word table first, a rare band's table only when a word from it is asked for.
public final class LMState {
    public let hidden: [Float]
    /// Tokens the model read, counting the leading <s> / <cut>.
    public let contextLength: Int
    /// The context began with <cut>: text before it is unknown.
    public let startsWithCut: Bool
    private let lm: ContextLM
    private var head: [Float]?
    private var tails: [[Float]?]

    public init(lm: ContextLM, hidden: [Float], contextLength: Int, startsWithCut: Bool) {
        self.lm = lm; self.hidden = hidden; self.contextLength = contextLength; self.startsWithCut = startsWithCut
        tails = Array(repeating: nil, count: lm.config.dims.count)
    }

    public convenience init(lm: ContextLM, tokens: [Int32]) {
        self.init(lm: lm, hidden: lm.hidden(tokens), contextLength: tokens.count, startsWithCut: tokens.first == ContextTokenizer.cut)
    }

    /// Log-probability of the likeliest next frequent word: how sure the model is of what comes next.
    public var topLogProb: Float {
        if head == nil { head = lm.headLogProbs(hidden) }
        return head![Int(ContextTokenizer.reserved)..<lm.bounds[1]].max() ?? 0
    }

    /// Natural-log probability that the next token is `token`.
    public func logProb(_ token: Int32) -> Float {
        let t = Int(token), b = lm.bounds
        guard t >= 0, t < lm.config.vocab else { return -30 }
        if head == nil { head = lm.headLogProbs(hidden) }
        if t < b[1] { return head![t] }
        var i = 1
        while i + 1 < b.count - 1 && t >= b[i + 1] { i += 1 }
        if tails[i] == nil { tails[i] = lm.tailLogProbs(i, hidden) }
        return head![b[1] + i - 1] + tails[i]![t - b[i]]
    }
}
