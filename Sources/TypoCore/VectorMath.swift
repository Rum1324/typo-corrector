import Foundation
import Accelerate

/// Array maths shared by the hand-written model inference (`NeuralRanker`, `ContextLM`).
enum VectorMath {
    /// GELU in its exact form, 0.5·x·(1 + erf(x/√2)), over a whole array. erf uses the
    /// Abramowitz–Stegun 7.1.26 polynomial (error below 1e-6 in single precision) so the loop
    /// vectorises; calling erff per element took most of the inference time.
    static func gelu(_ x: inout [Float]) {
        let n = x.count
        guard n > 0 else { return }
        var z = [Float](repeating: 0, count: n), e = [Float](repeating: 0, count: n)
        x.withUnsafeMutableBufferPointer { px in
            z.withUnsafeMutableBufferPointer { pz in
                e.withUnsafeMutableBufferPointer { pe in
                    var s: Float = 0.70710678
                    var count = Int32(n)
                    vDSP_vsmul(px.baseAddress!, 1, &s, pz.baseAddress!, 1, vDSP_Length(n))
                    vDSP_vabs(pz.baseAddress!, 1, pz.baseAddress!, 1, vDSP_Length(n))       // |x| / √2
                    vDSP_vsq(pz.baseAddress!, 1, pe.baseAddress!, 1, vDSP_Length(n))
                    vDSP_vneg(pe.baseAddress!, 1, pe.baseAddress!, 1, vDSP_Length(n))
                    vvexpf(pe.baseAddress!, pe.baseAddress!, &count)                        // exp(-x²/2)
                    for i in 0..<n {
                        let t = 1 / (1 + 0.3275911 * pz[i])
                        let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
                        let erfAbs = 1 - poly * pe[i]
                        let v = px[i]
                        px[i] = 0.5 * v * (1 + (v < 0 ? -erfAbs : erfAbs))
                    }
                }
            }
        }
    }
}
