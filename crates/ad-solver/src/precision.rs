//! DDF storage precision: the FP16C codec and the DDF-shifting convention.
//!
//! # Why store `f_i - w_i` instead of `f_i`
//!
//! A distribution function sits close to its lattice weight: at rest `f_i == w_i`
//! exactly, and in any flow we can simulate the deviation is `O(Ma)` — a few
//! percent at most. Storing `f_i` therefore burns the whole exponent range on a
//! constant, and every stored value carries the same absolute rounding error
//! `~w_i * 2^-m`. Storing the *shift* `g_i = f_i - w_i` centres the stored value
//! on zero, where floating point has enormous dynamic range, so the absolute
//! error scales with the deviation rather than with the mean. Lehmann (2022)
//! measures roughly an order of magnitude accuracy gain from this alone, which is
//! what makes 16-bit DDF storage viable at all.
//!
//! # The order of operations is the whole trick
//!
//! Shifting only pays if the shift is never undone. The moment you compute
//! `f_i = g_i + w_i` and then sum, you are back to catastrophic cancellation in
//! `rho = sum f_i` — a sum of `q` numbers of order `w_i` that must produce a
//! result whose *interesting part* is the `1e-4` deviation from 1. So:
//!
//! ```text
//!   drho   = sum_i g_i                 (small + small, no cancellation)
//!   rho    = 1 + drho                  (done once, at the end)
//!   rho*u  = sum_i c_i g_i             (sum_i c_i w_i == 0, so no correction)
//!   g_i^eq = w_i * (drho + rho*(3 c.u + 4.5 (c.u)^2 - 1.5 u.u))
//!   g_i'   = g_i - omega * (g_i - g_i^eq)
//! ```
//!
//! Note the equilibrium is built *already shifted*: `f^eq - w = w*((rho-1) +
//! rho*(...))`, never `f^eq` then minus `w`. And collision is exact in shifted
//! space because `f - f^eq == g - g^eq`, so the shift cancels identically and the
//! operator never sees a number of order 1. Every routine in this crate follows
//! that ordering; [`shifted_equilibrium`] is the canonical reference for it.
//!
//! # FP16C
//!
//! 1 sign / 4 exponent (bias 15) / 11 mantissa, no infinities and no NaNs, so the
//! all-ones exponent is a normal number and the range is `+/-(2 - 2^-11)`. That
//! is exactly the trade a shifted DDF wants: `g_i` cannot leave `+/-1` in any
//! stable simulation, so the four exponent bits IEEE binary16 spends reaching
//! 65504 are dead weight, and moving one of them into the mantissa halves the
//! relative rounding error.
//!
//! | format  | mantissa bits | max relative error (RNE) | max magnitude |
//! |---------|---------------|--------------------------|---------------|
//! | binary16| 10            | 2^-11 = 4.88e-4          | 65504         |
//! | FP16C   | 11            | 2^-12 = 2.44e-4          | 1.99951       |
//!
//! Everything here is integer bit manipulation on `u32`, deliberately: it must
//! run identically in WGSL on an adapter without the optional `f16` feature, and
//! [`wgsl_codec`] emits a line-for-line transliteration of
//! these functions. the `wgsl_source_mirrors_the_rust_codec` test guards the pairing at
//! the source level; `validation/lbm/gpu.rs` runs the WGSL against them on a real
//! device.
//!
//! Note that `f16` would not do instead, even where the adapter has it. The
//! encoding above spends its exponent differently from binary16, so an FP16C bit
//! pattern is not a binary16 *number*: every value in `[1.5, 2)` lands on what
//! binary16 reads as a NaN, and WGSL offers no bitcast between `f16` and `u32`
//! to move the bits across intact. What the 16 bits get stored *in* is a `u16`
//! element, one per cell; see [`ad_gpu::ddf`].

use ad_gpu::types::DdfPrecision;

/// Largest magnitude representable in FP16C: `2 - 2^-11`.
pub const FP16C_MAX: f32 = 1.999_511_7;
/// Smallest positive normal: `2^-14`.
pub const FP16C_MIN_NORMAL: f32 = 6.103_515_6e-5;
/// Smallest positive subnormal: `2^-25`.
pub const FP16C_MIN_SUBNORMAL: f32 = 2.980_232_2e-8;
/// Upper bound on the relative error of a round trip through FP16C, for values
/// at or above [`FP16C_MIN_NORMAL`]. Round-to-nearest-even on 11 stored mantissa
/// bits gives half an ulp, `2^-12`; the extra headroom covers the ulp boundary.
pub const FP16C_MAX_RELATIVE_ERROR: f32 = 2.5e-4;

/// Encode an `f32` as FP16C. Saturates rather than producing an infinity,
/// because the format has none: an overflowing DDF is already a diverged
/// simulation, and a finite clamp keeps the failure legible instead of turning
/// the whole field into NaN one step later.
#[inline]
pub fn f32_to_fp16c(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let e = (bits >> 23) & 0xff;

    // Inf / NaN have no encoding; saturate to the largest finite magnitude.
    if e == 0xff {
        return sign | 0x7fff;
    }

    // Normal range: the FP16C biased exponent is `e - 112`, so `e` in 113..=127
    // lands in 1..=15. Round to nearest even on the 12 mantissa bits we drop,
    // *before* re-reading the exponent, so a mantissa carry propagates into the
    // exponent for free.
    if e >= 101 {
        let round = 0x7ff + ((bits >> 12) & 1);
        let b = bits.wrapping_add(round);
        let e2 = (b >> 23) & 0xff;
        if e2 >= 128 {
            return sign | 0x7fff; // saturate
        }
        if e2 >= 113 {
            let m = (b & 0x007f_ffff) >> 12;
            return sign | (((e2 - 112) << 11) as u16) | m as u16;
        }
        // Subnormal: value = m * 2^-25 with m in 0..=2047. Reconstruct from the
        // *unrounded* significand so the rounding is applied at the right bit.
        let m32 = 0x0080_0000 | (bits & 0x007f_ffff);
        let shift = 125 - e; // >= 13 here, and <= 24 because e >= 101
        let half = 1u32 << (shift - 1);
        let m = (m32 + half - 1 + ((m32 >> shift) & 1)) >> shift;
        // `m == 0x800` means rounding pushed us up to the smallest normal, whose
        // encoding is exactly 0x0800 — the same bit pattern. No special case.
        return sign | m as u16;
    }

    // Below 2^-26: everything rounds to zero (2^-26 is half the smallest
    // subnormal, and round-to-nearest-even sends the tie to zero).
    sign
}

/// Decode FP16C back to `f32`. Exact: every FP16C value is representable in
/// `f32`, so this direction never rounds.
#[inline]
pub fn fp16c_to_f32(h: u16) -> f32 {
    let h = h as u32;
    let sign = (h & 0x8000) << 16;
    let e = (h >> 11) & 0xf;
    let m = h & 0x7ff;

    if e != 0 {
        return f32::from_bits(sign | ((e + 112) << 23) | (m << 12));
    }
    if m == 0 {
        return f32::from_bits(sign); // +/- 0
    }
    // Subnormal: value = m * 2^-25. Normalise by hand — `k` is the index of the
    // leading set bit, so `m = 2^k * (1 + frac)` and the f32 exponent is
    // `k - 25 + 127`.
    let k = 31 - m.leading_zeros();
    f32::from_bits(sign | ((k + 102) << 23) | ((m & !(1 << k)) << (23 - k)))
}

/// Round trip through FP16C, i.e. the value the solver actually stores and reads
/// back. Handy for CPU reference solvers that need to mimic GPU truncation.
#[inline]
pub fn quantise_fp16c(x: f32) -> f32 {
    fp16c_to_f32(f32_to_fp16c(x))
}

/// IEEE binary16, for the A/B comparison that justifies FP16C.
///
/// Present only so the test suite can prove FP16C is better on the values a DDF
/// actually takes; the solver never uses it.
pub fn f32_to_binary16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let e = (bits >> 23) & 0xff;
    if e == 0xff {
        return sign | 0x7c00 | (if bits & 0x007f_ffff != 0 { 0x200 } else { 0 });
    }
    // 2^-25 is half the smallest binary16 subnormal, and sits at e == 102.
    if e >= 102 {
        let round = 0xfff + ((bits >> 13) & 1);
        let b = bits.wrapping_add(round);
        let e2 = (b >> 23) & 0xff;
        if e2 >= 143 {
            return sign | 0x7c00; // overflow to infinity
        }
        if e2 >= 113 {
            return sign | (((e2 - 112) << 10) as u16) | ((b & 0x007f_ffff) >> 13) as u16;
        }
        let m32 = 0x0080_0000 | (bits & 0x007f_ffff);
        let shift = 126 - e;
        let half = 1u32 << (shift - 1);
        let m = (m32 + half - 1 + ((m32 >> shift) & 1)) >> shift;
        return sign | m as u16;
    }
    sign
}

/// Decode IEEE binary16.
pub fn binary16_to_f32(h: u16) -> f32 {
    let h = h as u32;
    let sign = (h & 0x8000) << 16;
    let e = (h >> 10) & 0x1f;
    let m = h & 0x3ff;
    if e == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (m << 13));
    }
    if e != 0 {
        return f32::from_bits(sign | ((e + 112) << 23) | (m << 13));
    }
    if m == 0 {
        return f32::from_bits(sign);
    }
    let k = 31 - m.leading_zeros();
    f32::from_bits(sign | ((k + 103) << 23) | ((m & !(1 << k)) << (23 - k)))
}

/// Round trip through IEEE binary16.
#[inline]
pub fn quantise_binary16(x: f32) -> f32 {
    binary16_to_f32(f32_to_binary16(x))
}

/// The shifted equilibrium `f_i^eq - w_i`, computed without ever forming `f_i^eq`.
///
/// This is the reference implementation of the ordering described in the module
/// comment; the WGSL in `shaders/lbm/collision.wgsl` mirrors it exactly. `drho`
/// is `rho - 1`, and it is passed separately from `rho` on purpose: the caller
/// obtained it as a sum of small numbers, and re-deriving it as `rho - 1.0`
/// inside here would throw that precision away again.
#[inline]
pub fn shifted_equilibrium(w: f32, c: [f32; 3], drho: f32, rho: f32, u: [f32; 3]) -> f32 {
    let cu = c[0] * u[0] + c[1] * u[1] + c[2] * u[2];
    let uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    w * (drho + rho * (3.0 * cu + 4.5 * cu * cu - 1.5 * uu))
}

/// WGSL source for the storage codec, matching [`f32_to_fp16c`] and
/// [`fp16c_to_f32`] operation for operation.
///
/// Emitted from Rust rather than kept as a static `.wgsl` file so the two halves
/// of the codec live in one place and cannot drift, in the same spirit as
/// [`ad_gpu::lattice::wgsl_prelude`]. `#if DDF_FP16C` in
/// `shaders/lbm/storage.wgsl` selects between this and the FP32 path.
pub fn wgsl_codec() -> String {
    // NOTE: kept textually close to the Rust above so a diff between the two is
    // readable. WGSL has no u16, so the encoded value lives in the low 16 bits
    // of a u32 throughout.
    r#"
// GENERATED by ad_solver::precision::wgsl_codec - do not edit.
// 1 sign / 4 exponent (bias 15) / 11 mantissa, no inf, no NaN, range +/-1.99951.

fn f32_to_fp16c(x: f32) -> u32 {
    let bits: u32 = bitcast<u32>(x);
    let sign: u32 = (bits >> 16u) & 0x8000u;
    let e: u32 = (bits >> 23u) & 0xffu;
    if (e == 0xffu) { return sign | 0x7fffu; }          // inf/NaN -> saturate
    if (e < 101u)  { return sign; }                      // underflows to zero
    let round: u32 = 0x7ffu + ((bits >> 12u) & 1u);      // round to nearest even
    let b: u32 = bits + round;
    let e2: u32 = (b >> 23u) & 0xffu;
    if (e2 >= 128u) { return sign | 0x7fffu; }           // saturate
    if (e2 >= 113u) {
        return sign | (((e2 - 112u) << 11u) & 0x7800u) | ((b & 0x007fffffu) >> 12u);
    }
    // Subnormal: value = m * 2^-25, rounded from the unrounded significand.
    let m32: u32 = 0x00800000u | (bits & 0x007fffffu);
    let shift: u32 = 125u - e;                           // in 13..=24
    let half: u32 = 1u << (shift - 1u);
    let m: u32 = (m32 + half - 1u + ((m32 >> shift) & 1u)) >> shift;
    return sign | m;                                     // m == 0x800 is the smallest normal
}

fn fp16c_to_f32(h: u32) -> f32 {
    let sign: u32 = (h & 0x8000u) << 16u;
    let e: u32 = (h >> 11u) & 0xfu;
    let m: u32 = h & 0x7ffu;
    if (e != 0u) { return bitcast<f32>(sign | ((e + 112u) << 23u) | (m << 12u)); }
    if (m == 0u) { return bitcast<f32>(sign); }
    let k: u32 = firstLeadingBit(m);
    return bitcast<f32>(sign | ((k + 102u) << 23u) | ((m & ~(1u << k)) << (23u - k)));
}
"#
    .to_string()
}

/// Distributions per `u32` word, for the *packed* FP16C fallback layout.
///
/// Only the fallback shares a word between cells. Where the adapter can address
/// 16 bits (`wgpu::Features::SHADER_I16`) each cell gets its own `u16` element
/// and nothing is packed, so this ratio does not describe that layout — see
/// [`ad_gpu::ddf`] and `ad_solver::shaders::DdfLayout`. The byte count is two
/// per distribution either way, which is what
/// [`DdfPrecision::bytes_per_ddf`] reports.
pub fn ddfs_per_word(p: DdfPrecision) -> u64 {
    match p {
        DdfPrecision::Fp32 => 1,
        DdfPrecision::Fp16c => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values a shifted DDF actually takes: `g_i = f_i - w_i`, which for
    /// `|u| <= 0.2` stays well inside `+/-0.2`, plus a sweep down to the
    /// subnormal floor and up to the saturation point.
    fn sweep() -> Vec<f32> {
        let mut v = Vec::new();
        let mut x = FP16C_MIN_NORMAL;
        while x < FP16C_MAX {
            v.push(x);
            v.push(-x);
            x *= 1.000_37; // ~2600 samples per octave-and-a-bit; hits many ulps
        }
        // Exact powers of two and their neighbourhoods, where exponent carries
        // and the normal/subnormal boundary live.
        for k in -25i32..=0 {
            let p = 2f32.powi(k);
            for s in [1.0, 1.000_1, 0.999_9, 1.5, 1.999] {
                v.push(p * s);
                v.push(-p * s);
            }
        }
        v.push(0.0);
        v.push(-0.0);
        v
    }

    #[test]
    fn fp16c_round_trips_within_half_an_ulp() {
        let mut worst = 0.0f32;
        let mut worst_at = 0.0f32;
        for x in sweep() {
            let back = quantise_fp16c(x);
            assert!(back.is_finite(), "{x:e} decoded to {back}");
            if x == 0.0 {
                assert_eq!(back, 0.0);
                continue;
            }
            if x.abs() < FP16C_MIN_NORMAL {
                // Subnormal: only an *absolute* bound is meaningful.
                assert!(
                    (back - x).abs() <= FP16C_MIN_SUBNORMAL * 0.5001,
                    "subnormal {x:e} -> {back:e}"
                );
                continue;
            }
            let rel = ((back - x) / x).abs();
            if rel > worst {
                worst = rel;
                worst_at = x;
            }
        }
        assert!(
            worst <= FP16C_MAX_RELATIVE_ERROR,
            "worst relative error {worst:e} at {worst_at:e}, want <= {FP16C_MAX_RELATIVE_ERROR:e}"
        );
        // ...and it really is a half-ulp code, not a lucky truncation: the worst
        // case must be close to 2^-12, otherwise the mantissa is not 11 bits.
        assert!(
            worst > 1.0e-4,
            "worst error {worst:e} is suspiciously small; is the sweep dense enough?"
        );
    }

    #[test]
    fn fp16c_beats_binary16_on_the_same_values() {
        let mut worst_c = 0.0f32;
        let mut worst_h = 0.0f32;
        let mut mean_c = 0.0f64;
        let mut mean_h = 0.0f64;
        let mut n = 0u64;
        for x in sweep() {
            if x.abs() < FP16C_MIN_NORMAL {
                continue;
            }
            let ec = ((quantise_fp16c(x) - x) / x).abs();
            let eh = ((quantise_binary16(x) - x) / x).abs();
            worst_c = worst_c.max(ec);
            worst_h = worst_h.max(eh);
            mean_c += ec as f64;
            mean_h += eh as f64;
            n += 1;
        }
        // One extra mantissa bit is exactly a factor of two, both in the worst
        // case and on average. Anything less means the encoder is dropping a bit.
        assert!(
            worst_h / worst_c > 1.9,
            "FP16C worst {worst_c:e} vs binary16 worst {worst_h:e}: expected ~2x better"
        );
        assert!(
            mean_h / mean_c > 1.9,
            "FP16C mean {:e} vs binary16 mean {:e}: expected ~2x better",
            mean_c / n as f64,
            mean_h / n as f64
        );
    }

    #[test]
    fn fp16c_saturates_instead_of_producing_infinities() {
        for x in [2.0f32, 10.0, 1e30, f32::INFINITY] {
            assert_eq!(quantise_fp16c(x), FP16C_MAX, "{x} should clamp");
            assert_eq!(quantise_fp16c(-x), -FP16C_MAX, "{} should clamp", -x);
        }
        assert!(
            quantise_fp16c(f32::NAN).is_finite(),
            "NaN must not survive the codec"
        );
    }

    #[test]
    fn fp16c_encoding_has_the_documented_layout() {
        // 1.0 is exponent 15 biased (unbiased 0), mantissa 0.
        assert_eq!(f32_to_fp16c(1.0), 0x7800);
        assert_eq!(fp16c_to_f32(0x7800), 1.0);
        // The largest finite value, 2 - 2^-11.
        assert_eq!(fp16c_to_f32(0x7fff), FP16C_MAX);
        // Smallest normal and smallest subnormal.
        assert_eq!(fp16c_to_f32(0x0800), FP16C_MIN_NORMAL);
        assert_eq!(fp16c_to_f32(0x0001), FP16C_MIN_SUBNORMAL);
        // Sign bit is the top bit and nothing else changes.
        for h in [0x0001u16, 0x0800, 0x1234, 0x7fff] {
            assert_eq!(fp16c_to_f32(h | 0x8000), -fp16c_to_f32(h));
        }
    }

    #[test]
    fn every_fp16c_bit_pattern_round_trips_exactly() {
        // Decoding is exact, so encode(decode(h)) == h for all 65536 patterns
        // (modulo the two zeros, which are distinct patterns with equal value).
        for h in 0u32..=0xffff {
            let h = h as u16;
            let x = fp16c_to_f32(h);
            assert!(x.is_finite(), "pattern {h:#06x} decoded to {x}");
            assert_eq!(
                f32_to_fp16c(x),
                h,
                "pattern {h:#06x} did not survive re-encoding"
            );
        }
    }

    #[test]
    fn binary16_reference_codec_is_itself_correct() {
        // The comparison is only worth anything if the baseline is right.
        assert_eq!(f32_to_binary16(1.0), 0x3c00);
        assert_eq!(binary16_to_f32(0x3c00), 1.0);
        assert_eq!(binary16_to_f32(0x7bff), 65504.0);
        assert_eq!(binary16_to_f32(0x0400), 6.103_515_6e-5);
        assert_eq!(
            quantise_binary16(65505.0),
            65504.0,
            "just below the midpoint, still finite"
        );
        assert_eq!(quantise_binary16(70000.0), f32::INFINITY);
        assert_eq!(
            binary16_to_f32(0x0001),
            5.960_464_5e-8,
            "smallest subnormal is 2^-24"
        );
    }

    #[test]
    fn shifted_equilibrium_reproduces_the_unshifted_form() {
        use ad_gpu::lattice::{D3Q19_DIRS, D3Q19_WEIGHTS};
        let u = [0.05f32, -0.02, 0.01];
        let rho = 1.0031f32;
        let drho = rho - 1.0;
        let uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
        for i in 0..19 {
            let c = [
                D3Q19_DIRS[i].x as f32,
                D3Q19_DIRS[i].y as f32,
                D3Q19_DIRS[i].z as f32,
            ];
            let cu = c[0] * u[0] + c[1] * u[1] + c[2] * u[2];
            let plain = D3Q19_WEIGHTS[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uu);
            let shifted = shifted_equilibrium(D3Q19_WEIGHTS[i], c, drho, rho, u);
            assert!(
                (shifted - (plain - D3Q19_WEIGHTS[i])).abs() < 1e-7,
                "direction {i}: shifted {shifted} vs plain-minus-weight {}",
                plain - D3Q19_WEIGHTS[i]
            );
        }
    }

    #[test]
    fn shifted_equilibrium_conserves_mass_and_momentum() {
        use ad_gpu::lattice::{D3Q19_DIRS, D3Q19_WEIGHTS};
        let u = [0.07f32, 0.03, -0.05];
        let rho = 0.994f32;
        let drho = rho - 1.0;
        let (mut sum, mut mom) = (0.0f64, [0.0f64; 3]);
        for i in 0..19 {
            let c = [
                D3Q19_DIRS[i].x as f32,
                D3Q19_DIRS[i].y as f32,
                D3Q19_DIRS[i].z as f32,
            ];
            let g = shifted_equilibrium(D3Q19_WEIGHTS[i], c, drho, rho, u);
            sum += g as f64;
            for a in 0..3 {
                mom[a] += (g * c[a]) as f64;
            }
        }
        // sum_i g_i^eq == rho - 1, and sum_i c_i g_i^eq == rho*u.
        assert!(
            (sum - drho as f64).abs() < 1e-6,
            "mass closure: {sum} vs {drho}"
        );
        for a in 0..3 {
            let want = (rho * u[a]) as f64;
            assert!(
                (mom[a] - want).abs() < 1e-6,
                "momentum {a}: {} vs {want}",
                mom[a]
            );
        }
    }

    #[test]
    fn wgsl_source_mirrors_the_rust_codec() {
        let s = wgsl_codec();
        // The magic numbers that define the format must appear in both halves;
        // if someone edits one side these are the constants that would change.
        for needle in [
            "0x7fffu",
            "0x7ffu",
            "112u",
            "0x007fffffu",
            "125u - e",
            "firstLeadingBit",
        ] {
            assert!(s.contains(needle), "WGSL codec is missing {needle}");
        }
        assert!(s.contains("fn f32_to_fp16c") && s.contains("fn fp16c_to_f32"));
    }
}
