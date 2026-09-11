//! Colour maps, built as 256-entry LUTs in linear light.
//!
//! # Why these maps and not others
//!
//! A colour map is a measuring instrument. Jet and the rainbow family are
//! excluded outright: their luminance is non-monotonic, so they invent
//! boundaries where the data is smooth (the notorious cyan and yellow bands)
//! and hide boundaries where the data is not. In a tool whose whole job is to
//! tell you whether a duct separates, that is a correctness bug, not a taste
//! preference. Turbo is available because people ask for it, and is labelled
//! not-colourblind-safe wherever it appears.
//!
//! - **inferno / magma** for volume rendering. They start at black and rise
//!   monotonically in lightness, which means the map is doing the opacity's job
//!   for it: low values are dark *and* transparent, so they compound instead of
//!   fighting. A map that starts bright (viridis) makes the volume look milky.
//! - **viridis / batlow** for surface and slice magnitudes: perceptually uniform,
//!   colourblind-safe, and light enough at the top to read annotations against.
//! - **cool-warm / vik** for signed data such as pressure, always with the range
//!   locked symmetric about zero. A diverging map with an off-centre zero is
//!   actively misleading — worse than a sequential map — because the eye reads
//!   the pale midpoint as "neutral" wherever it happens to land.
//!
//! # Why interpolate in Oklab
//!
//! Lerping 8-bit sRGB values between stops is the classic bug: sRGB is roughly
//! a gamma-2.2 encoding, so the average of two encoded values is much darker
//! than the encoding of the average. Blend `#0000FF` with `#FFFF00` in sRGB and
//! the midpoint comes out a muddy dark grey instead of a mid grey. Interpolating
//! in *linear light* fixes the darkening. Interpolating in **Oklab** fixes the
//! darkening *and* keeps the perceived lightness ramp even, which is what makes
//! a reconstructed map behave like the published one between its anchors. Both
//! are available; Oklab is the default.
//!
//! # About the stop tables
//!
//! The tables below are anchor reconstructions: the published maps sampled at
//! 9-11 points, with the space between anchors filled in perceptually. They are
//! not bit-exact copies of the reference LUTs. Everything this app relies on —
//! monotonic lightness, a symmetric diverging profile, CVD safety — is enforced
//! by construction and asserted in the tests below, so a small deviation from
//! the reference table cannot turn into a misleading picture.

use glam::Vec3;

/// Number of entries in an uploaded LUT. 256 is enough that linear filtering
/// between entries is invisible, and it makes the texture exactly 2 KiB at
/// `Rgba16Float`.
pub const LUT_SIZE: usize = 256;

/// Whether the map encodes a magnitude or a signed deviation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapKind {
    /// Monotonic lightness from dark to light.
    Sequential,
    /// Light in the middle, dark at both ends. Must be used with a symmetric
    /// range, see [`ColorMap::wants_symmetric_range`].
    Diverging,
}

/// How to blend between stops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Interpolation {
    /// Perceptually uniform. Default, and what makes reconstructions behave.
    #[default]
    Oklab,
    /// Plain lerp of linear-light RGB. Correct (no gamma darkening) but the
    /// lightness ramp between widely-spaced anchors can bow.
    LinearLight,
    /// Lerp of sRGB-encoded values. **Wrong**, and present only so the test
    /// suite can demonstrate the failure it causes.
    #[doc(hidden)]
    EncodedSrgb,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorMap {
    Inferno,
    Magma,
    Viridis,
    Batlow,
    CoolWarm,
    Vik,
    /// Not colourblind-safe. Offered because users ask for it by name.
    Turbo,
    /// Neutral ramp. Useful when the colour channel is carrying something else
    /// (shading, a second field) and you want the map out of the way.
    Grey,
}

impl ColorMap {
    pub const ALL: [ColorMap; 8] = [
        ColorMap::Inferno,
        ColorMap::Magma,
        ColorMap::Viridis,
        ColorMap::Batlow,
        ColorMap::CoolWarm,
        ColorMap::Vik,
        ColorMap::Turbo,
        ColorMap::Grey,
    ];

    /// Maps appropriate for volume rendering: they start at black, so the
    /// colour ramp reinforces the opacity ramp instead of fighting it.
    pub const VOLUME: [ColorMap; 3] = [ColorMap::Inferno, ColorMap::Magma, ColorMap::Grey];
    /// Maps appropriate for surfaces and slices of a magnitude.
    pub const MAGNITUDE: [ColorMap; 4] = [
        ColorMap::Viridis,
        ColorMap::Batlow,
        ColorMap::Inferno,
        ColorMap::Turbo,
    ];
    /// Maps appropriate for signed data such as pressure.
    pub const SIGNED: [ColorMap; 2] = [ColorMap::CoolWarm, ColorMap::Vik];

    pub fn name(self) -> &'static str {
        match self {
            ColorMap::Inferno => "inferno",
            ColorMap::Magma => "magma",
            ColorMap::Viridis => "viridis",
            ColorMap::Batlow => "batlow",
            ColorMap::CoolWarm => "cool-warm",
            ColorMap::Vik => "vik",
            ColorMap::Turbo => "turbo",
            ColorMap::Grey => "grey",
        }
    }

    /// Label for the UI. The one map that is not safe says so, every time.
    pub fn label(self) -> &'static str {
        match self {
            ColorMap::Turbo => "turbo (not colourblind-safe)",
            other => other.name(),
        }
    }

    pub fn kind(self) -> MapKind {
        match self {
            ColorMap::CoolWarm | ColorMap::Vik => MapKind::Diverging,
            _ => MapKind::Sequential,
        }
    }

    /// Diverging maps must be driven from a range centred on zero.
    pub fn wants_symmetric_range(self) -> bool {
        self.kind() == MapKind::Diverging
    }

    /// Safe under deuteranopia, protanopia and tritanopia.
    pub fn cvd_safe(self) -> bool {
        !matches!(self, ColorMap::Turbo)
    }

    /// Whether the map's luminance ramp is monotonic.
    ///
    /// Turbo is the exception, and this is the deeper reason it is a poor
    /// measuring instrument: it sweeps most of the hue circle and peaks in
    /// lightness around 70%, so it invents a bright band in the middle of the
    /// data. We do not "repair" it — silently reshaping a map the user asked for
    /// by name would be worse than letting it be what it is — we just never
    /// claim it is monotonic.
    pub fn luminance_monotonic(self) -> bool {
        !matches!(self, ColorMap::Turbo)
    }

    /// What to use instead when the accessibility toggle is on. The substitute
    /// preserves the *kind* of the map so a diverging field stays diverging.
    pub fn accessible_substitute(self) -> ColorMap {
        if self.cvd_safe() {
            self
        } else {
            match self.kind() {
                MapKind::Sequential => ColorMap::Viridis,
                MapKind::Diverging => ColorMap::Vik,
            }
        }
    }

    /// Colour for values below the mapped range. Electric cyan: absent from
    /// every map here, so clipping is unmistakable rather than blending into
    /// the bottom of the ramp.
    pub const UNDER: [u8; 3] = [0x00, 0xE5, 0xFF];
    /// Colour for values above the mapped range.
    pub const OVER: [u8; 3] = [0xFF, 0x00, 0xE5];

    /// Anchor stops, as `(position, sRGB-encoded 8-bit)`.
    pub fn stops(self) -> &'static [(f32, [u8; 3])] {
        match self {
            ColorMap::Inferno => &[
                (0.000, [0x00, 0x00, 0x04]),
                (0.125, [0x1B, 0x0C, 0x41]),
                (0.250, [0x4A, 0x0C, 0x6B]),
                (0.375, [0x78, 0x1C, 0x6D]),
                (0.500, [0xA5, 0x2C, 0x60]),
                (0.625, [0xCF, 0x44, 0x46]),
                (0.750, [0xED, 0x69, 0x25]),
                (0.875, [0xFB, 0x9A, 0x06]),
                (1.000, [0xFC, 0xFF, 0xA4]),
            ],
            ColorMap::Magma => &[
                (0.000, [0x00, 0x00, 0x04]),
                (0.125, [0x18, 0x0F, 0x3D]),
                (0.250, [0x44, 0x0F, 0x76]),
                (0.375, [0x72, 0x1F, 0x81]),
                (0.500, [0x9E, 0x2F, 0x7F]),
                (0.625, [0xCD, 0x40, 0x71]),
                (0.750, [0xF1, 0x60, 0x5D]),
                (0.875, [0xFD, 0x96, 0x68]),
                (1.000, [0xFC, 0xFD, 0xBF]),
            ],
            ColorMap::Viridis => &[
                (0.000, [0x44, 0x01, 0x54]),
                (0.125, [0x48, 0x28, 0x78]),
                (0.250, [0x3E, 0x4A, 0x89]),
                (0.375, [0x31, 0x68, 0x8E]),
                (0.500, [0x26, 0x82, 0x8E]),
                (0.625, [0x1F, 0x9E, 0x89]),
                (0.750, [0x35, 0xB7, 0x79]),
                (0.875, [0x6E, 0xCE, 0x58]),
                (1.000, [0xFD, 0xE7, 0x25]),
            ],
            ColorMap::Batlow => &[
                (0.000, [0x01, 0x19, 0x59]),
                (0.125, [0x12, 0x40, 0x5E]),
                (0.250, [0x1B, 0x60, 0x5A]),
                (0.375, [0x42, 0x7A, 0x47]),
                (0.500, [0x78, 0x93, 0x37]),
                (0.625, [0xAC, 0xA1, 0x3B]),
                (0.750, [0xD8, 0xA9, 0x60]),
                (0.875, [0xF5, 0xB8, 0x9A]),
                (1.000, [0xFA, 0xCC, 0xFA]),
            ],
            // Moreland's smooth cool-warm. Both ends land at the same lightness,
            // which is exactly the property that makes over- and under-pressure
            // read as equally significant.
            ColorMap::CoolWarm => &[
                (0.000, [0x3B, 0x4C, 0xC0]),
                (0.250, [0x90, 0xB2, 0xFE]),
                (0.500, [0xDD, 0xDD, 0xDD]),
                (0.750, [0xF5, 0x9C, 0x7D]),
                (1.000, [0xB4, 0x04, 0x26]),
            ],
            ColorMap::Vik => &[
                (0.000, [0x00, 0x12, 0x61]),
                (0.250, [0x3A, 0x6F, 0x99]),
                (0.500, [0xEB, 0xE6, 0xE4]),
                (0.750, [0xB1, 0x56, 0x36]),
                (1.000, [0x59, 0x00, 0x08]),
            ],
            ColorMap::Turbo => &[
                (0.000, [0x30, 0x12, 0x3B]),
                (0.100, [0x41, 0x49, 0xB0]),
                (0.200, [0x46, 0x81, 0xF5]),
                (0.300, [0x35, 0xAB, 0xF8]),
                (0.400, [0x1A, 0xD2, 0xD0]),
                (0.500, [0x35, 0xF3, 0x94]),
                (0.600, [0x7C, 0xFB, 0x4F]),
                (0.700, [0xC1, 0xE9, 0x2F]),
                (0.800, [0xF0, 0xBA, 0x38]),
                (0.900, [0xF0, 0x5B, 0x12]),
                (1.000, [0x7A, 0x04, 0x03]),
            ],
            ColorMap::Grey => &[(0.000, [0x00, 0x00, 0x00]), (1.000, [0xFF, 0xFF, 0xFF])],
        }
    }

    /// Sample at `t` in `[0, 1]`, returning **linear-light** RGB.
    pub fn sample(self, t: f32, interp: Interpolation) -> Vec3 {
        let stops = self.stops();
        let t = t.clamp(0.0, 1.0);
        let mut i = 0;
        while i + 2 < stops.len() && stops[i + 1].0 < t {
            i += 1;
        }
        let (t0, c0) = stops[i];
        let (t1, c1) = stops[i + 1];
        let u = if (t1 - t0).abs() < 1e-9 {
            0.0
        } else {
            (t - t0) / (t1 - t0)
        };
        blend(c0, c1, u.clamp(0.0, 1.0), interp)
    }

    /// Build the 256-entry LUT in linear light.
    ///
    /// Sequential maps get their luminance repaired to be non-decreasing (see
    /// [`enforce_monotone_luminance`]); the repair is normally a no-op, and the
    /// tests assert it never has to move a value far.
    pub fn lut(self, interp: Interpolation) -> Vec<Vec3> {
        let mut lut: Vec<Vec3> = (0..LUT_SIZE)
            .map(|i| self.sample(i as f32 / (LUT_SIZE - 1) as f32, interp))
            .collect();
        if self.kind() == MapKind::Sequential && self.luminance_monotonic() {
            enforce_monotone_luminance(&mut lut);
        }
        lut
    }
}

/// Relative luminance of a linear-light colour (Rec. 709 / sRGB primaries).
///
/// This — not the sRGB-encoded average, and not Oklab `L` — is what determines
/// whether a map still carries its information when printed in greyscale or
/// seen by a dichromat.
pub fn luminance(c: Vec3) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

/// Force luminance to be non-decreasing by scaling offending entries up.
///
/// A uniform scale of linear RGB changes luminance proportionally and leaves
/// chromaticity untouched, so this repairs the ramp without shifting hue. It
/// exists because an anchor reconstruction can dip by a fraction of a percent
/// between stops, and a *strictly* monotonic guarantee is what the rest of the
/// system (and the accessibility claim) is allowed to rely on.
pub fn enforce_monotone_luminance(lut: &mut [Vec3]) -> f32 {
    let mut worst = 0.0f32;
    for i in 1..lut.len() {
        let prev = luminance(lut[i - 1]);
        let cur = luminance(lut[i]);
        if cur < prev {
            worst = worst.max(prev - cur);
            let s = if cur > 1e-6 { prev / cur } else { 1.0 };
            lut[i] = (lut[i] * s).min(Vec3::ONE);
            // The clamp above can undershoot at the very top of the ramp; in
            // that case flatten onto the previous entry rather than dipping.
            if luminance(lut[i]) < prev - 1e-5 {
                lut[i] = lut[i - 1];
            }
        }
    }
    worst
}

fn blend(a: [u8; 3], b: [u8; 3], u: f32, interp: Interpolation) -> Vec3 {
    match interp {
        Interpolation::EncodedSrgb => {
            let a = Vec3::new(a[0] as f32, a[1] as f32, a[2] as f32) / 255.0;
            let b = Vec3::new(b[0] as f32, b[1] as f32, b[2] as f32) / 255.0;
            srgb_to_linear(a.lerp(b, u))
        }
        Interpolation::LinearLight => srgb8_to_linear(a).lerp(srgb8_to_linear(b), u),
        Interpolation::Oklab => {
            let la = linear_to_oklab(srgb8_to_linear(a));
            let lb = linear_to_oklab(srgb8_to_linear(b));
            oklab_to_linear(la.lerp(lb, u))
        }
    }
}

// -- colour space conversions ------------------------------------------------

pub fn srgb8_to_linear(c: [u8; 3]) -> Vec3 {
    srgb_to_linear(Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32) / 255.0)
}

/// The real piecewise sRGB transfer function, not `pow(x, 2.2)`. The linear toe
/// near black is exactly the part that matters for a map whose first entry is
/// `#000004`.
pub fn srgb_to_linear(c: Vec3) -> Vec3 {
    Vec3::new(
        srgb_to_linear1(c.x),
        srgb_to_linear1(c.y),
        srgb_to_linear1(c.z),
    )
}

pub fn linear_to_srgb(c: Vec3) -> Vec3 {
    Vec3::new(
        linear_to_srgb1(c.x),
        linear_to_srgb1(c.y),
        linear_to_srgb1(c.z),
    )
}

fn srgb_to_linear1(x: f32) -> f32 {
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb1(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    if x <= 0.003_130_8 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

/// Linear sRGB to Oklab (Björn Ottosson, 2020). `x` is `L`, `y`/`z` are `a`/`b`.
pub fn linear_to_oklab(c: Vec3) -> Vec3 {
    let l = 0.412_221_47 * c.x + 0.536_332_55 * c.y + 0.051_445_995 * c.z;
    let m = 0.211_903_5 * c.x + 0.680_699_5 * c.y + 0.107_396_96 * c.z;
    let s = 0.088_302_46 * c.x + 0.281_718_85 * c.y + 0.629_978_5 * c.z;
    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();
    Vec3::new(
        0.210_454_26 * l_ + 0.793_617_8 * m_ - 0.004_072_047 * s_,
        1.977_998_5 * l_ - 2.428_592_2 * m_ + 0.450_593_7 * s_,
        0.025_904_037 * l_ + 0.782_771_77 * m_ - 0.808_675_77 * s_,
    )
}

pub fn oklab_to_linear(c: Vec3) -> Vec3 {
    let l_ = c.x + 0.396_337_78 * c.y + 0.215_803_76 * c.z;
    let m_ = c.x - 0.105_561_346 * c.y - 0.063_854_17 * c.z;
    let s_ = c.x - 0.089_484_18 * c.y - 1.291_485_5 * c.z;
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;
    Vec3::new(
        4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
        -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s,
        -0.004_196_086 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    )
    .max(Vec3::ZERO)
}

// -- half-float packing ------------------------------------------------------

/// IEEE binary16 encoding, round-to-nearest-even, with subnormal support.
///
/// Written out rather than pulled in from a crate because it is twenty lines,
/// and because the LUT upload path is the one place where getting the subnormal
/// range wrong would silently flatten the darkest few entries of every
/// black-based volume map to zero.
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 0xFF {
        // Inf or NaN. Preserve NaN-ness by keeping a non-zero mantissa.
        return sign | 0x7C00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let unbiased = exp - 127 + 15;
    if unbiased >= 0x1F {
        return sign | 0x7C00; // overflow to infinity
    }
    if unbiased <= 0 {
        if unbiased < -10 {
            return sign; // underflows even the subnormal range
        }
        // Subnormal: restore the implicit leading 1 and shift it down.
        let m = mant | 0x0080_0000;
        let shift = (14 - unbiased) as u32;
        let half = (m >> shift) as u16;
        let round = (m >> (shift - 1)) & 1;
        let sticky = (m & ((1 << (shift - 1)) - 1)) != 0;
        return sign | (half + u16::from(round == 1 && (sticky || (half & 1) == 1)));
    }
    let half = (unbiased as u16) << 10 | (mant >> 13) as u16;
    let round = (mant >> 12) & 1;
    let sticky = (mant & 0x0FFF) != 0;
    sign | (half + u16::from(round == 1 && (sticky || (half & 1) == 1)))
}

/// Inverse of [`f32_to_f16_bits`]; used by the tests and by texture readback.
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x03FF) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal halves are exactly `mant * 2^-24`, and f32 represents every
        // one of them exactly, so no renormalisation loop is needed.
        let v = mant as f32 * (1.0 / 16_777_216.0);
        return if sign != 0 { -v } else { v };
    }
    if exp == 0x1F {
        return f32::from_bits(sign | 0x7F80_0000 | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
}

/// Pack an RGBA LUT into `Rgba16Float` texel bytes, ready for `write_texture`.
pub fn pack_rgba16f(entries: &[[f32; 4]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(entries.len() * 8);
    for e in entries {
        for c in e {
            out.extend_from_slice(&f32_to_f16_bits(*c).to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    /// No default anywhere in the renderer may be a rainbow map.
    ///
    /// Rainbow maps are not perceptually uniform, their non-monotonic luminance
    /// invents contrast the data does not contain, and they are not
    /// colourblind-safe. They stay selectable, because a user who wants one
    /// should be able to have one, but the tool must never choose one by itself.
    #[test]
    fn no_default_colour_map_is_a_rainbow() {
        let defaults = [
            (
                "volume transfer function",
                crate::transfer::TransferFunction::default().map,
            ),
            (
                "streaklines",
                crate::particles::StreaklineSettings::default().color_map,
            ),
            (
                "isosurface",
                crate::isosurface::IsosurfaceSettings::default().color_map,
            ),
            ("slice", crate::slice::SliceSettings::default().color_map),
        ];
        for (what, map) in defaults {
            assert_ne!(map, ColorMap::Turbo, "{what} defaults to a rainbow map");
        }
    }

    use super::*;

    #[test]
    fn sequential_maps_have_monotonic_luminance() {
        // The single property that separates a colour map from a decoration.
        for map in ColorMap::ALL {
            if map.kind() != MapKind::Sequential || !map.luminance_monotonic() {
                continue;
            }
            let lut = map.lut(Interpolation::Oklab);
            for i in 1..lut.len() {
                let a = luminance(lut[i - 1]);
                let b = luminance(lut[i]);
                assert!(
                    b >= a - 1e-6,
                    "{} dips in luminance at {i}: {a} -> {b}",
                    map.name()
                );
            }
            assert!(
                luminance(lut[lut.len() - 1]) > luminance(lut[0]) + 0.3,
                "{} is flat",
                map.name()
            );
        }
    }

    #[test]
    fn the_monotonicity_repair_is_essentially_a_no_op() {
        // If this starts failing, the anchor table has drifted somewhere real
        // and the repair is papering over it rather than polishing it.
        for map in ColorMap::ALL {
            if map.kind() != MapKind::Sequential || !map.luminance_monotonic() {
                continue;
            }
            let mut raw: Vec<Vec3> = (0..LUT_SIZE)
                .map(|i| map.sample(i as f32 / (LUT_SIZE - 1) as f32, Interpolation::Oklab))
                .collect();
            let worst = enforce_monotone_luminance(&mut raw);
            assert!(
                worst < 0.02,
                "{} needed a {worst} luminance repair",
                map.name()
            );
        }
    }

    #[test]
    fn volume_maps_start_at_black() {
        // Low values must be dark as well as transparent, or the volume
        // renders as milk.
        for map in ColorMap::VOLUME {
            let lut = map.lut(Interpolation::Oklab);
            assert!(
                luminance(lut[0]) < 0.005,
                "{} does not start black",
                map.name()
            );
        }
    }

    #[test]
    fn diverging_maps_are_light_in_the_middle_and_dark_at_both_ends() {
        for map in ColorMap::SIGNED {
            let lut = map.lut(Interpolation::Oklab);
            let mid = luminance(lut[LUT_SIZE / 2]);
            let lo = luminance(lut[0]);
            let hi = luminance(lut[LUT_SIZE - 1]);
            assert!(
                mid > lo + 0.2 && mid > hi + 0.2,
                "{} is not diverging",
                map.name()
            );
            // Equal-ish ends: an asymmetric diverging map makes one sign of the
            // pressure deviation look more important than the other.
            assert!(
                (lo - hi).abs() < 0.12,
                "{} ends differ in luminance: {lo} vs {hi}",
                map.name()
            );
            assert!(map.wants_symmetric_range());
        }
    }

    #[test]
    fn srgb_space_interpolation_is_measurably_darker_than_linear_light() {
        // Demonstrates the bug the module comment describes. Blue to yellow is
        // the worst case because both endpoints are dark in one channel.
        let a = [0x00, 0x00, 0xFF];
        let b = [0xFF, 0xFF, 0x00];
        let wrong = luminance(blend(a, b, 0.5, Interpolation::EncodedSrgb));
        let right = luminance(blend(a, b, 0.5, Interpolation::LinearLight));
        assert!(
            wrong < right * 0.75,
            "expected sRGB-space lerp to be much darker: {wrong} vs {right}"
        );
    }

    #[test]
    fn oklab_round_trips() {
        for c in [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.2, 0.5, 0.8),
            Vec3::new(0.94, 0.02, 0.31),
        ] {
            let back = oklab_to_linear(linear_to_oklab(c));
            assert!(
                (back - c).abs().max_element() < 1e-4,
                "{c} round-tripped to {back}"
            );
        }
    }

    #[test]
    fn srgb_transfer_round_trips() {
        for i in 0..=255u32 {
            let x = i as f32 / 255.0;
            let back = linear_to_srgb1(srgb_to_linear1(x));
            assert!((back - x).abs() < 1e-5, "{x} round-tripped to {back}");
        }
    }

    #[test]
    fn lut_endpoints_match_the_anchor_stops() {
        for map in ColorMap::ALL {
            let lut = map.lut(Interpolation::Oklab);
            let stops = map.stops();
            let first = srgb8_to_linear(stops[0].1);
            let last = srgb8_to_linear(stops[stops.len() - 1].1);
            assert!(
                (lut[0] - first).abs().max_element() < 1e-3,
                "{} start",
                map.name()
            );
            assert!(
                (lut[LUT_SIZE - 1] - last).abs().max_element() < 2e-2,
                "{} end: {} vs {}",
                map.name(),
                lut[LUT_SIZE - 1],
                last
            );
        }
    }

    #[test]
    fn accessibility_substitution_keeps_the_map_kind_and_is_a_fixed_point() {
        for map in ColorMap::ALL {
            let sub = map.accessible_substitute();
            assert!(
                sub.cvd_safe(),
                "{} substituted to an unsafe map",
                map.name()
            );
            assert_eq!(sub.kind(), map.kind(), "{} changed kind", map.name());
            // Applying it twice must not keep changing the map.
            assert_eq!(sub.accessible_substitute(), sub);
        }
        assert_eq!(ColorMap::Turbo.accessible_substitute(), ColorMap::Viridis);
        assert!(!ColorMap::Turbo.cvd_safe());
        assert!(ColorMap::Turbo.label().contains("not colourblind-safe"));
    }

    #[test]
    fn out_of_range_colours_are_far_from_every_map_entry() {
        // A clamp colour that sits inside the ramp is worse than none at all.
        // Turbo is excluded: it sweeps most of the hue circle, so *no* marker
        // is unambiguous against it. That is a property of turbo, and one more
        // reason the label says what it says.
        for marker in [ColorMap::UNDER, ColorMap::OVER] {
            let m = linear_to_oklab(srgb8_to_linear(marker));
            for map in ColorMap::ALL.iter().copied().filter(|m| m.cvd_safe()) {
                let mut nearest = f32::INFINITY;
                for c in map.lut(Interpolation::Oklab) {
                    nearest = nearest.min((linear_to_oklab(c) - m).length());
                }
                assert!(
                    nearest > 0.10,
                    "{marker:?} is only {nearest} from {}",
                    map.name()
                );
            }
        }
    }

    #[test]
    fn half_float_round_trips_across_the_useful_range() {
        for x in [
            0.0f32,
            1.0,
            0.5,
            -0.25,
            65504.0,
            6.1e-5,  // smallest normal
            5.96e-8, // smallest subnormal
            1.0 / 3.0,
            1234.0,
        ] {
            let back = f16_bits_to_f32(f32_to_f16_bits(x));
            let tol = (x.abs() * 1e-3).max(6e-8);
            assert!((back - x).abs() <= tol, "{x} round-tripped to {back}");
        }
        assert_eq!(
            f16_bits_to_f32(f32_to_f16_bits(f32::INFINITY)),
            f32::INFINITY
        );
        assert!(f16_bits_to_f32(f32_to_f16_bits(f32::NAN)).is_nan());
        // The dark end of inferno must survive the trip to f16.
        let dark = srgb8_to_linear([0x00, 0x00, 0x04]);
        let back = f16_bits_to_f32(f32_to_f16_bits(dark.z));
        assert!(back > 0.0, "darkest inferno entry flushed to zero");
    }

    #[test]
    fn packed_lut_has_the_expected_byte_length() {
        let entries: Vec<[f32; 4]> = vec![[0.25, 0.5, 0.75, 1.0]; LUT_SIZE];
        let bytes = pack_rgba16f(&entries);
        assert_eq!(bytes.len(), LUT_SIZE * 8);
        let first = u16::from_le_bytes([bytes[0], bytes[1]]);
        assert!((f16_bits_to_f32(first) - 0.25).abs() < 1e-4);
    }
}
