//! Line-integral convolution: the kernel, the animation, and the centreline
//! spline that the "slice follows the duct" mode rides on.
//!
//! This module is deliberately GPU-free. Everything here is the arithmetic that
//! `shaders/render/lic.wgsl` performs, written once in Rust so it can be tested
//! without an adapter, and mirrored there. [`slice`](crate::slice) is the only
//! consumer.
//!
//! # Why LIC at all, and why animated
//!
//! A colour-mapped slice tells you *how fast* the flow is. It tells you nothing
//! about *where it is going*, and in a bend that is the entire question. LIC
//! smears a noise field along the local streamlines, so the texture itself
//! becomes the flow direction — without the sampling bias of a hand-placed
//! streamline seed set, and at a cost that scales with pixels rather than with
//! seeds.
//!
//! Static LIC has one fatal ambiguity: it shows the streamline *axis* but not
//! which way along it the fluid travels. Left and right look identical. The fix
//! is to make the convolution kernel a travelling wave,
//!
//! ```text
//! k(s, t) = 0.5 * (1 + cos(2*pi*(s/L - phase(t))))
//! ```
//!
//! with `phase` advancing linearly in time. Bright bands then march along each
//! streamline in the direction of flow, which reads instantly and correctly.
//!
//! # Why the convolution is normalised
//!
//! The raw kernel weights do **not** sum to a phase-independent constant. For
//! `2L + 1` symmetric taps the sum is `0.5 * (2L + 1 + cos(2*pi*phase))`, so an
//! unnormalised convolution makes the whole image pulse in brightness once per
//! animation period — which looks exactly like a flickering exposure bug, and is
//! the sort of thing people spend an hour blaming on the tonemapper. Dividing by
//! the realised weight sum removes it exactly, and has the additional property
//! that a constant input returns that constant, so flat regions stay flat.
//!
//! # The honesty problem
//!
//! LIC on a plane can only show the **in-plane** component of the velocity. In a
//! 90-degree bend a large fraction of the flow pierces the plane, and a
//! confident-looking swirl texture drawn from a 10%-of-magnitude in-plane
//! residue is a lie told beautifully. [`LicSettings::honesty`] multiplies the LIC
//! contrast by `|u_in_plane| / |u|` so the texture fades to flat colour exactly
//! where the flow is leaving the plane. It costs nothing and it is the
//! difference between a figure you can publish and one you cannot.

use glam::Vec3;

/// Default half-length of the convolution, in steps. The useful band is 20-40:
/// shorter and the texture is noise, longer and every streamline blurs into its
/// neighbours and the image goes grey.
pub const DEFAULT_STEPS: u32 = 28;

/// Hard cap, mirrored in `lic.wgsl` as the loop bound. A fixed bound keeps the
/// shader's loop trip count uniform enough for the compiler to unroll, and stops
/// a bad uniform from hanging the GPU.
pub const MAX_STEPS: u32 = 64;

/// The travelling-ramp kernel, evaluated at normalised arc position
/// `s_over_l = s / L` in `[-1, 1]`.
///
/// Non-negative everywhere, unit peak, and periodic in `phase` with period 1.
#[inline]
pub fn ramp_kernel(s_over_l: f32, phase: f32) -> f32 {
    0.5 * (1.0 + (std::f32::consts::TAU * (s_over_l - phase)).cos())
}

/// Normalised convolution of `2L + 1` taps, ordered from `s = -L` to `s = +L`.
///
/// Returns the weighted mean, which is what keeps the image from pulsing with
/// the animation phase. An empty tap list returns 0.
pub fn convolve_normalised(taps: &[f32], phase: f32) -> f32 {
    let n = taps.len();
    if n < 2 {
        return taps.first().copied().unwrap_or(0.0);
    }
    let half = (n - 1) as f32 * 0.5;
    let mut acc = 0.0f32;
    let mut wsum = 0.0f32;
    for (i, v) in taps.iter().enumerate() {
        let s = (i as f32 - half) / half;
        let w = ramp_kernel(s, phase);
        acc += w * v;
        wsum += w;
    }
    if wsum <= 1e-20 {
        return 0.0;
    }
    acc / wsum
}

/// Independent noise cells the convolution actually averages over.
///
/// Taps closer together than one noise cell are the *same* sample, so the count
/// that matters is the arc length covered divided by the cell size, capped at
/// the number of taps. Mirrors `lic_effective_taps` in `lic.wgsl`.
///
/// `taps` is what the convolution **realised**, not what was configured. The
/// two diverge wherever the streamline is short — against a wall, at a
/// stagnation point, in the still air outside the duct — and applying a
/// long-convolution gain to a handful of samples binarises them into a stark
/// lattice. Those are precisely the regions worth looking at, so the
/// distinction is not a detail.
pub fn effective_taps(taps: f32, step_mm: f32, noise_scale_mm: f32) -> f32 {
    let n = taps.max(1.0);
    let covered = n * step_mm / noise_scale_mm.max(1e-4);
    covered.clamp(1.0, n)
}

/// Gain that restores usable contrast after the convolution.
///
/// This is the step everyone leaves out of a first LIC implementation, and its
/// absence looks like a broken shader rather than a missing scale factor. The
/// mean of `m` independent uniform(0,1) samples has standard deviation
/// `1 / sqrt(12 m)`; at the default 57 taps that is **0.038**, so the raw image
/// is a flat grey with a barely perceptible grain. Scaling so two standard
/// deviations fill half the range gives `sqrt(0.75 m)`, which puts essentially
/// the whole distribution inside `[0, 1]` with only the extreme tails clipping.
///
/// Mirrors `lic_contrast_gain` in `lic.wgsl`.
pub fn contrast_gain(effective_taps: f32) -> f32 {
    (0.75 * effective_taps.max(0.0)).sqrt().max(1.0)
}

/// Apply the gain about the mid-grey the convolution converges to.
pub fn apply_gain(raw: f32, gain: f32) -> f32 {
    (0.5 + (raw - 0.5) * gain).clamp(0.0, 1.0)
}

/// LIC controls. Shared by the CPU reference and the GPU uniform.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LicSettings {
    /// Half-length of the convolution in integration steps, each way. Clamped to
    /// [`MAX_STEPS`].
    pub steps: u32,
    /// Integration step along the streamline, in millimetres. Roughly one
    /// derived voxel is right: shorter oversamples a trilinear field, longer
    /// starts cutting corners on a curved streamline.
    pub step_mm: f32,
    /// Animation rate in cycles per second. Zero freezes the pattern, which is
    /// what a still screenshot wants.
    pub cycles_per_second: f32,
    /// How far the LIC modulates the colour: the composite is
    /// `colormap(scalar) * ((1 - contrast) + contrast * lic)`. 0.4 keeps the
    /// colour quantitative while the texture stays legible.
    pub contrast: f32,
    /// Millimetres per noise texel. Below about one voxel the texture aliases
    /// under camera motion; above about five it reads as blobs, not as fibres.
    pub noise_scale_mm: f32,
    /// Fade the LIC out where the flow pierces the plane. See the module docs;
    /// leave this on.
    pub honesty: bool,
    /// Exponent on `|u_in_plane| / |u|`. 1.0 is the honest linear weighting;
    /// higher is more aggressive about hiding out-of-plane regions.
    pub honesty_power: f32,
}

impl Default for LicSettings {
    fn default() -> Self {
        Self {
            steps: DEFAULT_STEPS,
            step_mm: 0.75,
            cycles_per_second: 0.35,
            contrast: 0.4,
            noise_scale_mm: 1.5,
            honesty: true,
            honesty_power: 1.0,
        }
    }
}

impl LicSettings {
    pub fn sanitise(&mut self) {
        self.steps = self.steps.clamp(2, MAX_STEPS);
        self.step_mm = self.step_mm.clamp(1.0e-3, 100.0);
        self.contrast = self.contrast.clamp(0.0, 1.0);
        self.noise_scale_mm = self.noise_scale_mm.max(1.0e-3);
        self.honesty_power = self.honesty_power.clamp(0.0, 8.0);
        if !self.cycles_per_second.is_finite() {
            self.cycles_per_second = 0.0;
        }
    }

    /// Animation phase at time `t` seconds. Wrapped into `[0, 1)` on the CPU so
    /// the shader never sees a large float whose fractional part has lost
    /// precision — after an hour at 60 fps a raw `t * rate` has only a few bits
    /// of phase left, and the animation visibly ratchets.
    pub fn phase_at(&self, t: f32) -> f32 {
        (t * self.cycles_per_second).rem_euclid(1.0)
    }
}

// -- centreline --------------------------------------------------------------

/// A Catmull-Rom spline through caller-supplied points, parameterised by arc
/// length.
///
/// Used by the "slice follows the duct centreline" mode: the plane normal is the
/// spline tangent, so dragging one slider walks a plane down the passage staying
/// perpendicular to it. For a bent duct that is far more informative than any
/// axis-aligned cut, because an axis-aligned cut through a 90-degree bend is
/// oblique to the flow over most of its area and every velocity it shows is
/// foreshortened by an angle that changes across the image.
///
/// Centripetal parameterisation (`alpha = 0.5`), for the same reason as
/// [`crate::camera::Flythrough`]: the uniform form cusps whenever two control
/// points are close together, and a cusp in a centreline means the plane normal
/// swings through a large angle over a millimetre of arc.
#[derive(Debug, Clone)]
pub struct Centreline {
    control: Vec<Vec3>,
    /// `(cumulative arc mm, point, unit tangent)`, densely resampled.
    table: Vec<(f32, Vec3, Vec3)>,
}

impl Centreline {
    /// Samples generated per spline segment when building the arc-length table.
    /// 32 keeps the arc-length error under a tenth of a percent for any duct
    /// bend anyone will draw, and the table is built once.
    const PER_SEGMENT: usize = 32;

    /// Build from at least two control points. Returns `None` otherwise, so the
    /// caller can fall back to an axis-aligned slice rather than panicking on an
    /// empty picking session.
    pub fn new(points: &[Vec3]) -> Option<Self> {
        if points.len() < 2 {
            return None;
        }
        let control: Vec<Vec3> = points.to_vec();
        let n = control.len();
        let mut table: Vec<(f32, Vec3, Vec3)> = Vec::with_capacity((n - 1) * Self::PER_SEGMENT + 1);

        let mut arc = 0.0f32;
        let mut prev: Option<Vec3> = None;
        for seg in 0..n - 1 {
            let p0 = control[seg.saturating_sub(1)];
            let p1 = control[seg];
            let p2 = control[seg + 1];
            let p3 = control[(seg + 2).min(n - 1)];
            // The last segment includes its endpoint; the others stop one short
            // so interior control points are not duplicated in the table.
            let last = seg == n - 2;
            let steps = Self::PER_SEGMENT + usize::from(last);
            for i in 0..steps {
                let u = i as f32 / Self::PER_SEGMENT as f32;
                let p = catmull_rom_centripetal(p0, p1, p2, p3, u);
                if let Some(q) = prev {
                    arc += (p - q).length();
                }
                prev = Some(p);
                table.push((arc, p, Vec3::ZERO));
            }
        }

        // Tangents by central difference on the table, so they are consistent
        // with the arc length rather than with the spline's own parameter.
        let m = table.len();
        for i in 0..m {
            let a = table[i.saturating_sub(1)].1;
            let b = table[(i + 1).min(m - 1)].1;
            let t = (b - a).normalize_or(Vec3::Z);
            table[i].2 = t;
        }

        Some(Self { control, table })
    }

    pub fn control_points(&self) -> &[Vec3] {
        &self.control
    }

    /// Total arc length, millimetres.
    pub fn length(&self) -> f32 {
        self.table.last().map(|e| e.0).unwrap_or(0.0)
    }

    /// Point and unit tangent at arc length `s` mm, clamped to the ends.
    pub fn sample(&self, s: f32) -> (Vec3, Vec3) {
        if self.table.is_empty() {
            return (Vec3::ZERO, Vec3::Z);
        }
        let total = self.length();
        let s = s.clamp(0.0, total);
        // Binary search: the table is monotone in arc length by construction.
        let mut lo = 0usize;
        let mut hi = self.table.len() - 1;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if self.table[mid].0 <= s {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let (sa, pa, ta) = self.table[lo];
        let (sb, pb, tb) = self.table[hi];
        let d = sb - sa;
        let u = if d.abs() < 1e-9 { 0.0 } else { (s - sa) / d };
        (pa.lerp(pb, u), ta.lerp(tb, u).normalize_or(ta))
    }

    /// Convenience: sample at a normalised position in `[0, 1]`.
    pub fn sample_normalised(&self, t: f32) -> (Vec3, Vec3) {
        self.sample(t.clamp(0.0, 1.0) * self.length())
    }

    /// An orthonormal frame at arc length `s`: `(origin, u, v)` with
    /// `u x v = tangent`, so `(u, v)` spans the cutting plane.
    ///
    /// The frame is built against a fixed world reference rather than by
    /// parallel transport. That gives a stable, reproducible basis for any `s`
    /// (transport would depend on where you started), at the cost of the frame
    /// spinning when the tangent passes near the reference axis — which is why
    /// the reference is swapped when they get close.
    pub fn frame(&self, s: f32) -> (Vec3, Vec3, Vec3) {
        let (p, t) = self.sample(s);
        let reference = if t.y.abs() > 0.9 { Vec3::X } else { Vec3::Y };
        let u = reference.cross(t).normalize_or(Vec3::X);
        let v = t.cross(u).normalize_or(Vec3::Y);
        (p, u, v)
    }
}

/// Centripetal Catmull-Rom on a 4-point stencil, `u` in `[0, 1]` across
/// `p1 -> p2`. Barry-Goldman pyramidal form, which is numerically better behaved
/// than expanding the basis polynomials.
fn catmull_rom_centripetal(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, u: f32) -> Vec3 {
    let knot = |ti: f32, a: Vec3, b: Vec3| ti + (a - b).length().sqrt().max(1e-4);
    let t0 = 0.0;
    let t1 = knot(t0, p1, p0);
    let t2 = knot(t1, p2, p1);
    let t3 = knot(t2, p3, p2);
    let t = t1 + u * (t2 - t1);

    let lerp = |a: Vec3, b: Vec3, ta: f32, tb: f32| -> Vec3 {
        let d = tb - ta;
        if d.abs() < 1e-9 {
            a
        } else {
            a * ((tb - t) / d) + b * ((t - ta) / d)
        }
    };
    let a1 = lerp(p0, p1, t0, t1);
    let a2 = lerp(p1, p2, t1, t2);
    let a3 = lerp(p2, p3, t2, t3);
    let b1 = lerp(a1, a2, t0, t2);
    let b2 = lerp(a2, a3, t1, t3);
    lerp(b1, b2, t1, t2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn the_kernel_is_a_non_negative_travelling_wave() {
        for phase in [0.0f32, 0.17, 0.5, 0.83, 0.999] {
            for i in 0..=200 {
                let s = -1.0 + i as f32 / 100.0;
                let k = ramp_kernel(s, phase);
                assert!((-1e-6..=1.0 + 1e-6).contains(&k), "k({s}, {phase}) = {k}");
            }
            // The peak sits exactly where the wave crest is, which is what makes
            // the animation read as motion along the streamline.
            assert!((ramp_kernel(phase, phase) - 1.0).abs() < 1e-5);
            // ...and it really moves: half a period away is a trough.
            assert!(ramp_kernel(phase + 0.5, phase) < 1e-5);
        }
    }

    #[test]
    fn the_kernel_is_periodic_in_phase() {
        for s in [-1.0f32, -0.3, 0.0, 0.42, 1.0] {
            for phase in [0.0f32, 0.25, 0.7] {
                assert!(
                    (ramp_kernel(s, phase) - ramp_kernel(s, phase + 1.0)).abs() < 1e-5,
                    "not periodic at s={s} phase={phase}"
                );
            }
        }
    }

    #[test]
    fn normalisation_removes_the_phase_pulse() {
        // The bug this exists to prevent: the raw weight sum varies by
        // `cos(2*pi*phase)` across the animation, so an unnormalised convolution
        // makes the entire image breathe once per cycle.
        let n = 2 * DEFAULT_STEPS as usize + 1;
        let taps = vec![0.37f32; n];

        let mut raw_min = f32::INFINITY;
        let mut raw_max = f32::NEG_INFINITY;
        for i in 0..64 {
            let phase = i as f32 / 64.0;
            // Normalised: exactly the input constant, at every phase.
            let got = convolve_normalised(&taps, phase);
            assert!(
                (got - 0.37).abs() < 1e-5,
                "phase {phase}: constant input gave {got}, not 0.37"
            );

            let half = (n - 1) as f32 * 0.5;
            let raw: f32 = (0..n)
                .map(|k| ramp_kernel((k as f32 - half) / half, phase) * 0.37)
                .sum();
            raw_min = raw_min.min(raw);
            raw_max = raw_max.max(raw);
        }
        // And confirm the pulse the normalisation is removing is real, so the
        // test above is not passing vacuously.
        assert!(
            (raw_max - raw_min) / raw_max > 0.01,
            "the unnormalised sum barely moved ({raw_min} .. {raw_max}); the test proves nothing"
        );
    }

    #[test]
    fn convolution_is_a_weighted_mean_and_stays_in_range() {
        // Whatever the phase, the result must lie between the smallest and the
        // largest tap. That is what stops the LIC composite from over- or
        // under-shooting the colour map.
        let taps: Vec<f32> = (0..41).map(|i| ((i * 7919) % 101) as f32 / 100.0).collect();
        let lo = taps.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = taps.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for i in 0..32 {
            let v = convolve_normalised(&taps, i as f32 / 32.0);
            assert!(v >= lo - 1e-5 && v <= hi + 1e-5, "{v} outside [{lo}, {hi}]");
        }
    }

    #[test]
    fn the_gain_turns_a_flat_grey_convolution_into_a_visible_texture() {
        // The failure this exists to prevent: with 57 taps the convolution of
        // white noise has a standard deviation of 0.038, so the plane comes out
        // uniform grey and looks like a shader that is not running. The gain has
        // to bring that back to something that fills the display range.
        let s = LicSettings::default();
        let m = effective_taps((2 * s.steps + 1) as f32, s.step_mm, s.noise_scale_mm);
        let gain = contrast_gain(m);

        // A cheap deterministic white-noise source: one independent sample per
        // noise cell, repeated for the taps that fall inside the same cell.
        let cells_per_tap = s.step_mm / s.noise_scale_mm;
        let mut raw = Vec::new();
        let mut boosted = Vec::new();
        for trial in 0..4000u32 {
            let taps: Vec<f32> = (0..2 * s.steps + 1)
                .map(|k| {
                    let cell = (k as f32 * cells_per_tap).floor() as u32;
                    let h = (trial.wrapping_mul(2654435761) ^ cell.wrapping_mul(40503))
                        .wrapping_mul(2246822519);
                    ((h >> 8) & 0xffff) as f32 / 65535.0
                })
                .collect();
            let v = convolve_normalised(&taps, 0.3);
            raw.push(v);
            boosted.push(apply_gain(v, gain));
        }

        let std_of = |v: &[f32]| {
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / v.len() as f32).sqrt()
        };
        let raw_std = std_of(&raw);
        let out_std = std_of(&boosted);

        assert!(
            raw_std < 0.08,
            "the raw convolution already has contrast {raw_std}; the test proves nothing"
        );
        assert!(
            out_std > 0.15,
            "after the gain the contrast is still only {out_std}; the plane will read as flat grey"
        );
        // ...and not so much that it clips to pure black and white, which throws
        // away the fibre structure the texture exists to show.
        assert!(
            out_std < 0.40,
            "the gain over-drives the texture to binary: {out_std}"
        );
        assert!(boosted.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    #[test]
    fn the_gain_never_dims_and_effective_taps_never_exceed_the_real_ones() {
        // Sub-unit gain would make a short convolution *less* visible than a
        // long one, which is backwards.
        for m in [0.0f32, 0.5, 1.0, 4.0, 57.0, 1000.0] {
            assert!(
                contrast_gain(m) >= 1.0,
                "gain at m={m} is {}",
                contrast_gain(m)
            );
        }
        assert!(contrast_gain(100.0) > contrast_gain(10.0));

        for steps in [2u32, 8, 28, 64] {
            let taps = (2 * steps + 1) as f32;
            // A noise cell far larger than the whole convolution: one sample.
            assert!((effective_taps(taps, 0.1, 1000.0) - 1.0).abs() < 1e-5);
            // A noise cell far finer than the step: every tap independent, and
            // never more than that however fine the noise gets.
            assert_eq!(effective_taps(taps, 1.0, 1e-6), taps);
        }
        // A stalled walk contributes one tap, and one sample must be left
        // alone: this is the case that otherwise turns still air into a stark
        // black-and-white lattice.
        assert_eq!(effective_taps(1.0, 0.75, 1.5), 1.0);
        assert_eq!(contrast_gain(effective_taps(1.0, 0.75, 1.5)), 1.0);
        assert_eq!(
            apply_gain(0.9, contrast_gain(effective_taps(1.0, 0.75, 1.5))),
            0.9
        );
        // Gain of a single sample is 1: nothing to average, nothing to restore.
        assert_eq!(contrast_gain(effective_taps(57.0, 0.1, 1000.0)), 1.0);
        // apply_gain is a no-op at unit gain and stays in range.
        assert_eq!(apply_gain(0.37, 1.0), 0.37);
        assert_eq!(apply_gain(0.9, 100.0), 1.0);
        assert_eq!(apply_gain(0.1, 100.0), 0.0);
    }

    #[test]
    fn phase_wraps_into_the_unit_interval() {
        let s = LicSettings {
            cycles_per_second: 0.35,
            ..Default::default()
        };
        for t in [0.0f32, 1.0, 61.0, 3600.0, 1.0e5] {
            let p = s.phase_at(t);
            assert!((0.0..1.0).contains(&p), "phase {p} at t={t}");
        }
        // A frozen animation must actually be frozen.
        let f = LicSettings {
            cycles_per_second: 0.0,
            ..Default::default()
        };
        assert_eq!(f.phase_at(1234.5), 0.0);
    }

    #[test]
    fn settings_sanitise_into_the_shader_contract() {
        let mut s = LicSettings {
            steps: 9999,
            step_mm: -1.0,
            contrast: 4.0,
            noise_scale_mm: 0.0,
            cycles_per_second: f32::NAN,
            honesty_power: -3.0,
            ..Default::default()
        };
        s.sanitise();
        assert!(s.steps <= MAX_STEPS && s.steps >= 2);
        assert!(s.step_mm > 0.0);
        assert_eq!(s.contrast, 1.0);
        assert!(s.noise_scale_mm > 0.0);
        assert_eq!(s.cycles_per_second, 0.0);
        assert!(s.honesty_power >= 0.0);
    }

    #[test]
    fn a_centreline_passes_through_its_control_points() {
        // The property Catmull-Rom is chosen for: the points you pick are the
        // plane positions you get.
        let pts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(30.0, 5.0, 0.0),
            Vec3::new(55.0, 5.0, 25.0),
            Vec3::new(55.0, 5.0, 60.0),
        ];
        let c = Centreline::new(&pts).unwrap();
        // 4000 samples over ~100 mm is 0.025 mm apart, so the sampling grid can
        // contribute at most 0.013 mm to the closest approach. At 400 samples
        // the grid alone accounts for 0.14 mm and the test measures itself
        // rather than the spline.
        const SAMPLES: u32 = 4000;
        for p in pts {
            // Closest approach of the sampled curve to each control point.
            let mut best = f32::INFINITY;
            for i in 0..=SAMPLES {
                let (q, _) = c.sample_normalised(i as f32 / SAMPLES as f32);
                best = best.min((q - p).length());
            }
            assert!(best < 0.05, "control point {p} is {best} mm off the curve");
        }
    }

    #[test]
    fn arc_length_is_monotone_and_matches_a_straight_line() {
        let straight = Centreline::new(&[
            Vec3::ZERO,
            Vec3::new(50.0, 0.0, 0.0),
            Vec3::new(100.0, 0.0, 0.0),
        ])
        .unwrap();
        assert!(
            (straight.length() - 100.0).abs() < 0.05,
            "straight length was {}",
            straight.length()
        );
        // Sampling at arc s must land s millimetres along.
        for s in [0.0f32, 12.5, 50.0, 99.0] {
            let (p, t) = straight.sample(s);
            assert!((p.x - s).abs() < 0.05, "arc {s} landed at {p}");
            assert!(t.dot(Vec3::X) > 0.999, "tangent {t} is not along the line");
        }

        // Monotone arc parameterisation on a curved path, and a unit tangent
        // everywhere: both are relied on by the binary search in `sample`.
        let bend = Centreline::new(&[
            Vec3::new(55.0, 0.0, 0.0),
            Vec3::new(52.0, 0.0, 20.0),
            Vec3::new(39.0, 0.0, 39.0),
            Vec3::new(20.0, 0.0, 52.0),
            Vec3::new(0.0, 0.0, 55.0),
        ])
        .unwrap();
        let mut last = -1.0f32;
        for i in 0..=200 {
            let s = i as f32 / 200.0 * bend.length();
            let (_, t) = bend.sample(s);
            assert!(
                (t.length() - 1.0).abs() < 1e-3,
                "tangent length {}",
                t.length()
            );
            assert!(s >= last);
            last = s;
        }
        // A quarter circle of radius 55 is 55 * pi / 2 long. The control points
        // are on the circle, so the spline should be within a percent of it.
        let exact = 55.0 * FRAC_PI_2;
        let err = (bend.length() - exact).abs() / exact;
        assert!(
            err < 0.02,
            "bend arc {} vs exact {exact} (rel {err})",
            bend.length()
        );
    }

    #[test]
    fn the_cutting_frame_is_orthonormal_and_spans_the_plane() {
        let c = Centreline::new(&[
            Vec3::new(55.0, 0.0, 0.0),
            Vec3::new(39.0, 0.0, 39.0),
            Vec3::new(0.0, 3.0, 55.0),
        ])
        .unwrap();
        for i in 0..=20 {
            let s = i as f32 / 20.0 * c.length();
            let (_, u, v) = c.frame(s);
            let (_, t) = c.sample(s);
            assert!((u.length() - 1.0).abs() < 1e-4);
            assert!((v.length() - 1.0).abs() < 1e-4);
            assert!(u.dot(v).abs() < 1e-3, "frame is not orthogonal at s={s}");
            // The plane spanned by (u, v) must be the one the tangent pierces.
            assert!(u.dot(t).abs() < 1e-3 && v.dot(t).abs() < 1e-3);
            assert!(u.cross(v).dot(t) > 0.99, "frame is left-handed at s={s}");
        }
    }

    #[test]
    fn a_degenerate_centreline_is_rejected_rather_than_panicking() {
        assert!(Centreline::new(&[]).is_none());
        assert!(Centreline::new(&[Vec3::ZERO]).is_none());
        // Two coincident points are legal input and must not divide by zero.
        let c = Centreline::new(&[Vec3::ZERO, Vec3::ZERO]).unwrap();
        let (p, t) = c.sample(0.5);
        assert!(p.is_finite() && t.is_finite());
    }
}
