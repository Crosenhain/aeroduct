//! The transfer function: value -> (colour, opacity).
//!
//! This is deliberately a *constrained* transfer function, not a research-grade
//! 2D editor. The full 2D histogram widget is a research tool: it is powerful,
//! it is what papers show, and in a design tool it is a trap — every session
//! starts with ten minutes of dragging polygons around before you can see your
//! duct. What people actually want, essentially always, is one of two things:
//!
//! - a *ramp*: low values fade out, high values glow. Four to six control
//!   points on a piecewise-linear opacity curve covers every variation of it.
//! - a *soft isosurface*: "show me where Q-tilde is about 0.5". That is a
//!   Gaussian bump, `alpha(v) = A exp(-((v - v0)/w)^2)`, and it gives the
//!   isosurface look with no marching cubes, no topology, no mesh, and — because
//!   it has a width rather than a hard threshold — no shimmering when the
//!   isolevel grazes a cell.
//!
//! Most users live in the second mode. It is the default for Q-criterion.
//!
//! # Where the units live
//!
//! The **soft isosurface centre and width are in data units**, not in normalised
//! `[0, 1]`. This is the whole point: if `v0` were normalised, re-ranging the
//! colour bar would silently move the isosurface, which is the same class of
//! usability failure that `Q_tilde` normalisation exists to fix. The
//! piecewise-linear curve *is* in normalised space, because a ramp is a
//! statement about the displayed range by definition.

use glam::Vec3;

use crate::colormap::{ColorMap, Interpolation, LUT_SIZE};

/// Opacity below this counts as "nothing here" for empty-space skipping. It is
/// the contract between the transfer function and [`crate::accel`]: a brick
/// whose values all map below this may be skipped entirely, so if this value
/// were too coarse the accelerator would visibly erase faint structure.
pub const SUPPORT_EPSILON: f32 = 1.0e-3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RangeScale {
    #[default]
    Linear,
    /// Log10. Useful for speed fields in a duct, where the interesting
    /// separated region is two decades below the core jet.
    Log,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OpacityMode {
    /// Piecewise-linear ramp over the normalised range.
    Curve,
    /// Gaussian bump in data units. The default for Q-criterion.
    #[default]
    SoftIso,
}

/// A piecewise-linear opacity ramp with 2-6 control points.
///
/// `points` is public and is plain `[t, alpha]` data on purpose: the UI crate
/// binds a drag-editor straight to it. Anything that reads it should call
/// [`OpacityCurve::sanitise`] first rather than assuming sortedness.
#[derive(Debug, Clone, PartialEq)]
pub struct OpacityCurve {
    pub points: Vec<[f32; 2]>,
}

/// Hard cap on control points. Six is enough for every ramp anyone actually
/// draws, and capping it keeps the GPU-side representation a fixed size.
pub const MAX_CURVE_POINTS: usize = 6;

impl Default for OpacityCurve {
    fn default() -> Self {
        // A gentle ramp: transparent through the low quarter, then rising.
        // Starting at exactly zero is not cosmetic — see `diverging` below.
        Self { points: vec![[0.0, 0.0], [0.25, 0.0], [0.65, 0.25], [1.0, 0.85]] }
    }
}

impl OpacityCurve {
    /// A V-shaped ramp: opaque at both ends, transparent in the middle.
    ///
    /// The right curve for **signed** data, and getting it wrong is one of the
    /// most convincing-looking failures in the whole renderer. A duct sits in a
    /// domain that is overwhelmingly still air, where the pressure deviation is
    /// zero — the exact middle of a symmetric range. Drive that through a
    /// sequential ramp and the middle of the ramp has, say, 15% opacity per
    /// step; over a few hundred raymarch steps that composites to fully opaque,
    /// in the pale midpoint colour of the diverging map. The result is a solid
    /// white box the size of the domain, and because it is smoothly shaded it
    /// looks like a render, not like a bug.
    pub fn diverging() -> Self {
        Self::new(vec![[0.0, 0.85], [0.36, 0.0], [0.64, 0.0], [1.0, 0.85]])
    }
}

impl OpacityCurve {
    pub fn new(points: Vec<[f32; 2]>) -> Self {
        let mut c = Self { points };
        c.sanitise();
        c
    }

    /// Sort by `t`, clamp into range, and enforce the point-count bounds.
    pub fn sanitise(&mut self) {
        for p in &mut self.points {
            p[0] = p[0].clamp(0.0, 1.0);
            p[1] = p[1].clamp(0.0, 1.0);
        }
        self.points.sort_by(|a, b| a[0].total_cmp(&b[0]));
        self.points.truncate(MAX_CURVE_POINTS);
        while self.points.len() < 2 {
            self.points.push([1.0, 1.0]);
        }
    }

    /// Evaluate at normalised `t`, clamping at both ends.
    pub fn eval(&self, t: f32) -> f32 {
        let p = &self.points;
        if p.is_empty() {
            return 0.0;
        }
        let t = t.clamp(0.0, 1.0);
        if t <= p[0][0] {
            return p[0][1];
        }
        if t >= p[p.len() - 1][0] {
            return p[p.len() - 1][1];
        }
        for w in p.windows(2) {
            let (a, b) = (w[0], w[1]);
            if t >= a[0] && t <= b[0] {
                let d = b[0] - a[0];
                let u = if d.abs() < 1e-9 { 0.0 } else { (t - a[0]) / d };
                return a[1] + (b[1] - a[1]) * u;
            }
        }
        p[p.len() - 1][1]
    }

    pub fn try_insert(&mut self, t: f32, alpha: f32) -> bool {
        if self.points.len() >= MAX_CURVE_POINTS {
            return false;
        }
        self.points.push([t, alpha]);
        self.sanitise();
        true
    }

    pub fn try_remove(&mut self, index: usize) -> bool {
        if self.points.len() <= 2 || index >= self.points.len() {
            return false;
        }
        self.points.remove(index);
        true
    }
}

/// `alpha(v) = amplitude * exp(-((v - center) / width)^2)`, in **data units**,
/// truncated to compact support.
///
/// # Why the truncation is not optional
///
/// A pure Gaussian never reaches zero. That sounds harmless and is not, because
/// the raymarch integrates along a path hundreds of steps long: an opacity of
/// 1.5% per step compounds to `1 - 0.985^200 = 95%` — fully opaque. The domain
/// around a duct is mostly *still air*, where `Q~` and `|u|` are exactly zero,
/// so a bump centred at `Q~ = 0.5` with a 0.25 shoulder still has 1.6% opacity
/// at zero and renders the entire 260 x 180 x 180 mm box as a solid black
/// brick. It looks exactly like a broken raymarcher.
///
/// So the bump is truncated at [`SoftIso::cutoff_widths`] and a pedestal is
/// subtracted so it reaches zero continuously rather than stepping:
///
/// ```text
/// alpha(v) = A * (exp(-x^2) - exp(-n^2)) / (1 - exp(-n^2))   for |x| < n
///          = 0                                               otherwise
/// ```
///
/// The default `n = 2` places the cutoff of the default preset exactly at
/// `Q~ = 0`, which is also the physically right place: `Q <= 0` is
/// strain-dominated and is not a vortex. Compact support is also what lets
/// [`crate::accel`] skip the empty 98% of the domain at all.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoftIso {
    pub center: f32,
    /// Shoulder width. Opacity falls to roughly `1/e` of the peak one width out.
    pub width: f32,
    pub amplitude: f32,
    /// Truncation radius in units of `width`. Below about 3 the truncation is
    /// what keeps the tail from fogging the domain; above it, it barely bites.
    pub cutoff_widths: f32,
}

impl Default for SoftIso {
    fn default() -> Self {
        // Q-tilde 0.5 with a 0.25 shoulder: right in the middle of the useful
        // 0.1-2 band, so the default view of a fresh sim shows vortex cores —
        // and with n = 2 the support is exactly [0, 1], so quiescent air is
        // exactly transparent.
        Self { center: 0.5, width: 0.25, amplitude: 0.9, cutoff_widths: 2.0 }
    }
}

impl SoftIso {
    pub fn eval(&self, v: f32) -> f32 {
        if self.width.abs() < 1e-9 {
            return 0.0;
        }
        let n = self.cutoff_widths.max(0.5);
        let x = (v - self.center) / self.width;
        if x.abs() >= n {
            return 0.0;
        }
        let pedestal = (-(n * n)).exp();
        self.amplitude * ((-(x * x)).exp() - pedestal) / (1.0 - pedestal)
    }

    /// Half-width beyond which opacity is exactly zero, or below `eps`. `None`
    /// when the bump never rises above `eps` at all.
    pub fn half_support(&self, eps: f32) -> Option<f32> {
        if self.amplitude <= eps {
            return None;
        }
        let n = self.cutoff_widths.max(0.5);
        let pedestal = (-(n * n)).exp();
        // Solve A * (exp(-x^2) - p) / (1 - p) = eps for x, then take whichever
        // is tighter: that or the hard cutoff.
        let target = eps * (1.0 - pedestal) / self.amplitude + pedestal;
        let from_eps = if target >= 1.0 {
            0.0
        } else {
            (-target.max(1e-30).ln()).max(0.0).sqrt()
        };
        Some(self.width.abs() * from_eps.min(n))
    }
}

/// The whole user-facing transfer function.
#[derive(Debug, Clone, PartialEq)]
pub struct TransferFunction {
    pub map: ColorMap,
    pub interpolation: Interpolation,
    /// `[lo, hi]` in data units.
    pub range: [f32; 2],
    pub scale: RangeScale,
    /// Force `range` symmetric about zero. Defaults to *on* for diverging maps;
    /// see the module docs in [`crate::colormap`] for why that is not optional
    /// in practice.
    pub symmetric_lock: bool,
    /// Global opacity multiplier, applied after the curve or the bump.
    pub density: f32,
    pub mode: OpacityMode,
    pub curve: OpacityCurve,
    pub iso: SoftIso,
    /// Paint values outside `range` in the clamp colours instead of clamping
    /// silently, so a user can see they are clipping.
    pub show_clamp_colors: bool,
}

impl Default for TransferFunction {
    fn default() -> Self {
        Self {
            map: ColorMap::Inferno,
            interpolation: Interpolation::default(),
            range: [0.0, 1.0],
            scale: RangeScale::Linear,
            symmetric_lock: false,
            density: 1.0,
            mode: OpacityMode::default(),
            curve: OpacityCurve::default(),
            iso: SoftIso::default(),
            show_clamp_colors: false,
        }
    }
}

impl TransferFunction {
    /// A sensible starting point for a scalar named `field`, with the data
    /// range it is expected to occupy.
    pub fn preset(field: crate::fields::DerivedField) -> Self {
        use crate::fields::DerivedField;
        match field {
            DerivedField::Speed => Self {
                map: ColorMap::Inferno,
                range: [0.0, 10.0],
                mode: OpacityMode::Curve,
                ..Default::default()
            },
            // The interesting one. Q-tilde is dimensionless by construction, so
            // this default survives a change of inlet velocity, which is exactly
            // the failure the normalisation exists to prevent.
            DerivedField::QCriterion => Self {
                map: ColorMap::Inferno,
                range: [0.0, 2.0],
                mode: OpacityMode::SoftIso,
                iso: SoftIso::default(),
                ..Default::default()
            },
            DerivedField::Vorticity => Self {
                map: ColorMap::Magma,
                range: [0.0, 8.0],
                mode: OpacityMode::Curve,
                ..Default::default()
            },
            // Signed, so: a diverging map, the symmetric lock on, and a V-shaped
            // opacity curve that is transparent at zero. All three go together;
            // any one of them alone is misleading.
            DerivedField::Pressure => Self {
                map: ColorMap::CoolWarm,
                range: [-50.0, 50.0],
                symmetric_lock: true,
                mode: OpacityMode::Curve,
                curve: OpacityCurve::diverging(),
                ..Default::default()
            },
        }
    }

    /// Enforce invariants that other code is allowed to rely on.
    pub fn sanitise(&mut self) {
        self.curve.sanitise();
        self.density = self.density.clamp(0.0, 32.0);
        if !self.range[0].is_finite() || !self.range[1].is_finite() {
            self.range = [0.0, 1.0];
        }
        if self.range[1] <= self.range[0] {
            self.range[1] = self.range[0] + 1e-6;
        }
        if self.symmetric_lock {
            let m = self.range[0].abs().max(self.range[1].abs()).max(1e-6);
            self.range = [-m, m];
        }
        if self.scale == RangeScale::Log {
            // Log needs a positive lower bound; fall back to four decades.
            if self.range[0] <= 0.0 {
                self.range[0] = self.range[1].abs().max(1e-6) * 1e-4;
            }
            if self.range[1] <= self.range[0] {
                self.range[1] = self.range[0] * 10.0;
            }
        }
    }

    /// Set the range from measured data extremes, honouring the symmetric lock.
    pub fn fit_range(&mut self, min: f32, max: f32) {
        self.range = [min, max];
        self.symmetric_lock |= self.map.wants_symmetric_range();
        self.sanitise();
    }

    /// Data value -> normalised position. **Not** clamped, so `< 0` and `> 1`
    /// still mean "below/above range" to the caller.
    pub fn normalise(&self, v: f32) -> f32 {
        match self.scale {
            RangeScale::Linear => (v - self.range[0]) / (self.range[1] - self.range[0]),
            RangeScale::Log => {
                let lo = self.range[0].max(1e-30).ln();
                let hi = self.range[1].max(self.range[0] * 1.000_001).ln();
                (v.max(1e-30).ln() - lo) / (hi - lo)
            }
        }
    }

    /// Inverse of [`TransferFunction::normalise`].
    pub fn denormalise(&self, t: f32) -> f32 {
        match self.scale {
            RangeScale::Linear => self.range[0] + t * (self.range[1] - self.range[0]),
            RangeScale::Log => {
                let lo = self.range[0].max(1e-30).ln();
                let hi = self.range[1].max(self.range[0] * 1.000_001).ln();
                (lo + t * (hi - lo)).exp()
            }
        }
    }

    /// Reference opacity for a data value: what one step of length `h_ref`
    /// through material of this value absorbs. The raymarcher corrects this for
    /// its actual step size; see [`correct_opacity`].
    pub fn opacity(&self, v: f32) -> f32 {
        let a = match self.mode {
            OpacityMode::Curve => self.curve.eval(self.normalise(v).clamp(0.0, 1.0)),
            OpacityMode::SoftIso => self.iso.eval(v),
        };
        (a * self.density).clamp(0.0, 1.0)
    }

    /// Linear-light colour for a data value.
    pub fn color(&self, v: f32) -> Vec3 {
        let t = self.normalise(v);
        if self.show_clamp_colors && t < 0.0 {
            return crate::colormap::srgb8_to_linear(ColorMap::UNDER);
        }
        if self.show_clamp_colors && t > 1.0 {
            return crate::colormap::srgb8_to_linear(ColorMap::OVER);
        }
        self.map.sample(t.clamp(0.0, 1.0), self.interpolation)
    }

    /// Bake the 256-entry RGBA LUT: colour from the map, alpha from the opacity
    /// mode. Baking alpha into the same texture is what keeps the raymarch inner
    /// loop down to one `textureSampleLevel` — the alternative, evaluating the
    /// curve in the shader, costs a branchy loop at every one of hundreds of
    /// samples per pixel.
    pub fn bake_lut(&self) -> Vec<[f32; 4]> {
        let colors = self.map.lut(self.interpolation);
        (0..LUT_SIZE)
            .map(|i| {
                let t = i as f32 / (LUT_SIZE - 1) as f32;
                let v = self.denormalise(t);
                let c = colors[i];
                [c.x, c.y, c.z, self.opacity(v)]
            })
            .collect()
    }

    /// Where in data space this transfer function is not transparent.
    ///
    /// This is what [`crate::accel`] binarises the brick min/max grid against,
    /// and two details of its shape are load-bearing.
    ///
    /// **The ends can be infinite.** If the top of the ramp is opaque then every
    /// value above the range is opaque too, because the LUT clamps. A finite
    /// upper bound would make the accelerator erase the core of the jet — the
    /// one region the user is looking at.
    ///
    /// **The middle can have a hole.** A diverging field driven by a V-shaped
    /// curve is transparent around zero and opaque at both extremes, and the
    /// zero band is exactly the still air that fills most of the domain. Without
    /// [`Support::gap`] a pressure view cannot be accelerated at all: the outer
    /// bracket alone covers everything, so every brick reads as active.
    pub fn support(&self) -> Option<Support> {
        match self.mode {
            OpacityMode::SoftIso => {
                let a = self.iso.amplitude * self.density;
                let half = SoftIso { amplitude: a, ..self.iso }.half_support(SUPPORT_EPSILON)?;
                Some(Support {
                    range: (self.iso.center - half, self.iso.center + half),
                    gap: None,
                })
            }
            OpacityMode::Curve => {
                let n = LUT_SIZE;
                let opaque: Vec<bool> = (0..n)
                    .map(|i| {
                        let t = i as f32 / (n - 1) as f32;
                        self.curve.eval(t) * self.density > SUPPORT_EPSILON
                    })
                    .collect();
                let lo = opaque.iter().position(|o| *o)?;
                let hi = opaque.iter().rposition(|o| *o)?;

                // Widen the outer bracket by one LUT entry so linear filtering
                // between a transparent and an opaque entry is never clipped.
                let lo_v = if lo == 0 {
                    f32::NEG_INFINITY
                } else {
                    self.denormalise((lo - 1) as f32 / (n - 1) as f32)
                };
                let hi_v = if hi == n - 1 {
                    f32::INFINITY
                } else {
                    self.denormalise((hi + 1) as f32 / (n - 1) as f32)
                };

                // Longest transparent run strictly inside the bracket. Taking
                // the longest rather than all of them keeps the GPU-side test to
                // two comparisons; a multi-modal curve simply skips less.
                let (mut best_a, mut best_b, mut best_len) = (0usize, 0usize, 0usize);
                let (mut run_a, mut run_len) = (0usize, 0usize);
                for i in lo..=hi {
                    if opaque[i] {
                        run_len = 0;
                        continue;
                    }
                    if run_len == 0 {
                        run_a = i;
                    }
                    run_len += 1;
                    if run_len > best_len {
                        best_len = run_len;
                        best_a = run_a;
                        best_b = i;
                    }
                }
                // Shrink by one entry each side: a value landing exactly on the
                // first transparent entry still filters against an opaque
                // neighbour on one side.
                let gap = if best_len >= 4 {
                    let to_v = |i: usize| self.denormalise(i as f32 / (n - 1) as f32);
                    Some((to_v(best_a + 1), to_v(best_b - 1)))
                } else {
                    None
                };

                Some(Support { range: (lo_v, hi_v), gap })
            }
        }
    }
}

/// The set of data values a transfer function can see.
///
/// `range` is the outer bracket; `gap`, when present, is a strictly-interior
/// band that is transparent. A value contributes when it is inside `range` and
/// **not** strictly inside `gap`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Support {
    pub range: (f32, f32),
    pub gap: Option<(f32, f32)>,
}

impl Support {
    /// Whether an interval of values — a brick's `[min, max]` — can contribute.
    ///
    /// Conservative in the safe direction: an interval that merely overlaps the
    /// gap is still active, because part of it lies outside.
    pub fn intersects(&self, min: f32, max: f32) -> bool {
        if min > max || max < self.range.0 || min > self.range.1 {
            return false;
        }
        match self.gap {
            Some((a, b)) => !(min > a && max < b),
            None => true,
        }
    }

    /// Whether a single value contributes.
    pub fn contains(&self, v: f32) -> bool {
        self.intersects(v, v)
    }

    /// `gap` as a pair that is empty when there is no gap, for the GPU uniform.
    pub fn gap_or_empty(&self) -> (f32, f32) {
        self.gap.unwrap_or((1.0, 0.0))
    }
}

/// Opacity correction for a step size that is not the reference step.
///
/// `alpha_c = 1 - (1 - alpha_ref)^(h / h_ref)`.
///
/// This is not a refinement, it is **load-bearing**. Without it, every
/// empty-space-skipping or adaptive-step optimisation changes the image, which
/// means you cannot tell an accelerator bug from an accelerator working. With
/// it, the composited result is invariant to step size, so a visual difference
/// between "skipping on" and "skipping off" is unambiguously a bug.
///
/// The derivation is Beer-Lambert: transmittance is `T = exp(-sigma h)`, so
/// `1 - alpha = exp(-sigma h)`; changing `h` raises the transmittance to the
/// power of the step ratio.
#[inline]
pub fn correct_opacity(alpha_ref: f32, h: f32, h_ref: f32) -> f32 {
    if h_ref <= 0.0 {
        return alpha_ref;
    }
    let a = alpha_ref.clamp(0.0, 1.0);
    if a >= 1.0 {
        return 1.0;
    }
    1.0 - (1.0 - a).powf(h / h_ref)
}

/// Front-to-back composite of `n` uniform steps of length `h` through material
/// of reference opacity `alpha_ref`. Used by the invariance test, and a useful
/// reference when debugging the shader's accumulation.
pub fn composite_uniform(alpha_ref: f32, h_ref: f32, h: f32, n: u32) -> f32 {
    let a = correct_opacity(alpha_ref, h, h_ref);
    let mut acc = 0.0f32;
    for _ in 0..n {
        acc += (1.0 - acc) * a;
    }
    acc
}

/// GPU mirror of the transfer function. The LUT itself is a separate texture.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TransferUniform {
    /// Lower bound, already log-transformed when `log_scale != 0`.
    pub lo: f32,
    /// `1 / (hi - lo)` in the same transformed space.
    pub inv_span: f32,
    pub log_scale: u32,
    pub show_clamp: u32,

    pub under_color: [f32; 4],
    pub over_color: [f32; 4],
}

impl TransferFunction {
    pub fn uniform(&self) -> TransferUniform {
        let (lo, hi) = match self.scale {
            RangeScale::Linear => (self.range[0], self.range[1]),
            RangeScale::Log => (self.range[0].max(1e-30).ln(), self.range[1].max(1e-29).ln()),
        };
        let under = crate::colormap::srgb8_to_linear(ColorMap::UNDER);
        let over = crate::colormap::srgb8_to_linear(ColorMap::OVER);
        TransferUniform {
            lo,
            inv_span: 1.0 / (hi - lo).max(1e-20),
            log_scale: u32::from(self.scale == RangeScale::Log),
            show_clamp: u32::from(self.show_clamp_colors),
            under_color: under.extend(1.0).to_array(),
            over_color: over.extend(1.0).to_array(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opacity_correction_is_invariant_to_step_size() {
        // The classic bug: halve the step and the volume doubles in density.
        // With the correction, marching 2 mm of material must absorb the same
        // amount however finely it is diced.
        let h_ref = 0.5f32;
        for alpha_ref in [0.02f32, 0.1, 0.35, 0.8, 0.99] {
            let coarse = composite_uniform(alpha_ref, h_ref, h_ref, 8);
            for div in [2u32, 4, 8, 16, 64] {
                let fine =
                    composite_uniform(alpha_ref, h_ref, h_ref / div as f32, 8 * div);
                assert!(
                    (fine - coarse).abs() < 1e-4,
                    "alpha_ref={alpha_ref} div={div}: {fine} vs {coarse}"
                );
            }
        }
    }

    #[test]
    fn uncorrected_compositing_visibly_drifts() {
        // Demonstrates what the correction is protecting against, so nobody
        // "simplifies" it away later.
        let naive = |a: f32, n: u32| {
            let mut acc = 0.0;
            for _ in 0..n {
                acc += (1.0 - acc) * a;
            }
            acc
        };
        let coarse = naive(0.1, 8);
        let fine = naive(0.1, 64);
        assert!(fine > coarse * 1.5, "expected a large drift, got {coarse} -> {fine}");
    }

    #[test]
    fn correction_handles_the_degenerate_ends() {
        assert_eq!(correct_opacity(0.0, 0.1, 1.0), 0.0);
        assert_eq!(correct_opacity(1.0, 0.1, 1.0), 1.0);
        assert_eq!(correct_opacity(0.5, 1.0, 1.0), 0.5);
        // Larger step absorbs more, smaller absorbs less. Monotone, always.
        assert!(correct_opacity(0.5, 2.0, 1.0) > 0.5);
        assert!(correct_opacity(0.5, 0.5, 1.0) < 0.5);
    }

    #[test]
    fn soft_iso_peaks_at_the_centre_and_decays_like_a_gaussian() {
        let iso = SoftIso { center: 0.5, width: 0.25, amplitude: 0.8, cutoff_widths: 4.0 };
        assert!((iso.eval(0.5) - 0.8).abs() < 1e-6);
        // One width out is about 1/e of the peak. This is the property that
        // makes "width" mean something the user can reason about. The pedestal
        // subtraction shifts it by well under a percent at n = 4.
        let one_width = iso.eval(0.75);
        assert!(
            (one_width - 0.8 / std::f32::consts::E).abs() < 0.005,
            "one width out gave {one_width}"
        );
        assert!((iso.eval(0.25) - iso.eval(0.75)).abs() < 1e-7, "must be symmetric");
        assert!(iso.eval(2.0) == 0.0);
    }

    #[test]
    fn a_soft_isosurface_is_exactly_transparent_in_still_air() {
        // The bug this cutoff exists to prevent: a pure Gaussian's tail is only
        // ~1.6% opaque at Q = 0, which compounds over a few hundred raymarch
        // steps into a solid black brick the size of the whole domain.
        let tf = TransferFunction::preset(crate::fields::DerivedField::QCriterion);
        assert_eq!(tf.mode, OpacityMode::SoftIso);
        assert_eq!(tf.opacity(0.0), 0.0, "quiescent air must be exactly transparent");
        assert_eq!(tf.opacity(-1.0), 0.0, "strain-dominated flow is not a vortex");
        // ...and the support agrees, so the accelerator skips it rather than
        // marching through it.
        let lo = tf.support().unwrap().range.0;
        assert!(lo >= -1e-6, "support reaches below zero: {lo}");

        // A hundred steps of the tail must accumulate to nothing.
        assert_eq!(composite_uniform(tf.opacity(0.0), 1.0, 1.0, 200), 0.0);

        // The same shape without a cutoff would fog the domain, which is the
        // point of the comparison.
        let uncut = SoftIso { cutoff_widths: 12.0, ..tf.iso };
        assert!(
            composite_uniform(uncut.eval(0.0), 1.0, 1.0, 200) > 0.9,
            "the untruncated tail should have been opaque; the test is not proving anything"
        );
    }

    #[test]
    fn soft_iso_centre_is_in_data_units_and_survives_re_ranging() {
        // The usability invariant: changing the displayed range must not move
        // the isosurface. This is the test that would catch someone "helpfully"
        // normalising the centre.
        let mut tf = TransferFunction {
            mode: OpacityMode::SoftIso,
            iso: SoftIso { center: 0.5, width: 0.2, amplitude: 1.0, cutoff_widths: 3.0 },
            range: [0.0, 2.0],
            ..Default::default()
        };
        let before = tf.opacity(0.5);
        tf.range = [0.0, 20.0];
        tf.sanitise();
        assert!((tf.opacity(0.5) - before).abs() < 1e-6);
        assert!(tf.opacity(0.5) > tf.opacity(1.5));
    }

    #[test]
    fn piecewise_curve_is_exact_at_control_points_and_linear_between() {
        let c = OpacityCurve::new(vec![[0.0, 0.0], [0.5, 1.0], [1.0, 0.2]]);
        assert!((c.eval(0.0) - 0.0).abs() < 1e-6);
        assert!((c.eval(0.5) - 1.0).abs() < 1e-6);
        assert!((c.eval(1.0) - 0.2).abs() < 1e-6);
        assert!((c.eval(0.25) - 0.5).abs() < 1e-6, "midpoint should be exactly halfway");
        assert!((c.eval(0.75) - 0.6).abs() < 1e-6);
        // Clamped, not extrapolated.
        assert!((c.eval(-1.0) - 0.0).abs() < 1e-6);
        assert!((c.eval(2.0) - 0.2).abs() < 1e-6);
    }

    #[test]
    fn curve_sanitises_unsorted_and_oversized_input() {
        let c = OpacityCurve::new(vec![
            [0.9, 0.5],
            [0.1, 0.2],
            [0.5, 0.9],
            [0.2, 0.1],
            [0.3, 0.4],
            [0.4, 0.3],
            [0.6, 0.1],
            [0.7, 0.0],
        ]);
        assert!(c.points.len() <= MAX_CURVE_POINTS);
        assert!(c.points.windows(2).all(|w| w[0][0] <= w[1][0]), "not sorted");
        let mut c2 = OpacityCurve::new(vec![[0.5, 0.5]]);
        assert!(c2.points.len() >= 2, "must never degenerate below two points");
        assert!(!c2.try_remove(0), "must refuse to drop below two points");
    }

    #[test]
    fn symmetric_lock_centres_the_range_on_zero() {
        // A diverging map with an off-centre zero is the failure mode this
        // exists to prevent.
        let mut tf = TransferFunction { map: ColorMap::CoolWarm, ..Default::default() };
        tf.fit_range(-12.0, 80.0);
        assert!(tf.symmetric_lock);
        assert!((tf.range[0] + tf.range[1]).abs() < 1e-6, "range {:?} is off-centre", tf.range);
        assert!((tf.normalise(0.0) - 0.5).abs() < 1e-6, "zero must land at the map midpoint");
    }

    #[test]
    fn log_scale_round_trips_and_spreads_decades_evenly() {
        let tf = TransferFunction {
            range: [0.01, 100.0],
            scale: RangeScale::Log,
            ..Default::default()
        };
        for v in [0.01f32, 0.1, 1.0, 10.0, 100.0] {
            let t = tf.normalise(v);
            assert!((tf.denormalise(t) - v).abs() < v * 1e-4, "{v} round-tripped to {}", tf.denormalise(t));
        }
        // Four decades over [0,1] means one decade per quarter.
        assert!((tf.normalise(0.1) - 0.25).abs() < 1e-5);
        assert!((tf.normalise(1.0) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn support_brackets_everything_the_transfer_function_can_see() {
        // The accelerator's correctness rests entirely on this: anything
        // outside `support()` must be exactly transparent.
        let tf = TransferFunction {
            mode: OpacityMode::SoftIso,
            iso: SoftIso { center: 0.5, width: 0.2, amplitude: 1.0, cutoff_widths: 3.0 },
            range: [0.0, 2.0],
            ..Default::default()
        };
        let Support { range: (lo, hi), gap } = tf.support().unwrap();
        assert!(gap.is_none(), "a single bump has no interior hole");
        assert!(lo < 0.5 && hi > 0.5);
        for i in 0..500 {
            let v = -2.0 + i as f32 * 0.01;
            if v < lo || v > hi {
                assert!(tf.opacity(v) <= SUPPORT_EPSILON, "opacity {} at v={v} is outside support ({lo}..{hi})", tf.opacity(v));
            }
        }
    }

    #[test]
    fn support_runs_to_infinity_when_the_ramp_top_is_opaque() {
        // Clamping means "above the range" is as opaque as the top of the range.
        // A finite support here would make the accelerator erase the core jet.
        let tf = TransferFunction {
            mode: OpacityMode::Curve,
            curve: OpacityCurve::new(vec![[0.0, 0.0], [0.5, 0.0], [1.0, 0.9]]),
            range: [0.0, 10.0],
            ..Default::default()
        };
        let Support { range: (lo, hi), .. } = tf.support().unwrap();
        assert_eq!(hi, f32::INFINITY);
        assert!(lo > 0.0 && lo < 10.0, "lo was {lo}");
        assert!(tf.opacity(1.0e6) > 0.5, "clamped high values must stay opaque");
    }

    #[test]
    fn every_preset_is_exactly_transparent_in_still_air() {
        // The domain around a duct is mostly motionless air: speed, vorticity
        // and Q are all exactly zero there, and so is the pressure deviation.
        // Any preset with non-zero opacity at those values composites into an
        // opaque box the size of the domain over a few hundred raymarch steps.
        // Two separate bugs of exactly this shape have already been fixed here;
        // this test is what stops a third.
        for field in crate::fields::DerivedField::ALL {
            let tf = TransferFunction::preset(field);
            assert_eq!(
                tf.opacity(0.0),
                0.0,
                "the {} preset fogs still air",
                field.name()
            );
            // ...and the accelerator agrees, so those bricks are skipped
            // rather than marched through.
            if let Some(support) = tf.support() {
                assert!(
                    !support.contains(0.0),
                    "the {} preset's support contains zero: {support:?}",
                    field.name()
                );
            }
        }
    }

    #[test]
    fn the_diverging_curve_is_symmetric_and_open_in_the_middle() {
        let c = OpacityCurve::diverging();
        assert_eq!(c.eval(0.5), 0.0, "zero must be transparent");
        // Symmetric about the midpoint, so over- and under-pressure of equal
        // magnitude are equally visible.
        for t in [0.0f32, 0.1, 0.2, 0.3, 0.45] {
            assert!(
                (c.eval(t) - c.eval(1.0 - t)).abs() < 1e-6,
                "asymmetric at t = {t}: {} vs {}",
                c.eval(t),
                c.eval(1.0 - t)
            );
        }
        assert!(c.eval(0.0) > 0.5 && c.eval(1.0) > 0.5, "both extremes must be visible");
    }

    #[test]
    fn a_fully_transparent_transfer_function_has_no_support() {
        let tf = TransferFunction { density: 0.0, ..Default::default() };
        assert!(tf.support().is_none());
    }

    #[test]
    fn baked_lut_agrees_with_direct_evaluation() {
        let tf = TransferFunction::preset(crate::fields::DerivedField::QCriterion);
        let lut = tf.bake_lut();
        assert_eq!(lut.len(), LUT_SIZE);
        for i in [0usize, 37, 128, 255] {
            let t = i as f32 / (LUT_SIZE - 1) as f32;
            let v = tf.denormalise(t);
            assert!((lut[i][3] - tf.opacity(v)).abs() < 1e-6, "alpha mismatch at {i}");
            let c = tf.color(v);
            assert!((lut[i][0] - c.x).abs() < 1e-5, "colour mismatch at {i}");
        }
    }

    #[test]
    fn uniform_matches_the_cpu_normalisation() {
        for scale in [RangeScale::Linear, RangeScale::Log] {
            let mut tf = TransferFunction { range: [0.5, 50.0], scale, ..Default::default() };
            tf.sanitise();
            let u = tf.uniform();
            for v in [0.6f32, 5.0, 25.0, 49.0] {
                let shader = {
                    let x = if u.log_scale != 0 { v.ln() } else { v };
                    (x - u.lo) * u.inv_span
                };
                assert!(
                    (shader - tf.normalise(v)).abs() < 1e-5,
                    "{scale:?} at {v}: shader {shader} vs cpu {}",
                    tf.normalise(v)
                );
            }
        }
        assert_eq!(std::mem::size_of::<TransferUniform>() % 16, 0);
    }
}
