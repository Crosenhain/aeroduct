//! The colour-bar legend and its draggable range handles.
//!
//! Almost no code, and a disproportionate usability win. Three gestures:
//!
//! * **drag an end** to rescale that end of the range,
//! * **double-click** to auto-range to the data,
//! * **lock** to freeze the range for an A/B comparison.
//!
//! The lock is the one that matters and the one people forget to build. Two
//! designs rendered with independently auto-ranged colour bars look identical
//! no matter how different they are, because auto-ranging *removes* exactly the
//! information you are trying to compare. A locked legend turns a colour
//! difference back into a real difference, so the lock is a first-class piece
//! of state here rather than a checkbox tucked in a settings panel — and it is
//! sticky across a field switch, because that is the moment it is most often
//! wanted and most easily lost.
//!
//! # Where the arithmetic lives
//!
//! Dragging a handle is a screen-space gesture on a possibly-logarithmic axis.
//! [`LegendRange::drag_handle`] does the transform through normalised space, so
//! a drag of the same number of pixels moves the same fraction of the bar
//! whether the scale is linear or log. Doing it in data space instead makes a
//! log legend's low end unusably twitchy and its high end immovable.

use ad_render::transfer::RangeScale;

/// Which end of the bar is being dragged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Handle {
    Low,
    High,
}

/// The legend's own state, separate from the transfer function it drives.
///
/// Kept separate because `TransferFunction::range` is what the shader reads,
/// while this is what the *widget* remembers — the lock, the pending
/// auto-range, which handle is being dragged. Merging them would put UI state
/// into a GPU-facing struct.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LegendRange {
    pub lo: f32,
    pub hi: f32,
    pub scale: RangeScale,
    /// Frozen: neither a drag nor an auto-range may change it.
    pub locked: bool,
    /// Keep `[-m, m]`. Mirrors `TransferFunction::symmetric_lock`, and dragging
    /// one end drags the other.
    pub symmetric: bool,
    /// Handle currently under the mouse button, for the drag.
    pub dragging: Option<Handle>,
}

impl Default for LegendRange {
    fn default() -> Self {
        Self {
            lo: 0.0,
            hi: 1.0,
            scale: RangeScale::Linear,
            locked: false,
            symmetric: false,
            dragging: None,
        }
    }
}

impl LegendRange {
    pub fn new(lo: f32, hi: f32, scale: RangeScale, symmetric: bool) -> Self {
        let mut r = Self { lo, hi, scale, symmetric, ..Default::default() };
        r.sanitise();
        r
    }

    pub fn span(&self) -> f32 {
        self.hi - self.lo
    }

    /// Data value -> position along the bar in `[0, 1]`.
    ///
    /// Mirrors `TransferFunction::normalise` deliberately: if the two ever
    /// disagree, the tick labels stop lining up with the colours, and that is
    /// the sort of bug people work around for months instead of reporting.
    pub fn normalise(&self, v: f32) -> f32 {
        match self.scale {
            RangeScale::Linear => (v - self.lo) / (self.hi - self.lo),
            RangeScale::Log => {
                let (lo, hi) = self.log_bounds();
                (v.max(1e-30).ln() - lo) / (hi - lo)
            }
        }
    }

    pub fn denormalise(&self, t: f32) -> f32 {
        match self.scale {
            RangeScale::Linear => self.lo + t * (self.hi - self.lo),
            RangeScale::Log => {
                let (lo, hi) = self.log_bounds();
                (lo + t * (hi - lo)).exp()
            }
        }
    }

    fn log_bounds(&self) -> (f32, f32) {
        let lo = self.lo.max(1e-30).ln();
        let hi = self.hi.max(self.lo * 1.000_001).max(1e-29).ln();
        (lo, hi.max(lo + 1e-6))
    }

    /// Drag one end by `delta_fraction` of the bar's length.
    ///
    /// Returns whether anything moved, so the caller only re-bakes the LUT when
    /// it has to. A locked legend always returns false — that is the whole
    /// contract of the lock, and enforcing it here rather than at the call site
    /// means no future panel can forget.
    pub fn drag_handle(&mut self, handle: Handle, delta_fraction: f32) -> bool {
        if self.locked || delta_fraction == 0.0 || !delta_fraction.is_finite() {
            return false;
        }
        let before = (self.lo, self.hi);
        match handle {
            Handle::Low => {
                let t = self.normalise(self.lo) + delta_fraction;
                self.lo = self.denormalise(t);
            }
            Handle::High => {
                let t = self.normalise(self.hi) + delta_fraction;
                self.hi = self.denormalise(t);
            }
        }
        if self.symmetric {
            // Dragging either end of a symmetric range moves both, keeping zero
            // at the midpoint of the diverging map. Anything else silently
            // shifts the neutral colour off zero, which is the failure the
            // symmetric lock exists to prevent in the first place.
            let m = self.lo.abs().max(self.hi.abs());
            self.lo = -m;
            self.hi = m;
        }
        self.sanitise();
        (self.lo, self.hi) != before
    }

    /// Snap the range onto measured extremes. Ignored while locked.
    ///
    /// The padding is 2% of the span, so the top of the data does not sit
    /// exactly on the last LUT entry where linear filtering has nothing to
    /// interpolate against.
    pub fn auto_range(&mut self, data_min: f32, data_max: f32) -> bool {
        if self.locked || !data_min.is_finite() || !data_max.is_finite() {
            return false;
        }
        let before = (self.lo, self.hi);
        let pad = ((data_max - data_min) * 0.02).abs().max(1e-6);
        self.lo = data_min - pad;
        self.hi = data_max + pad;
        if self.symmetric {
            let m = self.lo.abs().max(self.hi.abs());
            self.lo = -m;
            self.hi = m;
        }
        self.sanitise();
        (self.lo, self.hi) != before
    }

    /// Keep the invariants the shader and the tick generator rely on.
    pub fn sanitise(&mut self) {
        if !self.lo.is_finite() || !self.hi.is_finite() {
            self.lo = 0.0;
            self.hi = 1.0;
        }
        if self.hi <= self.lo {
            // A dragged handle can cross the other one. Rather than swapping,
            // which makes the bar flip under the cursor, hold a minimum span.
            self.hi = self.lo + (self.lo.abs().max(1.0) * 1e-4).max(1e-6);
        }
        if self.scale == RangeScale::Log {
            if self.lo <= 0.0 {
                // Four decades below the top: the usual useful window for a
                // duct speed field, where the separated region sits two decades
                // under the core jet.
                self.lo = self.hi.abs().max(1e-6) * 1e-4;
            }
            if self.hi <= self.lo {
                self.hi = self.lo * 10.0;
            }
        }
    }

    /// Copy into a renderer transfer function.
    pub fn apply_to(&self, tf: &mut ad_render::TransferFunction) {
        tf.range = [self.lo, self.hi];
        tf.scale = self.scale;
        tf.symmetric_lock = self.symmetric;
        tf.sanitise();
    }

    /// Read back from a renderer transfer function, keeping the lock.
    ///
    /// The lock surviving this is the point: switching field, or having Wave 3
    /// re-fit a range, must not quietly unfreeze a comparison.
    pub fn sync_from(&mut self, tf: &ad_render::TransferFunction) {
        if self.locked {
            return;
        }
        self.lo = tf.range[0];
        self.hi = tf.range[1];
        self.scale = tf.scale;
        self.symmetric = tf.symmetric_lock;
    }

    /// Nice tick values across the bar, at most `max_ticks` of them.
    ///
    /// Linear ticks land on 1/2/5 x 10^n so the labels are readable numbers;
    /// log ticks land on decades. Evenly spaced ticks would produce labels like
    /// `3.7142` and make the legend look like a debug readout.
    pub fn ticks(&self, max_ticks: usize) -> Vec<f32> {
        let n = max_ticks.max(2);
        match self.scale {
            RangeScale::Log => {
                let (lo, hi) = (self.lo.max(1e-30).log10(), self.hi.max(1e-29).log10());
                let step = (((hi - lo) / n as f32).ceil() as i32).max(1);
                let mut out = Vec::new();
                let mut e = lo.ceil() as i32;
                while (e as f32) <= hi && out.len() < n {
                    out.push(10f32.powi(e));
                    e += step;
                }
                out
            }
            RangeScale::Linear => {
                let raw = self.span() / n as f32;
                if !raw.is_finite() || raw <= 0.0 {
                    return vec![self.lo, self.hi];
                }
                let mag = 10f32.powf(raw.log10().floor());
                let norm = raw / mag;
                let step = if norm <= 1.0 {
                    mag
                } else if norm <= 2.0 {
                    2.0 * mag
                } else if norm <= 5.0 {
                    5.0 * mag
                } else {
                    10.0 * mag
                };
                let mut out = Vec::new();
                let mut v = (self.lo / step).ceil() * step;
                while v <= self.hi + step * 1e-4 && out.len() <= n + 1 {
                    out.push(v);
                    v += step;
                }
                out
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dragging_the_top_handle_rescales_only_that_end() {
        let mut r = LegendRange::new(0.0, 10.0, RangeScale::Linear, false);
        assert!(r.drag_handle(Handle::High, -0.2));
        assert!((r.lo - 0.0).abs() < 1e-6);
        assert!((r.hi - 8.0).abs() < 1e-5, "hi was {}", r.hi);
    }

    #[test]
    fn a_locked_legend_refuses_every_change() {
        // The A/B invariant. If this regresses, two designs compared side by
        // side get independently auto-ranged colour bars and look identical.
        let mut r = LegendRange::new(0.0, 10.0, RangeScale::Linear, false);
        r.locked = true;
        assert!(!r.drag_handle(Handle::High, 0.5));
        assert!(!r.drag_handle(Handle::Low, -0.5));
        assert!(!r.auto_range(-100.0, 100.0));
        assert_eq!((r.lo, r.hi), (0.0, 10.0));
        // ...and a sync from the renderer cannot unfreeze it either.
        let mut tf = ad_render::TransferFunction::default();
        tf.range = [-3.0, 77.0];
        r.sync_from(&tf);
        assert_eq!((r.lo, r.hi), (0.0, 10.0));
    }

    #[test]
    fn double_click_auto_ranges_with_a_little_headroom() {
        let mut r = LegendRange::new(0.0, 1.0, RangeScale::Linear, false);
        assert!(r.auto_range(2.0, 12.0));
        assert!(r.lo < 2.0 && r.hi > 12.0, "range {:?} does not contain the data", (r.lo, r.hi));
        assert!(r.lo > 1.5 && r.hi < 12.5, "the padding is far too generous");
    }

    #[test]
    fn a_log_drag_moves_the_same_fraction_of_the_bar_at_both_ends() {
        // Done in data space instead, the low end would be untouchable and the
        // high end would jump decades.
        let mut r = LegendRange::new(0.01, 100.0, RangeScale::Log, false);
        let t_before = r.normalise(1.0);
        r.drag_handle(Handle::High, 0.25);
        // A quarter-bar drag on a four-decade range adds one decade.
        assert!((r.hi - 1000.0).abs() / 1000.0 < 0.02, "hi was {}", r.hi);
        // ...and 1.0 has moved down the bar by the right amount.
        assert!(r.normalise(1.0) < t_before);
    }

    #[test]
    fn a_symmetric_range_stays_centred_on_zero_however_it_is_dragged() {
        // The pressure field's whole reason for existing: zero must stay on the
        // diverging map's neutral colour.
        let mut r = LegendRange::new(-50.0, 50.0, RangeScale::Linear, true);
        r.drag_handle(Handle::High, 0.3);
        assert!((r.lo + r.hi).abs() < 1e-4, "range {:?} is off-centre", (r.lo, r.hi));
        assert!((r.normalise(0.0) - 0.5).abs() < 1e-6);
        r.drag_handle(Handle::Low, -0.2);
        assert!((r.lo + r.hi).abs() < 1e-4);
        r.auto_range(-12.0, 80.0);
        assert!((r.lo + r.hi).abs() < 1e-4);
    }

    #[test]
    fn handles_crossing_hold_a_minimum_span_instead_of_flipping() {
        // Swapping instead would make the bar jump under the cursor mid-drag.
        let mut r = LegendRange::new(0.0, 10.0, RangeScale::Linear, false);
        r.drag_handle(Handle::High, -5.0);
        assert!(r.hi > r.lo, "range inverted: {:?}", (r.lo, r.hi));
        assert!(r.span() > 0.0 && r.span() < 0.1);
    }

    #[test]
    fn a_log_range_never_admits_a_non_positive_lower_bound() {
        let mut r = LegendRange::new(0.1, 100.0, RangeScale::Log, false);
        r.drag_handle(Handle::Low, -10.0);
        assert!(r.lo > 0.0, "log lower bound went to {}", r.lo);
        assert!(r.hi > r.lo);
        // A range that starts out invalid is repaired rather than trusted.
        let r2 = LegendRange::new(-5.0, 100.0, RangeScale::Log, false);
        assert!(r2.lo > 0.0);
    }

    #[test]
    fn normalise_and_denormalise_round_trip_on_both_scales() {
        for r in [
            LegendRange::new(-3.0, 7.0, RangeScale::Linear, false),
            LegendRange::new(0.01, 100.0, RangeScale::Log, false),
        ] {
            for t in [0.0f32, 0.25, 0.5, 1.0] {
                let v = r.denormalise(t);
                assert!((r.normalise(v) - t).abs() < 1e-4, "{t} -> {v} -> {}", r.normalise(v));
            }
        }
    }

    #[test]
    fn ticks_are_readable_numbers_inside_the_range() {
        let r = LegendRange::new(0.0, 10.0, RangeScale::Linear, false);
        let ticks = r.ticks(5);
        assert!(ticks.len() >= 3 && ticks.len() <= 8, "{ticks:?}");
        assert!(ticks.iter().all(|t| *t >= r.lo - 1e-3 && *t <= r.hi + 1e-3));
        // 1/2/5 spacing: on 0-10 with 5 ticks that is a step of 2.
        assert!((ticks[1] - ticks[0] - 2.0).abs() < 1e-4, "{ticks:?}");

        let log = LegendRange::new(0.01, 100.0, RangeScale::Log, false);
        let lt = log.ticks(6);
        assert!(lt.iter().all(|t| (t.log10() - t.log10().round()).abs() < 1e-4), "{lt:?}");
    }

    #[test]
    fn applying_to_a_transfer_function_round_trips_through_sync() {
        let mut tf = ad_render::TransferFunction::default();
        let r = LegendRange::new(2.0, 9.0, RangeScale::Linear, false);
        r.apply_to(&mut tf);
        assert_eq!(tf.range, [2.0, 9.0]);
        let mut back = LegendRange::default();
        back.sync_from(&tf);
        assert!((back.lo - 2.0).abs() < 1e-6 && (back.hi - 9.0).abs() < 1e-6);
    }
}
