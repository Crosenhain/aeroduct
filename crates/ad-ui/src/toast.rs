//! Transient notifications, drawn over the viewport.
//!
//! # Why this exists instead of a modal dialog
//!
//! The app has **no modal dialogs**, by design: the simulation is always
//! running and a modal would stop the user interacting with a live system to
//! acknowledge something they already knew. But some events genuinely must not
//! be missed — above all a **statistics reset**, because averaging across a
//! boundary-condition change produces a confidently wrong number with a small
//! error bar, which is the most dangerous output this app can produce.
//!
//! A toast threads that needle: impossible to miss, impossible to be blocked
//! by. The severities differ in how long they stay, and a
//! [`Severity::Critical`] toast does not fade at all until it is dismissed —
//! divergence and a lost device are not things to notice out of the corner of
//! an eye.
//!
//! # Coalescing
//!
//! Dragging the inlet slider produces a statistics reset on *every frame*.
//! Sixty toasts a second is not a notification, it is a strobe. Toasts carry an
//! optional [`Toast::key`]; pushing one whose key matches a live toast restarts
//! that toast's timer and increments its repeat count instead of stacking a new
//! one. The user sees one message that says "inlet velocity changed (x37)" and
//! stays on screen until a second after they let go of the slider.

use std::collections::VecDeque;

use crate::view::ResetCause;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Info,
    /// Something the user must know but need not act on. Statistics resets
    /// live here.
    Notice,
    Warning,
    /// Stays until dismissed.
    Critical,
}

impl Severity {
    /// Seconds on screen. `None` means "until dismissed".
    pub fn lifetime_s(self) -> Option<f32> {
        match self {
            Severity::Info => Some(3.0),
            Severity::Notice => Some(5.0),
            Severity::Warning => Some(9.0),
            Severity::Critical => None,
        }
    }

    pub fn color(self) -> [f32; 4] {
        match self {
            Severity::Info => [0.62, 0.68, 0.78, 1.0],
            Severity::Notice => [0.45, 0.76, 0.95, 1.0],
            Severity::Warning => [0.95, 0.72, 0.18, 1.0],
            Severity::Critical => [0.94, 0.33, 0.31, 1.0],
        }
    }

    pub fn prefix(self) -> &'static str {
        match self {
            Severity::Info => "",
            Severity::Notice => "",
            Severity::Warning => "warning: ",
            Severity::Critical => "ERROR: ",
        }
    }
}

/// One message on screen.
#[derive(Debug, Clone, PartialEq)]
pub struct Toast {
    pub severity: Severity,
    pub title: String,
    /// Optional second line, for the detail that would make the title too long.
    pub detail: String,
    /// Coalescing key. Two toasts with the same non-empty key are the same
    /// toast.
    pub key: String,
    /// Seconds this toast has been alive.
    pub age_s: f32,
    /// How many times it has been re-raised while alive.
    pub repeats: u32,
}

impl Toast {
    pub fn new(severity: Severity, title: impl Into<String>) -> Self {
        Self {
            severity,
            title: title.into(),
            detail: String::new(),
            key: String::new(),
            age_s: 0.0,
            repeats: 0,
        }
    }

    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = detail.into();
        self
    }

    /// Give the toast a coalescing key.
    pub fn with_key(mut self, key: impl Into<String>) -> Self {
        self.key = key.into();
        self
    }

    /// The statistics-reset toast. Constructed here rather than at each call
    /// site so every reset, wherever it comes from, is announced identically —
    /// and so the key is guaranteed to coalesce across a slider drag.
    pub fn statistics_reset(cause: ResetCause) -> Self {
        Toast::new(Severity::Notice, "Statistics reset")
            .with_detail(format!(
                "{} - averages and error bars start again from zero",
                cause.message()
            ))
            .with_key("stats-reset")
    }

    /// Remaining life as a fraction, for the fade-out. Always 1 for a critical
    /// toast, which never fades.
    pub fn remaining(&self) -> f32 {
        match self.severity.lifetime_s() {
            Some(life) => (1.0 - self.age_s / life.max(1e-3)).clamp(0.0, 1.0),
            None => 1.0,
        }
    }

    pub fn expired(&self) -> bool {
        matches!(self.severity.lifetime_s(), Some(life) if self.age_s >= life)
    }

    /// Title with the repeat count appended, so a coalesced burst still shows
    /// how big it was.
    pub fn display_title(&self) -> String {
        if self.repeats > 0 {
            format!("{}{} (x{})", self.severity.prefix(), self.title, self.repeats + 1)
        } else {
            format!("{}{}", self.severity.prefix(), self.title)
        }
    }
}

/// Hard cap on simultaneous toasts. Beyond this the oldest non-critical one is
/// dropped: a stack taller than the viewport hides the thing it is reporting
/// on.
pub const MAX_TOASTS: usize = 5;

/// The live queue.
#[derive(Debug, Clone, Default)]
pub struct Toasts {
    items: VecDeque<Toast>,
}

impl Toasts {
    pub fn new() -> Self {
        Self::default()
    }

    /// Raise a toast, coalescing onto a live one with the same key.
    pub fn push(&mut self, toast: Toast) {
        if !toast.key.is_empty() {
            if let Some(existing) = self.items.iter_mut().find(|t| t.key == toast.key) {
                existing.repeats += 1;
                existing.age_s = 0.0;
                // A repeat may be more severe than the original (a warning
                // following a notice); keep the worse of the two.
                existing.severity = existing.severity.max(toast.severity);
                existing.detail = toast.detail;
                return;
            }
        }
        self.items.push_back(toast);
        while self.items.len() > MAX_TOASTS {
            // Drop the oldest expirable one; never silently drop a critical.
            match self.items.iter().position(|t| t.severity != Severity::Critical) {
                Some(i) => {
                    self.items.remove(i);
                }
                None => {
                    self.items.pop_front();
                }
            }
        }
    }

    pub fn info(&mut self, title: impl Into<String>) {
        self.push(Toast::new(Severity::Info, title));
    }

    pub fn warn(&mut self, title: impl Into<String>) {
        self.push(Toast::new(Severity::Warning, title));
    }

    pub fn error(&mut self, title: impl Into<String>) {
        self.push(Toast::new(Severity::Critical, title));
    }

    /// Age every toast and drop the expired ones.
    pub fn update(&mut self, dt: f32) {
        for t in &mut self.items {
            t.age_s += dt.max(0.0);
        }
        self.items.retain(|t| !t.expired());
    }

    pub fn dismiss(&mut self, index: usize) {
        if index < self.items.len() {
            self.items.remove(index);
        }
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }

    pub fn iter(&self) -> impl Iterator<Item = &Toast> {
        self.items.iter()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_statistics_reset_names_its_cause() {
        let t = Toast::statistics_reset(ResetCause::InletVelocity);
        assert_eq!(t.severity, Severity::Notice);
        assert!(t.detail.contains("inlet velocity changed"), "{}", t.detail);
        assert!(t.detail.contains("start again"), "the consequence must be spelled out");
    }

    #[test]
    fn dragging_a_slider_produces_one_toast_not_sixty() {
        // The strobe this coalescing exists to prevent.
        let mut ts = Toasts::new();
        for _ in 0..60 {
            ts.push(Toast::statistics_reset(ResetCause::InletVelocity));
            ts.update(1.0 / 60.0);
        }
        assert_eq!(ts.len(), 1);
        let t = ts.iter().next().unwrap();
        assert_eq!(t.repeats, 59);
        assert!(t.display_title().contains("(x60)"), "{}", t.display_title());
        // ...and the timer restarted on the last push, so it is still fresh a
        // second into a one-second drag.
        assert!(t.age_s < 0.02, "the timer did not restart: {}", t.age_s);
    }

    #[test]
    fn coalesced_toasts_keep_the_worse_severity() {
        let mut ts = Toasts::new();
        ts.push(Toast::new(Severity::Notice, "boundary changed").with_key("k"));
        ts.push(Toast::new(Severity::Warning, "boundary changed").with_key("k"));
        assert_eq!(ts.len(), 1);
        assert_eq!(ts.iter().next().unwrap().severity, Severity::Warning);
        // ...and do not downgrade again.
        ts.push(Toast::new(Severity::Info, "boundary changed").with_key("k"));
        assert_eq!(ts.iter().next().unwrap().severity, Severity::Warning);
    }

    #[test]
    fn unkeyed_toasts_stack_rather_than_merging() {
        let mut ts = Toasts::new();
        ts.info("a");
        ts.info("b");
        assert_eq!(ts.len(), 2);
    }

    #[test]
    fn toasts_expire_on_their_own_schedule() {
        let mut ts = Toasts::new();
        ts.info("quick");
        ts.warn("slower");
        ts.update(4.0);
        assert_eq!(ts.len(), 1, "the info toast should have gone");
        assert_eq!(ts.iter().next().unwrap().severity, Severity::Warning);
        ts.update(6.0);
        assert!(ts.is_empty());
    }

    #[test]
    fn a_critical_toast_never_expires_on_its_own() {
        // Divergence must not scroll away while the user is looking elsewhere.
        let mut ts = Toasts::new();
        ts.error("solver diverged");
        ts.update(600.0);
        assert_eq!(ts.len(), 1);
        assert_eq!(ts.iter().next().unwrap().remaining(), 1.0, "a critical toast must not fade");
        ts.dismiss(0);
        assert!(ts.is_empty());
    }

    #[test]
    fn overflow_drops_the_oldest_expirable_toast_and_keeps_criticals() {
        let mut ts = Toasts::new();
        ts.error("diverged");
        for i in 0..MAX_TOASTS + 3 {
            ts.info(format!("msg {i}"));
        }
        assert_eq!(ts.len(), MAX_TOASTS);
        assert!(
            ts.iter().any(|t| t.severity == Severity::Critical),
            "the critical toast was evicted"
        );
    }

    #[test]
    fn fade_is_monotone_and_bounded() {
        let mut t = Toast::new(Severity::Info, "x");
        let mut last = 1.0;
        for _ in 0..40 {
            let r = t.remaining();
            assert!(r <= last + 1e-6 && (0.0..=1.0).contains(&r));
            last = r;
            t.age_s += 0.1;
        }
        assert_eq!(t.remaining(), 0.0);
        assert!(t.expired());
    }

    #[test]
    fn every_reset_cause_has_a_message() {
        for cause in [
            ResetCause::InletVelocity,
            ResetCause::InletAngle,
            ResetCause::InletOutletSwapped,
            ResetCause::VentChanged,
            ResetCause::GeometryChanged,
            ResetCause::DomainChanged,
            ResetCause::ResolutionChanged,
            ResetCause::FluidChanged,
            ResetCause::SolverRestarted,
            ResetCause::Manual,
        ] {
            assert!(!cause.message().is_empty(), "{cause:?} has no message");
            assert!(Toast::statistics_reset(cause).detail.contains(cause.message()));
        }
    }
}
