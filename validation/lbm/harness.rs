//! Shared scaffolding for the LBM validation cases.
//!
//! Included by each validation target with `#[path = "harness.rs"] mod harness;`.
//! The targets themselves are declared as `[[test]]` entries in
//! `crates/ad-solver/Cargo.toml`, so they live here (where the contract puts
//! validation) while still running under a plain `cargo test --workspace`.
//!
//! Dead code is expected: each target uses a subset of these helpers.
#![allow(dead_code)]

use ad_gpu::types::VelocitySet;
use ad_solver::{PaddedDomain, ReferenceLbm, SolverConfig};
use glam::{UVec3, Vec3};

/// Whether a GPU is available. GPU validation must *skip*, not fail, on a
/// machine without an adapter, so CI stays meaningful without one.
pub fn gpu() -> Option<ad_gpu::GpuContext> {
    ad_gpu::GpuContext::for_tests()
}

/// The same adapter, but a device deliberately denied 16-bit integer storage.
///
/// FP16C has two storage layouts: one `u16` per cell where the adapter supports
/// `SHADER_I16`, and two cells packed into a `u32` with atomic read-modify-write
/// stores where it does not. Only one of them can run on a given machine, so
/// without this the fallback is exercised precisely where nobody is testing —
/// and it is the path that runs on the weaker hardware, where a silent
/// correctness bug is least likely to be noticed.
///
/// Everything except `SHADER_I16` matches [`ad_gpu::GpuContext::new`]. The
/// `peak_bandwidth` is left unset because this context exists for correctness,
/// not for quoting throughput.
pub fn gpu_without_16bit_storage() -> Option<ad_gpu::GpuContext> {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    desc.backends = wgpu::Backends::PRIMARY;
    let instance = wgpu::Instance::new(desc);

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: ad_gpu::fallback_adapter_requested(),
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .ok()?;
    // Same rule as `GpuContext::for_tests`: a software adapter only on request.
    if adapter.get_info().device_type == wgpu::DeviceType::Cpu
        && !ad_gpu::fallback_adapter_requested()
    {
        eprintln!("skipping GPU test: only a software adapter");
        return None;
    }

    let available = adapter.features();
    let mut features = wgpu::Features::empty();
    let mut caps = ad_gpu::GpuCapabilities::default();
    for (f, slot) in [
        (wgpu::Features::SHADER_F16, &mut caps.shader_f16),
        (wgpu::Features::SUBGROUP, &mut caps.subgroups),
        (wgpu::Features::TIMESTAMP_QUERY, &mut caps.timestamps),
        (
            wgpu::Features::FLOAT32_FILTERABLE,
            &mut caps.float32_filterable,
        ),
    ] {
        if available.contains(f) {
            features |= f;
            *slot = true;
        }
    }

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("aeroduct device (16-bit storage withheld)"),
        required_features: features,
        required_limits: adapter.limits(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .ok()?;

    Some(ad_gpu::GpuContext::from_parts(
        instance, adapter, device, queue, caps,
    ))
}

/// A plane channel of `n_across` fluid cells between two bounce-back walls,
/// periodic along the flow (x) and span (z).
///
/// The span is a single cell: the exact solution is invariant along x and z, so
/// extra cells only cost time. Esoteric Pull handles a periodic axis of length 1
/// correctly — the link becomes a self-loop, and the pair's two slots stay
/// distinct, which is all the scheme requires.
///
/// The walls sit at the halfway points between the last fluid node and the halo,
/// so the channel width is exactly `n_across` lattice units. That is the number
/// the tau-independence test measures.
pub fn plane_channel(n_across: u32, cfg: SolverConfig) -> ReferenceLbm {
    let d = PaddedDomain::uniform_fluid(UVec3::new(1, n_across, 1), [true, false, true], cfg.set);
    ReferenceLbm::new(d, cfg)
}

/// A square duct `n_across` cells on a side, periodic along the flow.
///
/// Unlike a circular pipe, a square duct is represented *exactly* on a Cartesian
/// lattice: there is no staircase error at all, so a friction-factor comparison
/// measures the solver rather than the voxelisation.
pub fn square_duct(n_across: u32, cfg: SolverConfig) -> ReferenceLbm {
    let d = PaddedDomain::uniform_fluid(
        UVec3::new(1, n_across, n_across),
        [true, false, false],
        cfg.set,
    );
    ReferenceLbm::new(d, cfg)
}

/// Streamwise velocity across the channel, one entry per interior cell in y.
pub fn channel_profile(lbm: &ReferenceLbm) -> Vec<f32> {
    (0..lbm.domain.interior.y)
        .map(|y| lbm.macroscopic(UVec3::new(0, y, 0)).u.x)
        .collect()
}

/// Analytic plane-Poiseuille profile sampled at the lattice nodes.
///
/// With the walls halfway outside the first and last fluid node, node `j`
/// (0-based) sits at `y' = j + 1/2` measured from the lower wall, and the
/// channel width is `h = n_across`. `u = (F / (2 rho nu)) y' (h - y')`.
pub fn analytic_poiseuille(n_across: u32, force: f32, rho: f32, nu: f32) -> Vec<f32> {
    let h = n_across as f32;
    (0..n_across)
        .map(|j| {
            let y = j as f32 + 0.5;
            force / (2.0 * rho * nu) * y * (h - y)
        })
        .collect()
}

/// Peak of the analytic profile, `F h^2 / (8 rho nu)`.
pub fn analytic_u_max(n_across: u32, force: f32, rho: f32, nu: f32) -> f32 {
    let h = n_across as f32;
    force * h * h / (8.0 * rho * nu)
}

/// Body force that produces a given peak velocity in a channel `n_across` wide.
pub fn force_for_u_max(n_across: u32, u_max: f32, rho: f32, nu: f32) -> f32 {
    let h = n_across as f32;
    8.0 * rho * nu * u_max / (h * h)
}

/// Kinematic viscosity from the relaxation time.
pub fn nu_of(tau: f32) -> f32 {
    (tau - 0.5) / 3.0
}

/// Relative L2 error of `got` against `want`, normalised by the L2 norm of
/// `want`. This is the "L2 error" the acceptance criteria refer to.
pub fn l2_relative(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        num += ((g - w) as f64).powi(2);
        den += (*w as f64).powi(2);
    }
    (num / den.max(1e-300)).sqrt() as f32
}

/// Step until the velocity field stops changing, or `max_steps` is reached.
///
/// Returns the number of steps taken and the final relative change per step, so
/// a caller can assert that steady state was actually reached rather than
/// silently comparing a transient against an analytic answer.
pub fn run_to_steady(lbm: &mut ReferenceLbm, max_steps: u64, tol: f32) -> (u64, f32) {
    let sample = 200u64;
    let mut prev = channel_profile(lbm);
    let mut steps = 0u64;
    let mut change = f32::INFINITY;
    while steps < max_steps {
        for _ in 0..sample {
            lbm.step();
        }
        steps += sample;
        let now = channel_profile(lbm);
        let scale = now.iter().fold(0.0f32, |a, b| a.max(b.abs())).max(1e-20);
        change = now
            .iter()
            .zip(&prev)
            .fold(0.0f32, |a, (n, p)| a.max((n - p).abs()))
            / scale;
        prev = now;
        if change < tol {
            break;
        }
    }
    (steps, change)
}

/// The effective channel width the solver actually produced, in lattice units.
///
/// Invert `u_max = F h^2 / (8 rho nu)` for `h`. This is the diagnostic that
/// exposes a misplaced bounce-back wall: the *shape* of the profile stays
/// parabolic either way, but the width implied by its peak moves.
pub fn effective_width(u_max: f32, force: f32, rho: f32, nu: f32) -> f32 {
    (8.0 * rho * nu * u_max / force).sqrt()
}

/// Fit the peak of a parabola through the profile rather than taking the largest
/// sample, so an even number of cells (no node at the centreline) does not bias
/// the answer.
pub fn fitted_u_max(profile: &[f32]) -> f32 {
    // Least-squares fit of u = a + b*y + c*y^2 over the whole profile, then
    // evaluate at the vertex. The profile is a parabola by construction, so this
    // is exact for the ideal case and robust for the real one.
    let n = profile.len();
    let (mut s0, mut s1, mut s2, mut s3, mut s4) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
    let (mut t0, mut t1, mut t2) = (0.0f64, 0.0, 0.0);
    for (j, v) in profile.iter().enumerate() {
        let y = j as f64 + 0.5;
        let v = *v as f64;
        s0 += 1.0;
        s1 += y;
        s2 += y * y;
        s3 += y * y * y;
        s4 += y * y * y * y;
        t0 += v;
        t1 += v * y;
        t2 += v * y * y;
    }
    assert!(n >= 3, "need at least three samples to fit a parabola");
    // Solve the 3x3 normal equations by Cramer's rule.
    let m = [[s0, s1, s2], [s1, s2, s3], [s2, s3, s4]];
    let rhs = [t0, t1, t2];
    let det = det3(&m);
    let mut coef = [0.0f64; 3];
    for k in 0..3 {
        let mut mk = m;
        for r in 0..3 {
            mk[r][k] = rhs[r];
        }
        coef[k] = det3(&mk) / det;
    }
    let (a, b, c) = (coef[0], coef[1], coef[2]);
    let vertex = -b / (2.0 * c);
    (a + b * vertex + c * vertex * vertex) as f32
}

fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Bulk (area-averaged) velocity of a channel profile.
pub fn bulk_velocity(profile: &[f32]) -> f32 {
    profile.iter().sum::<f32>() / profile.len() as f32
}

/// Linear interpolation of a sampled profile at an arbitrary coordinate.
pub fn interp(samples: &[(f32, f32)], at: f32) -> f32 {
    if at <= samples[0].0 {
        return samples[0].1;
    }
    if at >= samples[samples.len() - 1].0 {
        return samples[samples.len() - 1].1;
    }
    for w in samples.windows(2) {
        let (x0, y0) = w[0];
        let (x1, y1) = w[1];
        if at >= x0 && at <= x1 {
            let t = (at - x0) / (x1 - x0).max(1e-20);
            return y0 + t * (y1 - y0);
        }
    }
    samples[samples.len() - 1].1
}

/// Default config for the analytic cases: no LES, FP32 storage, so the number
/// under test is the operator and nothing else.
pub fn clean_config(tau: f32) -> SolverConfig {
    SolverConfig {
        set: VelocitySet::D3Q19,
        tau0: tau,
        smagorinsky_c: 0.0,
        initial_velocity: Vec3::ZERO,
        ..Default::default()
    }
}
