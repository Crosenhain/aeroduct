# AeroDuct build contract

Read this before touching anything. It defines the interfaces the parallel build
agents share, and the file ownership that keeps them from colliding.

## What this app is

An interactive GPU airflow simulator for 3D-printed ducts. Load STL(s), place an
inlet, run a lattice-Boltzmann simulation on the GPU, and get a live visualisation
plus the engineering numbers that say whether the duct is any good.

The test part is `parts/Airflow redirector - Part 1.stl`: 85,180 triangles,
watertight 2-manifold, genus 1. It is a ~90 degree bend with a ~1.85:1 area
contraction, 145.0 x 72.2 x 68.9 mm, with two mouths flush against bounding-box
planes:

| | Mouth A | Mouth B |
|---|---|---|
| plane | `z = 0` | `y = 0` |
| open area | 2116 mm^2 (~139 x 15) | 1141 mm^2 (~74 x 15) |

Median internal passage width is **6.3 mm**, walls ~2 mm. Reynolds number is
1,600-15,500 over a 2-8 m/s vent range.

## Verified hardware baseline

RTX 4090, Vulkan backend, driver 616.56. Confirmed by running `ad-app`:

```
max_buffer_size = 4.00 GiB
max_storage_buffer_binding_size = 2.00 GiB     <-- the load-bearing one
max_storage_buffers_per_shader_stage = 524288
f16=true subgroups=true timestamps=true pipeline_cache=true f32_filterable=true
peak bandwidth 1008 GB/s
```

## Non-negotiables

1. **Never bypass `ad-gpu`.** It owns the device, the shared types, the WGSL
   preprocessor, the DDF allocation and the profiler. If you need something added
   to it, say so in your final report rather than editing it — a silent change to
   the contract breaks the other agents working in parallel right now.

2. **Distribution functions are structure-of-arrays, one storage buffer per
   lattice direction.** wgpu clamps a storage buffer *binding* to 2 GiB - 1
   (`MAX_I32_BINDING_SIZE` in `wgpu-hal`), below the adapter query, so it cannot
   be raised. Interleaved AoS would cap the grid at ~303 cubed; SoA reaches ~812
   cubed. Use `ad_gpu::DdfBuffers`.

3. **All DDF indexing goes through `ad_gpu::lattice::EsotericPull`.** Half of a
   cell's distributions physically live in a neighbour's array slot. Indexing the
   buffers directly will appear to work and produce subtly wrong physics. The
   scheme is derived in that module's doc comment and verified exhaustively by
   `esoteric_pull_store_then_load_round_trips`.

4. **Direction ordering is fixed**: `i` and `i+1` are opposites for odd `i`, so
   the pairs are `(1,2), (3,4), ... (17,18)`. `opposite(i)` is
   `((i-1) ^ 1) + 1`, **not** `i ^ 1`. Do not reorder the tables.

5. **WGSL storage textures are write-only in the portable spec.** Write through a
   storage binding, read through a *sampled* view plus a linear sampler. Two
   different views of the same texture, two different bind groups.

6. **Millimetres everywhere** in model and world space, because that is what CAD
   exports. Only `LatticeUnits` converts to SI.

7. **Test what you write.** Every module needs unit tests that would actually fail
   if the logic were wrong. Wave 0 shipped 21 tests and three of them caught real
   bugs in the first run, including a wrong `opposite()` and wrong streaming
   arithmetic. Prefer a test that checks a physical invariant (mass conservation,
   a moment closure, a round-trip) over one that checks a hardcoded number.

## Shared types — `ad-gpu`

Read `crates/ad-gpu/src/types.rs` in full before starting. Summary:

| type | purpose |
|---|---|
| `Bbox` | AABB in mm |
| `Grid` | dims, `dx_mm`, `origin_mm` (centre of cell 0,0,0); `linear()` is X-fastest |
| `VelocitySet` | `D3Q19` (default) or `D3Q27` |
| `DdfPrecision` | `Fp32` (verification) or `Fp16c` (production) |
| `flags::*` | per-cell `u8` bitfield: `SOLID`, `SOLID_BOUNDARY`, `INLET`, `OUTLET`, `SPONGE`, `EQUILIBRIUM` |
| `LatticeUnits` | SI <-> lattice conversion, plus `warnings()` |
| `SimUniforms` | GPU uniform mirror, `repr(C)`, 16-byte aligned |
| `BoundaryLink` | `(cell, direction, q)` for interpolated bounce-back |
| `FlowPatch` | planar inlet / outlet / measurement patch |
| `MetricSample` | one reduced measurement; the only element type in the readback ring |

And the helpers:

- `GpuContext::new_blocking(None)` — device with `adapter.limits()` requested.
- `ShaderLoader` / `ShaderDefines` — `#include`, `#if/#elif/#else/#endif`, and
  `#NAME` value substitution. `loader.add_virtual(name, src)` injects generated
  source; use it for `lattice::wgsl_prelude(set)` so the shader tables can never
  drift from the Rust ones.
- `Profiler::scope(name)` — attach to a compute pass descriptor; report with
  `summary(name, bytes)`, which gives percent-of-roofline.
- `ddf::bytes_per_cell(set, precision)` — `(storage, traffic_per_step)`.
  D3Q19/FP16C is `(55, 77)`.

## Physics decisions already made

Do not relitigate these; they are in the approved plan with citations.

- **D3Q19** default, D3Q27 switchable.
- **TRT**, magic parameter `Lambda = 3/16` (puts the bounce-back wall exactly
  halfway independent of viscosity). Fall back to 1/4 if unstable. Keep the
  collision operator behind a trait so regularized-BGK or cumulant can drop in.
- **Smagorinsky-Lilly** from the local non-equilibrium moments,
  `tau_eff = 0.5 * [tau0 + sqrt(tau0^2 + 18*sqrt(2)*Cs^2*sqrt(PiPi)/rho)]`,
  `Cs = 0.10..0.12`, clamp `tau_eff` to `[tau0, 1.0]`.
- **No wall function.** At `dx = 0.4 mm` the first fluid node sits at
  `y+ = 1.8..6.9` across the whole speed range, i.e. inside the viscous sublayer.
  A wall function assumes `y+ >= 30` and would impose a wrong shear stress.
- **Esoteric Pull** streaming, **FP32 arithmetic / FP16C storage** with
  DDF-shifting (store `f_i - w_i`; the order of operations matters).
- **Walls**: halfway bounce-back first, then single-node interpolated bounce-back
  (Marson et al., arXiv:2009.04604) using per-link `q`.
- **Inlet**: equilibrium for v1. Place it ~2 D_h upstream inside a straight
  extension so a boundary layer develops rather than injecting plug flow.
- **Outlet**: convective outflow + 20-cell sponge, pressure reference by
  anti-bounce-back. This is deliberate headroom for later acoustics work.
- **Box sides**: equilibrium boundaries, so the exit jet can entrain surrounding
  air. Never periodic, never no-slip.

## Resolution tiers

Domain 260 x 180 x 180 mm around the scene bbox.

| tier | dx | grid | cells | VRAM (D3Q19/FP16C) | steps/s |
|---|---|---|---|---|---|
| interactive | 0.75 mm | 347x240x240 | 20 M | 1.1 GB | ~460 |
| quality | 0.40 mm | 650x450x450 | 132 M | 7.2 GB | ~70 |
| max | 0.30 mm | 867x600x600 | 312 M | 17.2 GB | ~29 |

The *coarse* tier is the stability challenge, not the fine one: `tau` rises as
`dx` falls (0.50058 at 1 mm, 0.50194 at 0.3 mm), so test the collision operator at
0.75-1.0 mm first.

## Hand-calc targets

Inlet on mouth A, air at rho = 1.2:

| U_in | Q (L/s) | CFM | U_out | expected dp |
|---|---|---|---|---|
| 2 | 4.23 | 9.0 | 3.71 | 4-10 Pa |
| 3 | 6.35 | 13.5 | 5.56 | 9-22 Pa |
| 5 | 10.58 | 22.4 | 9.27 | 26-62 Pa |
| 8 | 16.93 | 35.9 | 14.84 | 66-159 Pa |

Loss coefficient target `K < 1`, ideally 0.3-0.6.

> **CORRECTION (measured, supersedes what this file said before).** This document
> originally described the part as "an uncut mitred bend" expected to run
> K = 2.0-3.5. **Both halves of that were wrong.**
>
> The bend is *well radiused*. Fitting a circle to the passage centroid track
> gives a radius of **57.4 mm** holding between 54 and 61 mm across the whole
> turn -- a genuine circular arc, not a corner -- against `D_h` = 23.8 mm, so
> **r/D_h = 2.4** (independently: the estimator crate measures 2.59). A mitre is
> r/D_h ~ 0. Handbook loss for a radiused bend at r/D > 1.5 is K = 0.2-0.3, an
> order of magnitude below what this file claimed. The part also turns about its
> *short* side and tapers over a gradual ~10 degree included angle: the cheapest
> version of all three features, not the most expensive.
>
> The K = 2.0-3.5 figure was carried in from a research summary and never checked
> against the geometry. It then set the expectation that a measured K of 12.5
> was merely "high" rather than obviously wrong.
>
> The 1D estimator (`ad-estimate`) puts the real figure at **K = 0.71 +/- 0.14**
> mouth-to-mouth, i.e. the duct is *inside* its design target. It reproduces the
> hand-calc table above at all four operating points once the discharge loss is
> added, and the offset is exactly one outlet velocity head -- a bookkeeping
> identity rather than a fitted agreement.
>
> Note also that the estimator's sources (ASHRAE CR3-3, Idelchik 6-7, Crane
> TP-410) put a *bare mitred rectangular* elbow at K = 1.0-1.6, not 2.0-3.5.
> That disagreement is about aspect-ratio convention and is **not resolved**; it
> is moot here only because this bend is radiused, so the mitre correlations do
> not apply at all.
>
> **Consequence for the solver:** the LBM's K = 12.5 +/- 0.9 is not a plausible
> reading of this geometry. Even the most pessimistic reading -- call the bend a
> sharp mitre *and* the taper a sudden step -- reaches only K = 2.45. Treat the
> LBM pressure drop as unvalidated until the resolution study closes.

## File ownership

Stay inside your lane. Do not create files outside it.

| agent | owns |
|---|---|
| geometry | `crates/ad-geom/**`, `shaders/geom/**`, `validation/geom/**` |
| solver | `crates/ad-solver/**`, `shaders/lbm/**`, `validation/lbm/**` |
| render core | `crates/ad-render/**`, `shaders/render/**`, `assets/**` |
| metrics | `crates/ad-metrics/**`, `shaders/metrics/**` |
| ui | `crates/ad-ui/**` |
| integration | `crates/ad-app/**` |

Nobody edits `crates/ad-gpu/**` or this file except the coordinator.

## Build

```bash
cargo build --workspace
cargo test --workspace
cargo run --release -p ad-app
```

`cargo test --workspace` must stay green. If you need a test that requires a GPU,
gate it so it skips cleanly when no adapter is available rather than failing CI.
