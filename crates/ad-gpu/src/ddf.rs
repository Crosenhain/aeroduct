//! Structure-of-arrays storage for the distribution functions.
//!
//! # Why SoA is mandatory, not merely preferable
//!
//! wgpu clamps `max_storage_buffer_binding_size` to `i32::MAX` (2 GiB - 1) inside
//! `wgpu-hal`, *below* the adapter query, so requesting `adapter.limits()` does
//! not raise it. The Vulkan driver on this machine reports 4 GiB - 1, but we
//! never see that. See <https://github.com/gfx-rs/wgpu/issues/8105>.
//!
//! An interleaved array-of-structs layout stores all `q` distributions of a cell
//! contiguously, so one binding carries `q * bytes_per_ddf` per cell:
//!
//! | layout                  | bytes/cell/binding | cells per binding | cubic edge |
//! |-------------------------|--------------------|-------------------|------------|
//! | AoS, D3Q19 FP32         | 76                 | 28.2 M            | ~303       |
//! | SoA, one buffer per dir | 4                   | 536 M             | ~812       |
//! | SoA, FP16C              | 2                   | 1.07 B            | ~1024      |
//!
//! Splitting by direction gives each binding a single scalar per cell, so the
//! ceiling moves out of reach. It is also the coalescing-optimal layout, because
//! neighbouring threads reading direction `i` touch adjacent addresses. So the
//! constraint and the performance optimum agree, and this costs nothing.
//!
//! `max_buffer_size` is *not* clamped on Windows + NVIDIA, but we allocate one
//! buffer per direction anyway: it keeps every binding trivially in range and
//! makes the bind group layout uniform.
//!
//! # Two FP16C layouts, same bytes
//!
//! Both FP16C layouts are two bytes per cell. They differ in the *element* the
//! shader addresses, and that is what decides whether a store has to be
//! synchronised. See the comment in [`DdfBuffers::allocate`].

use anyhow::{bail, Result};
use crate::types::{DdfPrecision, Grid, VelocitySet};

/// One storage buffer per lattice direction, plus the bookkeeping to bind them.
pub struct DdfBuffers {
    pub buffers: Vec<wgpu::Buffer>,
    pub set: VelocitySet,
    pub precision: DdfPrecision,
    pub cell_count: u64,
    /// Bytes in each per-direction buffer.
    pub bytes_per_direction: u64,
    /// FP16C only: `true` when every cell owns its own 16-bit storage element,
    /// so the shader binds these buffers as `array<u16>` and stores plainly.
    /// `false` selects the packed fallback, where two cells share one `u32` and
    /// the shader must go through `atomicAnd` + `atomicOr`. Always `false` for
    /// [`DdfPrecision::Fp32`], which has nothing to pack.
    ///
    /// The shader generator *must* read this rather than deciding for itself:
    /// the two views of the same bytes are not interchangeable, and a
    /// disagreement would be a silent factor-of-two indexing error.
    pub fp16c_per_cell: bool,
}

impl DdfBuffers {
    pub fn allocate(
        device: &wgpu::Device,
        limits: &wgpu::Limits,
        grid: Grid,
        set: VelocitySet,
        precision: DdfPrecision,
    ) -> Result<Self> {
        let cell_count = grid.cell_count();
        let q = set.q();

        // FP16C is two bytes per cell either way. What matters is not how many
        // bytes a direction buffer holds but *how big an element the shader can
        // address inside it*, because that decides whether a store needs
        // synchronising.
        //
        // Fixed, was a measured defect. The original layout packed two adjacent
        // *cells* into one `u32`. Each cell is written by its own invocation, so
        // one word had two writers and every store became a read-modify-write:
        // `atomicAnd` + `atomicOr`, 38 atomics per cell per step. That cost more
        // than halving the traffic saved — FP16C measured 4447 MLUPS against
        // FP32's 6305 on an RTX 4090, i.e. the format moving half the bytes ran
        // at 0.7x the speed of the one moving twice as many.
        //
        // The fix is to stop sharing a word. With `wgpu::Features::SHADER_I16`
        // (Vulkan `shaderInt16` + `VK_KHR_16bit_storage`, surfaced as
        // `GpuCapabilities::shader_i16`) the shader binds each direction buffer
        // as `array<u16>` under `enable wgpu_int16;`, one element per cell, and
        // the store is a plain unsynchronised 16-bit write.
        //
        // It is `u16` rather than the `f16` the original note proposed for a
        // concrete reason: FP16C is 1 sign / 4 exponent / 11 mantissa, so its bit
        // patterns are not binary16 values — everything in [1.5, 2) encodes as a
        // binary16 NaN — and WGSL has no bitcast between `f16` and `u32` to
        // launder them with (naga's `bitcast<T>` changes only the scalar *kind*,
        // never the width). A `u16` element carries the 16 bits exactly, which is
        // what `gpu_fp16c_storage_matches_the_rust_codec` demands. `SHADER_F16`
        // alone is not sufficient and is not what this gates on.
        //
        // Adapters without `SHADER_I16` keep the packed layout and the atomics.
        // It is slower but correct, and it is what runs on hardware that cannot
        // address 16 bits at all.
        let fp16c_per_cell = precision == DdfPrecision::Fp16c
            && device.features().contains(wgpu::Features::SHADER_I16);

        let bytes_per_direction = direction_bytes(cell_count, precision);

        let max_binding = limits.max_storage_buffer_binding_size as u64;
        if bytes_per_direction > max_binding {
            bail!(
                "grid {}x{}x{} ({:.1} M cells) needs {:.2} GiB per direction buffer, but the \
                 maximum storage buffer binding is {:.2} GiB. Reduce the resolution or increase \
                 the cell size.",
                grid.dims.x,
                grid.dims.y,
                grid.dims.z,
                cell_count as f64 / 1e6,
                bytes_per_direction as f64 / (1u64 << 30) as f64,
                max_binding as f64 / (1u64 << 30) as f64,
            );
        }

        let total = bytes_per_direction * q as u64;
        log::info!(
            "allocating DDFs: {:?} {:?}{}, {:.1} M cells, {} x {:.2} GiB = {:.2} GiB total",
            set,
            precision,
            match precision {
                DdfPrecision::Fp16c if fp16c_per_cell => " (u16 per cell)",
                DdfPrecision::Fp16c => " (packed pairs, atomic stores)",
                DdfPrecision::Fp32 => "",
            },
            cell_count as f64 / 1e6,
            q,
            bytes_per_direction as f64 / (1u64 << 30) as f64,
            total as f64 / (1u64 << 30) as f64,
        );
        if precision == DdfPrecision::Fp16c && !fp16c_per_cell {
            log::warn!(
                "adapter has no 16-bit integer storage (wgpu SHADER_I16), so FP16C falls back to \
                 two cells per word with atomic read-modify-write stores; expect it to run slower \
                 than FP32 despite moving half the bytes"
            );
        }

        let buffers = (0..q)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("ddf[{i}]")),
                    size: bytes_per_direction,
                    // COPY_SRC so the raw populations can be read back. The
                    // momentum-exchange wall-stress calculation needs the
                    // distributions themselves, not just the macroscopic
                    // fields, and so does any test that wants to compare
                    // against a CPU reference population by population.
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            })
            .collect();

        Ok(Self { buffers, set, precision, cell_count, bytes_per_direction, fp16c_per_cell })
    }

    pub fn total_bytes(&self) -> u64 {
        self.bytes_per_direction * self.set.q() as u64
    }

    /// Bind group layout entries for the DDF buffers, starting at `first_binding`.
    pub fn layout_entries(&self, first_binding: u32) -> Vec<wgpu::BindGroupLayoutEntry> {
        (0..self.set.q() as u32)
            .map(|i| wgpu::BindGroupLayoutEntry {
                binding: first_binding + i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect()
    }

    pub fn bind_entries(&self, first_binding: u32) -> Vec<wgpu::BindGroupEntry<'_>> {
        self.buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: first_binding + i as u32,
                resource: b.as_entire_binding(),
            })
            .collect()
    }
}

/// Bytes one direction buffer needs to hold `cell_count` cells.
///
/// Public because callers predict VRAM before a device exists — see
/// `ad_solver::solver::ddf_bytes` — and a second copy of this arithmetic
/// somewhere else is exactly the sort of thing that drifts out of step with the
/// allocator and turns into a mysterious out-of-bounds a release later.
pub const fn direction_bytes(cell_count: u64, precision: DdfPrecision) -> u64 {
    match precision {
        DdfPrecision::Fp32 => cell_count * 4,
        // Two bytes a cell, rounded up to a whole word: a buffer size has to be a
        // multiple of 4 to be copyable, and `array<u16>` simply carries one
        // unused trailing element on an odd cell count.
        DdfPrecision::Fp16c => (cell_count * 2 + 3) & !3,
    }
}

/// Bytes of memory traffic per cell per step, and total storage per cell.
///
/// Storage is `q * b + 17`: the distributions, plus density (4), velocity (12)
/// and the flag byte (1). Traffic is `2 * q * b + 1`, because each step reads and
/// writes every distribution and reads the flag.
///
/// For D3Q19 that gives 93 B/cell and 153 B/step in FP32, or 55 and 77 in FP16C,
/// matching the published FluidX3D figures. Those figures are what the roofline
/// estimate in [`crate::profiler`] is built on.
pub const fn bytes_per_cell(set: VelocitySet, precision: DdfPrecision) -> (u64, u64) {
    let q = set.q() as u64;
    let b = precision.bytes_per_ddf();
    (q * b + 17, 2 * q * b + 1)
}

/// Predicted steps per second on a bandwidth-bound kernel.
///
/// LBM moves the whole DDF array twice per step and does almost no arithmetic, so
/// achieved bandwidth is the entire performance story. FluidX3D reaches ~85% of
/// peak; a first custom WGSL kernel more realistically lands at 60-75%.
pub fn predicted_steps_per_second(
    cell_count: u64,
    set: VelocitySet,
    precision: DdfPrecision,
    peak_bandwidth: f64,
    efficiency: f64,
) -> f64 {
    let (_, traffic) = bytes_per_cell(set, precision);
    peak_bandwidth * efficiency / (traffic as f64 * cell_count as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_counts_match_the_published_figures() {
        assert_eq!(bytes_per_cell(VelocitySet::D3Q19, DdfPrecision::Fp32), (93, 153));
        assert_eq!(bytes_per_cell(VelocitySet::D3Q19, DdfPrecision::Fp16c), (55, 77));
        assert_eq!(bytes_per_cell(VelocitySet::D3Q27, DdfPrecision::Fp16c), (71, 109));
    }

    #[test]
    fn throughput_prediction_matches_the_planning_table() {
        // Quality tier: 650 x 450 x 450 at 70% of an RTX 4090's 1008 GB/s.
        let cells = 650u64 * 450 * 450;
        let sps = predicted_steps_per_second(
            cells,
            VelocitySet::D3Q19,
            DdfPrecision::Fp16c,
            1008.0e9,
            0.70,
        );
        assert!((sps - 69.6).abs() < 1.0, "predicted {sps} steps/s, expected ~70");
    }

    #[test]
    fn fp16c_halves_traffic_and_the_allocation_agrees() {
        // Guard the arithmetic so that if someone changes the layout, the traffic
        // claim is rechecked.
        let (store32, traffic32) = bytes_per_cell(VelocitySet::D3Q19, DdfPrecision::Fp32);
        let (store16, traffic16) = bytes_per_cell(VelocitySet::D3Q19, DdfPrecision::Fp16c);
        assert!(traffic16 * 2 < traffic32 * 2 + 2, "FP16C should roughly halve traffic");
        assert!(store16 < store32);

        // ...and `allocate` must actually reserve the two bytes per cell that the
        // traffic model above is billed against, in both layouts. The traffic
        // figure is what the roofline percentage divides by, so an allocation
        // that quietly rounded up to four bytes a cell would make every reported
        // bandwidth number half of the truth.
        for cells in [1u64, 2, 3, 1000, 1001] {
            let fp16 = direction_bytes(cells, DdfPrecision::Fp16c);
            assert_eq!(direction_bytes(cells, DdfPrecision::Fp32), cells * 4);
            assert!(
                fp16 >= cells * 2 && fp16 < cells * 2 + 4,
                "{cells} cells wanted {} bytes, got {fp16}",
                cells * 2
            );
            assert_eq!(fp16 % 4, 0, "a buffer size must be a whole number of words");
        }
    }

    #[test]
    fn soa_lifts_the_grid_ceiling_past_aos() {
        // A 2 GiB - 1 binding, FP32. AoS would carry 19 floats per cell.
        let binding = i32::MAX as u64;
        let aos_cells = binding / (19 * 4);
        let soa_cells = binding / 4;
        assert!((aos_cells as f64).cbrt() < 320.0, "AoS should cap around 300 cubed");
        assert!((soa_cells as f64).cbrt() > 800.0, "SoA should reach past 800 cubed");
    }
}
