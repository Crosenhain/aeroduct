//! GPU device acquisition.
//!
//! The single most important thing here is that we request `adapter.limits()`
//! rather than accepting the WebGPU defaults. The defaults cap `max_buffer_size`
//! at 256 MiB and `max_storage_buffer_binding_size` at 128 MiB, which would
//! silently limit the solver to roughly a 150-cubed grid and produce a baffling
//! allocation failure much later.

use anyhow::{Context as _, Result};
use std::sync::Arc;

/// Optional GPU features. None are required; each unlocks an optimisation, and we
/// record what we actually got so the rest of the app can branch on it.
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuCapabilities {
    /// Native f16 arithmetic. Not needed for FP16C storage (which goes through
    /// integer bit manipulation on u32) but useful elsewhere.
    pub shader_f16: bool,
    /// 16-bit integers in shaders (`enable wgpu_int16;`, `u16`), which on Vulkan
    /// also brings `VK_KHR_16bit_storage`. This is what lets an FP16C direction
    /// buffer be `array<u16>` with one element per cell instead of two cells
    /// packed into a `u32`; see [`crate::ddf`] for why that matters. `f16` is
    /// *not* a substitute: WGSL has no way to move a bit pattern between `f16`
    /// and `u32`, and the FP16C encoding is not a valid binary16 value.
    pub shader_i16: bool,
    /// Subgroup reductions, used by the metrics plane integrals.
    pub subgroups: bool,
    /// GPU timestamps, used to report achieved bandwidth against the roofline.
    pub timestamps: bool,
    /// Serialised pipeline cache. Compiling the 19-direction kernels is slow
    /// enough that this is worth having on Vulkan.
    pub pipeline_cache: bool,
    /// Sampling a Float32 texture with a filtering sampler.
    pub float32_filterable: bool,
    /// Peak theoretical memory bandwidth in bytes/sec, if we could identify the
    /// device. Used for the roofline percentage in the profiler, and to predict
    /// throughput at a resolution nobody has run yet.
    pub peak_bandwidth: Option<f64>,
    /// Dedicated device memory in bytes, if we could identify the device or
    /// `AERODUCT_VRAM_GB` says.
    ///
    /// wgpu has no portable query for it, and the resolution control needs a
    /// ceiling to refuse against. Past the end of VRAM a Vulkan driver does not
    /// necessarily fail the allocation: on Windows it may place it in system
    /// memory instead, where a bandwidth-bound solver runs at PCIe speed with
    /// no error anywhere. `None` skips that check, and nothing else stands in
    /// for it: running out of memory mid-build usually loses the device (see
    /// [`GpuContext::lost`]), which a transactional rebuild cannot undo.
    pub vram_bytes: Option<u64>,
}

pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub limits: wgpu::Limits,
    pub caps: GpuCapabilities,
    pub info: wgpu::AdapterInfo,
    /// Set, with wgpu's reason, once the device is lost. See [`Self::lost`].
    lost: Arc<std::sync::Mutex<Option<String>>>,
}

impl GpuContext {
    /// Acquire a device, preferring a discrete GPU on the Vulkan backend.
    ///
    /// `compatible_surface` should be supplied when a window already exists so the
    /// adapter chosen can actually present to it.
    pub async fn new(compatible_surface: Option<&wgpu::Surface<'_>>) -> Result<Self> {
        // PRIMARY is Vulkan + Metal + DX12. On this machine that resolves to
        // Vulkan, which is what we want; Metal keeps the macOS build honest.
        // WGPU_BACKEND and friends still override, which is handy for bisecting
        // a driver bug.
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        desc.backends = wgpu::Backends::PRIMARY;
        // `AERODUCT_MEMORY_BUDGET_PCT`: make allocations fail once this process
        // passes that percentage of the memory budget `VK_EXT_memory_budget`
        // reports. Off by default, deliberately. It looks like the ideal guard —
        // unlike the resolution control's estimate, the driver's budget knows
        // what every other process is using — but what it raises is a driver
        // out-of-memory, and in wgpu 30 that loses the device unless it comes
        // from creating a buffer, texture, sampler or query set (see
        // [`GpuContext::lost`]). Measured: a buffer creation failed softly, the
        // small staging uploads right behind it did not, the device was lost,
        // and the next unmap of a live metrics buffer found it destroyed. As a
        // guard it would turn a possibly slow allocation into a dead device, so
        // it exists only to rehearse that failure on demand.
        desc.memory_budget_thresholds.for_resource_creation =
            std::env::var("AERODUCT_MEMORY_BUDGET_PCT")
                .ok()
                .and_then(|v| v.trim().parse::<u8>().ok())
                .filter(|p| (1..=100).contains(p));
        if let Some(p) = desc.memory_budget_thresholds.for_resource_creation {
            log::warn!(
                "AERODUCT_MEMORY_BUDGET_PCT: allocations fail past {p}% of the driver's memory \
                 budget, and wgpu loses the device when one does"
            );
        }
        let instance = wgpu::Instance::new(desc);

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface,
                // Bucketing rounds the reported limits down to coarse tiers to
                // resist fingerprinting. That is a browser concern; here it would
                // just cost us grid resolution.
                apply_limit_buckets: false,
            })
            .await
            .context("no suitable GPU adapter found (is a Vulkan or Metal driver installed?)")?;

        let info = adapter.get_info();
        let adapter_features = adapter.features();
        let adapter_limits = adapter.limits();

        let mut features = wgpu::Features::empty();
        let mut caps = GpuCapabilities::default();

        let mut want = |f: wgpu::Features, slot: &mut bool| {
            if adapter_features.contains(f) {
                features |= f;
                *slot = true;
            }
        };
        want(wgpu::Features::SHADER_F16, &mut caps.shader_f16);
        want(wgpu::Features::SHADER_I16, &mut caps.shader_i16);
        want(wgpu::Features::SUBGROUP, &mut caps.subgroups);
        want(wgpu::Features::TIMESTAMP_QUERY, &mut caps.timestamps);
        want(wgpu::Features::PIPELINE_CACHE, &mut caps.pipeline_cache);
        want(
            wgpu::Features::FLOAT32_FILTERABLE,
            &mut caps.float32_filterable,
        );
        caps.peak_bandwidth = peak_bandwidth_for(&info.name);
        caps.vram_bytes = vram_override().or_else(|| vram_bytes_for(&info.name));

        // Ask for everything the adapter can actually do. See the module comment.
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("aeroduct device"),
                required_features: features,
                required_limits: adapter_limits.clone(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .context("failed to create GPU device with adapter limits")?;

        let limits = device.limits();
        log_environment(&info, &limits, &caps);

        let lost = watch_for_loss(&device);

        Ok(Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            limits,
            caps,
            info,
            lost,
        })
    }

    pub fn new_blocking(compatible_surface: Option<&wgpu::Surface<'_>>) -> Result<Self> {
        pollster::block_on(Self::new(compatible_surface))
    }

    /// The hard ceiling on a single storage buffer *binding*.
    ///
    /// wgpu clamps this to `i32::MAX` in `wgpu-hal` below the adapter query, so it
    /// cannot be raised even though the Vulkan driver reports 4 GiB - 1. This is
    /// the reason the DDFs must be laid out as structure-of-arrays; see
    /// [`crate::ddf`].
    pub fn max_binding_bytes(&self) -> u64 {
        self.limits.max_storage_buffer_binding_size as u64
    }

    pub fn max_buffer_bytes(&self) -> u64 {
        self.limits.max_buffer_size
    }

    /// Why the device was lost, if it has been.
    ///
    /// Worth asking after anything that can run out of memory. wgpu 30 treats a
    /// driver out-of-memory as fatal to the whole device everywhere except the
    /// creation of a buffer, texture, sampler or query set
    /// (`wgpu_core::device::Device::handle_hal_error`), so the staging uploads
    /// and submissions around a big allocation lose it; and a lost device frees
    /// every resource the app holds. Nothing can be recovered afterwards short
    /// of starting over.
    pub fn lost(&self) -> Option<String> {
        self.lost.lock().ok().and_then(|g| g.clone())
    }

    /// Wrap a device requested elsewhere — the validation harness does, to
    /// withhold features — with the same lost-device hook [`Self::new`] sets.
    pub fn from_parts(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        caps: GpuCapabilities,
    ) -> Self {
        let lost = watch_for_loss(&device);
        Self {
            limits: device.limits(),
            info: adapter.get_info(),
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            caps,
            lost,
        }
    }
}

/// Record a lost device, so the app can find out and stop cleanly rather than
/// discover it through the first validation error on a freed resource, which
/// wgpu's default handler turns into a panic somewhere unrelated. See
/// [`GpuContext::lost`].
fn watch_for_loss(device: &wgpu::Device) -> Arc<std::sync::Mutex<Option<String>>> {
    let lost = Arc::new(std::sync::Mutex::new(None::<String>));
    let flag = lost.clone();
    device.set_device_lost_callback(move |reason, message| {
        match reason {
            wgpu::DeviceLostReason::Destroyed => log::debug!("GPU device destroyed: {message}"),
            _ => log::error!("GPU device lost ({reason:?}): {message}"),
        }
        if let Ok(mut slot) = flag.lock() {
            slot.get_or_insert(format!("{reason:?}: {message}"));
        }
    });
    lost
}

/// Peak memory bandwidth for GPUs we recognise, so the profiler can report a
/// percentage of roofline rather than a bare GB/s that means nothing on its own.
fn peak_bandwidth_for(name: &str) -> Option<f64> {
    let n = name.to_ascii_lowercase();
    // GDDR6X 384-bit at 21 Gbps.
    if n.contains("4090") {
        return Some(1008.0e9);
    }
    if n.contains("4080") {
        return Some(717.0e9);
    }
    if n.contains("3090") {
        return Some(936.0e9);
    }
    if n.contains("5090") {
        return Some(1792.0e9);
    }
    None
}

fn log_environment(info: &wgpu::AdapterInfo, limits: &wgpu::Limits, caps: &GpuCapabilities) {
    log::info!(
        "GPU: {} ({:?}, {:?})",
        info.name,
        info.device_type,
        info.backend
    );
    log::info!("driver: {} {}", info.driver, info.driver_info);
    log::info!(
        "limits: max_buffer_size={:.2} GiB, max_storage_buffer_binding_size={:.2} GiB",
        limits.max_buffer_size as f64 / (1u64 << 30) as f64,
        limits.max_storage_buffer_binding_size as f64 / (1u64 << 30) as f64,
    );
    log::info!(
        "limits: max_storage_buffers_per_shader_stage={}, max_compute_invocations_per_workgroup={}",
        limits.max_storage_buffers_per_shader_stage,
        limits.max_compute_invocations_per_workgroup,
    );
    log::info!(
        "features: f16={} i16={} subgroups={} timestamps={} pipeline_cache={} f32_filterable={}",
        caps.shader_f16,
        caps.shader_i16,
        caps.subgroups,
        caps.timestamps,
        caps.pipeline_cache,
        caps.float32_filterable,
    );
    if let Some(bw) = caps.peak_bandwidth {
        log::info!("recognised peak bandwidth: {:.0} GB/s", bw / 1e9);
    } else {
        log::info!("peak bandwidth unknown for this device; roofline percentages disabled");
    }
    match caps.vram_bytes {
        Some(v) => log::info!("device memory: {:.1} GiB", v as f64 / (1u64 << 30) as f64),
        None => log::info!(
            "device memory size unknown; the resolution control will not check that a lattice \
             fits (set AERODUCT_VRAM_GB)"
        ),
    }

    // 19 SoA bindings plus flags, macroscopic fields and uniforms. If an adapter
    // ever reports fewer, the solver has to pack directions together and the
    // grid ceiling drops, so say so loudly rather than failing at pipeline
    // creation with an opaque validation error.
    if limits.max_storage_buffers_per_shader_stage < 24 {
        log::warn!(
            "adapter allows only {} storage buffers per stage; the D3Q19 solver wants 24",
            limits.max_storage_buffers_per_shader_stage
        );
    }
}

/// Dedicated memory for GPUs we recognise. Desktop parts only: the laptop
/// variants share the model number and not the memory — an "RTX 4090 Laptop
/// GPU" has 16 GB, not 24 — so anything calling itself a laptop part is left
/// unknown rather than guessed.
fn vram_bytes_for(name: &str) -> Option<u64> {
    const GIB: u64 = 1 << 30;
    let n = name.to_ascii_lowercase();
    if n.contains("laptop") || n.contains("mobile") {
        return None;
    }
    let gib = if n.contains("5090") {
        32
    } else if n.contains("4090") || n.contains("3090") {
        24
    } else if n.contains("4080") {
        16
    } else {
        return None;
    };
    Some(gib * GIB)
}

/// `AERODUCT_VRAM_GB`: the device memory size in GiB, for a GPU the table does
/// not know — or a smaller figure than the real one, to exercise the resolution
/// control's refusal path.
fn vram_override() -> Option<u64> {
    let gib = std::env::var("AERODUCT_VRAM_GB")
        .ok()?
        .parse::<f64>()
        .ok()?;
    (gib.is_finite() && gib > 0.0).then(|| (gib * (1u64 << 30) as f64) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_laptop_part_is_not_mistaken_for_the_desktop_card_it_is_named_after() {
        const GIB: u64 = 1 << 30;
        assert_eq!(vram_bytes_for("NVIDIA GeForce RTX 4090"), Some(24 * GIB));
        assert_eq!(vram_bytes_for("NVIDIA GeForce RTX 5090"), Some(32 * GIB));
        assert_eq!(
            vram_bytes_for("NVIDIA GeForce RTX 4080 SUPER"),
            Some(16 * GIB)
        );
        // Same model number, 16 GB rather than 24: guessing here would let the
        // resolution control plan for half as much again as the card has.
        assert_eq!(vram_bytes_for("NVIDIA GeForce RTX 4090 Laptop GPU"), None);
        assert_eq!(vram_bytes_for("AMD Radeon RX 7900 XTX"), None);
    }
}
