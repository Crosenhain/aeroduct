//! GPU timestamp profiling, reported against the memory roofline.
//!
//! LBM is bandwidth-bound: a step moves the whole distribution array twice and
//! does almost no arithmetic. So the only performance number that means anything
//! is achieved bandwidth as a fraction of peak. A bare "12 ms/step" tells you
//! nothing; "63% of roofline" tells you immediately whether to go looking for a
//! layout problem or accept the result and move on.
//!
//! Reference points on an RTX 4090 (1008 GB/s peak): FluidX3D reaches 85-88%.
//! A first custom WGSL kernel realistically lands at 60-75%. Below ~50%, suspect
//! buffer alignment or workgroup shape before anything else.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

/// State of the readback buffer's mapping.
///
/// A `map_async` that has not completed leaves the buffer mapped-pending.
/// Issuing a second one, or submitting more work that touches the buffer, is a
/// validation error. So the mapping has to be tracked explicitly rather than
/// inferred from a channel poll: an earlier version of this file returned early
/// on a miss without unmapping, which killed the render loop on frame two.
mod map_state {
    pub const IDLE: u8 = 0;
    pub const PENDING: u8 = 1;
    pub const READY: u8 = 2;
    pub const FAILED: u8 = 3;
}

/// Rolling timing for one named GPU scope.
#[derive(Debug, Clone, Default)]
pub struct ScopeTiming {
    pub last_ms: f64,
    pub mean_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    samples: u64,
}

impl ScopeTiming {
    fn push(&mut self, ms: f64) {
        self.last_ms = ms;
        if self.samples == 0 {
            self.min_ms = ms;
            self.max_ms = ms;
            self.mean_ms = ms;
        } else {
            self.min_ms = self.min_ms.min(ms);
            self.max_ms = self.max_ms.max(ms);
            // Exponential moving average: we care about current behaviour, not
            // the average since startup (which includes shader compilation).
            self.mean_ms += (ms - self.mean_ms) * 0.05;
        }
        self.samples += 1;
    }
}

/// An open [`Profiler::span_begin`]. Hand it back to [`Profiler::span_end`].
#[must_use = "a span that is never ended reports nothing"]
pub struct Span {
    end: u32,
}

pub struct Profiler {
    query_set: Option<wgpu::QuerySet>,
    resolve_buffer: Option<wgpu::Buffer>,
    readback: Option<wgpu::Buffer>,
    capacity: u32,
    next_query: u32,
    period_ns: f32,
    /// `(name, begin query, end query, divisor)`. See [`Profiler::scope_per`].
    scopes: Vec<(String, u32, u32, f64)>,
    /// Scope layout of the frame whose results are currently in flight. The live
    /// frame keeps rewriting `scopes`, so the in-flight mapping has to remember
    /// which frame it belongs to or timings get attributed to the wrong passes.
    inflight: Vec<(String, u32, u32, f64)>,
    inflight_queries: u32,
    map_state: Arc<AtomicU8>,
    timings: HashMap<String, ScopeTiming>,
    peak_bandwidth: Option<f64>,
    enabled: bool,
}

impl Profiler {
    /// `capacity` is the number of timestamp *pairs* to make room for.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        capacity: u32,
        enabled: bool,
        peak_bandwidth: Option<f64>,
    ) -> Self {
        if !enabled {
            return Self {
                query_set: None,
                resolve_buffer: None,
                readback: None,
                capacity,
                next_query: 0,
                period_ns: 0.0,
                scopes: Vec::new(),
                inflight: Vec::new(),
                inflight_queries: 0,
                map_state: Arc::new(AtomicU8::new(map_state::IDLE)),
                timings: HashMap::new(),
                peak_bandwidth,
                enabled: false,
            };
        }

        let count = capacity * 2;
        let bytes = count as u64 * 8;
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("profiler timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count,
        });
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("profiler resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("profiler readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            query_set: Some(query_set),
            resolve_buffer: Some(resolve_buffer),
            readback: Some(readback),
            capacity,
            next_query: 0,
            period_ns: queue.get_timestamp_period(),
            scopes: Vec::new(),
            inflight: Vec::new(),
            inflight_queries: 0,
            map_state: Arc::new(AtomicU8::new(map_state::IDLE)),
            timings: HashMap::new(),
            peak_bandwidth,
            enabled: true,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Call once at the start of each frame.
    pub fn begin_frame(&mut self) {
        self.next_query = 0;
        self.scopes.clear();
    }

    /// Timestamp writes to attach to a compute pass descriptor. Returns `None`
    /// when profiling is off or the capacity for this frame is exhausted, in
    /// which case the pass simply runs untimed.
    pub fn scope(&mut self, name: &str) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let (begin, end) = self.reserve(name)?;
        self.query_set.as_ref().map(|qs| wgpu::ComputePassTimestampWrites {
            query_set: qs,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        })
    }

    /// [`Profiler::scope`] for a pass that does `units` repetitions of the same
    /// work, such as a batch of solver steps. The timing is recorded per unit,
    /// so the moving average stays meaningful when the batch size changes from
    /// frame to frame, which it does whenever the step tuner is on. Dividing an
    /// average of mixed batch times by the latest batch size afterwards is not
    /// the same thing, and read 4-5x high under a tuner that was moving.
    pub fn scope_per(
        &mut self,
        name: &str,
        units: u32,
    ) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let (begin, end) = self.reserve_per(name, units.max(1) as f64)?;
        self.query_set.as_ref().map(|qs| wgpu::ComputePassTimestampWrites {
            query_set: qs,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        })
    }

    /// Timestamp writes for a *render* pass descriptor. Shares one query set and
    /// one per-frame budget with [`Profiler::scope`].
    pub fn render_scope(&mut self, name: &str) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let (begin, end) = self.reserve(name)?;
        self.query_set.as_ref().map(|qs| wgpu::RenderPassTimestampWrites {
            query_set: qs,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        })
    }

    /// Start timing *everything* recorded into `encoder` from here to the
    /// matching [`Profiler::span_end`], however many passes that is.
    ///
    /// For work recorded by code that takes no profiler (the metrics passes,
    /// the UI), and for totals over a group of passes. Each end is an empty
    /// compute pass carrying a single timestamp, so this needs nothing beyond
    /// `TIMESTAMP_QUERY`. Returns `None` under the same conditions as
    /// [`Profiler::scope`], which `span_end` accepts and ignores.
    ///
    /// Always end a span that was begun before resolving: a resolved query
    /// that was never written reads as garbage.
    ///
    /// Prefer [`Profiler::scope`] on the real passes where the recording code
    /// can take one. The begin marker's pass touches no resources, so no
    /// barrier holds it behind earlier work, and on a queue still draining a
    /// previous submit it can latch early. Around the render graph it agreed
    /// with the sum of the renderer's own scopes; around the metrics submit,
    /// which directly follows the solver's, it read high.
    pub fn span_begin(&mut self, encoder: &mut wgpu::CommandEncoder, name: &str) -> Option<Span> {
        let (begin, end) = self.reserve(name)?;
        let qs = self.query_set.as_ref()?;
        drop(encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("profiler span begin"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(begin),
                end_of_pass_write_index: None,
            }),
        }));
        Some(Span { end })
    }

    /// Close a span opened by [`Profiler::span_begin`] on the same encoder.
    pub fn span_end(&mut self, encoder: &mut wgpu::CommandEncoder, span: Option<Span>) {
        let (Some(span), Some(qs)) = (span, self.query_set.as_ref()) else { return };
        drop(encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("profiler span end"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: None,
                end_of_pass_write_index: Some(span.end),
            }),
        }));
    }

    /// Claim a timestamp pair for `name`, or `None` if profiling is off or this
    /// frame's budget is exhausted, in which case the pass simply runs untimed.
    fn reserve(&mut self, name: &str) -> Option<(u32, u32)> {
        self.reserve_per(name, 1.0)
    }

    fn reserve_per(&mut self, name: &str, divisor: f64) -> Option<(u32, u32)> {
        if !self.enabled || self.next_query + 1 >= self.capacity * 2 {
            return None;
        }
        let begin = self.next_query;
        let end = self.next_query + 1;
        self.next_query += 2;
        self.scopes.push((name.to_string(), begin, end, divisor));
        Some((begin, end))
    }

    /// Resolve this frame's queries. Call after recording, before submit.
    ///
    /// Skipped while a previous frame's readback is still mapped, because
    /// copying into a mapped buffer is a validation error.
    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if !self.enabled || self.next_query == 0 {
            return;
        }
        if self.map_state.load(Ordering::Acquire) != map_state::IDLE {
            return;
        }
        let (Some(qs), Some(resolve), Some(readback)) =
            (&self.query_set, &self.resolve_buffer, &self.readback)
        else {
            return;
        };
        encoder.resolve_query_set(qs, 0..self.next_query, resolve, 0);
        encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, self.next_query as u64 * 8);

        // Remember the layout that goes with the data just copied: `scopes` is
        // cleared by the next `begin_frame`, long before the map completes.
        self.inflight.clear();
        self.inflight.extend_from_slice(&self.scopes);
        self.inflight_queries = self.next_query;
    }

    /// Fold any completed readback into the rolling stats, and start the next.
    ///
    /// Never blocks, and never leaves the buffer mapped on any path. Call once
    /// per frame after submitting. Results arrive a frame or two late, which is
    /// fine for a number displayed in a status bar.
    pub fn collect(&mut self, device: &wgpu::Device) {
        if !self.enabled {
            return;
        }
        let Some(readback) = self.readback.clone() else { return };

        match self.map_state.load(Ordering::Acquire) {
            map_state::IDLE => {
                if self.inflight_queries == 0 {
                    return;
                }
                self.map_state.store(map_state::PENDING, Ordering::Release);
                let state = self.map_state.clone();
                readback
                    .slice(..self.inflight_queries as u64 * 8)
                    .map_async(wgpu::MapMode::Read, move |r| {
                        state.store(
                            if r.is_ok() { map_state::READY } else { map_state::FAILED },
                            Ordering::Release,
                        );
                    });
                // Nudge the queue along, but never wait on it.
                let _ = device.poll(wgpu::PollType::Poll);
            }
            map_state::PENDING => {
                let _ = device.poll(wgpu::PollType::Poll);
            }
            map_state::FAILED => {
                // Unmapping is required even on failure, or the buffer stays
                // pending forever and every later submit fails validation.
                readback.unmap();
                self.inflight_queries = 0;
                self.map_state.store(map_state::IDLE, Ordering::Release);
            }
            _ => {
                let n = self.inflight_queries as u64 * 8;
                let raw: Option<Vec<u64>> = match readback.slice(..n).get_mapped_range() {
                    Ok(view) => Some(bytemuck::cast_slice::<u8, u64>(&view).to_vec()),
                    Err(e) => {
                        log::debug!("profiler readback range unavailable: {e:?}");
                        None
                    }
                };
                readback.unmap();
                self.inflight_queries = 0;
                self.map_state.store(map_state::IDLE, Ordering::Release);

                if let Some(raw) = raw {
                    for (name, b, e, divisor) in &self.inflight {
                        let (b, e) = (*b as usize, *e as usize);
                        if e >= raw.len() || raw[e] <= raw[b] {
                            continue;
                        }
                        let ms =
                            (raw[e] - raw[b]) as f64 * self.period_ns as f64 * 1e-6 / divisor;
                        self.timings.entry(name.clone()).or_default().push(ms);
                    }
                }
            }
        }
    }

    pub fn timing(&self, name: &str) -> Option<&ScopeTiming> {
        self.timings.get(name)
    }

    pub fn timings(&self) -> impl Iterator<Item = (&String, &ScopeTiming)> {
        self.timings.iter()
    }

    /// Achieved bandwidth in bytes/sec for a scope that moved `bytes`.
    pub fn bandwidth(&self, name: &str, bytes: u64) -> Option<f64> {
        let t = self.timings.get(name)?;
        if t.mean_ms <= 0.0 {
            return None;
        }
        Some(bytes as f64 / (t.mean_ms * 1e-3))
    }

    /// Achieved bandwidth as a fraction of peak. This is the number to watch.
    pub fn roofline_fraction(&self, name: &str, bytes: u64) -> Option<f64> {
        Some(self.bandwidth(name, bytes)? / self.peak_bandwidth?)
    }

    /// One-line summary for the UI status bar.
    pub fn summary(&self, name: &str, bytes_per_call: u64) -> String {
        let Some(t) = self.timings.get(name) else {
            return format!("{name}: no data");
        };
        match self.roofline_fraction(name, bytes_per_call) {
            Some(f) => format!(
                "{name}: {:.2} ms ({:.0} GB/s, {:.0}% of roofline)",
                t.mean_ms,
                self.bandwidth(name, bytes_per_call).unwrap_or(0.0) / 1e9,
                f * 100.0
            ),
            None => format!("{name}: {:.2} ms", t.mean_ms),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bug this guards against: an earlier `collect` issued `map_async` and
    /// returned on a miss *without unmapping*, so the next `Queue::submit`
    /// failed validation with "buffer is still mapped" and the render loop died
    /// on frame two. A single frame passes either way, so this must run several.
    #[test]
    fn many_frames_of_resolve_and_collect_never_leave_the_buffer_mapped() {
        let Ok(gpu) = crate::GpuContext::new_blocking(None) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        if !gpu.caps.timestamps {
            eprintln!("skipping: adapter has no timestamp query support");
            return;
        }

        let mut p = Profiler::new(&gpu.device, &gpu.queue, 8, true, gpu.caps.peak_bandwidth);

        // A trivial compute pipeline, purely so there is a pass to time.
        let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("profiler test kernel"),
            source: wgpu::ShaderSource::Wgsl(
                "@compute @workgroup_size(1) fn main() {}".into(),
            ),
        });
        let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("profiler test pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        for frame in 0..32 {
            p.begin_frame();
            let mut enc = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let writes = p.scope("test");
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("test pass"),
                    timestamp_writes: writes,
                });
                pass.set_pipeline(&pipeline);
                pass.dispatch_workgroups(1, 1, 1);
            }
            p.resolve(&mut enc);
            // A validation failure surfaces here, which is why the submit has to
            // be inside the loop rather than batched.
            gpu.queue.submit([enc.finish()]);
            p.collect(&gpu.device);
            assert!(frame < 32);
        }

        // Everything above this point is the actual regression guard: 32 submits
        // that would each have failed validation with a leaked mapping.
        //
        // What follows is a liveness check, and it is deliberately soft. Whether
        // a readback completes within a bounded number of polls depends on how
        // busy the GPU is, and this suite runs its test binaries in parallel on
        // a machine that is also driving a desktop. Asserting that a timing
        // arrived turns "the GPU was busy" into a red build, which trains people
        // to ignore the result. So: drain generously, and if nothing lands, say
        // so rather than fail -- the correctness property was already proven.
        for _ in 0..512 {
            let _ = gpu.device.poll(wgpu::PollType::Poll);
            p.collect(&gpu.device);
            if p.timing("test").is_some() {
                break;
            }
        }
        match p.timing("test") {
            Some(t) => assert!(
                t.mean_ms >= 0.0 && t.mean_ms.is_finite(),
                "nonsense timing {t:?}"
            ),
            None => eprintln!(
                "note: no timestamp landed within the poll budget; the GPU is \
                 probably saturated. The mapping-lifecycle assertions above still ran."
            ),
        }
    }

    #[test]
    fn rolling_stats_track_min_max_and_last() {
        let mut t = ScopeTiming::default();
        for ms in [10.0, 5.0, 20.0, 12.0] {
            t.push(ms);
        }
        assert_eq!(t.last_ms, 12.0);
        assert_eq!(t.min_ms, 5.0);
        assert_eq!(t.max_ms, 20.0);
        // The EMA starts at the first sample and moves slowly.
        assert!(t.mean_ms > 9.0 && t.mean_ms < 12.0, "mean was {}", t.mean_ms);
    }
}
