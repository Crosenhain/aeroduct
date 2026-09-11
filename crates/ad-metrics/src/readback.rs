//! A non-blocking readback ring.
//!
//! # The rule
//!
//! **The render loop must never wait for a number in a status bar.** The metrics
//! reductions produce a few hundred bytes per frame; a `poll(Wait)` to collect
//! them would cost a full GPU round trip — several milliseconds at the
//! interactive tier, more than the frame budget — to save two frames of latency
//! on a value the user reads at 2 Hz. So nothing here ever blocks, and results
//! simply arrive two or three frames late, tagged with the step they came from
//! so a late arrival cannot be mistaken for a current one.
//!
//! # The `map_async` hazard
//!
//! `Buffer::map_async` leaves the buffer in a *mapped-pending* state until its
//! callback fires. While it is pending:
//!
//! * copying into it is a validation error, and
//! * on some backends the next `Queue::submit` that touches it fails outright.
//!
//! And unmapping is required **even when the map failed**. An earlier version of
//! [`ad_gpu::Profiler::collect`] returned early on a miss without unmapping, and
//! the render loop died on frame two. That is not a hypothetical; it is why that
//! module carries an explicit state machine and a test that runs 32 frames.
//!
//! This ring follows the same pattern, per slot:
//!
//! ```text
//! Free ──record()──▶ Recorded ──poll(): map_async──▶ Pending
//!   ▲                                                  │
//!   └──── unmap ◀── Ready (data taken) / Failed ◀──────┘
//! ```
//!
//! Only a `Free` slot may be recorded into, so the copy can never target a
//! mapped buffer. Every exit from `Pending` unmaps, success or failure.
//!
//! # Depth
//!
//! Two slots suffice on a well-behaved driver; three absorbs a frame where the
//! CPU runs ahead. More than that only adds latency. If [`ReadbackRing::record`]
//! ever returns `false` the ring is saturated — the GPU is more than `depth`
//! frames behind — and skipping that frame's measurement is the right response,
//! because the measurement would have been stale anyway.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use bytemuck::Pod;

mod state {
    /// Nothing in flight; safe to copy into.
    pub const FREE: u8 = 0;
    /// A copy has been recorded (and probably submitted); not yet mapped.
    pub const RECORDED: u8 = 1;
    /// `map_async` issued, callback not yet fired.
    pub const PENDING: u8 = 2;
    /// Mapped and readable.
    pub const READY: u8 = 3;
    /// The map failed. Must still be unmapped.
    pub const FAILED: u8 = 4;
}

struct Slot {
    buffer: wgpu::Buffer,
    /// Shared with the `map_async` callback, which may run on another thread.
    state: Arc<AtomicU8>,
    /// The solver step the recorded data belongs to.
    tag: u64,
}

/// One completed readback.
#[derive(Debug, Clone)]
pub struct Frame<T> {
    /// Solver step the data was captured at.
    pub step: u64,
    pub data: Vec<T>,
}

/// Ring of mapped staging buffers, `depth` deep, `elements` of `T` each.
pub struct ReadbackRing<T: Pod> {
    slots: Vec<Slot>,
    elements: usize,
    bytes: u64,
    next: usize,
    /// Frames dropped because every slot was busy. A steadily rising count means
    /// the ring is too shallow or the GPU is falling behind.
    dropped: u64,
    delivered: u64,
    _marker: PhantomData<T>,
}

impl<T: Pod> ReadbackRing<T> {
    /// `elements` is the number of `T` in one capture; `depth` is how many
    /// captures can be in flight. Use 2 or 3.
    pub fn new(device: &wgpu::Device, label: &str, elements: usize, depth: usize) -> Self {
        let elements = elements.max(1);
        let depth = depth.clamp(1, 8);
        // wgpu requires a mappable buffer size to be a multiple of COPY_BUFFER_ALIGNMENT.
        let raw = (elements * std::mem::size_of::<T>()) as u64;
        let bytes = raw.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
        let slots = (0..depth)
            .map(|i| Slot {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("{label} readback {i}")),
                    size: bytes,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                state: Arc::new(AtomicU8::new(state::FREE)),
                tag: 0,
            })
            .collect();
        Self {
            slots,
            elements,
            bytes,
            next: 0,
            dropped: 0,
            delivered: 0,
            _marker: PhantomData,
        }
    }

    pub fn depth(&self) -> usize {
        self.slots.len()
    }

    pub fn elements(&self) -> usize {
        self.elements
    }

    /// Captures skipped because the ring was full.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Captures successfully handed back.
    pub fn delivered(&self) -> u64 {
        self.delivered
    }

    /// Are any slots free? Cheap enough to call before doing the work that
    /// produces `src`, so a saturated ring can skip the whole reduction.
    pub fn has_capacity(&self) -> bool {
        self.slots.iter().any(|s| s.state.load(Ordering::Acquire) == state::FREE)
    }

    /// Record a copy of `src` (from `offset`) into a free slot.
    ///
    /// Returns `false` when every slot is busy, in which case *nothing was
    /// recorded* and the caller should not expect this step to come back. That
    /// is deliberately not an error: dropping a stale measurement is correct.
    pub fn record(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        offset: u64,
        step: u64,
    ) -> bool {
        let n = self.slots.len();
        for k in 0..n {
            let i = (self.next + k) % n;
            if self.slots[i].state.load(Ordering::Acquire) != state::FREE {
                continue;
            }
            encoder.copy_buffer_to_buffer(src, offset, &self.slots[i].buffer, 0, self.bytes);
            self.slots[i].tag = step;
            self.slots[i].state.store(state::RECORDED, Ordering::Release);
            self.next = (i + 1) % n;
            return true;
        }
        self.dropped += 1;
        false
    }

    /// Advance every slot's state machine and return any completed captures,
    /// oldest first.
    ///
    /// Never blocks. `device.poll(Poll)` only nudges the queue along; it does
    /// not wait. Call once per frame after submitting.
    pub fn poll(&mut self, device: &wgpu::Device) -> Vec<Frame<T>> {
        let mut out: Vec<Frame<T>> = Vec::new();
        let mut nudge = false;

        for slot in self.slots.iter_mut() {
            match slot.state.load(Ordering::Acquire) {
                state::RECORDED => {
                    slot.state.store(state::PENDING, Ordering::Release);
                    let flag = slot.state.clone();
                    slot.buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                        flag.store(
                            if r.is_ok() { state::READY } else { state::FAILED },
                            Ordering::Release,
                        );
                    });
                    nudge = true;
                }
                state::PENDING => nudge = true,
                state::FAILED => {
                    // Unmapping is mandatory even on failure, or this slot stays
                    // pending forever and every later submit that touches it
                    // fails validation.
                    slot.buffer.unmap();
                    slot.state.store(state::FREE, Ordering::Release);
                }
                state::READY => {
                    let data = match slot.buffer.slice(..).get_mapped_range() {
                        Ok(view) => {
                            let want = self.elements * std::mem::size_of::<T>();
                            Some(bytemuck::cast_slice::<u8, T>(&view[..want]).to_vec())
                        }
                        Err(e) => {
                            log::debug!("metrics readback range unavailable: {e:?}");
                            None
                        }
                    };
                    slot.buffer.unmap();
                    slot.state.store(state::FREE, Ordering::Release);
                    if let Some(data) = data {
                        out.push(Frame { step: slot.tag, data });
                    }
                }
                _ => {}
            }
        }

        if nudge {
            let _ = device.poll(wgpu::PollType::Poll);
        }
        out.sort_by_key(|f| f.step);
        self.delivered += out.len() as u64;
        out
    }

    /// Poll until every in-flight capture has landed. **Blocking; tests and
    /// one-shot batch runs only.** The frame loop must use [`Self::poll`].
    pub fn drain_blocking(&mut self, device: &wgpu::Device) -> Vec<Frame<T>> {
        let mut out = Vec::new();
        for _ in 0..256 {
            let busy = self
                .slots
                .iter()
                .any(|s| s.state.load(Ordering::Acquire) != state::FREE);
            if !busy && !out.is_empty() {
                break;
            }
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            out.extend(self.poll(device));
            if !busy {
                break;
            }
        }
        out.sort_by_key(|f| f.step);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu() -> Option<ad_gpu::GpuContext> {
        crate::test_gpu()
    }

    /// The bug this guards against is the one documented at the top of the
    /// module: a slot left mapped kills the *next* submit, not this one, so a
    /// single frame passes either way. This runs 64.
    #[test]
    fn sixty_four_frames_never_leave_a_buffer_mapped() {
        let Some(gpu) = gpu() else { return };
        let src = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ring test source"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut ring = ReadbackRing::<u32>::new(&gpu.device, "ring test", 64, 3);

        let mut seen = 0u64;
        for step in 0..64u64 {
            let payload: Vec<u32> = (0..64).map(|i| i + step as u32 * 1000).collect();
            gpu.queue.write_buffer(&src, 0, bytemuck::cast_slice(&payload));
            let mut enc = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            ring.record(&mut enc, &src, 0, step);
            // A validation failure from a still-mapped slot surfaces here.
            gpu.queue.submit([enc.finish()]);
            for f in ring.poll(&gpu.device) {
                assert_eq!(f.data.len(), 64);
                assert_eq!(f.data[0], f.step as u32 * 1000, "frame {} came back wrong", f.step);
                seen += 1;
            }
        }
        for f in ring.drain_blocking(&gpu.device) {
            assert_eq!(f.data[0], f.step as u32 * 1000);
            seen += 1;
        }
        assert!(seen > 32, "only {seen} of 64 frames were ever delivered");
    }

    #[test]
    fn a_saturated_ring_drops_rather_than_corrupting() {
        let Some(gpu) = gpu() else { return };
        let src = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("saturation source"),
            size: 64,
            usage: wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut ring = ReadbackRing::<u32>::new(&gpu.device, "saturation", 16, 2);

        // Record twice without ever polling: the third must be refused.
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        assert!(ring.record(&mut enc, &src, 0, 0));
        assert!(ring.record(&mut enc, &src, 0, 1));
        assert!(!ring.has_capacity());
        assert!(!ring.record(&mut enc, &src, 0, 2), "a full ring must refuse");
        assert_eq!(ring.dropped(), 1);
        gpu.queue.submit([enc.finish()]);

        // ...and after draining, it accepts again.
        let _ = ring.drain_blocking(&gpu.device);
        assert!(ring.has_capacity());
    }

    #[test]
    fn frames_are_returned_in_step_order_and_tagged() {
        let Some(gpu) = gpu() else { return };
        let src = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("order source"),
            size: 64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut ring = ReadbackRing::<u32>::new(&gpu.device, "order", 4, 3);
        for step in [100u64, 200, 300] {
            gpu.queue.write_buffer(&src, 0, bytemuck::cast_slice(&[step as u32; 4]));
            let mut enc = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            assert!(ring.record(&mut enc, &src, 0, step));
            gpu.queue.submit([enc.finish()]);
        }
        let frames = ring.drain_blocking(&gpu.device);
        assert_eq!(frames.len(), 3);
        assert_eq!(frames.iter().map(|f| f.step).collect::<Vec<_>>(), vec![100, 200, 300]);
        for f in &frames {
            assert_eq!(f.data[0], f.step as u32);
        }
    }
}
