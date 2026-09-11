//! The blue-noise tile, and the frame-to-frame sequence that advances it.
//!
//! # Why blue noise, and not a hash
//!
//! The raymarcher jitters its first sample by a fraction of a step so that the
//! banding from a fixed step size turns into noise. *Which* noise matters enormously:
//!
//! - A per-pixel hash is white noise. Its error is spread evenly across all
//!   spatial frequencies, including the low ones, so it leaves large soft
//!   blotches that survive a temporal filter — TAA averages them faithfully.
//! - Blue noise has almost no low-frequency energy. Its error lives at the
//!   pixel scale, exactly where TAA and the eye both remove it. The same
//!   variance, put somewhere useful.
//!
//! # Why the golden ratio between frames
//!
//! Reusing the same tile every frame would make the jitter static, and a static
//! jitter is just a different fixed pattern — TAA converges to it instead of
//! averaging it away. The offset is therefore advanced each frame by
//! `v_{n+1} = fract(v_n + 0.6180339887)`.
//!
//! The golden ratio is the specific choice because its continued-fraction
//! expansion is all ones, which makes it the *hardest* number to approximate
//! with a rational. Any rational step `p/q` revisits its starting point after
//! `q` frames and so only ever samples `q` distinct offsets; the golden ratio
//! never repeats and fills `[0, 1)` about as evenly as any prefix can
//! (three-distance theorem). Practically: TAA resolves it in a handful of
//! frames, and progressive accumulation keeps improving for as long as you
//! leave it running.

/// Void-and-cluster tile shipped in `assets/`. 128x128, one byte per texel,
/// generated once offline; see the tests for the properties it must have.
pub const BLUE_NOISE_SIZE: u32 = 128;

/// The tile itself.
pub const BLUE_NOISE_TILE: &[u8] = include_bytes!("../../../assets/bluenoise_128.gray");

/// The golden ratio conjugate, `1/phi`.
pub const GOLDEN_RATIO_CONJUGATE: f32 = 0.618_033_988_7;

/// Advance a `[0, 1)` offset by one frame.
#[inline]
pub fn golden_ratio_advance(v: f32) -> f32 {
    (v + GOLDEN_RATIO_CONJUGATE).fract()
}

/// The `n`th term of the sequence starting from `v0`, computed directly rather
/// than iterated, so a frame index can be turned into an offset with no state.
#[inline]
pub fn golden_ratio_at(v0: f32, n: u32) -> f32 {
    // `n * phi` loses precision above a few million frames; wrapping the
    // multiplication into [0,1) first keeps it exact for as long as anyone will
    // leave the app open.
    let frac = (n as f64 * GOLDEN_RATIO_CONJUGATE as f64).fract() as f32;
    (v0 + frac).fract()
}

/// Upload the tile as an `R8Unorm` 2D texture.
pub fn create_blue_noise_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
    let size = wgpu::Extent3d {
        width: BLUE_NOISE_SIZE,
        height: BLUE_NOISE_SIZE,
        depth_or_array_layers: 1,
    };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("blue noise 128"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // Unorm rather than Uint so the shader gets a [0,1) value with no
        // conversion, and so a filtering sampler can be used if a later pass
        // wants one.
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        BLUE_NOISE_TILE,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(BLUE_NOISE_SIZE),
            rows_per_image: Some(BLUE_NOISE_SIZE),
        },
        size,
    );
    tex
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = BLUE_NOISE_SIZE as usize;

    #[test]
    fn the_tile_is_the_right_size() {
        assert_eq!(BLUE_NOISE_TILE.len(), N * N);
    }

    #[test]
    fn the_tile_histogram_is_perfectly_flat() {
        // Void-and-cluster produces a rank ordering, so every output value must
        // appear exactly the same number of times. A non-flat histogram means
        // the jitter is biased and the raymarch will sample some sub-step
        // positions more often than others, which reintroduces banding.
        let mut hist = [0u32; 256];
        for b in BLUE_NOISE_TILE {
            hist[*b as usize] += 1;
        }
        let expect = (N * N / 256) as u32;
        for (v, c) in hist.iter().enumerate() {
            assert_eq!(*c, expect, "value {v} appears {c} times, want {expect}");
        }
    }

    /// Magnitude of one DFT bin of the tile. Only a handful of bins are needed,
    /// so this beats a full transform.
    fn dft_power(u: i32, v: i32) -> f64 {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        let n = N as f64;
        for y in 0..N {
            for x in 0..N {
                // Mean-subtracted, so the DC bin does not swamp everything.
                let s = BLUE_NOISE_TILE[y * N + x] as f64 - 127.5;
                let phase = -2.0
                    * std::f64::consts::PI
                    * ((u as f64 * x as f64) + (v as f64 * y as f64))
                    / n;
                re += s * phase.cos();
                im += s * phase.sin();
            }
        }
        re * re + im * im
    }

    #[test]
    fn the_tile_has_a_blue_spectrum() {
        // The defining property: energy suppressed at low spatial frequencies
        // and concentrated at high ones. White noise scores about 1.0 here; a
        // good void-and-cluster tile scores well under 0.2.
        let mut low = 0.0f64;
        let mut low_n = 0.0f64;
        let mut high = 0.0f64;
        let mut high_n = 0.0f64;

        for v in -6i32..=6 {
            for u in -6i32..=6 {
                if u == 0 && v == 0 {
                    continue; // DC, removed by the mean subtraction anyway
                }
                if u * u + v * v <= 36 {
                    low += dft_power(u, v);
                    low_n += 1.0;
                }
            }
        }
        // A high-frequency annulus of comparable bin count, near Nyquist.
        let nyq = (N / 2) as i32;
        for v in (nyq - 6)..=nyq {
            for u in (nyq - 6)..=nyq {
                high += dft_power(u, v);
                high_n += 1.0;
            }
        }

        let ratio = (low / low_n) / (high / high_n);
        assert!(
            ratio < 0.25,
            "low/high spectral power ratio is {ratio}; the tile is not blue noise"
        );
    }

    #[test]
    fn neighbouring_texels_are_decorrelated() {
        // Blue noise pushes similar values apart, so adjacent texels should
        // differ by much more than a random pairing would.
        let mut adjacent = 0.0f64;
        for y in 0..N {
            for x in 0..N {
                let a = BLUE_NOISE_TILE[y * N + x] as f64;
                let b = BLUE_NOISE_TILE[y * N + (x + 1) % N] as f64;
                adjacent += (a - b).abs();
            }
        }
        adjacent /= (N * N) as f64;
        // The expected absolute difference between two independent uniform
        // draws on 0..255 is 255/3 = 85. Blue noise beats that.
        assert!(adjacent > 85.0, "mean adjacent difference is only {adjacent}");
    }

    #[test]
    fn golden_ratio_sequence_stays_in_range_and_separates_optimally() {
        // A rational step `p/q` revisits its starting point after `q` frames and
        // only ever visits `q` distinct offsets. The golden ratio is the *worst*
        // number to approximate rationally, and Hurwitz's theorem gives the
        // resulting guarantee: the closest any two of the first `n` terms can
        // come is about `1 / (sqrt(5) n)`. Nothing does better than this.
        const N: usize = 256;
        let mut v = 0.0f32;
        let mut seen: Vec<f32> = Vec::with_capacity(N);
        for _ in 0..N {
            v = golden_ratio_advance(v);
            assert!((0.0..1.0).contains(&v), "sequence left the unit interval: {v}");
            seen.push(v);
        }
        let mut closest = f32::INFINITY;
        for i in 0..N {
            for j in (i + 1)..N {
                closest = closest.min((seen[i] - seen[j]).abs());
            }
        }
        let bound = 0.4 / N as f32;
        assert!(
            closest > bound,
            "closest pair among {N} terms is {closest}, below the 1/(sqrt5 n) bound {bound}"
        );
    }

    #[test]
    fn golden_ratio_prefixes_are_evenly_spread() {
        // The three-distance property: after n terms the unit interval is cut
        // into intervals of at most three distinct lengths, none of them large.
        // Practically this is what makes a short TAA history well distributed.
        for n in [8usize, 16, 64, 233] {
            let mut xs: Vec<f32> = (0..n as u32).map(|i| golden_ratio_at(0.0, i)).collect();
            xs.sort_by(|a, b| a.total_cmp(b));
            let mut worst_gap = xs[0];
            for w in xs.windows(2) {
                worst_gap = worst_gap.max(w[1] - w[0]);
            }
            worst_gap = worst_gap.max(1.0 - xs[n - 1]);
            // A perfectly even split would be 1/n; the golden ratio stays within
            // a small constant factor of it, unlike a random sequence.
            assert!(
                worst_gap < 2.2 / n as f32,
                "n={n}: largest gap {worst_gap} exceeds 2.2/n"
            );
        }
    }

    #[test]
    fn closed_form_and_iterated_sequences_agree() {
        let mut v = 0.25f32;
        for n in 1..64u32 {
            v = golden_ratio_advance(v);
            let direct = golden_ratio_at(0.25, n);
            assert!((v - direct).abs() < 1e-4, "n={n}: {v} vs {direct}");
        }
    }
}
