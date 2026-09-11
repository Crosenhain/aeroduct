// Pure line-integral-convolution arithmetic.
//
// Mirrored function for function from `crates/ad-render/src/lic.rs`, which is
// where the reasoning lives and where these are tested without an adapter.
//
// Declares NO bindings, on the same principle as `common.wgsl`: the streamline
// walk needs the velocity texture and therefore belongs in the file that owns
// the bind groups. `slice.wgsl` is the only consumer.

const LIC_TAU: f32 = 6.283185307179586;

// The travelling-ramp kernel, at normalised arc position `s / L` in [-1, 1].
//
// Non-negative, unit peak, period 1 in `phase`. The crest sits at `s/L = phase`,
// so advancing the phase marches bright bands *downstream* — which is the whole
// reason the animation exists. A static LIC shows the streamline axis but not
// which way the air is going along it, and in a bend that is the question.
fn lic_ramp_kernel(s_over_l: f32, phase: f32) -> f32 {
    return 0.5 * (1.0 + cos(LIC_TAU * (s_over_l - phase)));
}

// White noise on a lattice of `noise_scale_mm` cells in plane coordinates.
//
// Nearest-cell and not interpolated, deliberately: LIC needs input with power at
// the highest spatial frequency it can resolve. Smooth value noise convolves to
// almost nothing and the plane comes out looking like a soft grey wash.
//
// Plane coordinates rather than world: the texture then stays put as the plane
// is scrubbed along the duct, instead of crawling across it.
fn lic_noise(plane_mm: vec2<f32>, noise_scale_mm: f32) -> f32 {
    let c = vec2<i32>(floor(plane_mm / max(noise_scale_mm, 1.0e-4)));
    // Large odd multipliers, then the common integer hash. A plain
    // `x * A + y * B` alone leaves visible diagonal structure, which LIC
    // faithfully smears into stripes that look like flow features.
    let h = hash_u32(u32(c.x * 73856093) ^ u32(c.y * 19349663));
    return f32(h) * (1.0 / 4294967296.0);
}

// How much of the velocity actually lies in the cutting plane.
//
// The honesty term, and it is not decoration. LIC on a plane can only show the
// in-plane component; in a 90-degree bend a large fraction of the flow pierces
// the plane, and a confident swirl texture drawn from a 10%-of-magnitude residue
// is a lie told beautifully. Multiplying the contrast by this fades the texture
// to flat colour exactly where the flow is leaving.
fn lic_honesty(in_plane_speed: f32, speed: f32, power: f32) -> f32 {
    if (speed <= 1.0e-9) {
        return 0.0;
    }
    return pow(clamp(in_plane_speed / speed, 0.0, 1.0), power);
}

// Independent noise cells the convolution actually averages over.
//
// Taps closer together than one noise cell are the same sample, so the count
// that matters is the arc length covered divided by the cell size, never more
// than the number of taps.
//
// `taps` is the number the convolution *realised*, not the number configured.
// The difference bites wherever the streamline is short — against a wall, in a
// stagnation region, in still air outside the duct — and there the two are
// wildly apart: a stalled walk contributes one tap, and applying the full
// long-convolution gain to a single noise sample binarises it into a stark
// black-and-white lattice. Those are exactly the regions a duct designer is
// staring at, so getting this wrong puts the loudest artefact in the most
// important place.
fn lic_effective_taps(taps: f32, step_mm: f32, noise_scale_mm: f32) -> f32 {
    let n = max(taps, 1.0);
    let covered = n * step_mm / max(noise_scale_mm, 1.0e-4);
    return clamp(covered, 1.0, n);
}

// Gain that restores usable contrast after the convolution.
//
// The mean of `m` independent uniform(0,1) samples has standard deviation
// `1 / sqrt(12 m)`. At the default 57 taps that is 0.038 — the raw LIC image is
// a flat grey and the first reaction is to assume the shader is broken. Scaling
// so that two standard deviations fill half the range gives
// `gain = sqrt(0.75 m)`, which puts almost all of the distribution inside [0, 1]
// with only the extreme tails clipping.
fn lic_contrast_gain(effective_taps: f32) -> f32 {
    return max(sqrt(0.75 * effective_taps), 1.0);
}

// Apply the gain about the mid-grey the convolution converges to.
fn lic_apply_gain(raw: f32, gain: f32) -> f32 {
    return clamp(0.5 + (raw - 0.5) * gain, 0.0, 1.0);
}

// The composite: `colormap(scalar) * ((1 - c) + c * lic)`.
//
// Multiplicative and not additive, and never a blend towards the LIC's own grey.
// The colour has to stay the quantitative channel — the reader must be able to
// look up a hue in the legend and get a number — so the texture is only allowed
// to modulate its brightness. At the default contrast of 0.4 that is
// `base * (0.6 + 0.4 * lic)`.
fn lic_compose(base: vec3<f32>, lic: f32, contrast: f32) -> vec3<f32> {
    let c = clamp(contrast, 0.0, 1.0);
    return base * ((1.0 - c) + c * clamp(lic, 0.0, 1.0));
}
