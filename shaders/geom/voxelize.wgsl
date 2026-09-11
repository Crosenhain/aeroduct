// GPU signed-distance voxeliser.
//
// One pipeline produces all three things the rest of the app needs from the
// geometry, because they all fall out of the same distance field and computing
// them separately would be strictly more work:
//
//   * the solid mask (ad_gpu::flags),
//   * a narrow-band SDF for rendering and for the wall-distance the LES model
//     wants,
//   * the per-link `q` values for interpolated bounce-back.
//
// Stages, in order:
//   1. clear_region      scratch to "nothing found yet"
//   2. seed_distance     parallel over triangles, atomicMin of |distance|
//   3. assign_triangle   second pass to recover which triangle won
//   4. sign_band         angle-weighted pseudonormal gives the sign
//   5. seed_exterior +
//      sweep x/y/z       flood fill the sign outside the band
//   6. resolve_unknown   everything still unfilled is interior
//   7. classify          flags, link count, and the link list
//
// Work in stages 2 and 3 scales with surface *area*, not with volume, which is
// what makes a re-voxelisation cheap enough to run while dragging a part.

#include "geom/common.wgsl"
// Generated from ad_gpu::lattice and ad_gpu::flags respectively, so the shader
// tables and the per-cell bit values can never drift from the Rust ones.
#include "geom/lattice.wgsl"
#include "geom/flags.wgsl"

const WG: u32 = 64u;

// ---------------------------------------------------------------------------
// Region helpers
//
// Every grid-wide stage walks a cell range with a grid-stride loop rather than
// a one-thread-per-cell dispatch, because a 132 M cell grid needs 2 M
// workgroups and the limit is 65535 per dimension.
// ---------------------------------------------------------------------------

fn region_dims() -> vec3<u32> {
    return U.region_hi - U.region_lo + vec3<u32>(1u);
}

fn region_count() -> u32 {
    let d = region_dims();
    return d.x * d.y * d.z;
}

fn region_cell(i: u32) -> vec3<u32> {
    let d = region_dims();
    let x = i % d.x;
    let y = (i / d.x) % d.y;
    let z = i / (d.x * d.y);
    return U.region_lo + vec3<u32>(x, y, z);
}

// ---------------------------------------------------------------------------
// 1. Clear
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn clear_region(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = region_count();
    let stride = nwg.x * WG;
    for (var i = gid.x; i < n; i += stride) {
        let cell = cell_index(region_cell(i));
        atomicStore(&dist[cell], FAR_BITS);
        atomicStore(&tri_idx[cell], NO_TRI);
        state[cell] = ST_UNKNOWN;
        phi[cell] = 0.0;
    }
}

// ---------------------------------------------------------------------------
// 2 and 3. Narrow band
// ---------------------------------------------------------------------------

// The voxel range a triangle can possibly influence: its AABB grown by the band
// plus one cell, so no cell centre within the band is missed at the corners.
struct VoxelRange {
    lo: vec3<u32>,
    hi: vec3<u32>,
    empty: bool,
};

fn triangle_voxels(tri: Tri) -> VoxelRange {
    let pad = vec3<f32>(U.band_mm + U.dx_mm);
    let lo_mm = min(min(tri.a, tri.b), tri.c) - pad;
    let hi_mm = max(max(tri.a, tri.b), tri.c) + pad;

    // Cell centres sit at origin + dx * index, so invert that and round inward.
    let f_lo = ceil((lo_mm - U.origin_mm) / U.dx_mm);
    let f_hi = floor((hi_mm - U.origin_mm) / U.dx_mm);

    let r_lo = vec3<f32>(U.region_lo);
    let r_hi = vec3<f32>(U.region_hi);
    if (any(f_lo > r_hi) || any(f_hi < r_lo)) {
        return VoxelRange(vec3<u32>(0u), vec3<u32>(0u), true);
    }
    let lo = vec3<u32>(clamp(f_lo, r_lo, r_hi));
    let hi = vec3<u32>(clamp(f_hi, r_lo, r_hi));
    return VoxelRange(lo, hi, false);
}

// One workgroup per triangle, its 64 threads splitting that triangle's voxels.
// Triangles in a printed duct are all roughly one cell across, so the work per
// triangle is nearly uniform and this keeps the lanes busy without a
// load-balancing scheme.
@compute @workgroup_size(64)
fn seed_distance(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    for (var ti = wid.x; ti < U.tri_count; ti += nwg.x) {
        let t = active_tris[ti];
        let tri = tris[t];
        let r = triangle_voxels(tri);
        if (r.empty) { continue; }
        let n = r.hi - r.lo + vec3<u32>(1u);
        let nvox = n.x * n.y * n.z;

        for (var i = lid; i < nvox; i += WG) {
            let c = r.lo + vec3<u32>(i % n.x, (i / n.x) % n.y, i / (n.x * n.y));
            let p = cell_center(c);
            let cl = closest_point_on_tri(p, tri.a, tri.b, tri.c);
            let d = length(p - cl.p);
            if (d <= U.band_mm) {
                atomic_min_dist(cell_index(c), d);
            }
        }
    }
}

// Recover the winning triangle.
//
// WGSL has no 64-bit atomics, so distance and triangle index cannot be minimised
// together in one word without quantising the distance and capping the triangle
// count. Instead this second pass recomputes the same distances and claims the
// cells it ties for, lowest index winning.
//
// The comparison is deliberately a tolerance and not bit equality. Two separate
// shader compilations of the same expression are not required to produce
// identical rounding, and more importantly a cell whose closest point is an edge
// is *genuinely* equidistant from two triangles. Letting either one win is safe
// precisely because the sign then comes from that edge's pseudonormal, which is
// the same vector whichever of the two triangles is asked.
@compute @workgroup_size(64)
fn assign_triangle(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    for (var ti = wid.x; ti < U.tri_count; ti += nwg.x) {
        let t = active_tris[ti];
        let tri = tris[t];
        let r = triangle_voxels(tri);
        if (r.empty) { continue; }
        let n = r.hi - r.lo + vec3<u32>(1u);
        let nvox = n.x * n.y * n.z;

        for (var i = lid; i < nvox; i += WG) {
            let c = r.lo + vec3<u32>(i % n.x, (i / n.x) % n.y, i / (n.x * n.y));
            let p = cell_center(c);
            let cl = closest_point_on_tri(p, tri.a, tri.b, tri.c);
            let d = length(p - cl.p);
            if (d > U.band_mm) { continue; }
            let cell = cell_index(c);
            let best = bitcast<f32>(atomicLoad(&dist[cell]));
            if (d <= best * 1.000001 + 1.0e-9) {
                atomicMin(&tri_idx[cell], t);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Sign, by the angle-weighted pseudonormal
//
// This is the stage that decides whether the solver runs or explodes. Using the
// nearest *face* normal is wrong whenever the closest point lands on an edge or
// a vertex whose incident faces are more than 90 degrees apart, which for a
// printed duct with sharp lips is a large fraction of the near-wall cells. The
// failures are isolated solid cells floating in the passage and isolated fluid
// cells buried in the wall; both inject momentum from nowhere and the solver
// diverges within a few hundred steps.
//
// Baerentzen & Aanaes 2005 give the fix: at an edge use the sum of the two
// incident face normals, at a vertex the sum of the incident face normals each
// weighted by that face's interior angle at the vertex. Those vectors lie
// strictly inside the feature's normal cone, so `dot(p - closest, N)` has the
// right sign for every point. The weights are computed on the CPU in
// ad_geom::mesh::PseudoNormals; all this stage does is pick the right one.
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn sign_band(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = region_count();
    let stride = nwg.x * WG;
    for (var i = gid.x; i < n; i += stride) {
        let c = region_cell(i);
        let cell = cell_index(c);
        let t = atomicLoad(&tri_idx[cell]);
        if (t == NO_TRI) { continue; }

        let tri = tris[t];
        let p = cell_center(c);
        let cl = closest_point_on_tri(p, tri.a, tri.b, tri.c);
        let v = p - cl.p;
        let d = length(v);
        let pn = pseudonormal(t, cl.feature);

        if (dot(v, pn) > 0.0) {
            phi[cell] = d;
            state[cell] = ST_BAND_OUT;
        } else {
            phi[cell] = -d;
            state[cell] = ST_BAND_IN;
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Flood fill the sign outside the band
//
// The band is a closed shell at least two cells thick around a watertight
// surface, so its interior-signed cells separate the inside from the outside.
// Marking everything reachable from the domain boundary as exterior therefore
// leaves exactly the interior unmarked.
//
// Propagation runs through cells that are already known to be *outside* as well
// as through unknown ones. That matters for a concave pocket narrower than
// twice the band: such a pocket is unknown in the middle but walled off by
// outside-signed band cells, and a fill that stopped at any band cell would
// wrongly call it solid.
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn reset_fill(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = cell_count();
    let stride = nwg.x * WG;
    for (var i = gid.x; i < n; i += stride) {
        if (state[i] == ST_EXTERIOR) {
            state[i] = ST_UNKNOWN;
        }
    }
}

@compute @workgroup_size(64)
fn seed_exterior(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = cell_count();
    let stride = nwg.x * WG;
    let last = U.dims - vec3<u32>(1u);
    for (var i = gid.x; i < n; i += stride) {
        let x = i % U.dims.x;
        let y = (i / U.dims.x) % U.dims.y;
        let z = i / (U.dims.x * U.dims.y);
        let on_face = x == 0u || y == 0u || z == 0u || x == last.x || y == last.y || z == last.z;
        if (on_face && state[i] == ST_UNKNOWN) {
            state[i] = ST_EXTERIOR;
        }
    }
}

fn line_cell(a: u32, b: u32, i: u32) -> u32 {
    if (U.axis == 0u) {
        return cell_index(vec3<u32>(i, a, b));
    } else if (U.axis == 1u) {
        return cell_index(vec3<u32>(a, i, b));
    }
    return cell_index(vec3<u32>(a, b, i));
}

// One thread per line, sweeping it forwards then backwards. A line only ever
// touches its own cells, so no atomics are needed inside the sweep; three
// sweeps (X, Y, Z) make up one iteration, and iterations repeat until nothing
// changes.
@compute @workgroup_size(64)
fn sweep(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    var len: u32;
    var na: u32;
    var nb: u32;
    if (U.axis == 0u) {
        len = U.dims.x; na = U.dims.y; nb = U.dims.z;
    } else if (U.axis == 1u) {
        len = U.dims.y; na = U.dims.x; nb = U.dims.z;
    } else {
        len = U.dims.z; na = U.dims.x; nb = U.dims.y;
    }
    let lines = na * nb;
    let stride = nwg.x * WG;

    for (var l = gid.x; l < lines; l += stride) {
        let a = l % na;
        let b = l / na;
        var changed = false;

        var open = false;
        for (var i = 0u; i < len; i += 1u) {
            let cell = line_cell(a, b, i);
            let s = state[cell];
            if (s == ST_BAND_IN) {
                open = false;
            } else if (s == ST_UNKNOWN) {
                if (open) {
                    state[cell] = ST_EXTERIOR;
                    changed = true;
                }
            } else {
                open = true;
            }
        }

        open = false;
        for (var k = 0u; k < len; k += 1u) {
            let i = len - 1u - k;
            let cell = line_cell(a, b, i);
            let s = state[cell];
            if (s == ST_BAND_IN) {
                open = false;
            } else if (s == ST_UNKNOWN) {
                if (open) {
                    state[cell] = ST_EXTERIOR;
                    changed = true;
                }
            } else {
                open = true;
            }
        }

        if (changed) {
            atomicMax(&counters[3], 1u);
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Everything the fill never reached is interior.
//
// Cells outside the band get a clamped magnitude rather than a true distance.
// That is all a narrow-band field promises, and it is all the solver and the
// renderer need: the wall model only looks at cells adjacent to the surface,
// and the ray marcher only needs a conservative step outside the band.
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn resolve_unknown(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = cell_count();
    let stride = nwg.x * WG;
    let far = U.band_mm + U.dx_mm;
    for (var i = gid.x; i < n; i += stride) {
        let s = state[i];
        if (s == ST_UNKNOWN) {
            phi[i] = -far;
        } else if (s == ST_EXTERIOR) {
            phi[i] = far;
        }
    }
}

// ---------------------------------------------------------------------------
// 7. Flags and boundary links
//
// Run twice: once with link_capacity = 0 to learn the exact count, then again
// with the output buffer sized. Writing the flags twice is harmless because
// every write is an atomicOr of the same bits.
// ---------------------------------------------------------------------------

fn write_flag(cell: u32, value: u32) {
    // Four cells per word, little-endian, so the buffer reads as a plain u8
    // array on the CPU side and as packed u32 in the solver.
    let word = cell >> 2u;
    let shift = (cell & 3u) * 8u;
    atomicOr(&cell_flags[word], value << shift);
}

@compute @workgroup_size(64)
fn classify(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = cell_count();
    let stride = nwg.x * WG;
    let last = vec3<i32>(U.dims) - vec3<i32>(1);

    for (var i = gid.x; i < n; i += stride) {
        let phi_f = phi[i];
        if (phi_f <= 0.0) {
            write_flag(i, FLAG_SOLID);
            atomicAdd(&counters[1], 1u);
            continue;
        }

        let c = vec3<i32>(
            i32(i % U.dims.x),
            i32((i / U.dims.x) % U.dims.y),
            i32(i / (U.dims.x * U.dims.y)),
        );

        var boundary = false;
        for (var d = 1u; d < Q; d += 1u) {
            let nb = c + C[d];
            if (any(nb < vec3<i32>(0)) || any(nb > last)) { continue; }
            let nc = cell_index(vec3<u32>(nb));
            let phi_n = phi[nc];
            if (phi_n > 0.0) { continue; }

            boundary = true;
            // Linear interpolation along the link. phi is a Euclidean distance,
            // so this is a fraction of the link length for the diagonal
            // directions too, which is exactly what interpolated bounce-back
            // wants.
            let q = clamp(phi_f / (phi_f - phi_n), 1.0 / 255.0, 1.0);
            let slot = atomicAdd(&counters[0], 1u);
            if (slot < U.link_capacity) {
                let qq = u32(round(q * 255.0));
                links[slot] = Link(i, d | (qq << 8u));
            } else if (U.link_capacity != 0u) {
                atomicAdd(&counters[4], 1u);
            }
        }

        if (boundary) {
            write_flag(i, FLAG_SOLID_BOUNDARY);
            atomicAdd(&counters[2], 1u);
        }
    }
}
