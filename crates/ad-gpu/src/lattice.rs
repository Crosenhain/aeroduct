//! Lattice velocity sets and the Esoteric Pull streaming index scheme.
//!
//! # Direction ordering is load-bearing
//!
//! Directions are ordered so that `i` and `i+1` are opposites for every odd `i`:
//! `(1,2)`, `(3,4)`, ... `(17,18)`. Esoteric Pull depends on this pairing, and so
//! does bounce-back via [`opposite`]. Do not reorder these tables without
//! reworking both.
//!
//! # Esoteric Pull
//!
//! Lehmann 2022, *Computation* 10(6) 92. Half the distribution functions stream
//! at the end of one stream-collide kernel and the other half at the beginning of
//! the next, so only one copy of the DDFs is ever needed. That halves storage
//! versus the usual two-population ping-pong, and it makes bounce-back implicit:
//! a solid neighbour just means a cell reads back its own opposite slot, costing
//! zero extra memory traffic and zero branches.
//!
//! The cost is that half of a cell's DDFs physically live in a neighbour's array
//! slot. Every read or write must therefore go through [`EsotericPull`] rather
//! than indexing the buffer directly. That includes post-processing: anything
//! wanting "all q distributions at cell x at time t" must ask this module.

use glam::IVec3;

/// Everything a velocity set needs to describe itself.
pub struct LatticeDef {
    pub q: usize,
    pub directions: &'static [IVec3],
    pub weights: &'static [f32],
}

/// D3Q19 direction vectors, in opposite-adjacent order.
pub const D3Q19_DIRS: [IVec3; 19] = [
    IVec3::new(0, 0, 0),
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(0, -1, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(0, 0, -1),
    IVec3::new(1, 1, 0),
    IVec3::new(-1, -1, 0),
    IVec3::new(1, 0, 1),
    IVec3::new(-1, 0, -1),
    IVec3::new(0, 1, 1),
    IVec3::new(0, -1, -1),
    IVec3::new(1, -1, 0),
    IVec3::new(-1, 1, 0),
    IVec3::new(1, 0, -1),
    IVec3::new(-1, 0, 1),
    IVec3::new(0, 1, -1),
    IVec3::new(0, -1, 1),
];

const W0_19: f32 = 1.0 / 3.0;
const W1_19: f32 = 1.0 / 18.0;
const W2_19: f32 = 1.0 / 36.0;

pub const D3Q19_WEIGHTS: [f32; 19] = [
    W0_19, W1_19, W1_19, W1_19, W1_19, W1_19, W1_19, W2_19, W2_19, W2_19, W2_19, W2_19, W2_19,
    W2_19, W2_19, W2_19, W2_19, W2_19, W2_19,
];

/// D3Q27: the D3Q19 set plus the eight body diagonals, same pairing rule.
pub const D3Q27_DIRS: [IVec3; 27] = [
    IVec3::new(0, 0, 0),
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(0, -1, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(0, 0, -1),
    IVec3::new(1, 1, 0),
    IVec3::new(-1, -1, 0),
    IVec3::new(1, 0, 1),
    IVec3::new(-1, 0, -1),
    IVec3::new(0, 1, 1),
    IVec3::new(0, -1, -1),
    IVec3::new(1, -1, 0),
    IVec3::new(-1, 1, 0),
    IVec3::new(1, 0, -1),
    IVec3::new(-1, 0, 1),
    IVec3::new(0, 1, -1),
    IVec3::new(0, -1, 1),
    IVec3::new(1, 1, 1),
    IVec3::new(-1, -1, -1),
    IVec3::new(1, 1, -1),
    IVec3::new(-1, -1, 1),
    IVec3::new(1, -1, 1),
    IVec3::new(-1, 1, -1),
    IVec3::new(-1, 1, 1),
    IVec3::new(1, -1, -1),
];

const W0_27: f32 = 8.0 / 27.0;
const W1_27: f32 = 2.0 / 27.0;
const W2_27: f32 = 1.0 / 54.0;
const W3_27: f32 = 1.0 / 216.0;

pub const D3Q27_WEIGHTS: [f32; 27] = [
    W0_27, W1_27, W1_27, W1_27, W1_27, W1_27, W1_27, W2_27, W2_27, W2_27, W2_27, W2_27, W2_27,
    W2_27, W2_27, W2_27, W2_27, W2_27, W2_27, W3_27, W3_27, W3_27, W3_27, W3_27, W3_27, W3_27,
    W3_27,
];

pub const D3Q19: LatticeDef =
    LatticeDef { q: 19, directions: &D3Q19_DIRS, weights: &D3Q19_WEIGHTS };
pub const D3Q27: LatticeDef =
    LatticeDef { q: 27, directions: &D3Q27_DIRS, weights: &D3Q27_WEIGHTS };

impl super::types::VelocitySet {
    pub fn def(self) -> &'static LatticeDef {
        match self {
            super::types::VelocitySet::D3Q19 => &D3Q19,
            super::types::VelocitySet::D3Q27 => &D3Q27,
        }
    }
}

/// Index of the direction opposite to `i`.
///
/// With the pairing `(1,2), (3,4), ...` this is "flip the low bit of `i-1`",
/// *not* the `i ^ 1` you would use if the pairs were `(0,1), (2,3), ...`.
#[inline]
pub const fn opposite(i: usize) -> usize {
    if i == 0 {
        0
    } else {
        ((i - 1) ^ 1) + 1
    }
}

/// The odd member of the pair `i` belongs to. Pairs are `(1,2), (3,4), ...`, so
/// this maps 1 and 2 to 1, 3 and 4 to 3, and so on.
#[inline]
pub const fn pair_first(i: usize) -> usize {
    ((i - 1) & !1usize) + 1
}

/// The Esoteric Pull index scheme, in Rust, so the CPU-side reference solver and
/// the validation harness agree with the shader by construction.
///
/// # Derivation
///
/// Take a pair `(m, m+1)` with `c_{m+1} = -c_m`, and the link joining cell `n` to
/// `n + c_m`. Exactly two populations ever travel that link: `f_m` from `n`
/// towards `n + c_m`, and `f_{m+1}` back the other way. Esoteric Pull gives the
/// link one slot per population and swaps which is which on alternate steps, so
/// the pair of cells sharing the link never write the same address in the same
/// step. That is what makes it thread-safe with a single copy of the DDFs.
///
/// Canonically assign the link to its lower cell, so the link `(n, n + c_m)` uses
/// slots indexed by `n`, and the link arriving at `n` from `n - c_m` uses slots
/// indexed by `n - c_m`. Then, with `p` the current step parity:
///
/// ```text
/// load  f_m     <- slot(n - c_m, p ? m   : m+1)
/// load  f_{m+1} <- slot(n,       p ? m+1 : m  )
///
/// store f_m     -> slot(n,       p ? m+1 : m  )
/// store f_{m+1} -> slot(n - c_m, p ? m   : m+1)
/// ```
///
/// Two things fall out. Store uses the *complement* of the parity expression load
/// used, which is exactly what makes the next step pick the value up from the
/// right neighbour. And a cell's load and store for one pair touch the same two
/// addresses, which is the in-place property: read a slot, write the pair's other
/// member back into it.
///
/// `slot(cell, dir) = dir * cell_count + cell` is structure-of-arrays; see
/// [`crate::ddf`] for why SoA is mandatory here rather than merely preferable.
///
/// The round-trip property is verified exhaustively over a small periodic lattice
/// in this module's tests. If you change any of this, that test is the gate.
#[derive(Debug, Clone, Copy)]
pub struct EsotericPull {
    pub cell_count: u64,
    pub q: usize,
}

impl EsotericPull {
    pub fn new(cell_count: u64, q: usize) -> Self {
        Self { cell_count, q }
    }

    /// Flat offset of one distribution within the SoA layout.
    #[inline]
    pub fn slot(&self, cell: u64, dir: usize) -> u64 {
        dir as u64 * self.cell_count + cell
    }

    /// Which slot to *read* distribution `i` of `cell` from.
    ///
    /// `neighbour(d)` must return the linear index of `cell` shifted by direction
    /// `d`. Only ever called with the direction opposite the pair's odd member.
    pub fn load_slot(
        &self,
        cell: u64,
        i: usize,
        odd_step: bool,
        neighbour: impl Fn(usize) -> u64,
    ) -> u64 {
        if i == 0 {
            return self.slot(cell, 0);
        }
        let m = pair_first(i);
        if i == m {
            // Upstream link: the one arriving from n - c_m.
            self.slot(neighbour(m + 1), if odd_step { m } else { m + 1 })
        } else {
            self.slot(cell, if odd_step { m + 1 } else { m })
        }
    }

    /// Which slot to *write* distribution `i` of `cell` to.
    pub fn store_slot(
        &self,
        cell: u64,
        i: usize,
        odd_step: bool,
        neighbour: impl Fn(usize) -> u64,
    ) -> u64 {
        if i == 0 {
            return self.slot(cell, 0);
        }
        let m = pair_first(i);
        if i == m {
            self.slot(cell, if odd_step { m + 1 } else { m })
        } else {
            self.slot(neighbour(m + 1), if odd_step { m } else { m + 1 })
        }
    }
}

/// Emit the lattice tables as a WGSL prelude.
///
/// Generated rather than hand-written so the shader and the Rust reference can
/// never disagree about direction ordering or weights, which would be a silent
/// and extremely hard-to-find class of bug.
pub fn wgsl_prelude(set: super::types::VelocitySet) -> String {
    let def = set.def();
    let mut s = String::with_capacity(2048);
    s.push_str("// GENERATED by ad-gpu::lattice::wgsl_prelude - do not edit\n");
    s.push_str(&format!("const Q: u32 = {}u;\n", def.q));
    s.push_str(&format!("const C: array<vec3<i32>, {}> = array<vec3<i32>, {}>(\n", def.q, def.q));
    for d in def.directions {
        s.push_str(&format!("    vec3<i32>({}, {}, {}),\n", d.x, d.y, d.z));
    }
    s.push_str(");\n");
    s.push_str(&format!("const W: array<f32, {}> = array<f32, {}>(\n", def.q, def.q));
    for w in def.weights {
        s.push_str(&format!("    {:.17},\n", w));
    }
    s.push_str(");\n");
    s.push_str(
        "\n// Opposite direction, given the (1,2), (3,4), ... pairing of the tables\n\
         // above. Note this is NOT i ^ 1, which would be right only if the pairs\n\
         // started at index 0.\n\
         fn opposite(i: u32) -> u32 { return select(((i - 1u) ^ 1u) + 1u, 0u, i == 0u); }\n\
         \n\
         // The odd member of the pair i belongs to.\n\
         fn pair_first(i: u32) -> u32 { return ((i - 1u) & ~1u) + 1u; }\n",
    );
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::VelocitySet;

    fn check_set(dirs: &[IVec3], weights: &[f32]) {
        // Weights sum to one, or the equilibrium is not normalised.
        let sum: f64 = weights.iter().map(|w| *w as f64).sum();
        assert!((sum - 1.0).abs() < 1e-6, "weights summed to {sum}");

        // Pairing: i and i+1 must be opposites for odd i.
        for i in (1..dirs.len()).step_by(2) {
            assert_eq!(dirs[i], -dirs[i + 1], "directions {i} and {} are not opposites", i + 1);
            assert_eq!(weights[i], weights[i + 1], "paired weights differ at {i}");
            assert_eq!(opposite(i), i + 1);
            assert_eq!(opposite(i + 1), i);
        }

        // Directions are distinct.
        for i in 0..dirs.len() {
            for j in (i + 1)..dirs.len() {
                assert_ne!(dirs[i], dirs[j], "duplicate direction at {i} and {j}");
            }
        }

        // First moment vanishes and the second moment is isotropic with c_s^2 = 1/3.
        let mut m1 = [0.0f64; 3];
        let mut m2 = [[0.0f64; 3]; 3];
        for (d, w) in dirs.iter().zip(weights) {
            let (w, c) = (*w as f64, [d.x as f64, d.y as f64, d.z as f64]);
            for a in 0..3 {
                m1[a] += w * c[a];
                for b in 0..3 {
                    m2[a][b] += w * c[a] * c[b];
                }
            }
        }
        for a in 0..3 {
            // Tolerance is set by the f32 weights, not by the arithmetic: the
            // tables are stored as f32 so the closure holds to ~1e-7, not 1e-15.
            assert!(m1[a].abs() < 1e-6, "first moment nonzero: {m1:?}");
            for b in 0..3 {
                let want = if a == b { 1.0 / 3.0 } else { 0.0 };
                assert!((m2[a][b] - want).abs() < 1e-6, "second moment wrong at {a},{b}: {m2:?}");
            }
        }
    }

    #[test]
    fn opposite_and_pair_first_match_the_table_ordering() {
        assert_eq!(opposite(0), 0);
        for (i, want) in [(1, 2), (2, 1), (3, 4), (4, 3), (17, 18), (18, 17)] {
            assert_eq!(opposite(i), want, "opposite({i})");
        }
        for (i, want) in [(1, 1), (2, 1), (3, 3), (4, 3), (17, 17), (18, 17)] {
            assert_eq!(pair_first(i), want, "pair_first({i})");
        }
    }

    #[test]
    fn d3q19_is_well_formed() {
        check_set(&D3Q19_DIRS, &D3Q19_WEIGHTS);
    }

    #[test]
    fn d3q27_is_well_formed() {
        check_set(&D3Q27_DIRS, &D3Q27_WEIGHTS);
    }

    /// The property that makes Esoteric Pull correct: what one step stores, the
    /// next step's load must retrieve from the neighbour it streamed to.
    #[test]
    fn esoteric_pull_store_then_load_round_trips() {
        // A 4x4x4 periodic lattice is enough to exercise every link.
        const N: i32 = 4;
        let cells = (N * N * N) as u64;
        let ep = EsotericPull::new(cells, 19);
        let lin = |c: IVec3| -> u64 {
            let w = |v: i32| ((v % N) + N) % N;
            ((w(c.z) * N + w(c.y)) * N + w(c.x)) as u64
        };

        for odd in [false, true] {
            for zi in 0..N {
                for yi in 0..N {
                    for xi in 0..N {
                        let here = IVec3::new(xi, yi, zi);
                        let n_here = lin(here);
                        for i in 1..19usize {
                            // Where this cell parks distribution i on this step.
                            let stored = ep
                                .store_slot(n_here, i, odd, |d| lin(here + D3Q19_DIRS[d]));
                            // Streaming carries f_i from here to here + c_i, so
                            // that cell must read the same slot for the same
                            // direction on the next step.
                            let there = here + D3Q19_DIRS[i];
                            let n_there = lin(there);
                            let loaded = ep
                                .load_slot(n_there, i, !odd, |d| lin(there + D3Q19_DIRS[d]));
                            assert_eq!(
                                stored, loaded,
                                "link {i} from {here:?} (odd={odd}) stored to slot {stored} but \
                                 the downstream cell {there:?} loads slot {loaded}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn wgsl_prelude_mentions_every_direction() {
        let src = wgsl_prelude(VelocitySet::D3Q19);
        assert!(src.contains("const Q: u32 = 19u;"));
        // 19 direction entries, plus the two array type annotations on the
        // declaration line.
        assert_eq!(src.matches("vec3<i32>").count(), 19 + 2);
        assert!(src.contains("fn opposite"));
        assert!(src.contains("fn pair_first"));
    }
}
