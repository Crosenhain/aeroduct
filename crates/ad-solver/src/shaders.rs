//! Generated WGSL: the DDF bindings, the storage codec, and the Esoteric Pull
//! load/store bodies.
//!
//! # Why these are generated rather than written by hand
//!
//! WGSL has no arrays of bindings, so the `q` structure-of-arrays DDF buffers
//! must be `q` separate global declarations, and any access has to name one of
//! them literally. Written by hand that is 19 (or 27) near-identical blocks
//! whose only content is index arithmetic — precisely the sort of thing that
//! goes wrong once and then produces plausible-looking physics forever.
//!
//! Worse, a `switch` over the direction index would be *divergent*: bounce-back
//! flips which of a pair's two slots a lane reads, so neighbouring lanes want
//! different buffers. A 19-way divergent switch on the hot path is not
//! acceptable. Unrolling the pair loop turns it into a two-way branch that is
//! uniform everywhere except at a wall.
//!
//! # How the addressing is derived
//!
//! Not by restating the rule — by *asking* [`ad_gpu::lattice::EsotericPull`].
//! A `probe` helper runs the real Rust implementation with a sentinel cell count, so
//! `slot()` becomes an invertible encoding of `(direction, which cell)`, and
//! reads the answer back out. The generated shader is therefore derived from the
//! same code that `esoteric_pull_store_then_load_round_trips` covers
//! exhaustively, and cannot drift from it.

use ad_gpu::lattice::{pair_first, EsotericPull};
use ad_gpu::types::{DdfPrecision, VelocitySet};
use std::cell::Cell;
use std::fmt::Write as _;

/// Sentinel cell count. Large enough that `dir * SENTINEL + cell` is uniquely
/// decodable for the two cell indices we probe with (0 = this cell,
/// 1 = the neighbour), and small enough not to overflow.
const SENTINEL: u64 = 1 << 20;

/// One resolved address: which direction buffer, and whether it is indexed by
/// this cell or by the neighbour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Addr {
    pub dir: usize,
    /// `Some(d)` means "index by the neighbour in direction `d`".
    pub neighbour_dir: Option<usize>,
}

/// Ask [`EsotericPull`] where direction `i` lives, for both step parities.
///
/// Public because it is the *only* sanctioned way to learn the transport
/// addressing, and a second backend must not re-derive it. `ad-cuda` generates
/// its CUDA C from this same function, so the two backends' unrolled load/store
/// bodies come from one implementation and cannot drift apart — which is the
/// whole reason this generator exists rather than a hand-written table.
pub fn probe(q: usize, i: usize, odd: bool, store: bool) -> Addr {
    let ep = EsotericPull::new(SENTINEL, q);
    let asked: Cell<Option<usize>> = Cell::new(None);
    let neighbour = |d: usize| {
        asked.set(Some(d));
        1u64
    };
    let idx = if store {
        ep.store_slot(0, i, odd, neighbour)
    } else {
        ep.load_slot(0, i, odd, neighbour)
    };
    let dir = (idx / SENTINEL) as usize;
    let cell = idx % SENTINEL;
    Addr {
        dir,
        neighbour_dir: if cell == 1 { asked.get() } else { None },
    }
}

/// The `enable` directives the generated bindings need, if any.
///
/// WGSL requires every `enable` to precede every declaration in the module, so
/// this cannot live next to the bindings it belongs to; `ad_solver::solver::
/// build_loader` puts it at the very top of the first virtual include instead.
pub fn enable_directives(layout: DdfLayout) -> &'static str {
    match layout {
        DdfLayout::Fp16cPerCell => "enable wgpu_int16;\n",
        _ => "",
    }
}

/// How the DDF buffers are addressed, which is not implied by the precision
/// alone: FP16C has two layouts over identical bytes.
///
/// Chosen by [`ad_gpu::DdfBuffers::allocate`] from the device's features, and
/// carried here rather than re-derived, because the allocator and the shader
/// disagreeing about it is a silent factor-of-two indexing error rather than a
/// compile failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DdfLayout {
    /// One `u32` per cell, holding the population bit-for-bit.
    Fp32,
    /// One `u16` per cell. Needs `wgpu::Features::SHADER_I16`; the store is a
    /// plain unsynchronised 16-bit write.
    Fp16cPerCell,
    /// Two cells packed into one `u32`, for adapters without 16-bit storage.
    /// One word has two writers, so every store is an atomic read-modify-write.
    Fp16cPacked,
}

impl DdfLayout {
    /// The layout `DdfBuffers` chose, given the precision and whether it managed
    /// to give every cell its own element.
    pub fn of(precision: DdfPrecision, fp16c_per_cell: bool) -> Self {
        match (precision, fp16c_per_cell) {
            (DdfPrecision::Fp32, _) => DdfLayout::Fp32,
            (DdfPrecision::Fp16c, true) => DdfLayout::Fp16cPerCell,
            (DdfPrecision::Fp16c, false) => DdfLayout::Fp16cPacked,
        }
    }

    /// A name for `ShaderDefines`, so the preprocessor cache key distinguishes
    /// two builds that differ only in storage layout.
    pub const fn shader_define(self) -> &'static str {
        match self {
            DdfLayout::Fp32 => "DDF_LAYOUT_FP32",
            DdfLayout::Fp16cPerCell => "DDF_LAYOUT_FP16C_U16",
            DdfLayout::Fp16cPacked => "DDF_LAYOUT_FP16C_PACKED",
        }
    }
}

/// The `q` storage-buffer declarations plus per-direction typed accessors.
fn bindings(set: VelocitySet, layout: DdfLayout) -> String {
    let q = set.q();
    let mut s = String::with_capacity(8192);
    s.push_str("// GENERATED by ad_solver::shaders - do not edit.\n");

    match layout {
        DdfLayout::Fp32 => {
            for i in 0..q {
                let _ = writeln!(
                    s,
                    "@group(1) @binding({i}) var<storage, read_write> ddf{i}: array<u32>;"
                );
            }
            s.push_str(
                "\n// FP32 storage. The value stored is still the *shifted* population\n\
                 // g = f - w; see ad_solver::precision for why that matters even when\n\
                 // the storage is exact.\n",
            );
            for i in 0..q {
                let _ = writeln!(
                    s,
                    "fn ddf_get_{i}(cell: u32) -> f32 {{ return bitcast<f32>(ddf{i}[cell]); }}"
                );
                let _ = writeln!(
                    s,
                    "fn ddf_put_{i}(cell: u32, v: f32) {{ ddf{i}[cell] = bitcast<u32>(v); }}"
                );
            }
        }
        DdfLayout::Fp16cPerCell => {
            for i in 0..q {
                let _ = writeln!(
                    s,
                    "@group(1) @binding({i}) var<storage, read_write> ddf{i}: array<u16>;"
                );
            }
            s.push_str(
                "\n// FP16C storage, one u16 per cell (ad_gpu::ddf allocates two bytes per\n\
                 // cell per direction). Nothing is shared between invocations, so the\n\
                 // store is a plain 16-bit write and there is no synchronisation at all.\n\
                 //\n\
                 // The element is u16 and not f16 because the two are not\n\
                 // interchangeable here. FP16C is 1 sign / 4 exponent / 11 mantissa, so\n\
                 // an encoded value is a bit pattern, not a binary16 number: everything\n\
                 // in [1.5, 2) encodes as what binary16 would read as a NaN. WGSL offers\n\
                 // no bitcast between f16 and u32 to launder the bits through, whereas a\n\
                 // u16 carries all sixteen of them exactly. `enable wgpu_int16;` is\n\
                 // emitted at the top of the module by ad_solver::shaders.\n\
                 //\n\
                 // The value stored is the *shifted* population g = f - w, encoded by the\n\
                 // codec ad_solver::precision emits. Shifting is what makes 16-bit\n\
                 // storage viable; see that module for why the order of operations that\n\
                 // produces g must not be rearranged.\n",
            );
            for i in 0..q {
                let _ = writeln!(
                    s,
                    "fn ddf_get_{i}(cell: u32) -> f32 {{ return fp16c_to_f32(u32(ddf{i}[cell])); }}"
                );
                let _ = writeln!(
                    s,
                    "fn ddf_put_{i}(cell: u32, v: f32) {{ ddf{i}[cell] = u16(f32_to_fp16c(v)); }}"
                );
            }
        }
        DdfLayout::Fp16cPacked => {
            for i in 0..q {
                let _ = writeln!(
                    s,
                    "@group(1) @binding({i}) var<storage, read_write> ddf{i}: array<atomic<u32>>;"
                );
            }
            s.push_str(
                "\n// FP16C fallback for adapters without 16-bit storage (wgpu SHADER_I16).\n\
                 // Two *cells* share one u32, so the word straddling cells 2k and 2k+1 is\n\
                 // written by two different invocations.\n\
                 //\n\
                 // Esoteric Pull guarantees each 16-bit half is owned by exactly one cell\n\
                 // for the whole step - a cell reads and writes the same address, and the\n\
                 // two cells sharing a link swap which slot they own each step - so the\n\
                 // halves never race with each other. Only the read-modify-write of the\n\
                 // containing word does. atomicAnd with a mask that leaves the other half\n\
                 // untouched, followed by atomicOr of our own bits, is therefore correct\n\
                 // under every interleaving: neither invocation ever touches a bit the\n\
                 // other owns.\n\
                 //\n\
                 // Correct, but 2q atomics per cell per step: measurably slower than FP32\n\
                 // despite moving half the bytes. This path exists so an adapter that\n\
                 // cannot address 16 bits still runs, not because it is a good idea.\n",
            );
            for i in 0..q {
                let _ = writeln!(
                    s,
                    "fn ddf_get_{i}(cell: u32) -> f32 {{\n    \
                       let sh = (cell & 1u) * 16u;\n    \
                       return fp16c_to_f32((atomicLoad(&ddf{i}[cell >> 1u]) >> sh) & 0xffffu);\n\
                     }}"
                );
                let _ = writeln!(
                    s,
                    "fn ddf_put_{i}(cell: u32, v: f32) {{\n    \
                       let sh = (cell & 1u) * 16u;\n    \
                       let h = f32_to_fp16c(v) << sh;\n    \
                       atomicAnd(&ddf{i}[cell >> 1u], ~(0xffffu << sh));\n    \
                       atomicOr(&ddf{i}[cell >> 1u], h);\n\
                     }}"
                );
            }
        }
    }
    s
}

/// The unrolled load and store bodies.
///
/// `load_ddf` fills `g[i]` with the shifted population arriving along direction
/// `i`, applying the bounce-back parity flip on any link whose bit is set in
/// `mask`. `store_ddf` writes them back; stores are never flipped.
fn transport(set: VelocitySet) -> String {
    let q = set.q();
    let mut s = String::with_capacity(16384);
    s.push_str(
        "// GENERATED by ad_solver::shaders::transport - do not edit.\n\
         //\n\
         // Addresses come from ad_gpu::lattice::EsotericPull itself (see the module\n\
         // docs); this file is a transcription of what that code returns, not a\n\
         // re-derivation of it.\n\
         //\n\
         // `mask` bit i set means the upstream neighbour of direction i is solid, in\n\
         // which case the *load* uses the opposite step parity. That reads back the\n\
         // population this cell pushed into the wall link last step, which is exactly\n\
         // halfway bounce-back - at zero extra memory traffic and with a two-way\n\
         // branch that is uniform everywhere except at a wall.\n\n",
    );

    // ---- load ----
    s.push_str("fn load_ddf(cell: u32, c: vec3<u32>, odd: bool, mask: u32, g: ptr<function, array<f32, Q_CONST>>) {\n");
    let a0 = probe(q, 0, false, false);
    assert_eq!(a0.dir, 0);
    s.push_str("    (*g)[0] = ddf_get_0(cell);\n");

    let mut i = 1;
    while i < q {
        let m = pair_first(i);
        assert_eq!(m, i, "pairs must start on odd indices");
        // Both members of the pair share one neighbour lookup.
        let even = probe(q, i, false, false);
        let odd = probe(q, i, true, false);
        let nd = even.neighbour_dir.or(odd.neighbour_dir).expect("odd member must use a neighbour");
        let _ = writeln!(s, "    {{ // pair ({i}, {})", i + 1);
        let _ = writeln!(s, "        let nb = neighbour_index(c, {nd}u);");
        for k in 0..2 {
            let d = i + k;
            let e = probe(q, d, false, false);
            let o = probe(q, d, true, false);
            let idx_e = if e.neighbour_dir.is_some() { "nb" } else { "cell" };
            let idx_o = if o.neighbour_dir.is_some() { "nb" } else { "cell" };
            assert_eq!(idx_e, idx_o, "load address base must not depend on parity");
            let _ = writeln!(
                s,
                "        if (odd != (((mask >> {d}u) & 1u) != 0u)) {{ (*g)[{d}] = ddf_get_{}({idx_o}); }} \
                 else {{ (*g)[{d}] = ddf_get_{}({idx_e}); }}",
                o.dir, e.dir
            );
        }
        s.push_str("    }\n");
        i += 2;
    }
    s.push_str("}\n\n");

    // ---- store ----
    s.push_str("fn store_ddf(cell: u32, c: vec3<u32>, odd: bool, g: ptr<function, array<f32, Q_CONST>>) {\n");
    s.push_str("    ddf_put_0(cell, (*g)[0]);\n");
    let mut i = 1;
    while i < q {
        let even = probe(q, i, false, true);
        let odd_a = probe(q, i, true, true);
        let even_b = probe(q, i + 1, false, true);
        let odd_b = probe(q, i + 1, true, true);
        let nd = even_b
            .neighbour_dir
            .or(odd_b.neighbour_dir)
            .expect("even member's store must use a neighbour");
        let _ = writeln!(s, "    {{ // pair ({i}, {})", i + 1);
        let _ = writeln!(s, "        let nb = neighbour_index(c, {nd}u);");
        let _ = writeln!(
            s,
            "        if (odd) {{ ddf_put_{}(cell, (*g)[{i}]); ddf_put_{}(nb, (*g)[{}]); }} \
             else {{ ddf_put_{}(cell, (*g)[{i}]); ddf_put_{}(nb, (*g)[{}]); }}",
            odd_a.dir,
            odd_b.dir,
            i + 1,
            even.dir,
            even_b.dir,
            i + 1
        );
        assert!(even.neighbour_dir.is_none() && odd_a.neighbour_dir.is_none());
        s.push_str("    }\n");
        i += 2;
    }
    s.push_str("}\n");
    s
}

/// The [`ad_gpu::types::flags`] bitfield, emitted so the shader cannot hold a
/// stale copy of a value that lives in another crate.
fn flag_constants() -> String {
    use ad_gpu::types::flags;
    let mut s = String::from("// GENERATED from ad_gpu::types::flags\n");
    for (name, value) in [
        ("SOLID", flags::SOLID),
        ("SOLID_BOUNDARY", flags::SOLID_BOUNDARY),
        ("INLET", flags::INLET),
        ("OUTLET", flags::OUTLET),
        ("SPONGE", flags::SPONGE),
        ("EQUILIBRIUM", flags::EQUILIBRIUM),
    ] {
        let _ = writeln!(s, "const FLAG_{name}: u32 = {value}u;");
    }
    s
}

/// Everything generated, as one virtual include.
///
/// Does *not* include the `enable` directives the layout may need; those have to
/// precede every declaration in the module, so [`enable_directives`] emits them
/// separately and `build_loader` places them first.
pub fn generated_prelude(set: VelocitySet, layout: DdfLayout) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "const Q_CONST: u32 = {}u;", set.q());
    s.push_str(&flag_constants());
    s.push('\n');
    s.push_str(&bindings(set, layout));
    s.push('\n');
    s.push_str(&transport(set));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_recovers_the_documented_scheme() {
        // The doc comment on EsotericPull states:
        //   load  f_m     <- slot(n - c_m, p ? m   : m+1)
        //   load  f_{m+1} <- slot(n,       p ? m+1 : m  )
        //   store f_m     -> slot(n,       p ? m+1 : m  )
        //   store f_{m+1} -> slot(n - c_m, p ? m   : m+1)
        // If probing ever disagrees with that, the generator is reading the wrong
        // thing and every shader it emits is wrong.
        for q in [19usize, 27] {
            let mut m = 1;
            while m < q {
                for odd in [false, true] {
                    let want_a = if odd { m } else { m + 1 };
                    let want_b = if odd { m + 1 } else { m };

                    let l_m = probe(q, m, odd, false);
                    assert_eq!(l_m.dir, want_a);
                    assert_eq!(l_m.neighbour_dir, Some(m + 1), "f_m loads from n - c_m");

                    let l_m1 = probe(q, m + 1, odd, false);
                    assert_eq!(l_m1.dir, want_b);
                    assert_eq!(l_m1.neighbour_dir, None, "f_m+1 loads from this cell");

                    let s_m = probe(q, m, odd, true);
                    assert_eq!(s_m.dir, want_b);
                    assert_eq!(s_m.neighbour_dir, None);

                    let s_m1 = probe(q, m + 1, odd, true);
                    assert_eq!(s_m1.dir, want_a);
                    assert_eq!(s_m1.neighbour_dir, Some(m + 1));
                }
                m += 2;
            }
        }
    }

    /// A load and the store that feeds it must share an address: that is the
    /// in-place property, and it is what makes the bounce-back flip work.
    #[test]
    fn load_and_store_of_a_pair_touch_the_same_two_addresses() {
        for q in [19usize, 27] {
            let mut m = 1;
            while m < q {
                for odd in [false, true] {
                    assert_eq!(probe(q, m, odd, false), probe(q, m + 1, odd, true));
                    assert_eq!(probe(q, m + 1, odd, false), probe(q, m, odd, true));
                }
                m += 2;
            }
        }
    }

    #[test]
    fn generated_wgsl_declares_every_buffer_and_touches_every_direction() {
        for set in [VelocitySet::D3Q19, VelocitySet::D3Q27] {
            for p in [DdfLayout::Fp32, DdfLayout::Fp16cPerCell, DdfLayout::Fp16cPacked] {
                let s = generated_prelude(set, p);
                let q = set.q();
                for i in 0..q {
                    assert!(
                        s.contains(&format!("@binding({i}) var<storage")),
                        "{set:?}/{p:?}: missing binding {i}"
                    );
                    assert!(s.contains(&format!("fn ddf_get_{i}(")));
                    assert!(s.contains(&format!("fn ddf_put_{i}(")));
                    assert!(
                        s.contains(&format!("(*g)[{i}]")),
                        "{set:?}/{p:?}: direction {i} never appears in transport"
                    );
                }
                assert_eq!(
                    s.matches("fn ").count(),
                    2 * q + 2,
                    "{set:?}/{p:?}: unexpected function count"
                );
                // Only the packed fallback may synchronise. If atomics ever
                // reappear on the per-cell path, the whole point of this layout
                // has been lost and the format is slower than FP32 again.
                if p == DdfLayout::Fp16cPacked {
                    assert!(s.contains("atomicAnd") && s.contains("atomicOr"));
                } else {
                    assert!(!s.contains("atomic"), "{set:?}/{p:?} must not synchronise");
                }
                // And each layout must address the element size it was allocated
                // for: a u16 array indexed by cell, or a u32 array of pairs.
                match p {
                    DdfLayout::Fp32 => assert!(s.contains("array<u32>")),
                    DdfLayout::Fp16cPerCell => {
                        assert!(s.contains("array<u16>"));
                        assert!(!s.contains("cell >> 1u"), "per-cell layout must not pack");
                        assert_eq!(enable_directives(p), "enable wgpu_int16;\n");
                    }
                    DdfLayout::Fp16cPacked => {
                        assert!(s.contains("array<atomic<u32>>"));
                        assert!(s.contains("cell >> 1u"));
                        assert_eq!(enable_directives(p), "");
                    }
                }
            }
        }
    }

    /// Every direction buffer must be *written* exactly as many times as it is
    /// read, or some slot is being dropped.
    #[test]
    fn transport_reads_and_writes_each_buffer_the_same_number_of_times() {
        for set in [VelocitySet::D3Q19, VelocitySet::D3Q27] {
            let s = transport(set);
            for i in 0..set.q() {
                let gets = s.matches(&format!("ddf_get_{i}(")).count();
                let puts = s.matches(&format!("ddf_put_{i}(")).count();
                assert_eq!(gets, puts, "{set:?}: buffer {i} read {gets} times, written {puts}");
            }
        }
    }
}
