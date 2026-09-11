//! Domain padding, the per-cell link mask, and the boundary-condition taxonomy.
//!
//! # Why the domain is padded
//!
//! Esoteric Pull parks half of every cell's distributions in the slot belonging
//! to the neighbour at `n - c_m` (see [`ad_gpu::lattice::EsotericPull`]). For a
//! cell on the low face of the domain that neighbour is outside the grid, and
//! there is nowhere to put the population — including the one that bounce-back
//! needs to hand straight back.
//!
//! Wrapping the index instead is *not* an option even for a closed box: the cell
//! on the opposite face already owns both slots at that address on every step,
//! and the two cells would write each other's data. That is a silent
//! wrong-physics bug, not a crash.
//!
//! So every non-periodic axis gets one cell of solid halo at each end. One is
//! exactly enough: the most negative offset any direction produces is `(-1,-1,-1)`
//! (D3Q27), and halo cells never execute, so they never need neighbours of their
//! own. The cost is `(1 + 2/N)^3 - 1`, about 2.5% of cells at the interactive
//! tier. Periodic axes get no halo and use genuine modular indexing, where the
//! ownership argument above works out correctly.
//!
//! # Why there is a link mask
//!
//! The bounce-back rule is "load direction `i` with the *opposite* step parity
//! when the upstream neighbour `n - c_i` is solid" (derived in
//! [`crate::reference`]). Evaluating that literally costs `q - 1` scattered
//! neighbour-flag reads per cell per step. The geometry is static, so instead we
//! precompute one `u32` per cell whose bit `i` is set exactly when direction `i`
//! must flip. That turns 18 scattered byte reads into one coalesced word read,
//! for 4 bytes per cell of extra traffic — about 5% on top of the D3Q19/FP16C
//! per-step figure of 77 B/cell.
//!
//! The mask is also the natural place to add Wave-3's interpolated bounce-back:
//! a link that needs the per-link `q` from [`ad_gpu::BoundaryLink`] is already
//! identified here, so the shader gains a second lookup on masked links only and
//! nothing else moves.

use ad_gpu::lattice::LatticeDef;
use ad_gpu::types::{flags, BoundaryLink, VelocitySet};
use glam::{IVec3, UVec3};

/// Interior grid plus the solid halo Esoteric Pull needs on open axes.
#[derive(Debug, Clone)]
pub struct PaddedDomain {
    /// Dimensions the caller asked for.
    pub interior: UVec3,
    /// Dimensions actually allocated, `interior + 2` on each non-periodic axis.
    pub padded: UVec3,
    /// Interior cell `(i,j,k)` lives at padded `(i,j,k) + offset`.
    pub offset: UVec3,
    pub periodic: [bool; 3],
    /// One byte per padded cell. The halo shell is [`flags::SOLID`].
    pub flags: Vec<u8>,
    /// One `u32` per padded cell; bit `i` means "flip the load parity for
    /// direction `i`", i.e. this link ends on a solid cell.
    pub link_mask: Vec<u32>,
}

impl PaddedDomain {
    /// Build the padded domain from the interior classification produced by the
    /// geometry crate.
    ///
    /// `mask` is one [`flags`] byte per interior cell, X-fastest, and must be
    /// `interior.x * interior.y * interior.z` long.
    pub fn new(interior: UVec3, periodic: [bool; 3], mask: &[u8], set: VelocitySet) -> Self {
        let n_interior = (interior.x as usize) * (interior.y as usize) * (interior.z as usize);
        assert_eq!(
            mask.len(),
            n_interior,
            "mask has {} entries but the grid has {n_interior} cells",
            mask.len()
        );

        let pad = UVec3::new(
            if periodic[0] { 0 } else { 1 },
            if periodic[1] { 0 } else { 1 },
            if periodic[2] { 0 } else { 1 },
        );
        let padded = interior + pad * 2;
        let n_padded = (padded.x as usize) * (padded.y as usize) * (padded.z as usize);

        // The halo starts solid; interior cells overwrite their own entry.
        let mut cell_flags = vec![flags::SOLID; n_padded];
        for z in 0..interior.z {
            for y in 0..interior.y {
                for x in 0..interior.x {
                    let src = ((z * interior.y + y) * interior.x + x) as usize;
                    let p = UVec3::new(x, y, z) + pad;
                    let dst = ((p.z * padded.y + p.y) * padded.x + p.x) as usize;
                    cell_flags[dst] = mask[src];
                }
            }
        }

        let mut me = Self {
            interior,
            padded,
            offset: pad,
            periodic,
            flags: cell_flags,
            link_mask: Vec::new(),
        };
        me.rebuild_link_mask(set.def());
        me
    }

    /// A domain of pure fluid, for the analytic validation cases.
    pub fn uniform_fluid(interior: UVec3, periodic: [bool; 3], set: VelocitySet) -> Self {
        let n = (interior.x as usize) * (interior.y as usize) * (interior.z as usize);
        Self::new(interior, periodic, &vec![flags::FLUID; n], set)
    }

    pub fn padded_cell_count(&self) -> u64 {
        self.padded.x as u64 * self.padded.y as u64 * self.padded.z as u64
    }

    pub fn interior_cell_count(&self) -> u64 {
        self.interior.x as u64 * self.interior.y as u64 * self.interior.z as u64
    }

    /// Linear index within the padded grid. X-fastest, matching
    /// [`ad_gpu::Grid::linear`].
    #[inline]
    pub fn linear(&self, c: UVec3) -> u32 {
        (c.z * self.padded.y + c.y) * self.padded.x + c.x
    }

    /// Padded linear index of an interior cell.
    #[inline]
    pub fn interior_linear(&self, c: UVec3) -> u32 {
        self.linear(c + self.offset)
    }

    /// Neighbour of padded cell `c` in direction `d`, wrapping on every axis.
    ///
    /// Wrapping unconditionally is safe *and* free: on a padded axis the wrap can
    /// only trigger for a halo cell, and halo cells never execute. Doing it this
    /// way means no invocation can ever compute an out-of-range index, which
    /// removes a whole class of bounds bug from the shader.
    #[inline]
    pub fn neighbour(&self, c: UVec3, d: IVec3) -> UVec3 {
        let n = self.padded.as_ivec3();
        let p = c.as_ivec3() + d;
        UVec3::new(
            (((p.x % n.x) + n.x) % n.x) as u32,
            (((p.y % n.y) + n.y) % n.y) as u32,
            (((p.z % n.z) + n.z) % n.z) as u32,
        )
    }

    #[inline]
    pub fn coords(&self, cell: u32) -> UVec3 {
        let x = cell % self.padded.x;
        let y = (cell / self.padded.x) % self.padded.y;
        let z = cell / (self.padded.x * self.padded.y);
        UVec3::new(x, y, z)
    }

    /// Recompute [`Self::link_mask`] from [`Self::flags`]. Call after changing
    /// the geometry.
    pub fn rebuild_link_mask(&mut self, def: &LatticeDef) {
        let n = self.padded_cell_count() as usize;
        let mut mask = vec![0u32; n];
        for cell in 0..n {
            let c = self.coords(cell as u32);
            if !flags::is_fluid(self.flags[cell]) {
                continue;
            }
            let mut m = 0u32;
            for i in 1..def.q {
                // The population arriving along direction i comes from n - c_i.
                let up = self.neighbour(c, -def.directions[i]);
                if !flags::is_fluid(self.flags[self.linear(up) as usize]) {
                    m |= 1 << i;
                }
            }
            mask[cell] = m;
        }
        self.link_mask = mask;
    }

    /// Distance in cells from the nearest closed domain face, saturating at
    /// `limit`. Periodic axes are excluded, since there is no face there.
    ///
    /// This is what grades the sponge. Grading by distance-to-face rather than
    /// by a per-cell field keeps the sponge free of extra memory traffic, and it
    /// is the right shape for the case that matters: an absorbing layer wrapped
    /// around the open box.
    pub fn face_distance(&self, c: UVec3, limit: u32) -> u32 {
        let mut d = limit;
        for a in 0..3 {
            if self.periodic[a] {
                continue;
            }
            let n = self.padded[a];
            let v = c[a];
            d = d.min(v).min(n.saturating_sub(1).saturating_sub(v));
        }
        d.min(limit)
    }

    /// Sponge relaxation weight at a padded cell, in `[0, strength]`.
    ///
    /// Quadratic grading from zero at the inner edge of the layer to the full
    /// strength at the face. A quadratic ramp is the standard choice: a step
    /// change in damping reflects almost as much as the wall it replaced.
    ///
    /// **Gated on [`flags::SPONGE`].** The distance is only the profile; the
    /// flag is what says where the absorbing layer is. Applying the sponge by
    /// distance alone damps wherever the domain happens to be narrow, which in a
    /// duct that passes within `cells` of a face means damping the flow being
    /// measured — measured at 12% of the streamwise flux, and divergent when
    /// composed with the outlet's anti-bounce-back.
    pub fn sponge_sigma(&self, c: UVec3, cell_flags: u8, cells: u32, strength: f32) -> f32 {
        if cell_flags & flags::SPONGE == 0 || cells == 0 || strength <= 0.0 {
            return 0.0;
        }
        let d = self.face_distance(c, cells);
        if d >= cells {
            return 0.0;
        }
        let t = (cells - d) as f32 / cells as f32;
        strength * t * t
    }
}

/// Which boundary treatment a cell gets, decoded from the [`flags`] bitfield.
///
/// The ordering here is the dispatch priority in the shader: `SOLID` wins over
/// everything (it does not execute at all), then `INLET`, then `OUTLET`, then
/// `EQUILIBRIUM`. `SPONGE` is orthogonal and composes with any of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellKind {
    Solid,
    Fluid,
    /// Equilibrium at the prescribed velocity and the locally computed density.
    /// Cannot blow up, which is why it is the v1 inlet.
    Inlet,
    /// Convective outflow, with the pressure pinned by anti-bounce-back.
    Outlet,
    /// Open box side: relaxed to `(rho_ref, 0)` so the exit jet can entrain
    /// surrounding air instead of being confined.
    Equilibrium,
}

impl CellKind {
    #[inline]
    pub fn of(f: u8) -> Self {
        if f & flags::SOLID != 0 {
            CellKind::Solid
        } else if f & flags::INLET != 0 {
            CellKind::Inlet
        } else if f & flags::OUTLET != 0 {
            CellKind::Outlet
        } else if f & flags::EQUILIBRIUM != 0 {
            CellKind::Equilibrium
        } else {
            CellKind::Fluid
        }
    }
}

/// Per-link wall distances, indexed for the shader.
///
/// Halfway bounce-back assumes `q = 1/2` on every link. Single-node interpolated
/// bounce-back (Marson et al., arXiv:2009.04604) instead uses the real `q` and
/// recovers second-order accuracy on a curved wall without needing the second
/// fluid node behind it — which matters here because the median passage is only
/// ~6.3 mm across and there frequently *is* no second node.
///
/// Wave 1 does not consume this. It is built and uploaded anyway so that turning
/// it on in Wave 3 is a shader change and a bind-group entry, not a rewrite: the
/// lookup key is `(cell, direction)` and the shader already knows both at the
/// point where it would need `q`.
#[derive(Debug, Clone, Default)]
pub struct LinkTable {
    /// `q` quantised to a byte, one entry per (cell, direction) that has one.
    /// Sorted by cell then direction so a binary search or a per-cell offset
    /// table can find a link cheaply.
    pub links: Vec<BoundaryLink>,
}

impl LinkTable {
    /// Re-index links from interior coordinates into the padded grid.
    pub fn remap(links: &[BoundaryLink], interior: UVec3, domain: &PaddedDomain) -> Self {
        let mut out: Vec<BoundaryLink> = links
            .iter()
            .map(|l| {
                let x = l.cell % interior.x;
                let y = (l.cell / interior.x) % interior.y;
                let z = l.cell / (interior.x * interior.y);
                BoundaryLink { cell: domain.interior_linear(UVec3::new(x, y, z)), ..*l }
            })
            .collect();
        out.sort_by_key(|l| (l.cell, l.direction));
        Self { links: out }
    }

    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }

    /// `q` for one link, or `0.5` (the halfway assumption) when absent.
    pub fn q(&self, cell: u32, direction: u8) -> f32 {
        match self.links.binary_search_by_key(&(cell, direction), |l| (l.cell, l.direction)) {
            Ok(i) => self.links[i].q(),
            Err(_) => 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ad_gpu::lattice::D3Q19_DIRS;

    #[test]
    fn open_axes_are_padded_and_periodic_axes_are_not() {
        let d = PaddedDomain::uniform_fluid(
            UVec3::new(4, 5, 6),
            [true, false, false],
            VelocitySet::D3Q19,
        );
        assert_eq!(d.padded, UVec3::new(4, 7, 8));
        assert_eq!(d.offset, UVec3::new(0, 1, 1));
        assert_eq!(d.padded_cell_count(), 4 * 7 * 8);
    }

    #[test]
    fn the_halo_is_solid_and_the_interior_is_not() {
        let d = PaddedDomain::uniform_fluid(UVec3::new(3, 3, 3), [false; 3], VelocitySet::D3Q19);
        assert_eq!(d.padded, UVec3::splat(5));
        for z in 0..5u32 {
            for y in 0..5u32 {
                for x in 0..5u32 {
                    let on_shell = x == 0 || y == 0 || z == 0 || x == 4 || y == 4 || z == 4;
                    let f = d.flags[d.linear(UVec3::new(x, y, z)) as usize];
                    assert_eq!(
                        !flags::is_fluid(f),
                        on_shell,
                        "cell ({x},{y},{z}) shell={on_shell} flags={f:#04x}"
                    );
                }
            }
        }
    }

    /// One cell of halo is exactly enough: every slot a fluid cell writes must
    /// land inside the allocation. This is the property that would fail silently
    /// (as an out-of-bounds clamp, or as aliasing onto another cell's link) if
    /// the padding were wrong.
    #[test]
    fn every_slot_a_fluid_cell_touches_is_inside_the_padded_grid() {
        for periodic in [[false; 3], [true, false, false], [true, true, true]] {
            for set in [VelocitySet::D3Q19, VelocitySet::D3Q27] {
                let d = PaddedDomain::uniform_fluid(UVec3::new(5, 4, 3), periodic, set);
                let def = set.def();
                for cell in 0..d.padded_cell_count() as u32 {
                    if !flags::is_fluid(d.flags[cell as usize]) {
                        continue;
                    }
                    let c = d.coords(cell);
                    for i in 1..def.q {
                        // n - c_m for the pair's odd member is the only address
                        // that leaves the cell; check it directly.
                        let m = ad_gpu::lattice::pair_first(i);
                        let target = c.as_ivec3() - def.directions[m];
                        for a in 0..3 {
                            if !periodic[a] {
                                assert!(
                                    target[a] >= 0 && target[a] < d.padded[a] as i32,
                                    "{set:?} {periodic:?}: fluid cell {c:?} direction {i} \
                                     addresses {target:?}, outside {:?}",
                                    d.padded
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn link_mask_marks_exactly_the_links_that_end_on_solid() {
        let mut d = PaddedDomain::uniform_fluid(UVec3::new(6, 6, 6), [true; 3], VelocitySet::D3Q19);
        // Plant one solid cell in the middle of a fully periodic box.
        let solid = d.linear(UVec3::new(3, 3, 3));
        d.flags[solid as usize] = flags::SOLID;
        d.rebuild_link_mask(VelocitySet::D3Q19.def());

        for cell in 0..d.padded_cell_count() as u32 {
            if cell == solid {
                assert_eq!(d.link_mask[cell as usize], 0, "solid cells carry no mask");
                continue;
            }
            let c = d.coords(cell);
            for i in 1..19usize {
                let up = d.neighbour(c, -D3Q19_DIRS[i]);
                let want = d.linear(up) == solid;
                let got = d.link_mask[cell as usize] & (1 << i) != 0;
                assert_eq!(got, want, "cell {c:?} direction {i}");
            }
        }
        // The solid cell has 18 fluid neighbours, each of which sees exactly one
        // blocked link.
        let blocked: u32 = d.link_mask.iter().map(|m| m.count_ones()).sum();
        assert_eq!(blocked, 18, "one blocked link per neighbour of the solid cell");
    }

    #[test]
    fn sponge_grades_quadratically_to_zero_at_the_inner_edge() {
        let d = PaddedDomain::uniform_fluid(UVec3::new(40, 40, 40), [false; 3], VelocitySet::D3Q19);
        let sp = flags::SPONGE;
        let mid = UVec3::splat(20);
        assert_eq!(d.sponge_sigma(mid, sp, 10, 0.5), 0.0, "the interior must be untouched");
        // At the face itself (padded index 0) the layer is at full strength.
        let face = UVec3::new(0, 20, 20);
        assert!((d.sponge_sigma(face, sp, 10, 0.5) - 0.5).abs() < 1e-6);
        // ...but only where the flag says so. A cell in the same place without
        // the flag must see nothing at all.
        assert_eq!(d.sponge_sigma(face, flags::FLUID, 10, 0.5), 0.0);
        // Monotone and continuous going in.
        let mut prev = f32::INFINITY;
        for x in 0..12u32 {
            let s = d.sponge_sigma(UVec3::new(x, 20, 20), sp, 10, 0.5);
            assert!(s <= prev + 1e-9, "sponge is not monotone at x={x}");
            prev = s;
        }
        assert_eq!(d.sponge_sigma(UVec3::new(10, 20, 20), sp, 10, 0.5), 0.0);
    }

    #[test]
    fn cell_kind_dispatch_priority_is_solid_inlet_outlet_equilibrium() {
        assert_eq!(CellKind::of(flags::FLUID), CellKind::Fluid);
        assert_eq!(CellKind::of(flags::SOLID | flags::INLET), CellKind::Solid);
        assert_eq!(CellKind::of(flags::INLET | flags::OUTLET), CellKind::Inlet);
        assert_eq!(CellKind::of(flags::OUTLET | flags::EQUILIBRIUM), CellKind::Outlet);
        assert_eq!(CellKind::of(flags::EQUILIBRIUM | flags::SPONGE), CellKind::Equilibrium);
        // SPONGE alone is not a boundary; it only modifies one.
        assert_eq!(CellKind::of(flags::SPONGE), CellKind::Fluid);
    }

    #[test]
    fn link_table_remaps_into_the_padded_grid_and_defaults_to_halfway() {
        let interior = UVec3::new(4, 4, 4);
        let d = PaddedDomain::uniform_fluid(interior, [false; 3], VelocitySet::D3Q19);
        let src = vec![BoundaryLink {
            cell: (2 * 4 + 1) * 4 + 3, // interior (3,1,2)
            direction: 5,
            q_quantised: BoundaryLink::quantise(0.25),
            _pad: [0; 2],
        }];
        let t = LinkTable::remap(&src, interior, &d);
        let padded_cell = d.interior_linear(UVec3::new(3, 1, 2));
        assert_eq!(t.links[0].cell, padded_cell);
        assert!((t.q(padded_cell, 5) - 0.25).abs() < 1.0 / 255.0);
        assert_eq!(t.q(padded_cell, 6), 0.5, "absent links fall back to halfway");
    }
}
