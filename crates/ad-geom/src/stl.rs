//! STL loading: binary and ASCII, detected by content.
//!
//! # Why detection is by content and not by extension
//!
//! The `.stl` extension says nothing about the encoding, and neither does the
//! header. The ASCII grammar starts with the word `solid`, so a naive "starts
//! with `solid`" test looks right — but a large fraction of binary exporters
//! (SolidWorks among them) write a header that begins with the word `solid`
//! too, because the 80-byte header is free-form and they put a product string
//! in it. The reliable discriminator is arithmetic: a binary STL is exactly
//! `84 + 50 * n` bytes, where `n` is the little-endian `u32` at offset 80. That
//! is checked first, and only if it fails do we look at the text.
//!
//! # Why the file's normals are ignored
//!
//! Every facet carries a normal, and it is routinely wrong: zeros, unnormalised
//! vectors, or a direction that contradicts the vertex order. The winding is the
//! authority here and the normal is recomputed from it; the disagreement is
//! counted and reported so a genuinely inside-out file is visible rather than
//! silently producing an inverted distance field.

use crate::mesh::{MeshHealth, TriMesh};
use anyhow::{bail, Context as _, Result};
use glam::Vec3;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

/// Vertex welding quantum, millimetres.
///
/// 1e-4 mm is a hundred times finer than any printer can resolve and a hundred
/// times coarser than the `f32` spacing at duct-sized coordinates (~1e-5 mm at
/// 150 mm), so it merges the round-off differences an exporter introduces
/// without merging anything the designer meant to keep apart.
pub const WELD_QUANTUM_MM: f32 = 1.0e-4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StlFormat {
    Binary,
    Ascii,
}

/// The result of a load, including everything that was noticed on the way in.
#[derive(Debug, Clone)]
pub struct StlLoad {
    pub mesh: TriMesh,
    pub format: StlFormat,
    /// The 80-byte header of a binary file, trimmed. Empty for ASCII.
    pub header: String,
    /// Triangles in the file, before degenerate ones were dropped.
    pub raw_triangle_count: usize,
    /// Triangles dropped because two or more of their corners welded together.
    pub dropped_degenerate: usize,
    /// Vertex references before welding: three per raw triangle.
    pub raw_vertex_count: usize,
    /// The file was wound inside-out and every triangle has been reversed.
    ///
    /// A closed surface with negative signed volume describes the *complement*
    /// of the solid. Voxelising it directly turns the whole domain solid except
    /// for the part, which is a spectacular and confusing failure, so it is
    /// corrected here and reported rather than left for the user to discover.
    pub flipped: bool,
    health: MeshHealth,
}

impl StlLoad {
    /// The full mesh report. Computed once at load, after any winding fix.
    pub fn health(&self) -> &MeshHealth {
        &self.health
    }

    pub fn summary(&self) -> String {
        format!(
            "{:?} STL{}: {} triangles, {} vertex refs welded to {}{}{}",
            self.format,
            if self.header.is_empty() { String::new() } else { format!(" \"{}\"", self.header) },
            self.mesh.triangle_count(),
            self.raw_vertex_count,
            self.mesh.vertex_count(),
            if self.dropped_degenerate > 0 {
                format!(", {} degenerate triangles dropped", self.dropped_degenerate)
            } else {
                String::new()
            },
            if self.flipped { ", wound inside-out and corrected" } else { "" },
        )
    }
}

pub fn load_stl(path: impl AsRef<Path>) -> Result<StlLoad> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    parse_stl(&bytes).with_context(|| format!("parsing {}", path.display()))
}

/// Parse an STL from memory, detecting the encoding from the bytes.
pub fn parse_stl(bytes: &[u8]) -> Result<StlLoad> {
    match detect_format(bytes)? {
        StlFormat::Binary => parse_binary(bytes),
        StlFormat::Ascii => parse_ascii(bytes),
    }
}

/// Decide binary vs ASCII, with an error message that says *why* if neither
/// fits. Getting this wrong produces either a parse failure on line 1 or, worse,
/// 4 million triangles of garbage, so the failure path is worth some words.
fn detect_format(bytes: &[u8]) -> Result<StlFormat> {
    if bytes.len() >= 84 {
        let n = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as u64;
        let expected = 84 + 50 * n;
        if expected == bytes.len() as u64 {
            return Ok(StlFormat::Binary);
        }
        // Not a length match. If it also does not look like text, the file is
        // truncated or padded and the arithmetic is the most useful thing to
        // say about it.
        if !looks_like_ascii_stl(bytes) {
            bail!(
                "not a valid STL: the header claims {n} triangles, which needs 84 + 50*{n} = \
                 {expected} bytes, but the file is {} bytes ({} {}). The file is neither valid \
                 binary STL nor ASCII STL text.",
                bytes.len(),
                (expected as i64 - bytes.len() as i64).unsigned_abs(),
                if expected > bytes.len() as u64 { "short" } else { "extra" },
            );
        }
    }

    if looks_like_ascii_stl(bytes) {
        return Ok(StlFormat::Ascii);
    }
    bail!(
        "not an STL file: {} bytes, too short for a binary header and no `solid`/`facet` \
         keywords found",
        bytes.len()
    )
}

fn looks_like_ascii_stl(bytes: &[u8]) -> bool {
    // Only the head needs checking, and only for text: an ASCII STL must have a
    // `facet` keyword, and a binary file that happens to spell `solid` in its
    // header will not have one in the first few kilobytes (its triangle data is
    // raw floats).
    let head = &bytes[..bytes.len().min(4096)];
    let text = String::from_utf8_lossy(head);
    let starts_solid = text.trim_start().starts_with("solid");
    starts_solid && text.contains("facet") && text.contains("vertex")
}

fn parse_binary(bytes: &[u8]) -> Result<StlLoad> {
    let n = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    let header = String::from_utf8_lossy(&bytes[..80])
        .trim_matches(|c: char| c.is_whitespace() || c == '\0')
        .to_string();

    // One 50-byte record per triangle: 12 floats then a 2-byte attribute count.
    // Parsed in parallel; at 85k triangles this is the single most expensive
    // part of loading.
    let records = &bytes[84..84 + 50 * n];
    let parsed: Vec<([Vec3; 3], Vec3)> = records
        .par_chunks_exact(50)
        .map(|r| {
            let f = |o: usize| f32::from_le_bytes([r[o], r[o + 1], r[o + 2], r[o + 3]]);
            let v = |o: usize| Vec3::new(f(o), f(o + 4), f(o + 8));
            ([v(12), v(24), v(36)], v(0))
        })
        .collect();

    Ok(finish(parsed, StlFormat::Binary, header))
}

fn parse_ascii(bytes: &[u8]) -> Result<StlLoad> {
    let text = std::str::from_utf8(bytes).context("ASCII STL is not valid UTF-8")?;

    let mut parsed: Vec<([Vec3; 3], Vec3)> = Vec::new();
    let mut corners: Vec<Vec3> = Vec::with_capacity(3);
    let mut normal = Vec3::ZERO;
    let mut tokens = text.split_ascii_whitespace();

    // A keyword-driven scan rather than a line grammar: real files vary in
    // indentation, line endings and whether `outer loop` is on its own line,
    // and none of that changes the meaning.
    while let Some(tok) = tokens.next() {
        match tok {
            "normal" => {
                normal = read_vec3(&mut tokens).context("facet normal")?;
            }
            "vertex" => {
                corners.push(read_vec3(&mut tokens).context("vertex")?);
                if corners.len() == 3 {
                    parsed.push(([corners[0], corners[1], corners[2]], normal));
                    corners.clear();
                }
            }
            _ => {}
        }
    }
    if !corners.is_empty() {
        bail!(
            "ASCII STL ended mid-facet: {} loose vertices after {} complete triangles",
            corners.len(),
            parsed.len()
        );
    }
    if parsed.is_empty() {
        bail!("ASCII STL contained no triangles");
    }

    Ok(finish(parsed, StlFormat::Ascii, String::new()))
}

fn read_vec3<'a>(tokens: &mut impl Iterator<Item = &'a str>) -> Result<Vec3> {
    let mut v = [0.0f32; 3];
    for (i, slot) in v.iter_mut().enumerate() {
        let t = tokens.next().with_context(|| format!("expected 3 numbers, ran out at {i}"))?;
        *slot = t.parse::<f32>().with_context(|| format!("`{t}` is not a number"))?;
    }
    Ok(Vec3::from(v))
}

/// Weld the triangle soup into an indexed mesh and package the result.
fn finish(parsed: Vec<([Vec3; 3], Vec3)>, format: StlFormat, header: String) -> StlLoad {
    let raw_triangle_count = parsed.len();
    let raw_vertex_count = raw_triangle_count * 3;

    let mut welder = Welder::with_capacity(raw_vertex_count);
    let mut indices = Vec::with_capacity(raw_triangle_count);
    let mut file_normals = Vec::with_capacity(raw_triangle_count);
    let mut dropped_degenerate = 0;

    for (corners, normal) in parsed {
        let i = [
            welder.insert(corners[0]),
            welder.insert(corners[1]),
            welder.insert(corners[2]),
        ];
        // Two corners welding together means the triangle had an edge shorter
        // than the weld quantum: it carries no area and no normal, and leaving
        // it in would show up as a non-manifold edge later.
        if i[0] == i[1] || i[1] == i[2] || i[2] == i[0] {
            dropped_degenerate += 1;
            continue;
        }
        indices.push(i);
        file_normals.push(normal);
    }

    let mut mesh = TriMesh::new(welder.into_positions(), indices);
    mesh.file_normals = Some(file_normals);

    // A watertight surface with negative signed volume is inside-out; every
    // downstream sign would be inverted. Only meaningful once we know the mesh
    // is closed, so the topology is computed first and kept.
    let mut flipped = false;
    let mut health = mesh.health();
    if health.is_watertight_manifold() && health.signed_volume_mm3 < 0.0 {
        log::warn!(
            "STL is wound inside-out (signed volume {:.1} mm^3); reversing every triangle",
            health.signed_volume_mm3
        );
        mesh.flip_winding();
        health = mesh.health();
        flipped = true;
    }

    StlLoad {
        mesh,
        format,
        header,
        raw_triangle_count,
        dropped_degenerate,
        raw_vertex_count,
        flipped,
        health,
    }
}

/// Spatial-hash vertex welder.
///
/// Snapping to a grid of [`WELD_QUANTUM_MM`] and hashing the integer cell is not
/// quite enough on its own: two positions a nanometre apart can still straddle a
/// cell boundary and hash differently. So a miss falls back to probing the 26
/// neighbouring cells before giving up and allocating a new vertex. That costs
/// 27 hash lookups per *unique* vertex and one per repeat, which for a
/// three-times-shared vertex is a good trade for not silently tearing the mesh
/// along an arbitrary plane.
struct Welder {
    map: HashMap<[i32; 3], u32>,
    positions: Vec<Vec3>,
}

impl Welder {
    fn with_capacity(n: usize) -> Self {
        Self { map: HashMap::with_capacity(n / 2), positions: Vec::with_capacity(n / 2) }
    }

    #[inline]
    fn key(p: Vec3) -> [i32; 3] {
        [
            (p.x / WELD_QUANTUM_MM).round() as i32,
            (p.y / WELD_QUANTUM_MM).round() as i32,
            (p.z / WELD_QUANTUM_MM).round() as i32,
        ]
    }

    fn insert(&mut self, p: Vec3) -> u32 {
        let k = Self::key(p);
        if let Some(i) = self.map.get(&k) {
            return *i;
        }
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if (dx, dy, dz) == (0, 0, 0) {
                        continue;
                    }
                    let nk = [k[0] + dx, k[1] + dy, k[2] + dz];
                    if let Some(i) = self.map.get(&nk) {
                        // Confirm it is genuinely within tolerance before
                        // merging; the neighbouring cell may hold a vertex up
                        // to two quanta away.
                        if (self.positions[*i as usize] - p).abs().max_element()
                            <= WELD_QUANTUM_MM
                        {
                            let i = *i;
                            self.map.insert(k, i);
                            return i;
                        }
                    }
                }
            }
        }
        let i = self.positions.len() as u32;
        self.positions.push(p);
        self.map.insert(k, i);
        i
    }

    fn into_positions(self) -> Vec<Vec3> {
        self.positions
    }
}

/// Serialise a mesh as a binary STL. Used by the tests to round-trip, and handy
/// for dumping a subdivided or transformed mesh for inspection in a slicer.
pub fn write_binary_stl(mesh: &TriMesh, header: &str) -> Vec<u8> {
    let n = mesh.triangle_count();
    let mut out = Vec::with_capacity(84 + 50 * n);
    let mut head = [0u8; 80];
    let hb = header.as_bytes();
    head[..hb.len().min(80)].copy_from_slice(&hb[..hb.len().min(80)]);
    out.extend_from_slice(&head);
    out.extend_from_slice(&(n as u32).to_le_bytes());
    for t in 0..n {
        let normal = mesh.face_normal(t);
        for v in [normal, mesh.triangle(t)[0], mesh.triangle(t)[1], mesh.triangle(t)[2]] {
            out.extend_from_slice(&v.x.to_le_bytes());
            out.extend_from_slice(&v.y.to_le_bytes());
            out.extend_from_slice(&v.z.to_le_bytes());
        }
        out.extend_from_slice(&0u16.to_le_bytes());
    }
    out
}

/// Serialise a mesh as an ASCII STL.
pub fn write_ascii_stl(mesh: &TriMesh, name: &str) -> String {
    let mut s = format!("solid {name}\n");
    for t in 0..mesh.triangle_count() {
        let n = mesh.face_normal(t);
        let [a, b, c] = mesh.triangle(t);
        s.push_str(&format!("  facet normal {} {} {}\n    outer loop\n", n.x, n.y, n.z));
        for v in [a, b, c] {
            s.push_str(&format!("      vertex {} {} {}\n", v.x, v.y, v.z));
        }
        s.push_str("    endloop\n  endfacet\n");
    }
    s.push_str(&format!("endsolid {name}\n"));
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn binary_is_detected_even_when_the_header_says_solid() {
        // The exact trap: a binary file whose 80-byte header begins with the
        // word `solid`, which is what several CAD packages emit.
        let m = primitives::box_mesh(Vec3::ZERO, Vec3::splat(10.0));
        let bytes = write_binary_stl(&m, "solid created by a CAD package");
        assert_eq!(detect_format(&bytes).unwrap(), StlFormat::Binary);
        let load = parse_stl(&bytes).unwrap();
        assert_eq!(load.format, StlFormat::Binary);
        assert_eq!(load.mesh.triangle_count(), 12);
    }

    #[test]
    fn ascii_is_detected_by_content_not_by_the_first_word() {
        // The bytes at offset 80 of an ASCII file are some arbitrary text, which
        // the binary reader would happily interpret as a triangle count of
        // several hundred million. The length arithmetic rejects it, and the
        // `facet` / `vertex` keywords settle it.
        let m = primitives::box_mesh(Vec3::ZERO, Vec3::splat(10.0));
        let text = write_ascii_stl(&m, "cube");
        assert!(text.len() > 84);
        assert_eq!(detect_format(text.as_bytes()).unwrap(), StlFormat::Ascii);
        let load = parse_stl(text.as_bytes()).unwrap();
        assert_eq!(load.format, StlFormat::Ascii);
        assert_eq!(load.mesh.triangle_count(), 12);
    }

    #[test]
    fn an_inside_out_file_is_corrected_and_reported() {
        let mut m = primitives::box_mesh(Vec3::ZERO, Vec3::splat(10.0));
        m.flip_winding();
        assert!(m.signed_volume() < 0.0);

        let load = parse_stl(&write_binary_stl(&m, "inverted")).unwrap();
        assert!(load.flipped, "an inside-out file must be corrected");
        assert!(
            load.health().signed_volume_mm3 > 0.0,
            "volume is still {}",
            load.health().signed_volume_mm3
        );
        assert!(load.summary().contains("inside-out"));

        // ...and a correctly wound file must be left alone.
        let ok = parse_stl(&write_binary_stl(
            &primitives::box_mesh(Vec3::ZERO, Vec3::splat(10.0)),
            "fine",
        ))
        .unwrap();
        assert!(!ok.flipped);
    }

    #[test]
    fn a_truncated_binary_file_reports_the_byte_arithmetic() {
        let m = primitives::box_mesh(Vec3::ZERO, Vec3::splat(10.0));
        let mut bytes = write_binary_stl(&m, "cube");
        bytes.truncate(bytes.len() - 20);
        let err = parse_stl(&bytes).unwrap_err().to_string();
        assert!(err.contains("84 + 50*12"), "unhelpful error: {err}");
        assert!(err.contains("684"), "expected byte count in: {err}");
    }

    #[test]
    fn binary_and_ascii_round_trip_to_the_same_mesh() {
        let m = primitives::uv_sphere(Vec3::new(1.0, 2.0, 3.0), 5.0, 16, 8);
        let from_bin = parse_stl(&write_binary_stl(&m, "s")).unwrap();
        let from_txt = parse_stl(write_ascii_stl(&m, "s").as_bytes()).unwrap();

        for load in [&from_bin, &from_txt] {
            assert_eq!(load.mesh.triangle_count(), m.triangle_count());
            // Welding must recover exactly the original shared vertices.
            assert_eq!(load.mesh.vertex_count(), m.vertex_count());
            assert!(load.mesh.topology().is_watertight_manifold());
            assert!((load.mesh.signed_volume() - m.signed_volume()).abs() < 1e-2);
        }
    }

    #[test]
    fn welding_merges_vertices_that_differ_below_the_quantum() {
        // A triangle soup where the shared corners are perturbed by a fraction
        // of the weld quantum, which is what an exporter's round-off looks like.
        let m = primitives::box_mesh(Vec3::ZERO, Vec3::splat(10.0));
        let mut bytes = write_binary_stl(&m, "cube");
        for t in 0..12usize {
            for c in 0..3usize {
                let off = 84 + 50 * t + 12 + 12 * c;
                for a in 0..3usize {
                    let o = off + 4 * a;
                    let v = f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
                    let jitter = ((t * 3 + c + a) as f32 * 0.037).sin() * WELD_QUANTUM_MM * 0.4;
                    bytes[o..o + 4].copy_from_slice(&(v + jitter).to_le_bytes());
                }
            }
        }
        let load = parse_stl(&bytes).unwrap();
        assert_eq!(load.mesh.vertex_count(), 8, "jittered corners failed to weld");
        assert!(load.mesh.topology().is_watertight_manifold());
    }

    #[test]
    fn file_normals_are_reported_but_not_believed() {
        let m = primitives::box_mesh(Vec3::ZERO, Vec3::splat(10.0));
        let mut bytes = write_binary_stl(&m, "cube");
        // Invert every stored normal. The winding still says outward, so the
        // volume must stay positive and the disagreement must be counted.
        for t in 0..12usize {
            for a in 0..3usize {
                let o = 84 + 50 * t + 4 * a;
                let v = f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
                bytes[o..o + 4].copy_from_slice(&(-v).to_le_bytes());
            }
        }
        let load = parse_stl(&bytes).unwrap();
        let h = load.health();
        assert_eq!(h.normal_inversions, 12);
        assert_eq!(h.normal_disagreements, 12);
        assert!(h.signed_volume_mm3 > 0.0, "the winding must win, not the file normal");
        assert!(h.report().contains("disagree"));
    }

    #[test]
    fn degenerate_triangles_are_dropped_not_kept() {
        let mut m = primitives::box_mesh(Vec3::ZERO, Vec3::splat(10.0));
        // A sliver whose two corners are closer together than the weld quantum.
        let a = m.positions.len() as u32;
        m.positions.push(Vec3::new(20.0, 0.0, 0.0));
        m.positions.push(Vec3::new(20.0, 0.0, WELD_QUANTUM_MM * 0.1));
        m.positions.push(Vec3::new(21.0, 0.0, 0.0));
        m.indices.push([a, a + 1, a + 2]);
        let load = parse_stl(&write_binary_stl(&m, "x")).unwrap();
        assert_eq!(load.raw_triangle_count, 13);
        assert_eq!(load.dropped_degenerate, 1);
        assert_eq!(load.mesh.triangle_count(), 12);
    }

    #[test]
    fn garbage_is_rejected_with_a_reason() {
        let err = parse_stl(b"this is not an stl").unwrap_err().to_string();
        assert!(err.contains("not an STL file"), "{err}");
    }
}
