//! A minimal PNG writer for the screenshot action.
//!
//! Stored (uncompressed) deflate blocks inside a zlib wrapper: thirty lines and
//! no dependency, versus pulling an image crate into the app for one button.
//! The files are large, and that is the right trade for a screenshot that is
//! written once and looked at immediately.

use std::io;
use std::path::Path;

pub fn write_rgba(path: &Path, w: u32, h: u32, rgba: &[u8]) -> io::Result<()> {
    assert_eq!(rgba.len(), (w as usize) * (h as usize) * 4);
    let mut raw = Vec::with_capacity((w * h * 4 + h) as usize);
    for y in 0..h as usize {
        raw.push(0u8); // filter type: none
        let start = y * w as usize * 4;
        raw.extend_from_slice(&rgba[start..start + w as usize * 4]);
    }

    let mut out = Vec::new();
    out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);

    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit RGBA, no interlace
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &zlib_stored(&raw));
    chunk(&mut out, b"IEND", &[]);

    std::fs::write(path, out)
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

fn zlib_stored(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    let mut i = 0;
    loop {
        let n = (data.len() - i).min(65535);
        let last = i + n >= data.len();
        out.push(u8::from(last));
        out.extend_from_slice(&(n as u16).to_le_bytes());
        out.extend_from_slice(&(!(n as u16)).to_le_bytes());
        out.extend_from_slice(&data[i..i + n]);
        i += n;
        if last {
            break;
        }
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for byte in data {
        a = (a + *byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for byte in data {
        crc ^= *byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_header_and_the_checksums_are_well_formed() {
        // Not a decode test — just enough that a corrupt file cannot be written
        // silently, which is the failure mode of a hand-rolled encoder.
        let dir = std::env::temp_dir().join("aeroduct-png-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.png");
        let pixels = vec![0x40u8; 4 * 4 * 4];
        write_rgba(&path, 4, 4, &pixels).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        // Walk the chunks and verify every CRC.
        let mut i = 8usize;
        let mut kinds = Vec::new();
        while i + 8 <= bytes.len() {
            let len = u32::from_be_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
            let kind = &bytes[i + 4..i + 8];
            let data = &bytes[i + 8..i + 8 + len];
            let want = u32::from_be_bytes(
                bytes[i + 8 + len..i + 12 + len].try_into().unwrap(),
            );
            let mut input = kind.to_vec();
            input.extend_from_slice(data);
            assert_eq!(crc32(&input), want, "bad CRC on {:?}", std::str::from_utf8(kind));
            kinds.push(String::from_utf8_lossy(kind).to_string());
            i += 12 + len;
        }
        assert_eq!(kinds, vec!["IHDR", "IDAT", "IEND"]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn adler_matches_the_reference_value() {
        // "Wikipedia" -> 0x11E60398, the canonical worked example.
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }
}
