//! Shared by the reference tests.
use std::path::Path;

use ndarray::{ArrayD, IxDyn};

/// Minimal .npy reader: little-endian u8 / i64 / f32, C order.
pub fn npy(path: &Path) -> ArrayD<f64> {
    let bytes = std::fs::read(path).unwrap();
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
    assert!(!header.contains("'fortran_order': True"), "Fortran-order npy: {header}");
    let data = &bytes[10 + header_len..];
    let shape_str = &header[header.find("'shape': (").unwrap() + 10..];
    let shape: Vec<usize> =
        shape_str[..shape_str.find(')').unwrap()].split(',').filter_map(|s| s.trim().parse().ok()).collect();
    let values: Vec<f64> = if header.contains("'|u1'") {
        data.iter().map(|&b| b as f64).collect()
    } else if header.contains("'<i8'") {
        data.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap()) as f64).collect()
    } else if header.contains("'<f4'") {
        data.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap()) as f64).collect()
    } else {
        panic!("unsupported npy header {header}");
    };
    ArrayD::from_shape_vec(IxDyn(&shape), values).unwrap()
}

