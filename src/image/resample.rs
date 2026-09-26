//! Image resizing that reproduces Pillow's `Image.resize` bit for bit on
//! 8-bit images (libImaging/Resample.c): same filters and support scaling,
//! 22-bit fixed-point coefficients, horizontal pass then vertical pass.
//! Hugging Face's image processors resize through Pillow, so matching it
//! matters for matching reference model inputs.
use std::f64::consts::PI;

/// Resampling filter; `from_pil` maps Pillow's (and HF configs') integer codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Filter {
    Nearest,
    Box,
    Bilinear,
    Hamming,
    Bicubic,
    Lanczos,
}

impl Filter {
    /// Pillow codes: NEAREST 0, LANCZOS 1, BILINEAR 2, BICUBIC 3, BOX 4, HAMMING 5.
    pub fn from_pil(code: i64) -> Option<Filter> {
        Some(match code {
            0 => Filter::Nearest,
            1 => Filter::Lanczos,
            2 => Filter::Bilinear,
            3 => Filter::Bicubic,
            4 => Filter::Box,
            5 => Filter::Hamming,
            _ => return None,
        })
    }

    pub fn parse(name: &str) -> Option<Filter> {
        Some(match name.to_ascii_lowercase().as_str() {
            "nearest" => Filter::Nearest,
            "box" => Filter::Box,
            "bilinear" | "linear" | "triangle" => Filter::Bilinear,
            "hamming" => Filter::Hamming,
            "bicubic" | "cubic" => Filter::Bicubic,
            "lanczos" => Filter::Lanczos,
            _ => return None,
        })
    }

    fn support(self) -> f64 {
        match self {
            Filter::Nearest | Filter::Box => 0.5,
            Filter::Bilinear | Filter::Hamming => 1.0,
            Filter::Bicubic => 2.0,
            Filter::Lanczos => 3.0,
        }
    }

    fn weight(self, x: f64) -> f64 {
        fn sinc(x: f64) -> f64 {
            if x == 0.0 { 1.0 } else { (x * PI).sin() / (x * PI) }
        }
        match self {
            Filter::Nearest | Filter::Box => {
                if x > -0.5 && x <= 0.5 { 1.0 } else { 0.0 }
            }
            Filter::Bilinear => {
                let x = x.abs();
                if x < 1.0 { 1.0 - x } else { 0.0 }
            }
            Filter::Hamming => {
                let x = x.abs();
                if x == 0.0 {
                    1.0
                } else if x >= 1.0 {
                    0.0
                } else {
                    let x = x * PI;
                    // Pillow uses float constants here.
                    x.sin() / x * (0.54f32 as f64 + 0.46f32 as f64 * x.cos())
                }
            }
            Filter::Bicubic => {
                const A: f64 = -0.5;
                let x = x.abs();
                if x < 1.0 {
                    ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
                } else if x < 2.0 {
                    (((x - 5.0) * x + 8.0) * x - 4.0) * A
                } else {
                    0.0
                }
            }
            Filter::Lanczos => {
                if (-3.0..3.0).contains(&x) { sinc(x) * sinc(x / 3.0) } else { 0.0 }
            }
        }
    }
}

const PRECISION_BITS: u32 = 32 - 8 - 2;

/// Per output pixel: first input index, tap count and fixed-point weights.
struct Coeffs {
    bounds: Vec<(usize, usize)>,
    ksize: usize,
    weights: Vec<i32>,
}

fn precompute(in_size: usize, out_size: usize, filter: Filter) -> Coeffs {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = filter.support() * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(out_size);
    let mut weights = vec![0i32; out_size * ksize];
    let mut k = vec![0f64; ksize];
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        // C casts truncate toward zero.
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize - xmin;
        let mut total = 0.0;
        for (x, w) in k.iter_mut().enumerate().take(xmax) {
            *w = filter.weight((x as f64 + xmin as f64 - center + 0.5) * ss);
            total += *w;
        }
        for x in 0..ksize {
            let w = if x < xmax && total != 0.0 { k[x] / total } else if x < xmax { k[x] } else { 0.0 };
            let fixed = w * (1u32 << PRECISION_BITS) as f64;
            weights[xx * ksize + x] = if w < 0.0 { (fixed - 0.5) as i32 } else { (fixed + 0.5) as i32 };
        }
        bounds.push((xmin, xmax));
    }
    Coeffs { bounds, ksize, weights }
}

fn clip8(v: i64) -> u8 {
    (v >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Resizes an interleaved 8-bit image (`channels` bytes per pixel) like Pillow.
pub fn resize_u8(src: &[u8], width: usize, height: usize, channels: usize, out_w: usize, out_h: usize, filter: Filter) -> Vec<u8> {
    assert_eq!(src.len(), width * height * channels);
    if (out_w, out_h) == (width, height) {
        return src.to_vec();
    }
    if filter == Filter::Nearest {
        return resize_nearest(src, width, height, channels, out_w, out_h);
    }
    let horiz = precompute(width, out_w, filter);
    let vert = precompute(height, out_h, filter);
    let round = 1i64 << (PRECISION_BITS - 1);

    // Horizontal pass over the rows the vertical pass reads.
    let (mut cur, mut cur_w) = (src.to_vec(), width);
    let mut vert_bounds = vert.bounds.clone();
    if out_w != width {
        let first = vert.bounds[0].0;
        let last = vert.bounds[out_h - 1].0 + vert.bounds[out_h - 1].1;
        for b in &mut vert_bounds {
            b.0 -= first;
        }
        let rows = last - first;
        let mut tmp = vec![0u8; out_w * rows * channels];
        for y in 0..rows {
            let row = &src[(y + first) * width * channels..(y + first + 1) * width * channels];
            for x in 0..out_w {
                let (xmin, n) = horiz.bounds[x];
                let k = &horiz.weights[x * horiz.ksize..x * horiz.ksize + n];
                for c in 0..channels {
                    let mut ss = round;
                    for (i, &w) in k.iter().enumerate() {
                        ss += row[(xmin + i) * channels + c] as i64 * w as i64;
                    }
                    tmp[(y * out_w + x) * channels + c] = clip8(ss);
                }
            }
        }
        (cur, cur_w) = (tmp, out_w);
    }
    if out_h == height {
        return cur;
    }
    let mut out = vec![0u8; out_w * out_h * channels];
    for y in 0..out_h {
        let (ymin, n) = vert_bounds[y];
        let k = &vert.weights[y * vert.ksize..y * vert.ksize + n];
        for x in 0..cur_w {
            for c in 0..channels {
                let mut ss = round;
                for (i, &w) in k.iter().enumerate() {
                    ss += cur[((ymin + i) * cur_w + x) * channels + c] as i64 * w as i64;
                }
                out[(y * out_w + x) * channels + c] = clip8(ss);
            }
        }
    }
    out
}

/// Pillow's nearest-neighbour resize (ImagingScaleAffine): source index
/// floor((x + 0.5) * scale), accumulated in doubles like Pillow does.
fn resize_nearest(src: &[u8], width: usize, height: usize, channels: usize, out_w: usize, out_h: usize) -> Vec<u8> {
    let index = |in_size: usize, out_size: usize| -> Vec<usize> {
        let step = in_size as f64 / out_size as f64;
        let mut pos = step * 0.5;
        (0..out_size)
            .map(|_| {
                let i = (pos as usize).min(in_size - 1);
                pos += step;
                i
            })
            .collect()
    };
    let (xs, ys) = (index(width, out_w), index(height, out_h));
    let mut out = Vec::with_capacity(out_w * out_h * channels);
    for &y in &ys {
        for &x in &xs {
            out.extend_from_slice(&src[(y * width + x) * channels..(y * width + x + 1) * channels]);
        }
    }
    out
}
