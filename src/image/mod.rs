//! Images to tensors.
//!
//! The pipeline follows Hugging Face's (slow, Pillow-based) image processors:
//! convert to RGB, resize in 8 bits with Pillow's resampler, center crop,
//! rescale (in f64, stored as f32), normalize (in f32), then order channels
//! and lay them out.
pub mod resample;

use ::image::{DynamicImage, RgbImage};
use ndarray::{Array3, ArrayD};
use serde_json::Value;

use crate::error::{Error, Result};
pub use resample::Filter;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelOrder {
    Rgb,
    Bgr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// `[channels, height, width]` (PyTorch).
    Chw,
    /// `[height, width, channels]`.
    Hwc,
}

/// Output size of a resize, static or derived from the input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Size {
    /// Exactly this size.
    Exact { width: u32, height: u32 },
    /// Shorter edge becomes `edge`, keeping the aspect ratio (HF
    /// `shortest_edge`); the longer edge is capped at `max_longest` if given.
    ShortestEdge { edge: u32, max_longest: Option<u32> },
    /// Longer edge becomes `edge`, keeping the aspect ratio.
    LongestEdge(u32),
    /// Each side rounded to a multiple of `multiple`, scaled to keep the pixel
    /// count within bounds (Qwen2-VL's `smart_resize`); for patch-based models.
    Multiple { multiple: u32, min_pixels: u64, max_pixels: u64 },
}

/// Python's round(): half to even.
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 { 2.0 * (x / 2.0).round() } else { r }
}

impl Size {
    /// (width, height) for an input of `width` x `height`.
    pub fn output(&self, width: u32, height: u32) -> (u32, u32) {
        match *self {
            Size::Exact { width, height } => (width, height),
            Size::ShortestEdge { edge, max_longest } => {
                // transformers' get_resize_output_image_size(default_to_square=False).
                let (short, long) = if width <= height { (width, height) } else { (height, width) };
                let (mut new_short, mut new_long) = (edge, (edge as f64 * long as f64 / short as f64) as u32);
                if let Some(max) = max_longest {
                    if new_long > max {
                        new_short = (max as f64 * new_short as f64 / new_long as f64) as u32;
                        new_long = max;
                    }
                }
                if width <= height { (new_short, new_long) } else { (new_long, new_short) }
            }
            Size::LongestEdge(edge) => {
                let scale = edge as f64 / width.max(height) as f64;
                ((width as f64 * scale + 0.5) as u32, (height as f64 * scale + 0.5) as u32)
            }
            Size::Multiple { multiple, min_pixels, max_pixels } => {
                let (h, w, f) = (height as f64, width as f64, multiple as f64);
                let mut hb = round_half_even(h / f) * f;
                let mut wb = round_half_even(w / f) * f;
                if hb * wb > max_pixels as f64 {
                    let beta = (h * w / max_pixels as f64).sqrt();
                    hb = ((h / beta / f).floor() * f).max(f);
                    wb = ((w / beta / f).floor() * f).max(f);
                } else if hb * wb < min_pixels as f64 {
                    let beta = (min_pixels as f64 / (h * w)).sqrt();
                    hb = (h * beta / f).ceil() * f;
                    wb = (w * beta / f).ceil() * f;
                }
                (wb as u32, hb as u32)
            }
        }
    }
}

/// Resizes an RGB image like Pillow's `Image.resize(size, filter)`.
pub fn resize(image: &RgbImage, width: u32, height: u32, filter: Filter) -> RgbImage {
    let (w, h) = image.dimensions();
    let data = resample::resize_u8(image.as_raw(), w as usize, h as usize, 3, width as usize, height as usize, filter);
    RgbImage::from_raw(width, height, data).expect("resize output size")
}

/// Bilinear resize without antialiasing, like torch's `F.interpolate(mode="bilinear",
/// align_corners=False, antialias=False)` on 8-bit images (what torchvision's
/// `resize(..., antialias=False)` runs): separable, horizontal pass first, 8-bit
/// intermediate, int16 fixed-point weights. Close to OpenCV's `INTER_LINEAR`;
/// downscaling skips pixels instead of averaging them (unlike Pillow).
pub fn resize_bilinear_no_antialias(image: &RgbImage, width: u32, height: u32) -> RgbImage {
    let (w, h) = image.dimensions();
    let mut data = image.as_raw().clone();
    if width != w {
        data = linear_pass(&data, w as usize, h as usize, width as usize, true);
    }
    if height != h {
        data = linear_pass(&data, width as usize, h as usize, height as usize, false);
    }
    RgbImage::from_raw(width, height, data).expect("resize output size")
}

/// One axis of `resize_bilinear_no_antialias` on interleaved RGB (`w` x `h`).
fn linear_pass(src: &[u8], w: usize, h: usize, out: usize, horizontal: bool) -> Vec<u8> {
    let input = if horizontal { w } else { h };
    // torch's _compute_indices_min_size_weights (linear, support 1) in f64.
    let scale = input as f64 / out as f64;
    let mut taps = Vec::with_capacity(out);
    let mut max_weight = 0f64;
    for i in 0..out {
        let real = (scale * (i as f64 + 0.5) - 0.5).max(0.0);
        let index = (real.floor() as usize).min(input - 1);
        let lambda = (real - index as f64).clamp(0.0, 1.0);
        let size = ((index + 2).min(input) - index).min(2);
        let mut weights = [0f64; 2];
        let mut slot = 0;
        for j in 0..2 {
            let x = (j as f64 - lambda).abs();
            let wt = if x < 1.0 { 1.0 - x } else { 0.0 };
            if index + j == 0 {
                slot = 0;
            } else if index + j >= input - 1 {
                slot = size - 1;
            }
            weights[slot] += wt;
            max_weight = max_weight.max(weights[slot]);
            slot += 1;
        }
        taps.push((index, size, weights));
    }
    let mut precision = 0;
    while precision < 22 && ((0.5 + max_weight * (1u64 << (precision + 1)) as f64) as i64) < (1 << 15) {
        precision += 1;
    }
    let fixed: Vec<(usize, usize, [i32; 2])> =
        taps.iter().map(|&(i, n, wt)| (i, n, wt.map(|v| (0.5 + v * (1u64 << precision) as f64) as i32))).collect();
    let (ow, oh) = if horizontal { (out, h) } else { (w, out) };
    let mut dst = vec![0u8; ow * oh * 3];
    for y in 0..oh {
        for x in 0..ow {
            let (index, size, wt) = fixed[if horizontal { x } else { y }];
            for c in 0..3 {
                let mut acc = 1i32 << (precision - 1);
                for (k, &weight) in wt.iter().enumerate().take(size) {
                    let (sx, sy) = if horizontal { (index + k, y) } else { (x, index + k) };
                    acc += src[(sy * w + sx) * 3 + c] as i32 * weight;
                }
                dst[(y * ow + x) * 3 + c] = (acc >> precision).clamp(0, 255) as u8;
            }
        }
    }
    dst
}

/// Center crop like transformers' `center_crop`: the crop starts at
/// `(size - crop) // 2` (floor division), and where the crop is larger than
/// the image the outside is zero (HF pads the image centered, rounding the
/// padding up, which lands on the same offsets).
pub fn center_crop(image: &RgbImage, width: u32, height: u32) -> RgbImage {
    let (w, h) = (image.width() as i64, image.height() as i64);
    let left = (w - width as i64).div_euclid(2);
    let top = (h - height as i64).div_euclid(2);
    RgbImage::from_fn(width, height, |x, y| {
        let (ix, iy) = (x as i64 + left, y as i64 + top);
        if (0..w).contains(&ix) && (0..h).contains(&iy) {
            *image.get_pixel(ix as u32, iy as u32)
        } else {
            ::image::Rgb([0, 0, 0])
        }
    })
}

/// Letterbox like Ultralytics' `LetterBox` (YOLO input): scale to fit
/// `width` x `height` keeping the aspect ratio, center, and fill the rest with
/// `fill`. The geometry (sizes, rounding, padding split) matches Ultralytics;
/// the resampling is Pillow's, not OpenCV's. Undo on boxes with
/// [`crate::detect::unletterbox`].
pub fn letterbox(image: &RgbImage, width: u32, height: u32, fill: [u8; 3], filter: Filter) -> RgbImage {
    let (w, h) = (image.width() as f64, image.height() as f64);
    let r = (height as f64 / h).min(width as f64 / w);
    let nw = ((w * r).round_ties_even() as u32).clamp(1, width);
    let nh = ((h * r).round_ties_even() as u32).clamp(1, height);
    let left = ((width - nw) as f64 / 2.0 - 0.1).round_ties_even().max(0.0) as i64;
    let top = ((height - nh) as f64 / 2.0 - 0.1).round_ties_even().max(0.0) as i64;
    let scaled = if (nw, nh) == image.dimensions() { image.clone() } else { resize(image, nw, nh, filter) };
    let mut out = RgbImage::from_pixel(width, height, ::image::Rgb(fill));
    ::image::imageops::replace(&mut out, &scaled, left, top);
    out
}

/// An image preprocessing pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageProcessor {
    pub resize: Option<(Size, Filter)>,
    /// (width, height)
    pub center_crop: Option<(u32, u32)>,
    /// Multiply pixel values by this (e.g. 1/255).
    pub rescale: Option<f64>,
    /// Per-channel (mean, std), in RGB order.
    pub normalize: Option<([f32; 3], [f32; 3])>,
    pub channel_order: ChannelOrder,
    pub layout: Layout,
}

impl Default for ImageProcessor {
    fn default() -> Self {
        ImageProcessor {
            resize: None,
            center_crop: None,
            rescale: Some(1.0 / 255.0),
            normalize: None,
            channel_order: ChannelOrder::Rgb,
            layout: Layout::Chw,
        }
    }
}

impl ImageProcessor {
    /// The 8-bit image after resize and crop.
    pub fn prepare(&self, image: &DynamicImage) -> RgbImage {
        let mut rgb = image.to_rgb8();
        if let Some((size, filter)) = self.resize {
            let (w, h) = size.output(rgb.width(), rgb.height());
            rgb = resize(&rgb, w, h, filter);
        }
        if let Some((w, h)) = self.center_crop {
            rgb = center_crop(&rgb, w, h);
        }
        rgb
    }

    /// `[3, H, W]` (or `[H, W, 3]`) f32 tensor.
    pub fn process(&self, image: &DynamicImage) -> Array3<f32> {
        self.to_tensor(&self.prepare(image))
    }

    /// Rescale, normalize, order and lay out an already prepared image.
    pub fn to_tensor(&self, rgb: &RgbImage) -> Array3<f32> {
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let channel = |c: usize| match self.channel_order {
            ChannelOrder::Rgb => c,
            ChannelOrder::Bgr => 2 - c,
        };
        let value = |x: usize, y: usize, c: usize| -> f32 {
            let v = rgb.get_pixel(x as u32, y as u32)[c];
            let mut v = match self.rescale {
                Some(s) => (v as f64 * s) as f32,
                None => v as f32,
            };
            if let Some((mean, std)) = self.normalize {
                v = (v - mean[c]) / std[c];
            }
            v
        };
        match self.layout {
            Layout::Chw => Array3::from_shape_fn((3, h, w), |(c, y, x)| value(x, y, channel(c))),
            Layout::Hwc => Array3::from_shape_fn((h, w, 3), |(y, x, c)| value(x, y, channel(c))),
        }
    }

    /// Stacks processed images into `[batch, ...]`; all must end up the same size.
    pub fn process_batch(&self, images: &[DynamicImage]) -> Result<ArrayD<f32>> {
        let tensors: Vec<Array3<f32>> = images.iter().map(|i| self.process(i)).collect();
        let views: Vec<_> = tensors.iter().map(|t| t.view()).collect();
        Ok(ndarray::stack(ndarray::Axis(0), &views)
            .map_err(|_| Error::Image("images have different sizes after processing".into()))?
            .into_dyn())
    }

    /// From a Hugging Face `preprocessor_config.json` (image processor part).
    pub fn from_hf_config(config: &Value) -> Result<ImageProcessor> {
        let get = |k: &str| config.get(k).filter(|v| !v.is_null());
        let flag = |k: &str, default: bool| get(k).and_then(Value::as_bool).unwrap_or(default);
        let uint = |v: &Value, k: &str| v.get(k).and_then(Value::as_u64).map(|n| n as u32);
        let mut p = ImageProcessor::default();

        if flag("do_resize", true) {
            let filter = match get("resample") {
                Some(v) => v
                    .as_i64()
                    .and_then(Filter::from_pil)
                    .ok_or_else(|| Error::Config(format!("unsupported resample {v}")))?,
                None => Filter::Bicubic,
            };
            let patch_multiple = get("patch_size")
                .and_then(Value::as_u64)
                .map(|p| p * get("merge_size").and_then(Value::as_u64).unwrap_or(1));
            let size = match (get("size"), get("min_pixels"), get("max_pixels"), patch_multiple) {
                (_, Some(min), Some(max), Some(multiple)) => Some(Size::Multiple {
                    multiple: multiple as u32,
                    min_pixels: min.as_u64().unwrap_or(0),
                    max_pixels: max.as_u64().unwrap_or(u64::MAX),
                }),
                (Some(size), ..) => {
                    if let Some(edge) = size.as_u64() {
                        // Legacy int size: shortest edge when a crop follows (CLIP), else square.
                        Some(if get("crop_size").is_some() {
                            Size::ShortestEdge { edge: edge as u32, max_longest: None }
                        } else {
                            Size::Exact { width: edge as u32, height: edge as u32 }
                        })
                    } else if let (Some(h), Some(w)) = (uint(size, "height"), uint(size, "width")) {
                        Some(Size::Exact { width: w, height: h })
                    } else if let Some(edge) = uint(size, "shortest_edge") {
                        Some(Size::ShortestEdge { edge, max_longest: uint(size, "longest_edge") })
                    } else if let Some(edge) = uint(size, "longest_edge") {
                        Some(Size::LongestEdge(edge))
                    } else {
                        return Err(Error::Config(format!("unsupported size {size}")));
                    }
                }
                _ => None,
            };
            p.resize = size.map(|s| (s, filter));
        }
        if flag("do_center_crop", false) {
            let crop = get("crop_size").ok_or_else(|| Error::Config("do_center_crop without crop_size".into()))?;
            p.center_crop = Some(match crop.as_u64() {
                Some(n) => (n as u32, n as u32),
                None => (
                    uint(crop, "width").ok_or_else(|| Error::Config("crop_size.width".into()))?,
                    uint(crop, "height").ok_or_else(|| Error::Config("crop_size.height".into()))?,
                ),
            });
        }
        p.rescale = if flag("do_rescale", true) {
            Some(get("rescale_factor").and_then(Value::as_f64).unwrap_or(1.0 / 255.0))
        } else {
            None
        };
        if flag("do_normalize", false) {
            let triple = |k: &str| -> Result<[f32; 3]> {
                let v = get(k).ok_or_else(|| Error::Config(format!("do_normalize without {k}")))?;
                match v.as_array().map(|a| a.iter().filter_map(Value::as_f64).map(|x| x as f32).collect::<Vec<_>>()) {
                    Some(a) if a.len() == 3 => Ok([a[0], a[1], a[2]]),
                    _ => match v.as_f64() {
                        Some(x) => Ok([x as f32; 3]),
                        None => Err(Error::Config(format!("{k} must be 3 numbers"))),
                    },
                }
            };
            p.normalize = Some((triple("image_mean")?, triple("image_std")?));
        }
        Ok(p)
    }
}

/// Decodes an image file (PNG, JPEG, WebP, ...).
pub fn decode(bytes: &[u8]) -> Result<DynamicImage> {
    Ok(::image::load_from_memory(bytes)?)
}
