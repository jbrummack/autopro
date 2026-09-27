//! Image "splitting" (tiling) for high-resolution vision encoders: downscale
//! a big image to fit a size limit, round it up to a multiple of the tile
//! size, cut it into non-overlapping square tiles, and append one more tile —
//! the whole (pre-split) image resized down to a single tile — as a
//! low-resolution global view. This is how docling's VLM pipeline (Granite
//! Docling, built on the Idefics3 / SmolVLM architecture) feeds a
//! high-resolution page image to a fixed-resolution vision encoder instead of
//! squashing the whole page into one tile.
//!
//! Matches transformers' `Idefics3ImageProcessor` / `SmolVLMImageProcessor`
//! with `do_image_splitting=True` (`_resize_output_size_rescale_to_max_len`,
//! `_resize_output_size_scale_below_upper_bound`, `resize_for_vision_encoder`,
//! `split_images` in `image_processing_idefics3.py`). As of this writing that
//! file requires torch/torchvision (resizing through a torchvision backend);
//! this follows the historical PIL/numpy implementation instead (same
//! formulas, ported function by function), consistent with the rest of this
//! crate and with the resampler Granite Docling's `preprocessor_config.json`
//! actually asks for (`"resample": 1` = Pillow's `LANCZOS`).
//! `ibm-granite/granite-docling-258M`'s config: `size.longest_edge=2048`,
//! `max_image_size.longest_edge=512`, `do_image_splitting=true`.
use ::image::DynamicImage;
use ndarray::Array3;
use serde_json::Value;

use super::{ChannelOrder, Filter, ImageProcessor, resize};
use crate::error::{Error, Result};

/// 4k resolution, an absolute cap on the initial (aspect-preserving) resize.
const MAX_IMAGE_SIZE: u32 = 4096;

/// `_resize_output_size_rescale_to_max_len`: the longer edge becomes
/// `max_len`, the other edge scaled to keep the aspect ratio and rounded up
/// to an even number.
fn rescale_to_max_len(height: u32, width: u32, max_len: u32) -> (u32, u32) {
    let aspect_ratio = width as f64 / height as f64;
    let (h, w) = if width >= height {
        let w = max_len;
        let mut h = (w as f64 / aspect_ratio) as u32;
        if !h.is_multiple_of(2) {
            h += 1;
        }
        (h, w)
    } else {
        let h = max_len;
        let mut w = (h as f64 * aspect_ratio) as u32;
        if !w.is_multiple_of(2) {
            w += 1;
        }
        (h, w)
    };
    (h.max(1), w.max(1))
}

/// `_resize_output_size_scale_below_upper_bound`: clamps the longer edge to
/// `max_len` if it still overshoots (only bites near [`MAX_IMAGE_SIZE`]).
fn scale_below_upper_bound(height: u32, width: u32, max_len: u32) -> (u32, u32) {
    let aspect_ratio = width as f64 / height as f64;
    let (h, w) = if width >= height && width > max_len {
        (( max_len as f64 / aspect_ratio) as u32, max_len)
    } else if height > width && height > max_len {
        (max_len, (max_len as f64 * aspect_ratio) as u32)
    } else {
        (height, width)
    };
    (h.max(1), w.max(1))
}

/// `resize_for_vision_encoder`: rounds both edges up to a multiple of
/// `tile_edge`. The order matters for an exact match: the longer edge is
/// rounded up first, the other edge is derived from *that* rounded edge (via
/// the original aspect ratio) and then rounded up itself.
fn size_for_tiles(height: u32, width: u32, tile_edge: u32) -> (u32, u32) {
    let aspect_ratio = width as f64 / height as f64;
    if width >= height {
        let w = width.div_ceil(tile_edge) * tile_edge;
        let h = (w as f64 / aspect_ratio) as u32;
        (h.max(1).div_ceil(tile_edge) * tile_edge, w)
    } else {
        let h = height.div_ceil(tile_edge) * tile_edge;
        let w = (h as f64 * aspect_ratio) as u32;
        (h, w.max(1).div_ceil(tile_edge) * tile_edge)
    }
}

/// A tiled image: `rows * cols` non-overlapping `tile_edge` x `tile_edge`
/// tiles in row-major order (row 0 col 0, row 0 col 1, ...), followed by one
/// more tile — the whole pre-split image resized down to a single tile — when
/// the image was actually split. Otherwise `rows = cols = 0` and there is
/// exactly one tile (the image was already small enough).
#[derive(Debug, Clone, PartialEq)]
pub struct Tiles {
    /// One `[3, tile_edge, tile_edge]` (or `[tile_edge, tile_edge, 3]`) tensor
    /// per tile.
    pub tiles: Vec<Array3<f32>>,
    pub rows: u32,
    pub cols: u32,
}

impl Tiles {
    /// Whether the last tile in `tiles` is the whole-image global view rather
    /// than a grid tile.
    pub fn has_global(&self) -> bool {
        self.rows > 0 && self.cols > 0
    }
}

/// Splits a big image into fixed-size tiles for a tiled vision encoder
/// (docling's VLM pipeline). See the module docs for the algorithm.
#[derive(Debug, Clone, PartialEq)]
pub struct TileProcessor {
    /// Longest edge after the initial downscale (transformers'
    /// `size.longest_edge`).
    pub resize_longest_edge: u32,
    /// Tile side in pixels (transformers' `max_image_size.longest_edge`).
    pub tile_edge: u32,
    pub filter: Filter,
    /// `false` squares the whole image into one `tile_edge` x `tile_edge`
    /// tile instead of splitting it (`do_image_splitting=False`).
    pub do_image_splitting: bool,
    /// Multiply pixel values by this (e.g. 1/255).
    pub rescale: Option<f64>,
    /// Per-channel (mean, std), in RGB order.
    pub normalize: Option<([f32; 3], [f32; 3])>,
    pub channel_order: ChannelOrder,
}

impl TileProcessor {
    /// Tiles an already-decoded image.
    pub fn process(&self, image: &DynamicImage) -> Tiles {
        let rgb = image.to_rgb8();
        let (width, height) = rgb.dimensions();
        let (h1, w1) = rescale_to_max_len(height, width, self.resize_longest_edge);
        let (h1, w1) = scale_below_upper_bound(h1, w1, MAX_IMAGE_SIZE);
        let base = if (w1, h1) == (width, height) { rgb } else { resize(&rgb, w1, h1, self.filter) };

        let to_tensor = ImageProcessor {
            rescale: self.rescale,
            normalize: self.normalize,
            channel_order: self.channel_order,
            ..ImageProcessor::default()
        };

        if !self.do_image_splitting {
            let square = resize(&base, self.tile_edge, self.tile_edge, self.filter);
            return Tiles { tiles: vec![to_tensor.to_tensor(&square)], rows: 0, cols: 0 };
        }

        let (h2, w2) = size_for_tiles(base.height(), base.width(), self.tile_edge);
        let sized = if (w2, h2) == base.dimensions() { base } else { resize(&base, w2, h2, self.filter) };

        if h2 <= self.tile_edge && w2 <= self.tile_edge {
            return Tiles { tiles: vec![to_tensor.to_tensor(&sized)], rows: 0, cols: 0 };
        }

        let (rows, cols) = (h2 / self.tile_edge, w2 / self.tile_edge);
        let mut tiles = Vec::with_capacity((rows * cols + 1) as usize);
        for row in 0..rows {
            for col in 0..cols {
                let crop = ::image::imageops::crop_imm(&sized, col * self.tile_edge, row * self.tile_edge, self.tile_edge, self.tile_edge);
                tiles.push(to_tensor.to_tensor(&crop.to_image()));
            }
        }
        let global = resize(&sized, self.tile_edge, self.tile_edge, self.filter);
        tiles.push(to_tensor.to_tensor(&global));
        Tiles { tiles, rows, cols }
    }

    /// From a Hugging Face `preprocessor_config.json` (`Idefics3ImageProcessor`
    /// / `SmolVLMImageProcessor`, e.g. Granite Docling's). Assumes
    /// `do_resize=True` and `do_convert_rgb=True` (both classes' defaults,
    /// and what docling ships).
    pub fn from_hf_config(config: &Value) -> Result<TileProcessor> {
        let get = |k: &str| config.get(k).filter(|v| !v.is_null());
        let flag = |k: &str, default: bool| get(k).and_then(Value::as_bool).unwrap_or(default);
        let edge = |k: &str, default: u32| -> Result<u32> {
            match get(k) {
                Some(v) => v
                    .get("longest_edge")
                    .and_then(Value::as_u64)
                    .map(|n| n as u32)
                    .ok_or_else(|| Error::Config(format!("{k} must have a longest_edge"))),
                None => Ok(default),
            }
        };
        let triple = |k: &str, default: [f32; 3]| -> Result<[f32; 3]> {
            match get(k) {
                Some(v) => match v.as_array().map(|a| a.iter().filter_map(Value::as_f64).map(|x| x as f32).collect::<Vec<_>>()) {
                    Some(a) if a.len() == 3 => Ok([a[0], a[1], a[2]]),
                    _ => Err(Error::Config(format!("{k} must be 3 numbers"))),
                },
                None => Ok(default),
            }
        };
        let filter = match get("resample") {
            Some(v) => v.as_i64().and_then(Filter::from_pil).ok_or_else(|| Error::Config(format!("unsupported resample {v}")))?,
            None => Filter::Lanczos,
        };
        Ok(TileProcessor {
            resize_longest_edge: edge("size", 4 * 364)?,
            tile_edge: edge("max_image_size", 364)?,
            filter,
            do_image_splitting: flag("do_image_splitting", true),
            rescale: if flag("do_rescale", true) { Some(get("rescale_factor").and_then(Value::as_f64).unwrap_or(1.0 / 255.0)) } else { None },
            normalize: if flag("do_normalize", true) {
                Some((triple("image_mean", [0.5; 3])?, triple("image_std", [0.5; 3])?))
            } else {
                None
            },
            channel_order: ChannelOrder::Rgb,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Hand-computed from the ported transformers formulas (Granite Docling's
    // size.longest_edge=2048, max_image_size.longest_edge=512).
    #[test]
    fn granite_docling_geometry() {
        // Portrait page 1700x2200: initial resize keeps the aspect ratio with
        // longest edge 2048 (2048 * 1700/2200 = 1582.5 -> 1582, already even).
        let (h1, w1) = rescale_to_max_len(2200, 1700, 2048);
        assert_eq!((h1, w1), (2048, 1582));
        let (h1, w1) = scale_below_upper_bound(h1, w1, 4096);
        assert_eq!((h1, w1), (2048, 1582)); // well under MAX_IMAGE_SIZE, unchanged
        let (h2, w2) = size_for_tiles(h1, w1, 512);
        assert_eq!((h2, w2), (2048, 2048)); // both edges round up to a multiple of 512
        assert_eq!((h2 / 512, w2 / 512), (4, 4));
    }

    #[test]
    fn small_image_is_not_split() {
        // With a tile_edge equal to the initial resize target, the rounded-up
        // size can never exceed one tile: a single frame, no split.
        let (h1, w1) = rescale_to_max_len(300, 400, 2048);
        let (h1, w1) = scale_below_upper_bound(h1, w1, 4096);
        assert_eq!((h1, w1), (1536, 2048)); // longest edge (400>300) rescaled to 2048, 300*2048/400=1536
        let (h2, w2) = size_for_tiles(h1, w1, 2048);
        assert_eq!((h2, w2), (2048, 2048));
        assert!(h2 <= 2048 && w2 <= 2048);
    }

    #[test]
    fn process_splits_a_page_into_tiles_with_a_global_view() {
        // Granite Docling's config: longest edge always lands on a multiple
        // of the tile size (2048 = 4*512), so any page (even a small one,
        // upscaled by the initial resize) ends up split.
        let p = TileProcessor {
            resize_longest_edge: 2048,
            tile_edge: 512,
            filter: Filter::Lanczos,
            do_image_splitting: true,
            rescale: Some(1.0 / 255.0),
            normalize: Some(([0.5; 3], [0.5; 3])),
            channel_order: ChannelOrder::Rgb,
        };
        for (w, h) in [(1700, 2200), (100, 80)] {
            let tiles = p.process(&DynamicImage::new_rgb8(w, h));
            assert_eq!((tiles.rows, tiles.cols), (4, 4), "{w}x{h}");
            assert!(tiles.has_global());
            assert_eq!(tiles.tiles.len(), 4 * 4 + 1);
            for t in &tiles.tiles {
                assert_eq!(t.shape(), [3, 512, 512]);
            }
        }
    }

    #[test]
    fn process_squares_a_single_tile_when_it_already_fits() {
        // tile_edge == resize_longest_edge: the rounded-up size can never
        // exceed one tile, for any input image.
        let p = TileProcessor {
            resize_longest_edge: 2048,
            tile_edge: 2048,
            filter: Filter::Lanczos,
            do_image_splitting: true,
            rescale: Some(1.0 / 255.0),
            normalize: Some(([0.5; 3], [0.5; 3])),
            channel_order: ChannelOrder::Rgb,
        };
        for (w, h) in [(1700, 2200), (100, 80), (5000, 40)] {
            let tiles = p.process(&DynamicImage::new_rgb8(w, h));
            assert_eq!((tiles.rows, tiles.cols), (0, 0), "{w}x{h}");
            assert!(!tiles.has_global());
            assert_eq!(tiles.tiles.len(), 1);
            assert_eq!(tiles.tiles[0].shape(), [3, 2048, 2048]);
        }
    }

    #[test]
    fn from_hf_config_reads_granite_docling() {
        // ibm-granite/granite-docling-258M's preprocessor_config.json.
        let config: Value = serde_json::from_str(
            r#"{
                "do_image_splitting": true, "do_resize": true, "size": {"longest_edge": 2048},
                "max_image_size": {"longest_edge": 512}, "do_rescale": true,
                "rescale_factor": 0.00392156862745098, "do_normalize": true,
                "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "resample": 1,
                "do_pad": true, "image_processor_type": "Idefics3ImageProcessor"
            }"#,
        )
        .unwrap();
        let p = TileProcessor::from_hf_config(&config).unwrap();
        assert_eq!(p.resize_longest_edge, 2048);
        assert_eq!(p.tile_edge, 512);
        assert_eq!(p.filter, Filter::Lanczos);
        assert!(p.do_image_splitting);
        assert_eq!(p.rescale, Some(1.0 / 255.0));
        assert_eq!(p.normalize, Some(([0.5; 3], [0.5; 3])));
    }

    #[test]
    fn size_for_tiles_rounds_the_derived_edge_up_too() {
        // width >= height branch: width rounds to a tile multiple first, then
        // height is derived from the *rounded* width and rounded up itself.
        let (h, w) = size_for_tiles(100, 1000, 364);
        assert_eq!(w, 1092); // ceil(1000/364)*364
        // height = int(1092 / (1000/100)) = int(109.2) = 109 -> ceil(109/364)*364 = 364
        assert_eq!(h, 364);
    }
}
