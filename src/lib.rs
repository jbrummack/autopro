//! autopro: turns raw model inputs into tensors and model outputs into results, in the spirit of Hugging
//! Face's `AutoProcessor`, and matching its (slow, reference) processors.
//!
//! - [`image`]: decode, resize (Pillow-exact), crop, rescale, normalize, RGB/BGR, CHW/HWC.
//! - [`audio`]: WAV decoding, resampling, waveforms and (log-)mel spectrograms (Whisper-compatible).
//! - [`text`]: SentencePiece `.model` files (native implementation) and `tokenizer.json`.
//!
//! Postprocessing:
//! - [`detect`]: box formats, IoU, NMS (torchvision semantics), detector heads to detections (Ultralytics).
//! - [`cluster`]: DBSCAN (scikit-learn semantics) and centroids, e.g. for patch embeddings; distances on BLAS.
//! - [`ocr`]: text boxes from DB probability maps, reading order, text-line crops, CTC decoding (PaddleOCR).

pub mod audio;
pub mod cluster;
pub mod detect;
pub mod error;
pub mod image;
pub mod linalg;
pub mod ocr;
pub mod text;

pub use error::{Error, Result};

