# autopro

Model pre- and postprocessing in Rust, in the spirit of Hugging Face's
`AutoProcessor`: images, audio and text to tensors (`ndarray`), matching the
Hugging Face reference (Pillow-based) processors; detector outputs to
detections, embeddings to clusters and OCR maps to text lines, matching
torchvision / Ultralytics, scikit-learn and PaddleOCR / OpenCV.

## Images — `autopro::image`

```rust
let processor = ImageProcessor::from_hf_config(&serde_json::from_str(&config)?)?; // preprocessor_config.json
let pixels = processor.process(&autopro::image::decode(&bytes)?);                 // [3, H, W] f32
```

- Decode (PNG, JPEG, WebP, … via `image`), convert to RGB (alpha dropped like Pillow).
- Resize, **bit-exact with Pillow** (`nearest`, `box`, `bilinear`, `hamming`,
  `bicubic`, `lanczos`; antialiased downscaling, 8-bit fixed point), because HF
  processors resize through Pillow.
- Sizes: static (`Exact`) or dynamic (`ShortestEdge` with optional longest cap,
  `LongestEdge`, `Multiple` for patch models incl. Qwen2-VL `smart_resize`).
- HF center crop (floor offsets, zero padding), rescale, normalize, RGB/BGR, CHW/HWC.
- `letterbox` for YOLO inputs (Ultralytics geometry; Pillow resampling, not OpenCV's).
- `TileProcessor` (`image::tiling`): docling's VLM pipeline (Granite Docling,
  Idefics3 / SmolVLM architecture) splits a page image into fixed-size tiles
  plus a low-res global view instead of squashing it into one square; matches
  `Idefics3ImageProcessor` / `SmolVLMImageProcessor` (`do_image_splitting`),
  same geometry as `preprocessor_config.json` (`size.longest_edge`,
  `max_image_size.longest_edge`).

## Audio — `autopro::audio`

- `Audio::from_wav_bytes` (integer PCM scaled like soundfile, mono mixdown),
  `resample` (windowed sinc; not bit-compatible with librosa/soxr).
- `WaveformProcessor`: raw waveform, zero-mean/unit-variance, padding + mask (Wav2Vec2).
- `MelSpectrogram`: STFT + mel filter bank + log (HTK/Slaney/Kaldi scales,
  Slaney norm, ln/log10/dB), transformers' `spectrogram()` semantics.
- `WhisperFeatures`: `WhisperFeatureExtractor` (80 or 128 mels, 30 s).

## Text — `autopro::text`

- `Tokenizer::from_bytes` detects the format:
  - SentencePiece `.model`: native implementation of the library's normalizer
    (precompiled charsmap), unigram Viterbi and BPE, user-defined symbols and
    byte fallback — no C++ dependency.
  - `tokenizer.json`: via `tokenizers` (pure-Rust regex backend).
- `TextOptions`: lowercase, special tokens, truncation (keeping special
  tokens, like HF), padding (fixed or longest) → ids + attention mask.
- `Tokenizer::decode(ids, skip_special)`: text from ids (SentencePiece like
  `DecodeIds`: byte pieces, `▁` as spaces, dummy prefix removed).

## Detection — `autopro::detect`

```rust
let d = Detector { score_threshold: 0.25, iou_threshold: 0.45, ..Detector::default() }
    .detect(rows.slice(s![.., ..4]), rows.slice(s![.., 4..]));   // YOLOv8 head, transposed
let boxes = unletterbox(d.boxes.view(), (640, 640), (width, height));
```

- `nms`: greedy NMS like torchvision's `nms`, per class like `batched_nms`
  (IoU > threshold suppresses; kept indices by descending score).
- `Detector`: Ultralytics' `non_max_suppression` (best class or multi-label,
  score threshold, candidate cap, class-wise or agnostic NMS, max detections);
  returns boxes (xyxy), scores, classes and each detection's source row.
- `convert` (xyxy / xywh / cxcywh), `iou`, `scale`, `clip`, `unletterbox`.

## OCR — `autopro::ocr`

```rust
let found = DbDecoder { threshold: 0.2, box_threshold: 0.45, unclip_ratio: 1.4, ..DbDecoder::default() }
    .decode(prob.view(), width, height);                     // DB map [H, W] -> quads on the original image
for i in sort_boxes(&found.boxes) {                          // reading order
    let line = crop_text_line(&image, &found.boxes[i]);     // straightened text line
    let (ids, score) = ctc_greedy(probs.view());             // recognizer output [T, classes]
}
```

- `DbDecoder`: transformers' PP-OCR `post_process_object_detection`
  (PaddleOCR's `DBPostProcess`, fast box score, polygon-offset unclip).
- The OpenCV functions it needs, ported rather than approximated:
  `find_contours` (`RETR_LIST`, `CHAIN_APPROX_SIMPLE`: same contours, order and
  points), `min_area_rect` (convex hull order, float32 rotating calipers and
  `boxPoints`, including clang's FMA contraction on arm64), `fill_poly`.
- `sort_boxes`, `crop_text_line`: PaddleOCR's `sort_quad_boxes` and
  `get_minarea_rect_crop` (`getPerspectiveTransform` + bicubic
  `warpPerspective` with OpenCV's fixed-point weights, replicated border).
- `ctc_greedy`: best class per step, repeats merged, blanks dropped, mean
  probability.
- `image::resize_bilinear_no_antialias`: torch's `antialias=False` bilinear on
  8-bit images (separable, int16 weights; what torchvision's resize runs).

## Clustering — `autopro::cluster`

```rust
let c = Dbscan { eps: 0.08, min_samples: 8, metric: Metric::Cosine }.fit(patches.view());   // [n, d]
let means = centroids(patches.view(), &c.labels, c.n_clusters);
```

- `Dbscan`: scikit-learn's DBSCAN (same labels, core points and cluster
  order), euclidean or cosine.
- `radius_neighbors`: one matrix product per block of rows (BLAS), with pairs
  close to `eps` rechecked in f64 so f32 rounding never decides membership.
- Speed (M-series, release): 1024 x 768 patches ~10 ms; 4096 x 768 ~125 ms
  with Accelerate vs ~400 ms without (`cargo run --release --example cluster_bench`).

BLAS features: `platform-blas` (default) calls Accelerate's `sgemm` on Apple
platforms; `blas` enables ndarray's BLAS backend (link a BLAS yourself, e.g.
`blas-src` with OpenBLAS); otherwise ndarray's `matrixmultiply`.

## Tests against the references

`tests/reference.rs` compares everything with the Python implementations:

```sh
cd reference
nix develop --command python make_fixtures.py ../tests/fixtures path/to/some/sentencepiece.model
cd .. && cargo test
```

The flake (Python with transformers, tokenizers, sentencepiece, Pillow,
numpy, soundfile; no torch/CUDA) only lives in the Nix store: remove it with
`nix store gc`. Current results: Pillow resize bit-exact for all filters and
sizes tested, HF image processors (CLIP, ViT, TIPSv2 configs) within 1e-6,
SentencePiece ids and decoded text identical (unigram and BPE, byte fallback,
user symbols, Unicode normalization), `tokenizer.json` ids/truncation/padding/decoding identical,
Whisper log-mel within 1e-4, Wav2Vec2 within 1e-5.

Postprocessing fixtures (torchvision NMS / `batched_nms`, Ultralytics'
`non_max_suppression` on torchvision, scikit-learn DBSCAN) come from a second
shell with CPU-only torch (no CUDA):

```sh
cd reference
nix develop .#post --command python make_post_fixtures.py ../tests/fixtures/post
```

Results: NMS indices, detections (boxes, scores, classes, rows) and DBSCAN
labels / core points identical.

OCR fixtures (OpenCV 4.13, torchvision, the DB / crop code of transformers
5.17 and PaddleOCR 3.x) come from the same shell:

```sh
nix develop .#post --command python make_ocr_fixtures.py ../tests/fixtures/ocr
```

Results (arm64 macOS): contours, their min-area rectangles, `fillPoly` masks,
reading order and the no-antialias resize identical; DB boxes within 1 pixel,
scores within 1e-5; crops the same size with at most 0.2% of values off by up
to 4 (last-bit differences in the homography move a few samples to the
neighboring 1/32 pixel); `minAreaRect` on arbitrary point sets identical up to
ties between equal-area rectangles. OpenCV builds without FMA contraction
(e.g. x86) round some rectangles differently in the last bit.

Image-tiling fixtures (`Idefics3ImageProcessor`'s splitting, ported to Pillow
since that file now requires torch/torchvision) need only Pillow and numpy,
no flake:

```sh
cd reference && python make_docling_fixtures.py ../tests/fixtures/docling
```

Results: tile counts and pixel values identical to the ported reference.

Not covered yet: `tokenizer_config.json` / feature-extractor config loading,
"fast" (torchvision) image processors, MP3/FLAC decoding, EXIF orientation.
