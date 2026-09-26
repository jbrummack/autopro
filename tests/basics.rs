//! Checks that don't need the generated reference fixtures.
use autopro::{
    audio::WhisperFeatures,
    image::{Size, center_crop},
    text::{Padding, TextOptions, Tokenizer},
};

#[test]
fn resize_policies() {
    // HF shortest_edge: the long side is truncated, not rounded.
    assert_eq!(Size::ShortestEdge { edge: 224, max_longest: None }.output(640, 480), (298, 224));
    // Long side 298 > 250: short becomes int(250 * 224 / 298) = 187.
    assert_eq!(Size::ShortestEdge { edge: 224, max_longest: Some(250) }.output(480, 640), (187, 250));
    assert_eq!(Size::LongestEdge(100).output(640, 480), (100, 75));
    // Qwen2-VL smart_resize (factor 28); expected values from transformers source.
    let multiple = |min, max| Size::Multiple { multiple: 28, min_pixels: min, max_pixels: max };
    assert_eq!(multiple(56 * 56, 28 * 28 * 1280).output(640, 480), (644, 476));
    assert_eq!(multiple(56 * 56, 200_000).output(1920, 1080), (588, 308));
    assert_eq!(multiple(100_000, 1_000_000).output(100, 50), (448, 224));
}

#[test]
fn center_crop_pads_like_hf() {
    let img = image::RgbImage::from_fn(5, 3, |x, y| image::Rgb([x as u8 + 1, y as u8 + 1, 0]));
    let out = center_crop(&img, 3, 5);
    // Width: (5 - 3) // 2 = 1 -> columns 1..4. Height: (3 - 5) // 2 = -1 -> one zero row on top.
    assert_eq!(out.get_pixel(0, 0).0, [0, 0, 0]);
    assert_eq!(out.get_pixel(0, 1).0, [2, 1, 0]);
    assert_eq!(out.get_pixel(2, 3).0, [4, 3, 0]);
    assert_eq!(out.get_pixel(2, 4).0, [0, 0, 0]);
}

#[test]
fn truncation_keeps_special_tokens() {
    let json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
      "normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"decoder":null,
      "post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"[SEP]","type_id":0}}],
        "pair":[],"special_tokens":{"[CLS]":{"id":"[CLS]","ids":[1],"tokens":["[CLS]"]},"[SEP]":{"id":"[SEP]","ids":[2],"tokens":["[SEP]"]}}},
      "model":{"type":"WordLevel","vocab":{"[PAD]":0,"[CLS]":1,"[SEP]":2,"[UNK]":3,"a":4,"b":5,"c":6},"unk_token":"[UNK]"}}"#;
    let tokenizer = Tokenizer::from_bytes(json.as_bytes()).unwrap();
    let options = TextOptions { max_length: Some(4), padding: Padding::Fixed(6), ..Default::default() };
    let enc = options.encode_batch(&tokenizer, &["a b c a b"]).unwrap();
    assert_eq!(enc.ids.row(0).to_vec(), [1, 4, 5, 2, 0, 0]);
    assert_eq!(enc.attention_mask.row(0).to_vec(), [1, 1, 1, 1, 0, 0]);
}

#[test]
fn whisper_shape() {
    let features = WhisperFeatures::new(80).compute(&vec![0.0; 16000]);
    assert_eq!(features.dim(), (80, 3000));
}
