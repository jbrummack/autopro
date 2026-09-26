"""Generates autopro's test fixtures with the Hugging Face / Pillow / sentencepiece
reference implementations.

    cd reference && nix develop --command python make_fixtures.py ../tests/fixtures [path/to/tokenizer.model]

Writes inputs (PNG, WAV, tokenizer files) and expected outputs (.npy) plus
manifest.json, which tests/reference.rs reads.
"""

import json
import os
import shutil
import sys
import warnings

import numpy as np
import sentencepiece as spm
import soundfile as sf
from PIL import Image

warnings.filterwarnings("ignore")
out = sys.argv[1]
spm_model = sys.argv[2] if len(sys.argv) > 2 else None
os.makedirs(out, exist_ok=True)
manifest = {"images": [], "resize": [], "processors": [], "sentencepiece": [], "tokenizers": [], "audio": []}


def save(name, array):
    np.save(os.path.join(out, name), array)
    return name


# ---------------------------------------------------------------- images

def test_image(w, h, seed, mode="RGB"):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([x * 255 / w, y * 255 / h, (x + y) % 37 * 7], axis=-1)
    noise = rng.integers(-40, 40, size=(h, w, 3))
    rgb = np.clip(base + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb, "RGB")
    if mode == "RGBA":
        alpha = rng.integers(0, 256, size=(h, w, 1)).astype(np.uint8)
        image = Image.fromarray(np.concatenate([rgb, alpha], axis=-1), "RGBA")
    elif mode == "L":
        image = image.convert("L")
    return image


images = {
    "rgb_97x61": test_image(97, 61, 0),
    "rgb_320x240": test_image(320, 240, 1),
    "rgba_50x80": test_image(50, 80, 2, "RGBA"),
    "gray_33x47": test_image(33, 47, 3, "L"),
}
for name, image in images.items():
    image.save(os.path.join(out, f"{name}.png"))
    manifest["images"].append({"name": name, "file": f"{name}.png"})

filters = {0: "nearest", 1: "lanczos", 2: "bilinear", 3: "bicubic", 4: "box", 5: "hamming"}
for name in ["rgb_97x61", "rgb_320x240"]:
    rgb = images[name].convert("RGB")
    for code, filter_name in filters.items():
        for size in [(224, 224), (31, 45), (150, 100), (97 * 3, 61 * 2)]:
            resized = np.array(rgb.resize(size, resample=code))
            file = save(f"resize_{name}_{filter_name}_{size[0]}x{size[1]}.npy", resized)
            manifest["resize"].append({"image": name, "filter": filter_name, "width": size[0], "height": size[1], "expected": file})

from transformers import CLIPImageProcessor, ViTImageProcessor  # noqa: E402

processor_configs = {
    # OpenAI CLIP: shortest edge 224 bicubic, center crop, CLIP normalization.
    "clip": {
        "do_resize": True, "size": {"shortest_edge": 224}, "resample": 3, "do_center_crop": True,
        "crop_size": {"height": 224, "width": 224}, "do_rescale": True, "rescale_factor": 1 / 255,
        "do_normalize": True, "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711], "do_convert_rgb": True,
    },
    # ViT: fixed 224x224 bilinear, mean/std 0.5.
    "vit": {
        "do_resize": True, "size": {"height": 224, "width": 224}, "resample": 2, "do_rescale": True,
        "rescale_factor": 1 / 255, "do_normalize": True, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
    },
    # TIPSv2: 448x448 bilinear, rescale only.
    "tips": {
        "do_resize": True, "size": {"height": 448, "width": 448}, "resample": 2, "do_rescale": True,
        "rescale_factor": 1 / 255, "do_normalize": False,
    },
}
classes = {"clip": CLIPImageProcessor, "vit": ViTImageProcessor, "tips": ViTImageProcessor}
for pname, config in processor_configs.items():
    processor = classes[pname](**config)
    for iname, image in images.items():
        if pname == "tips" and iname != "rgb_320x240":
            continue  # 448 px float outputs are large; one image is enough
        pixels = processor(images=image.convert("RGB"), return_tensors="np")["pixel_values"][0]
        file = save(f"processor_{pname}_{iname}.npy", pixels.astype(np.float32))
        manifest["processors"].append({"processor": pname, "config": config, "image": iname, "expected": file})

# ------------------------------------------------------------------ text

texts = [
    "a photo of a bus",
    "A dog running on the beach at sunset, with waves in the background.",
    "  leading and   inner   spaces  ",
    "tabs\tand\nnewlines",
    "Ünïcödé àccents and ß",
    "full-width ＡＢＣ　１２３",
    "emoji 🙂 and CJK 日本語のテキスト",
    "numbers 1234567 3.14159 and symbols #@$%^&*()",
    "",
    "   ",
    "Hello WORLD",
    "user_sym and <custom> tokens",
]

sp_models = {}
if spm_model:
    shutil.copy(spm_model, os.path.join(out, "tips.model"))
    sp_models["tips"] = "tips.model"

corpus = os.path.join(out, "corpus.txt")
with open(corpus, "w") as f:
    for i in range(300):
        f.write(f"sentence {i} with some words like photo dog beach sunset waves running numbers {i * 7}\n")
        f.write("the quick brown fox jumps over the lazy dog. Ünïcödé àccents ß\n")
for kind in ["bpe", "unigram"]:
    prefix = os.path.join(out, f"spm_{kind}")
    spm.SentencePieceTrainer.train(
        input=corpus, model_prefix=prefix, vocab_size=450, model_type=kind, byte_fallback=True,
        user_defined_symbols=["user_sym", "<custom>"], character_coverage=0.99, minloglevel=2,
    )
    os.remove(prefix + ".vocab")
    sp_models[kind] = f"spm_{kind}.model"

for mname, file in sp_models.items():
    sp = spm.SentencePieceProcessor(model_file=os.path.join(out, file))
    for text in texts:
        for bos, eos in [(False, False), (True, True)]:
            manifest["sentencepiece"].append({
                "model": file, "text": text, "add_bos": bos, "add_eos": eos,
                "ids": sp.encode(text, add_bos=bos, add_eos=eos),
                "decoded": sp.decode(sp.encode(text, add_bos=bos, add_eos=eos)),
            })

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

wordpiece = Tokenizer(models.WordPiece(unk_token="[UNK]"))
wordpiece.normalizer = normalizers.BertNormalizer(lowercase=True)
wordpiece.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
wordpiece.train([corpus], trainers.WordPieceTrainer(vocab_size=300, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]))
wordpiece.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]", special_tokens=[("[CLS]", wordpiece.token_to_id("[CLS]")), ("[SEP]", wordpiece.token_to_id("[SEP]"))],
)
wordpiece.decoder = decoders.WordPiece()
bytebpe = Tokenizer(models.BPE())
bytebpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
bytebpe.train([corpus], trainers.BpeTrainer(vocab_size=400, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
bytebpe.post_processor = processors.ByteLevel(trim_offsets=False)
bytebpe.decoder = decoders.ByteLevel()

for tname, tok in [("wordpiece", wordpiece), ("bytebpe", bytebpe)]:
    file = f"{tname}.json"
    tok.save(os.path.join(out, file))
    fast = PreTrainedTokenizerFast(tokenizer_file=os.path.join(out, file), pad_token="[PAD]" if tname == "wordpiece" else "<|endoftext|>")
    for text in texts:
        for special in [False, True]:
            ids = tok.encode(text, add_special_tokens=special).ids
            case = {"tokenizer": file, "text": text, "add_special_tokens": special, "ids": ids,
                    "decoded": tok.decode(ids, skip_special_tokens=True)}
            enc = fast(text, add_special_tokens=special, truncation=True, max_length=8, padding="max_length")
            case["max_length_8"] = {"ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
            manifest["tokenizers"].append(case)
os.remove(corpus)

# ----------------------------------------------------------------- audio

from transformers import Wav2Vec2FeatureExtractor, WhisperFeatureExtractor  # noqa: E402

sr = 16000
t = np.arange(int(sr * 3.7)) / sr
signal = 0.4 * np.sin(2 * np.pi * (200 + 900 * t) * t) + 0.1 * np.random.default_rng(0).normal(size=t.size)
sf.write(os.path.join(out, "chirp_16k.wav"), signal.astype(np.float32), sr, subtype="PCM_16")
samples, _ = sf.read(os.path.join(out, "chirp_16k.wav"), dtype="float32")
for n_mels in [80, 128]:
    fe = WhisperFeatureExtractor(feature_size=n_mels)
    features = fe(samples, sampling_rate=sr, return_tensors="np")["input_features"][0]
    manifest["audio"].append({"kind": "whisper", "wav": "chirp_16k.wav", "n_mels": n_mels,
                              "expected": save(f"whisper_{n_mels}.npy", features.astype(np.float32))})
fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=sr, padding_value=0.0, do_normalize=True, return_attention_mask=True)
enc = fe(samples, sampling_rate=sr, padding="max_length", max_length=64000, return_tensors="np")
manifest["audio"].append({"kind": "wav2vec2", "wav": "chirp_16k.wav", "length": 64000,
                          "expected": save("wav2vec2_values.npy", enc["input_values"][0].astype(np.float32)),
                          "mask": save("wav2vec2_mask.npy", enc["attention_mask"][0].astype(np.int64))})

with open(os.path.join(out, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1, ensure_ascii=False)
print({k: len(v) for k, v in manifest.items()})
