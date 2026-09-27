"""Generates autopro's image-tiling fixtures, ported from transformers'
Idefics3ImageProcessor / SmolVLMImageProcessor (image_processing_idefics3.py):
_resize_output_size_rescale_to_max_len, _resize_output_size_scale_below_upper_bound,
resize_for_vision_encoder, split_images. That file now resizes through a
torchvision backend; this uses Pillow instead (same formulas), since this
crate already matches Pillow bit-exactly and docling's models (e.g. Granite
Docling) ask for Pillow's LANCZOS (preprocessor_config.json `"resample": 1`)
anyway. Needs only Pillow and numpy, unlike reference/make_fixtures.py.

    python make_docling_fixtures.py ../tests/fixtures/docling

Writes inputs (PNG) and expected outputs (.npy) plus manifest.json, which
tests/docling_reference.rs reads.
"""

import json
import math
import os
import sys

import numpy as np
from PIL import Image

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
manifest = {"tiling": []}

MAX_IMAGE_SIZE = 4096


def _rescale_to_max_len(height, width, max_len):
    aspect_ratio = width / height
    if width >= height:
        width = max_len
        height = int(width / aspect_ratio)
        if height % 2 != 0:
            height += 1
    else:
        height = max_len
        width = int(height * aspect_ratio)
        if width % 2 != 0:
            width += 1
    return max(height, 1), max(width, 1)


def _scale_below_upper_bound(height, width, max_len):
    aspect_ratio = width / height
    if width >= height and width > max_len:
        width = max_len
        height = int(width / aspect_ratio)
    elif height > width and height > max_len:
        height = max_len
        width = int(height * aspect_ratio)
    return max(height, 1), max(width, 1)


def _size_for_tiles(height, width, tile_edge):
    aspect_ratio = width / height
    if width >= height:
        width = math.ceil(width / tile_edge) * tile_edge
        height = int(width / aspect_ratio)
        height = math.ceil(height / tile_edge) * tile_edge
    else:
        height = math.ceil(height / tile_edge) * tile_edge
        width = int(height * aspect_ratio)
        width = math.ceil(width / tile_edge) * tile_edge
    return height, width


def tile_image(image, resize_longest_edge, tile_edge, do_image_splitting, filt=Image.LANCZOS):
    rgb = image.convert("RGB")
    h1, w1 = _rescale_to_max_len(rgb.height, rgb.width, resize_longest_edge)
    h1, w1 = _scale_below_upper_bound(h1, w1, MAX_IMAGE_SIZE)
    base = rgb.resize((w1, h1), filt) if (w1, h1) != (rgb.width, rgb.height) else rgb

    if not do_image_splitting:
        return [base.resize((tile_edge, tile_edge), filt)], 0, 0

    h2, w2 = _size_for_tiles(base.height, base.width, tile_edge)
    sized = base.resize((w2, h2), filt) if (w2, h2) != (base.width, base.height) else base

    if h2 <= tile_edge and w2 <= tile_edge:
        return [sized], 0, 0

    rows, cols = h2 // tile_edge, w2 // tile_edge
    tiles = [
        sized.crop((col * tile_edge, row * tile_edge, (col + 1) * tile_edge, (row + 1) * tile_edge))
        for row in range(rows)
        for col in range(cols)
    ]
    tiles.append(sized.resize((tile_edge, tile_edge), filt))
    return tiles, rows, cols


def test_image(w, h, seed):
    """Same generator as make_fixtures.py's, kept local so this script has no
    sibling-file dependency."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([x * 255 / w, y * 255 / h, (x + y) % 37 * 7], axis=-1)
    noise = rng.integers(-40, 40, size=(h, w, 3))
    rgb = np.clip(base + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


images = {"rgb_97x61": test_image(97, 61, 0), "rgb_320x240": test_image(320, 240, 1)}
for name, image in images.items():
    image.save(os.path.join(out, f"{name}.png"))

tiling_configs = {
    # Small so most cases stay cheap; one real config validates the shipped values.
    "tiny_split": {"resize_longest_edge": 96, "tile_edge": 32, "do_image_splitting": True},
    "tiny_square": {"resize_longest_edge": 64, "tile_edge": 64, "do_image_splitting": False},
    "granite_docling": {"resize_longest_edge": 2048, "tile_edge": 512, "do_image_splitting": True},
}
for iname, image in images.items():
    for cname, cfg in tiling_configs.items():
        if cname == "granite_docling" and iname != "rgb_97x61":
            continue  # full-resolution case is large; one image is enough
        tiles, rows, cols = tile_image(image, cfg["resize_longest_edge"], cfg["tile_edge"], cfg["do_image_splitting"])
        stacked = np.stack(
            [((np.asarray(t, dtype=np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1) for t in tiles], axis=0
        )
        file = f"tiling_{cname}_{iname}.npy"
        np.save(os.path.join(out, file), stacked)
        manifest["tiling"].append({**cfg, "config": cname, "image": iname, "rows": rows, "cols": cols, "expected": file})

with open(os.path.join(out, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)
print({k: len(v) for k, v in manifest.items()})
