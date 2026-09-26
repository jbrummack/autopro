"""Generates autopro's OCR fixtures with OpenCV, torchvision and PaddleOCR's /
transformers' postprocessing code (copied below; transformers 5.17
PPOCRV5ServerDetImageProcessor, PaddleOCR 3.x CropByPolys / sort_quad_boxes).

    cd reference && nix develop .#post --command python make_ocr_fixtures.py ../tests/fixtures/ocr

Writes inputs and expected outputs (.npy) plus manifest.json, which
tests/ocr_reference.rs reads.
"""

import json
import math
import os
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms.v2.functional as tvF

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
manifest = {"contours": [], "min_area_rect": [], "fill_poly": [], "db": [], "crops": [], "resize": [], "sort": []}
rng = np.random.default_rng(0)


def save(name, array):
    np.save(os.path.join(out, name), np.ascontiguousarray(array))
    return name


def rotated_rect(cx, cy, w, h, angle):
    return cv2.boxPoints(((cx, cy), (w, h), angle))


def text_map(height, width, n, holes=True):
    """A soft probability map with n rotated 'text lines', some touching the border, some with holes."""
    prob = np.zeros((height, width), np.float32)
    for _ in range(n):
        w, h = rng.uniform(8, width * 0.6), rng.uniform(4, 30)
        cx, cy = rng.uniform(-10, width + 10), rng.uniform(-10, height + 10)
        angle = rng.choice([0.0, 0.0, rng.uniform(-45, 45), 90.0])
        pts = rotated_rect(cx, cy, w, h, angle).round().astype(np.int32)
        cv2.fillPoly(prob, [pts], float(rng.uniform(0.4, 0.98)))
        if holes and rng.random() < 0.3:
            cv2.fillPoly(prob, [rotated_rect(cx, cy, w * 0.3, h * 0.3, angle).round().astype(np.int32)], 0.0)
    prob = cv2.GaussianBlur(prob, (5, 5), 1.2)
    prob += rng.normal(0, 0.03, prob.shape).astype(np.float32)
    return np.clip(prob, 0, 1).astype(np.float32)


# ---------------------------------------------------------------- contours

for i, (h, w, n) in enumerate([(40, 50, 6), (120, 160, 20), (300, 240, 60), (64, 64, 0)]):
    if n:
        bitmap = text_map(h, w, n) > 0.3
    else:  # single pixels, thin lines, a ring, touching the border
        bitmap = np.zeros((h, w), bool)
        bitmap[rng.integers(0, h, 30), rng.integers(0, w, 30)] = True
        bitmap[10, 5:40] = True
        bitmap[5:50, 50] = True
        cv2.circle(bitmap.view(np.uint8), (30, 35), 12, 1, 2)
        bitmap[0:3, :] = True
    name = save(f"contours_{i}.npy", bitmap.astype(np.uint8))
    contours, _ = cv2.findContours(bitmap.astype(np.uint8) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    manifest["contours"].append({
        "bitmap": name, "contours": [c.reshape(-1, 2).tolist() for c in contours],
        # minAreaRect of each contour: its float32 rounding depends on the hull's order.
        "boxes": [cv2.boxPoints(cv2.minAreaRect(c)).tolist() for c in contours],
    })

# ---------------------------------------------------------------- minAreaRect

for i in range(200):
    n = int(rng.integers(3, 40))
    if i % 2:
        pts = rng.integers(0, 500, size=(n, 2)).astype(np.float32)
    else:
        rect = rotated_rect(*rng.uniform(50, 450, 2), *rng.uniform(2, 200, 2), rng.uniform(-90, 90))
        t = rng.uniform(0, 1, size=(n, 1))
        k = rng.integers(0, 4, n)
        pts = (rect[k] * (1 - t) + rect[(k + 1) % 4] * t).astype(np.float32)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    manifest["min_area_rect"].append({"points": pts.tolist(), "corners": cv2.boxPoints(((cx, cy), (w, h), angle)).tolist(),
                                      "short": float(min(w, h))})

# ---------------------------------------------------------------- fillPoly

for i in range(60):
    mask = np.zeros((48, 64), np.uint8)
    if i % 3 == 0:
        pts = rng.integers(-5, 70, size=(4, 2)).astype(np.int32)
    else:
        pts = rotated_rect(*rng.uniform(10, 50, 2), *rng.uniform(1, 40, 2), rng.uniform(-90, 90)).astype(np.int32)
    cv2.fillPoly(mask, pts.reshape(1, -1, 2), 1)
    manifest["fill_poly"].append({"points": pts.tolist(), "mask": save(f"fill_{i}.npy", mask)})

# ---------------------------------------------------------------- DB postprocessing (transformers 5.17)


def unclip(contour_box, unclip_ratio):
    polygon = contour_box.reshape(-1, 2).astype(np.float32)
    perimeter = cv2.arcLength(polygon, True)
    area = cv2.contourArea(polygon)
    offset_distance = area * unclip_ratio / perimeter
    x, y = polygon[:, 0], polygon[:, 1]
    is_counter_clockwise = (x @ np.roll(y, -1) - y @ np.roll(x, -1)) > 0.0
    edges = np.roll(polygon, -1, axis=0) - polygon
    edge_lengths = np.linalg.norm(edges, axis=1, keepdims=True)
    edge_directions = edges / np.maximum(edge_lengths, 1e-6)
    if is_counter_clockwise:
        normals = np.stack([edge_directions[:, 1], -edge_directions[:, 0]], axis=1)
    else:
        normals = np.stack([-edge_directions[:, 1], edge_directions[:, 0]], axis=1)
    shifted_points = polygon + offset_distance * normals
    prev_shifted_points = np.roll(shifted_points, 1, axis=0)
    prev_edge_directions = np.roll(edge_directions, 1, axis=0)
    cross_product = prev_edge_directions[:, 0] * edge_directions[:, 1] - prev_edge_directions[:, 1] * edge_directions[:, 0]
    is_parallel_mask = np.abs(cross_product) < 1e-6
    cross_product_safe = np.where(is_parallel_mask, 1.0, cross_product)
    vec_to_current = shifted_points - prev_shifted_points
    intersection_param = (vec_to_current[:, 0] * edge_directions[:, 1] - vec_to_current[:, 1] * edge_directions[:, 0]) / cross_product_safe
    new_vertices = prev_shifted_points + prev_edge_directions * intersection_param[:, None]
    if np.any(is_parallel_mask):
        prev_normals = np.roll(normals, 1, axis=0)
        fallback_points = polygon + 0.5 * offset_distance * (prev_normals + normals)
        new_vertices[is_parallel_mask] = fallback_points[is_parallel_mask]
    return np.array([new_vertices.astype(np.float32)])


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(cv2.boxPoints(bounding_box), key=lambda x: x[0])
    i1, i4 = (0, 1) if points[1][1] > points[0][1] else (1, 0)
    i2, i3 = (2, 3) if points[3][1] > points[2][1] else (3, 2)
    return [points[i1], points[i2], points[i3], points[i4]], min(bounding_box[1])


def get_box_score(bitmap, polygon_bounding_box):
    height, width = bitmap.shape[:2]
    box = polygon_bounding_box.copy()
    xmin = max(0, min(math.floor(box[:, 0].min()), width - 1))
    xmax = max(0, min(math.ceil(box[:, 0].max()), width - 1))
    ymin = max(0, min(math.floor(box[:, 1].min()), height - 1))
    ymax = max(0, min(math.ceil(box[:, 1].max()), height - 1))
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]


def boxes_from_bitmap(prediction, bitmap, dest_width, dest_height, box_threshold, unclip_ratio, min_size, max_candidates):
    height, width = bitmap.shape
    width_scale = dest_width / width
    height_scale = dest_height / height
    outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = outs[1] if len(outs) == 3 else outs[0]
    boxes, scores = [], []
    for index in range(min(len(contours), max_candidates)):
        points, short_side_length = get_mini_boxes(contours[index])
        if short_side_length < min_size:
            continue
        points = np.array(points)
        score = get_box_score(prediction, points.reshape(-1, 2))
        if box_threshold > score:
            continue
        box = unclip(points, unclip_ratio).reshape(-1, 1, 2)
        box, short_side_length = get_mini_boxes(box)
        if short_side_length < min_size + 2:
            continue
        box = np.array(box)
        for i in range(box.shape[0]):
            box[i, 0] = max(0, min(round(box[i, 0] * width_scale), dest_width))
            box[i, 1] = max(0, min(round(box[i, 1] * height_scale), dest_height))
        boxes.append(box.astype(np.int16))
        scores.append(score)
    return np.array(boxes, dtype=np.int16), scores


db_cases = [(96, 128, 10, (130, 100)), (320, 256, 50, (250, 311)), (640, 480, 120, (1224, 1584)), (160, 160, 25, (160, 160))]
for i, (h, w, n, (dw, dh)) in enumerate(db_cases):
    prob = text_map(h, w, n)
    for thresh, box_thresh, unclip_ratio in [(0.3, 0.6, 1.5), (0.2, 0.45, 1.4)]:
        # As transformers passes them: the original size as float32.
        dest_w, dest_h = np.float32(dw), np.float32(dh)
        boxes, scores = boxes_from_bitmap(prob, prob > thresh, dest_w, dest_h, box_thresh, unclip_ratio, 3, 1000)
        manifest["db"].append({
            "prob": save(f"db_{i}.npy", prob), "width": dw, "height": dh, "threshold": thresh,
            "box_threshold": box_thresh, "unclip_ratio": unclip_ratio,
            "boxes": boxes.tolist(), "scores": [float(s) for s in scores],
        })

# ---------------------------------------------------------------- PaddleOCR reading order and crops


def sort_quad_boxes(boxes):
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][0][1], boxes[i][0][0]))
    for i in range(len(order) - 1):
        for j in range(i, -1, -1):
            a, b = boxes[order[j]], boxes[order[j + 1]]
            if abs(b[0][1] - a[0][1]) < 10 and b[0][0] < a[0][0]:
                order[j], order[j + 1] = order[j + 1], order[j]
            else:
                break
    return order


def get_rotate_crop_image(img, points):
    img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if dst_img.shape[0] * 1.0 / dst_img.shape[1] >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def get_minarea_rect_crop(img, points):
    bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
    a, d = (0, 1) if points[1][1] > points[0][1] else (1, 0)
    b, c = (2, 3) if points[3][1] > points[2][1] else (3, 2)
    return get_rotate_crop_image(img, np.array([points[a], points[b], points[c], points[d]]))


yy, xx = np.mgrid[0:240, 0:320]
image = np.stack([(xx * 0.8 + 20 * np.sin(yy / 7.0)), (yy + 30 * np.cos(xx / 11.0)), (xx + yy) * 0.5], axis=-1)
image = np.clip(image + rng.normal(0, 25, image.shape), 0, 255).astype(np.uint8)
save("crop_image.npy", image)
quads = []
for i in range(40):
    w, h = rng.uniform(5, 200), rng.uniform(5, 60)
    if i % 4 == 3:
        w, h = h, w * 1.2  # tall: rotated after cropping
    angle = 0.0 if i % 5 == 0 else rng.uniform(-40, 40)
    quad = rotated_rect(rng.uniform(0, 320), rng.uniform(0, 240), w, h, angle)[[1, 2, 3, 0]].round()
    quad[:, 0] = quad[:, 0].clip(0, 320)
    quad[:, 1] = quad[:, 1].clip(0, 240)
    quads.append(quad.astype(np.float32))
    crop = np.ascontiguousarray(get_minarea_rect_crop(image, quad))
    manifest["crops"].append({"quad": quad.tolist(), "crop": save(f"crop_{i}.npy", crop)})
manifest["sort"].append({"quads": [q.tolist() for q in quads], "order": sort_quad_boxes([q for q in quads])})

# ---------------------------------------------------------------- resize (torchvision, antialias=False)

for (w, h) in [(33, 48), (320, 48), (97, 61), (700, 48), (12, 30)]:
    resized = tvF.resize(torch.from_numpy(image).permute(2, 0, 1), [h, w], interpolation=tvF.InterpolationMode.BILINEAR, antialias=False)
    manifest["resize"].append({"width": w, "height": h, "image": save(f"resize_{w}x{h}.npy", resized.permute(1, 2, 0).numpy())})

json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"))
print({k: len(v) for k, v in manifest.items()}, "cv2", cv2.__version__, "torch", torch.__version__)
