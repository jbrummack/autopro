"""Generates autopro's postprocessing fixtures with torchvision and scikit-learn.

    cd reference && nix develop .#post --command python make_post_fixtures.py ../tests/fixtures/post

Writes inputs and expected outputs (.npy) plus manifest.json, which
tests/post_reference.rs reads.
"""

import json
import os
import sys

import numpy as np
import torch
import torchvision
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
manifest = {"nms": [], "detector": [], "dbscan": []}


def save(name, array):
    np.save(os.path.join(out, name), np.ascontiguousarray(array))
    return name


def random_boxes(rng, n, size=640.0):
    """xyxy boxes, clustered so that many overlap."""
    centers = rng.uniform(0, size, size=(max(n // 8, 1), 2))
    c = centers[rng.integers(0, len(centers), n)] + rng.normal(0, 12, size=(n, 2))
    wh = rng.uniform(8, 120, size=(n, 2))
    return np.concatenate([c - wh / 2, c + wh / 2], axis=1).astype(np.float32)


# ---------------------------------------------------------------- NMS

rng = np.random.default_rng(0)
for n, n_classes in [(50, 1), (300, 1), (300, 5), (1500, 20)]:
    boxes = random_boxes(rng, n)
    # Distinct scores: torch's sort order for ties is unspecified.
    scores = rng.permutation(n).astype(np.float32) / n + np.float32(0.5 / n)
    classes = rng.integers(0, n_classes, n)
    name = f"nms_{n}_{n_classes}"
    save(f"{name}_boxes.npy", boxes)
    save(f"{name}_scores.npy", scores)
    save(f"{name}_classes.npy", classes.astype(np.int64))
    for iou in [0.3, 0.5, 0.7]:
        b, s = torch.from_numpy(boxes), torch.from_numpy(scores)
        if n_classes == 1:
            keep = torchvision.ops.nms(b, s, iou)
        else:
            keep = torchvision.ops.batched_nms(b, s, torch.from_numpy(classes), iou)
        manifest["nms"].append({
            "name": name, "iou": iou, "per_class": n_classes > 1, "keep": keep.tolist(),
        })


# ---------------------------------------------------------------- detector heads
# Ultralytics' non_max_suppression (ultralytics/utils/nms.py, detection only),
# with torchvision's nms as the kernel.

def xywh2xyxy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ultralytics_nms(prediction, conf_thres, iou_thres, agnostic, multi_label, max_det, max_nms=30000, max_wh=7680):
    nc = prediction.shape[1] - 4
    xc = prediction[:, 4:].amax(1) > conf_thres
    prediction = prediction.transpose(-1, -2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = []
    for xi, x in enumerate(prediction):
        rows = torch.nonzero(xc[xi]).flatten()
        x = x[xc[xi]]
        box, cls = x.split((4, nc), 1)
        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
            rows = rows[i]
        else:
            conf, j = cls.max(1, keepdim=True)
            keep = conf.view(-1) > conf_thres
            x = torch.cat((box, conf, j.float()), 1)[keep]
            rows = rows[keep]
        if x.shape[0] > max_nms:
            order = x[:, 4].argsort(descending=True)[:max_nms]
            x, rows = x[order], rows[order]
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        i = torchvision.ops.nms(x[:, :4] + c, x[:, 4], iou_thres)[:max_det]
        output.append((x[i], rows[i]))
    return output


rng = np.random.default_rng(1)
for n, nc in [(400, 3), (2000, 10)]:
    boxes = random_boxes(rng, n)
    cxcywh = np.concatenate([(boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]], axis=1)
    # Sparse, distinct class probabilities.
    logits = rng.normal(-3, 2, size=(n, nc))
    probs = (1 / (1 + np.exp(-logits))).astype(np.float32)
    head = np.concatenate([cxcywh, probs], axis=1).T[None].astype(np.float32)  # [1, 4 + nc, n], YOLOv8 layout
    name = f"yolo_{n}_{nc}"
    save(f"{name}.npy", head[0])
    for conf, iou, agnostic, multi, max_det in [
        (0.25, 0.45, False, False, 300), (0.1, 0.6, False, True, 300), (0.25, 0.45, True, False, 20),
    ]:
        (det, rows), = ultralytics_nms(torch.from_numpy(head.copy()), conf, iou, agnostic, multi, max_det)
        manifest["detector"].append({
            "head": f"{name}.npy", "score_threshold": conf, "iou_threshold": iou, "class_agnostic": agnostic,
            "multi_label": multi, "max_detections": max_det,
            "boxes": det[:, :4].tolist(), "scores": det[:, 4].tolist(),
            "classes": det[:, 5].long().tolist(), "indices": rows.tolist(),
        })


# ---------------------------------------------------------------- DBSCAN

def dbscan_case(name, x, eps, min_samples, metric):
    fitted = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm="brute").fit(x)
    manifest["dbscan"].append({
        "points": name, "eps": eps, "min_samples": min_samples, "metric": metric,
        "labels": fitted.labels_.tolist(), "core": fitted.core_sample_indices_.tolist(),
    })


blobs, _ = make_blobs(n_samples=600, centers=6, n_features=16, cluster_std=1.5, random_state=0)
blobs = blobs.astype(np.float32)
save("blobs.npy", blobs)
for eps, min_samples in [(6.0, 5), (7.0, 10), (5.5, 3)]:
    dbscan_case("blobs.npy", blobs, eps, min_samples, "euclidean")

# "Patch embeddings": 32x32 patches of 384-d tokens from a few prototypes
# (regions of an image) plus noise, like a ViT's last layer.
rng = np.random.default_rng(2)
prototypes = rng.normal(0, 1, size=(5, 384))
region = (np.add.outer(np.arange(32) // 11, np.arange(32) // 13) % 5).reshape(-1)
patches = (prototypes[region] * rng.uniform(0.5, 2.0, size=(1024, 1)) + rng.normal(0, 0.6, size=(1024, 384))).astype(np.float32)
save("patches.npy", patches)
for eps, min_samples in [(0.3, 8), (0.25, 4)]:
    dbscan_case("patches.npy", patches, eps, min_samples, "cosine")
for eps in [16.0, 17.0]:
    dbscan_case("patches.npy", patches, eps, 8, "euclidean")

with open(os.path.join(out, "manifest.json"), "w") as f:
    json.dump(manifest, f)
print({k: len(v) for k, v in manifest.items()})
