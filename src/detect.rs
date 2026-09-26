//! Object detection postprocessing: box formats, IoU, non-maximum suppression
//! (torchvision's `nms` / `batched_nms` semantics) and decoding of detector
//! heads into detections (Ultralytics' `non_max_suppression`).
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

/// How a box's four numbers are laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoxFormat {
    /// Corners: x1, y1, x2, y2.
    Xyxy,
    /// Top-left corner and size: x, y, w, h.
    Xywh,
    /// Center and size: cx, cy, w, h (YOLO, DETR).
    Cxcywh,
}

impl BoxFormat {
    pub fn parse(s: &str) -> Option<BoxFormat> {
        match s {
            "xyxy" => Some(BoxFormat::Xyxy),
            "xywh" => Some(BoxFormat::Xywh),
            "cxcywh" => Some(BoxFormat::Cxcywh),
            _ => None,
        }
    }
}

/// Converts boxes [n, 4] between formats (torchvision `box_convert`).
pub fn convert(boxes: ArrayView2<f32>, from: BoxFormat, to: BoxFormat) -> Array2<f32> {
    assert_eq!(boxes.ncols(), 4, "boxes must be [n, 4]");
    let mut out = boxes.to_owned();
    if from == to {
        return out;
    }
    for mut b in out.rows_mut() {
        let [x1, y1, x2, y2] = match from {
            BoxFormat::Xyxy => [b[0], b[1], b[2], b[3]],
            BoxFormat::Xywh => [b[0], b[1], b[0] + b[2], b[1] + b[3]],
            BoxFormat::Cxcywh => [b[0] - 0.5 * b[2], b[1] - 0.5 * b[3], b[0] + 0.5 * b[2], b[1] + 0.5 * b[3]],
        };
        let v = match to {
            BoxFormat::Xyxy => [x1, y1, x2, y2],
            BoxFormat::Xywh => [x1, y1, x2 - x1, y2 - y1],
            BoxFormat::Cxcywh => [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1],
        };
        b.assign(&ArrayView1::from(&v));
    }
    out
}

fn area(b: ArrayView1<f32>) -> f32 {
    (b[2] - b[0]) * (b[3] - b[1])
}

fn intersection(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    w * h
}

/// Pairwise IoU of xyxy boxes a [n, 4] and b [m, 4]: [n, m] (torchvision `box_iou`).
pub fn iou(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
    Array2::from_shape_fn((a.nrows(), b.nrows()), |(i, j)| {
        let inter = intersection(a.row(i), b.row(j));
        inter / (area(a.row(i)) + area(b.row(j)) - inter)
    })
}

/// Indices by descending score; ties keep index order.
fn by_score(scores: ArrayView1<f32>) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&i, &j| scores[j].total_cmp(&scores[i]));
    order
}

/// Greedy NMS over xyxy boxes [n, 4]: keeps the best-scoring box and drops
/// every later one whose IoU with it exceeds `iou_threshold`, per class if
/// `classes` is given (boxes of different classes never suppress each other).
/// Returns kept indices by descending score, like torchvision's `nms` and
/// `batched_nms`.
pub fn nms(boxes: ArrayView2<f32>, scores: ArrayView1<f32>, classes: Option<&[usize]>, iou_threshold: f32) -> Vec<usize> {
    assert_eq!(boxes.ncols(), 4, "boxes must be [n, 4]");
    assert_eq!(boxes.nrows(), scores.len(), "one score per box");
    if let Some(c) = classes {
        assert_eq!(c.len(), scores.len(), "one class per box");
    }
    let order = by_score(scores);
    let areas: Vec<f32> = boxes.rows().into_iter().map(area).collect();
    let mut suppressed = vec![false; order.len()];
    let mut keep = Vec::new();
    for (n, &i) in order.iter().enumerate() {
        if suppressed[i] {
            continue;
        }
        keep.push(i);
        let bi = boxes.row(i);
        for &j in &order[n + 1..] {
            if suppressed[j] || classes.is_some_and(|c| c[i] != c[j]) {
                continue;
            }
            let inter = intersection(bi, boxes.row(j));
            if inter / (areas[i] + areas[j] - inter) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }
    keep
}

/// Rows of `x` at `indices`.
pub fn take_rows<T: Clone>(x: ArrayView2<T>, indices: &[usize]) -> Array2<T> {
    x.select(Axis(0), indices)
}

/// Detections in xyxy pixel coordinates, best first.
#[derive(Debug, Clone, PartialEq)]
pub struct Detections {
    /// [n, 4] xyxy.
    pub boxes: Array2<f32>,
    pub scores: Array1<f32>,
    pub classes: Vec<usize>,
    /// Row of each detection in the decoder's input (e.g. to pick mask coefficients).
    pub indices: Vec<usize>,
}

impl Detections {
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }
}

/// Turns a detector head's candidates into detections, like Ultralytics'
/// `non_max_suppression` (YOLOv5/v8/11): per candidate the best class (or
/// every class above the threshold with `multi_label`), a score threshold,
/// at most `max_candidates` best into class-wise NMS, at most `max_detections` out.
#[derive(Debug, Clone)]
pub struct Detector {
    /// Layout of the candidate boxes.
    pub format: BoxFormat,
    /// Keep class scores strictly above this.
    pub score_threshold: f32,
    pub iou_threshold: f32,
    /// NMS across all classes instead of per class.
    pub class_agnostic: bool,
    /// One detection per (box, class) above the threshold instead of per box.
    pub multi_label: bool,
    pub max_candidates: usize,
    pub max_detections: usize,
}

impl Default for Detector {
    fn default() -> Self {
        Detector {
            format: BoxFormat::Cxcywh,
            score_threshold: 0.25,
            iou_threshold: 0.45,
            class_agnostic: false,
            multi_label: false,
            max_candidates: 30000,
            max_detections: 300,
        }
    }
}

impl Detector {
    /// `boxes` [n, 4] in `self.format`, `class_scores` [n, classes]
    /// (probabilities, e.g. after a sigmoid). For YOLOv8-style outputs
    /// [4 + classes, n], transpose and split first.
    pub fn detect(&self, boxes: ArrayView2<f32>, class_scores: ArrayView2<f32>) -> Detections {
        assert_eq!(boxes.nrows(), class_scores.nrows(), "one row of class scores per box");
        let boxes = convert(boxes, self.format, BoxFormat::Xyxy);
        // Candidates as (row, class, score), in row order like Ultralytics.
        let mut candidates: Vec<(usize, usize, f32)> = Vec::new();
        for (row, scores) in class_scores.rows().into_iter().enumerate() {
            if self.multi_label {
                candidates.extend(
                    scores.iter().enumerate().filter(|(_, s)| **s > self.score_threshold).map(|(c, &s)| (row, c, s)),
                );
            } else if let Some((c, &s)) =
                scores.iter().enumerate().fold(None, |best: Option<(usize, &f32)>, (c, s)| match best {
                    Some((_, b)) if *s <= *b => best,
                    _ => Some((c, s)),
                })
                && s > self.score_threshold
            {
                candidates.push((row, c, s));
            }
        }
        if candidates.len() > self.max_candidates {
            let scores = Array1::from_iter(candidates.iter().map(|c| c.2));
            let order = by_score(scores.view());
            candidates = order[..self.max_candidates].iter().map(|&i| candidates[i]).collect();
        }
        let rows: Vec<usize> = candidates.iter().map(|c| c.0).collect();
        let classes: Vec<usize> = candidates.iter().map(|c| c.1).collect();
        let scores = Array1::from_iter(candidates.iter().map(|c| c.2));
        let cand_boxes = take_rows(boxes.view(), &rows);
        let mut keep =
            nms(cand_boxes.view(), scores.view(), (!self.class_agnostic).then_some(&classes[..]), self.iou_threshold);
        keep.truncate(self.max_detections);
        Detections {
            boxes: take_rows(cand_boxes.view(), &keep),
            scores: keep.iter().map(|&i| scores[i]).collect(),
            classes: keep.iter().map(|&i| classes[i]).collect(),
            indices: keep.iter().map(|&i| rows[i]).collect(),
        }
    }
}

/// Clamps xyxy boxes to an image of `width` x `height`.
pub fn clip(boxes: &mut Array2<f32>, width: f32, height: f32) {
    for mut b in boxes.rows_mut() {
        b[0] = b[0].clamp(0.0, width);
        b[1] = b[1].clamp(0.0, height);
        b[2] = b[2].clamp(0.0, width);
        b[3] = b[3].clamp(0.0, height);
    }
}

/// Multiplies x coordinates by `sx` and y by `sy` (any format), e.g. to map
/// normalized DETR boxes to pixels or undo a plain resize.
pub fn scale(boxes: ArrayView2<f32>, sx: f32, sy: f32) -> Array2<f32> {
    let factors = ndarray::arr1(&[sx, sy, sx, sy]);
    &boxes * &factors
}

/// Maps xyxy boxes from a letterboxed image of `model` size (w, h) back to the
/// `original` image (w, h) and clips them, like Ultralytics' `scale_boxes`
/// (the inverse of [`crate::image::letterbox`]).
pub fn unletterbox(boxes: ArrayView2<f32>, model: (u32, u32), original: (u32, u32)) -> Array2<f32> {
    let (mw, mh) = (model.0 as f64, model.1 as f64);
    let (ow, oh) = (original.0 as f64, original.1 as f64);
    let gain = (mh / oh).min(mw / ow);
    let pad_x = ((mw - ow * gain) / 2.0 - 0.1).round_ties_even();
    let pad_y = ((mh - oh * gain) / 2.0 - 0.1).round_ties_even();
    let mut out = boxes.to_owned();
    for mut b in out.rows_mut() {
        for (i, pad) in [pad_x, pad_y, pad_x, pad_y].into_iter().enumerate() {
            b[i] = ((b[i] as f64 - pad) / gain) as f32;
        }
    }
    clip(&mut out, ow as f32, oh as f32);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn formats_round_trip() {
        let xyxy = array![[10.0f32, 20.0, 50.0, 80.0]];
        let cxcywh = convert(xyxy.view(), BoxFormat::Xyxy, BoxFormat::Cxcywh);
        assert_eq!(cxcywh, array![[30.0, 50.0, 40.0, 60.0]]);
        assert_eq!(convert(xyxy.view(), BoxFormat::Xyxy, BoxFormat::Xywh), array![[10.0, 20.0, 40.0, 60.0]]);
        assert_eq!(convert(cxcywh.view(), BoxFormat::Cxcywh, BoxFormat::Xyxy), xyxy);
    }

    #[test]
    fn nms_suppresses_overlaps_per_class() {
        let boxes = array![[0.0f32, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [20.0, 20.0, 30.0, 30.0], [0.0, 0.0, 10.0, 9.0]];
        let scores = array![0.9f32, 0.8, 0.7, 0.95];
        assert_eq!(nms(boxes.view(), scores.view(), None, 0.5), [3, 2]);
        assert_eq!(nms(boxes.view(), scores.view(), Some(&[0, 0, 0, 1]), 0.5), [3, 0, 2]);
        // IoU exactly at the threshold is kept (only IoU > threshold suppresses).
        let iou03 = iou(boxes.slice(ndarray::s![0..1, ..]), boxes.slice(ndarray::s![3..4, ..]))[[0, 0]];
        assert_eq!(nms(boxes.view(), scores.view(), None, iou03), [3, 0, 1, 2]);
    }

    #[test]
    fn detector_picks_best_class_and_thresholds() {
        let boxes = array![[5.0f32, 5.0, 10.0, 10.0], [5.5, 5.0, 10.0, 10.0], [50.0, 50.0, 10.0, 10.0]];
        let scores = array![[0.1f32, 0.9], [0.8, 0.3], [0.2, 0.1]];
        let d = Detector::default().detect(boxes.view(), scores.view());
        assert_eq!((d.indices.clone(), d.classes.clone()), (vec![0, 1], vec![1, 0]));
        assert_eq!(d.boxes.row(0).to_vec(), [0.0, 0.0, 10.0, 10.0]);
        let d = Detector { class_agnostic: true, ..Detector::default() }.detect(boxes.view(), scores.view());
        assert_eq!(d.indices, [0]);
        let d = Detector { multi_label: true, iou_threshold: 0.99, ..Detector::default() }.detect(boxes.view(), scores.view());
        assert_eq!(d.classes, [1, 0, 1]);
    }

    #[test]
    fn unletterbox_inverts_letterbox_geometry() {
        // 200x100 letterboxed into 64x64: gain 0.32, 32 rows of padding split 16/16.
        let model = array![[0.0f32, 16.0, 64.0, 48.0]];
        let back = unletterbox(model.view(), (64, 64), (200, 100));
        assert_eq!(back, array![[0.0, 0.0, 200.0, 100.0]]);
    }
}
