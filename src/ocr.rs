//! OCR postprocessing, as in PaddleOCR (and transformers' PP-OCR processors):
//! text boxes from a DB (differentiable binarization) probability map,
//! reading order, text-line crops for the recognizer and greedy CTC
//! decoding.
//!
//! The geometry reproduces the OpenCV functions PaddleOCR calls:
//! `findContours` (Suzuki-Abe border following, `RETR_LIST`),
//! `minAreaRect` + `boxPoints`, `fillPoly` (for box scores),
//! `getPerspectiveTransform` + `warpPerspective` (bicubic, replicated
//! border, OpenCV's fixed-point weights).
use ::image::{Rgb, RgbImage};
use ndarray::{Array2, ArrayView2};

/// (x, y).
pub type Point = [f32; 2];
/// A text box: top-left, top-right, bottom-right, bottom-left.
pub type Quad = [Point; 4];

// ---------------------------------------------------------------- contours

/// Offsets (row, column) of the 8 neighbors, clockwise from east (y down).
const NEIGHBORS: [(isize, isize); 8] = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)];

fn direction(from: (isize, isize), to: (isize, isize)) -> usize {
    let d = (to.0 - from.0, to.1 - from.1);
    NEIGHBORS.iter().position(|&n| n == d).expect("neighbor")
}

/// Every border of the foreground (`true`) regions, outer borders and hole
/// borders alike, as OpenCV's `findContours(mask, RETR_LIST,
/// CHAIN_APPROX_SIMPLE)` returns them: same contours, same order, same
/// points ((x, y) pixel coordinates; straight runs reduced to their ends).
/// Pixels outside the image count as background.
pub fn find_contours(mask: ArrayView2<bool>) -> Vec<Vec<[i32; 2]>> {
    let (h, w) = mask.dim();
    let (rows, cols) = (h + 2, w + 2);
    // 0 background, 1 unvisited foreground, nbd visited, -nbd visited with
    // background to the right (Suzuki & Abe's labels).
    let mut f = vec![0i32; rows * cols];
    for ((y, x), &v) in mask.indexed_iter() {
        f[(y + 1) * cols + x + 1] = v as i32;
    }
    let at = |p: (isize, isize)| p.0 as usize * cols + p.1 as usize;
    let mut contours = Vec::new();
    let mut nbd = 1;
    // OpenCV's scan: a border starts where a pixel differs from the one
    // before it (as that one is after any border through it was traced).
    for i in 1..rows as isize - 1 {
        let mut prev = 0;
        for j in 1..cols as isize {
            let mut p = f[at((i, j))];
            if prev == 0 && p == 1 {
                nbd += 1;
                contours.push(follow_border(&mut f, cols, (i, j), (i, j - 1), nbd)); // outer border
                p = f[at((i, j))];
            } else if p == 0 && prev >= 1 {
                nbd += 1;
                contours.push(follow_border(&mut f, cols, (i, j - 1), (i, j), nbd)); // hole border
            }
            prev = p;
        }
    }
    // OpenCV links each new contour in front of the previous ones.
    contours.reverse();
    contours
}

/// Traces one border from `start`, whose background neighbor is `from`,
/// labeling it `nbd`; returns the CHAIN_APPROX_SIMPLE points.
fn follow_border(f: &mut [i32], cols: usize, start: (isize, isize), from: (isize, isize), nbd: i32) -> Vec<[i32; 2]> {
    let at = |p: (isize, isize)| p.0 as usize * cols + p.1 as usize;
    let step = |p: (isize, isize), d: usize| (p.0 + NEIGHBORS[d].0, p.1 + NEIGHBORS[d].1);
    let point = |p: (isize, isize)| [p.1 as i32 - 1, p.0 as i32 - 1];
    // 3.1: clockwise from `from` for the first foreground neighbor.
    let d0 = direction(start, from);
    let Some(first) = (0..8).map(|k| (d0 + k) % 8).find(|&d| f[at(step(start, d))] != 0) else {
        f[at(start)] = -nbd;
        return vec![point(start)];
    };
    let p1 = step(start, first);
    let (mut p2, mut p3) = (p1, start);
    let mut points = Vec::new();
    // As if the border had arrived at the start from p1.
    let mut last_dir = Some((first + 4) % 8);
    loop {
        // 3.3: counterclockwise around p3, starting after p2.
        let d2 = direction(p3, p2);
        let mut east_empty = false;
        let mut next = None;
        for k in 1..=8 {
            let d = (d2 + 8 - k) % 8;
            let q = step(p3, d);
            if f[at(q)] != 0 {
                next = Some((q, d));
                break;
            }
            if d == 0 {
                east_empty = true;
            }
        }
        let (p4, d) = next.expect("the border has a next pixel");
        // 3.4
        if east_empty {
            f[at(p3)] = -nbd;
        } else if f[at(p3)] == 1 {
            f[at(p3)] = nbd;
        }
        // Chain approximation: keep the pixels where the direction changes.
        if last_dir != Some(d) {
            points.push(point(p3));
        }
        last_dir = Some(d);
        // 3.5
        if p4 == start && p3 == p1 {
            break;
        }
        p2 = p3;
        p3 = p4;
    }
    points
}

// ---------------------------------------------------------------- rotated rectangles

/// A rectangle of any rotation: its corners (cyclic order) and side lengths.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotatedRect {
    pub corners: Quad,
    pub size: [f32; 2],
}

/// One monotone chain of OpenCV's Sklansky hull over points sorted by x
/// (`p(i)`), from `start` to `end`: indices into the sorted order.
fn sklansky(p: &dyn Fn(isize) -> [f64; 2], integer: bool, start: isize, end: isize, nsign: i32, sign2: i32) -> Vec<isize> {
    let sign = |v: f64| (v > 0.0) as i32 - (v < 0.0) as i32;
    let incr = if end > start { 1 } else { -1 };
    if start == end || p(start) == p(end) {
        return vec![start];
    }
    let (mut pprev, mut pcur) = (start, start + incr);
    let mut pnext = pcur + incr;
    let mut stack = vec![pprev, pcur, pnext];
    let mut size = 3usize;
    let set = |stack: &mut Vec<isize>, i: usize, v: isize| {
        if i >= stack.len() {
            stack.resize(i + 1, 0);
        }
        stack[i] = v;
    };
    let end = end + incr;
    while pnext != end {
        let (cury, nexty) = (p(pcur)[1], p(pnext)[1]);
        let by = nexty - cury;
        if sign(by) != nsign {
            let mut a = [p(pcur)[0] - p(pprev)[0], cury - p(pprev)[1]];
            let mut b = [p(pnext)[0] - p(pcur)[0], by];
            if !integer {
                // cv::normalize on float vectors.
                let unit = |v: [f64; 2]| {
                    let n = (v[0] * v[0] + v[1] * v[1]).sqrt();
                    let s = if n != 0.0 { 1.0 / n } else { 0.0 };
                    [(v[0] * s) as f32 as f64, (v[1] * s) as f32 as f64]
                };
                (a, b) = (unit(a), unit(b));
            }
            let convexity = a[1] * b[0] - a[0] * b[1];
            if sign(convexity) == sign2 && (a[0] != 0.0 || a[1] != 0.0) {
                pprev = pcur;
                pcur = pnext;
                pnext += incr;
                set(&mut stack, size, pnext);
                size += 1;
            } else if pprev == start {
                pcur = pnext;
                stack[1] = pcur;
                pnext += incr;
                set(&mut stack, 2, pnext);
            } else {
                stack[size - 2] = pnext;
                pcur = pprev;
                pprev = stack[size - 4];
                size -= 1;
            }
        } else {
            pnext += incr;
            stack[size - 1] = pnext;
        }
    }
    stack.truncate(size - 1);
    stack
}

/// OpenCV's `convexHull(points, clockwise = false)`: the vertices in its
/// order and orientation (which decide the rounding in `minAreaRect`).
/// `integer` selects its integer-point arithmetic.
fn convex_hull(points: &[[f64; 2]], integer: bool) -> Vec<[f64; 2]> {
    let total = points.len();
    if total == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..total).collect();
    order.sort_by(|&a, &b| points[a][0].total_cmp(&points[b][0]).then(points[a][1].total_cmp(&points[b][1])).then(a.cmp(&b)));
    let p = |i: isize| points[order[i as usize]];
    let (mut miny, mut maxy) = (0isize, 0isize);
    for i in 1..total as isize {
        if p(miny)[1] > p(i)[1] {
            miny = i;
        }
        if p(maxy)[1] < p(i)[1] {
            maxy = i;
        }
    }
    let last = total as isize - 1;
    let mut hull: Vec<isize> = Vec::new();
    if p(0) == p(last) {
        hull.push(0);
    } else {
        // Upper half, swapped for counterclockwise output.
        let tr = sklansky(&p, integer, 0, maxy, -1, 1);
        let tl = sklansky(&p, integer, last, maxy, -1, -1);
        hull.extend(&tl[..tl.len() - 1]);
        hull.extend(tr[1..].iter().rev());
        let stop = if tr.len() > 2 { tr[1] } else if tl.len() > 2 { tl[tl.len() - 2] } else { -1 };
        let mut bl = sklansky(&p, integer, 0, miny, 1, -1);
        let mut br = sklansky(&p, integer, last, miny, 1, 1);
        if stop >= 0 {
            let check = if bl.len() > 2 {
                bl[1]
            } else if bl.len() + br.len() > 2 {
                br[2 - bl.len()]
            } else {
                -1
            };
            if check == stop || (check >= 0 && p(check) == p(stop)) {
                // All points on one line: the lower part mirrors the upper one.
                bl.truncate(2);
                br.truncate(2);
            }
        }
        hull.extend(&bl[..bl.len() - 1]);
        hull.extend(br[1..].iter().rev());
    }
    // Input indices, cyclically shifted to ascend or descend if they can.
    let mut hull: Vec<usize> = hull.into_iter().map(|i| order[i as usize]).collect();
    let n = hull.len();
    if n >= 3 {
        let (mut min_i, mut max_i, mut lt) = (0, 0, 0);
        for i in 1..n {
            let idx = hull[i];
            lt += (hull[i - 1] < idx) as usize;
            if lt > 1 && lt + 2 <= i {
                break;
            }
            if idx < hull[min_i] {
                min_i = i;
            }
            if idx > hull[max_i] {
                max_i = i;
            }
        }
        let dist = min_i.abs_diff(max_i);
        if (dist == 1 || dist == n - 1) && (lt <= 1 || lt + 2 >= n) {
            let ascending = (max_i + 1) % n == min_i;
            let start = if ascending { min_i } else { max_i };
            let rotated: Vec<usize> = (0..n).map(|k| hull[(start + k) % n]).collect();
            if start > 0 && rotated.windows(2).all(|w| ascending == (w[0] < w[1])) {
                hull = rotated;
            }
        }
    }
    hull.into_iter().map(|i| points[i]).collect()
}

/// The smallest-area rectangle around the points: OpenCV's `minAreaRect`
/// (rotating calipers in float32, ties going to the last candidate) with the
/// corners of `boxPoints`, in its order.
pub fn min_area_rect(points: &[Point]) -> RotatedRect {
    let pts: Vec<[f64; 2]> = points.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
    // OpenCV takes the integer path for int32 points (contours, integer boxes).
    let integer = points.iter().all(|p| p[0].fract() == 0.0 && p[1].fract() == 0.0);
    let hull: Vec<[f32; 2]> = convex_hull(&pts, integer).iter().map(|p| [p[0] as f32, p[1] as f32]).collect();
    let (center, size, angle) = match hull.len() {
        0 => ([0.0, 0.0], [0.0, 0.0], -90.0),
        1 => (hull[0], [0.0, 0.0], -90.0),
        2 => {
            let center = [(hull[0][0] + hull[1][0]) * 0.5, (hull[0][1] + hull[1][1]) * 0.5];
            let (dx, dy) = ((hull[0][0] - hull[1][0]) as f64, (hull[0][1] - hull[1][1]) as f64);
            let len = (dx * dx + dy * dy).sqrt() as f32;
            let (mut size, mut angle) = ([0.0, len], -std::f64::consts::FRAC_PI_2);
            if dx == 0.0 {
                size = [len, 0.0];
            } else if dy < 0.0 {
                angle = dy.atan2(dx);
                size = [len, 0.0];
            } else if dy > 0.0 {
                angle = -dx.atan2(dy);
            }
            (center, size, (angle * 180.0 / std::f64::consts::PI) as f32)
        }
        _ => {
            let [corner, v1, v2] = rotating_calipers(&hull);
            let center = [corner[0] + (v1[0] + v2[0]) * 0.5, corner[1] + (v1[1] + v2[1]) * 0.5];
            let mut size = [
                (v2[0] as f64).mul_add(v2[0] as f64, v2[1] as f64 * v2[1] as f64).sqrt() as f32,
                (v1[0] as f64).mul_add(v1[0] as f64, v1[1] as f64 * v1[1] as f64).sqrt() as f32,
            ];
            let mut angle = -std::f64::consts::FRAC_PI_2;
            if v1[0] == 0.0 && v1[1] > 0.0 {
                size.swap(0, 1);
            } else {
                angle = -(v1[0] as f64).atan2(v1[1] as f64);
            }
            (center, size, (angle * 180.0 / std::f64::consts::PI) as f32)
        }
    };
    RotatedRect { corners: box_points(center, size, angle), size }
}

/// OpenCV's `rotatingCalipers(..., CALIPERS_MINAREARECT)` on a hull with
/// positive orientation: a corner and the two side vectors.
fn rotating_calipers(points: &[[f32; 2]]) -> [[f32; 2]; 3] {
    let n = points.len();
    let mut vect = vec![[0f32; 2]; n];
    let mut inv_len = vec![0f32; n];
    let (mut left, mut bottom, mut right, mut top) = (0, 0, 0, 0);
    let mut pt0 = points[0];
    let (mut left_x, mut right_x, mut top_y, mut bottom_y) = (pt0[0], pt0[0], pt0[1], pt0[1]);
    for i in 0..n {
        if pt0[0] < left_x {
            (left_x, left) = (pt0[0], i);
        }
        if pt0[0] > right_x {
            (right_x, right) = (pt0[0], i);
        }
        if pt0[1] > top_y {
            (top_y, top) = (pt0[1], i);
        }
        if pt0[1] < bottom_y {
            (bottom_y, bottom) = (pt0[1], i);
        }
        let pt = points[(i + 1) % n];
        let (dx, dy) = ((pt[0] - pt0[0]) as f64, (pt[1] - pt0[1]) as f64);
        vect[i] = [dx as f32, dy as f32];
        inv_len[i] = (1.0 / dx.mul_add(dx, dy * dy).sqrt()) as f32;
        pt0 = pt;
    }
    let (mut base_a, mut base_b);
    let mut seq = [bottom, right, top, left];
    let mut min_area = f32::MAX;
    let mut best = (0usize, 0f32, 0f32, 0f32, 0f32, 0usize);
    // firstVecIsRight: v1 rotated 90° clockwise points away from v2.
    let is_right = |v1: [f32; 2], v2: [f32; 2]| v1[1].mul_add(v2[0], -v1[0] * v2[1]) < 0.0;
    for _ in 0..n {
        let rot = [
            vect[seq[0]],
            [vect[seq[1]][1], -vect[seq[1]][0]],
            [-vect[seq[2]][0], -vect[seq[2]][1]],
            [-vect[seq[3]][1], vect[seq[3]][0]],
        ];
        let mut main = 0;
        for i in 1..4 {
            if is_right(rot[i], rot[main]) {
                main = i;
            }
        }
        let p = seq[main];
        let (lx, ly) = (vect[p][0] * inv_len[p], vect[p][1] * inv_len[p]);
        (base_a, base_b) = match main {
            0 => (lx, ly),
            1 => (ly, -lx),
            2 => (-lx, -ly),
            _ => (-ly, lx),
        };
        seq[main] = (seq[main] + 1) % n;
        let (dx, dy) = (points[seq[1]][0] - points[seq[3]][0], points[seq[1]][1] - points[seq[3]][1]);
        let width = dx.mul_add(base_a, dy * base_b);
        let (dx, dy) = (points[seq[2]][0] - points[seq[0]][0], points[seq[2]][1] - points[seq[0]][1]);
        let height = (-dx).mul_add(base_b, dy * base_a);
        let area = width * height;
        if area <= min_area {
            min_area = area;
            best = (seq[3], base_a, width, base_b, height, seq[0]);
        }
    }
    let (li, a1, width, b1, height, bi) = best;
    let (a2, b2) = (-b1, a1);
    let c1 = a1.mul_add(points[li][0], points[li][1] * b1);
    let c2 = a2.mul_add(points[bi][0], points[bi][1] * b2);
    let idet = 1.0 / a1.mul_add(b2, -(a2 * b1));
    let px = c1.mul_add(b2, -(c2 * b1)) * idet;
    let py = a1.mul_add(c2, -(a2 * c1)) * idet;
    [[px, py], [a1 * width, b1 * width], [a2 * height, b2 * height]]
}

/// OpenCV's `RotatedRect::points` (what `boxPoints` returns).
fn box_points(center: [f32; 2], size: [f32; 2], angle: f32) -> Quad {
    let rad = angle as f64 * std::f64::consts::PI / 180.0;
    let b = rad.cos() as f32 * 0.5;
    let a = rad.sin() as f32 * 0.5;
    let [cx, cy] = center;
    let [w, h] = size;
    let p0 = [(-b).mul_add(w, (-a).mul_add(h, cx)), (-a).mul_add(w, b.mul_add(h, cy))];
    let p1 = [(-b).mul_add(w, a.mul_add(h, cx)), (-a).mul_add(w, (-b).mul_add(h, cy))];
    [p0, p1, [2.0 * cx - p0[0], 2.0 * cy - p0[1]], [2.0 * cx - p1[0], 2.0 * cy - p1[1]]]
}

/// Four corners as PaddleOCR orders them (`get_mini_boxes`): sorted by x,
/// then top-left, top-right, bottom-right, bottom-left.
pub fn order_corners(corners: Quad) -> Quad {
    let mut p = corners.to_vec();
    p.sort_by(|a, b| a[0].total_cmp(&b[0])); // stable, like Python's sorted
    let (a, d) = if p[1][1] > p[0][1] { (0, 1) } else { (1, 0) };
    let (b, c) = if p[3][1] > p[2][1] { (2, 3) } else { (3, 2) };
    [p[a], p[b], p[c], p[d]]
}

/// `get_mini_boxes`: the ordered min-area rectangle and its shorter side.
fn mini_box(points: &[Point]) -> (Quad, f32) {
    let r = min_area_rect(points);
    (order_corners(r.corners), r.size[0].min(r.size[1]))
}

// ---------------------------------------------------------------- polygon filling

/// OpenCV's `clipLine`: clips the segment to a w x h image; false if it's outside.
fn clip_line(w: i64, h: i64, p1: &mut [i64; 2], p2: &mut [i64; 2]) -> bool {
    let (right, bottom) = (w - 1, h - 1);
    let code = |p: &[i64; 2]| (p[0] < 0) as i32 + (p[0] > right) as i32 * 2 + (p[1] < 0) as i32 * 4 + (p[1] > bottom) as i32 * 8;
    let (mut c1, mut c2) = (code(p1), code(p2));
    if (c1 & c2) == 0 && (c1 | c2) != 0 {
        // Casts truncate toward zero, like C's.
        if c1 & 12 != 0 {
            let a = if c1 < 8 { 0 } else { bottom };
            p1[0] += ((a - p1[1]) as f64 * (p2[0] - p1[0]) as f64 / (p2[1] - p1[1]) as f64) as i64;
            p1[1] = a;
            c1 = (p1[0] < 0) as i32 + (p1[0] > right) as i32 * 2;
        }
        if c2 & 12 != 0 {
            let a = if c2 < 8 { 0 } else { bottom };
            p2[0] += ((a - p2[1]) as f64 * (p2[0] - p1[0]) as f64 / (p2[1] - p1[1]) as f64) as i64;
            p2[1] = a;
            c2 = (p2[0] < 0) as i32 + (p2[0] > right) as i32 * 2;
        }
        if (c1 & c2) == 0 && (c1 | c2) != 0 {
            if c1 != 0 {
                let a = if c1 == 1 { 0 } else { right };
                p1[1] += ((a - p1[0]) as f64 * (p2[1] - p1[1]) as f64 / (p2[0] - p1[0]) as f64) as i64;
                p1[0] = a;
                c1 = 0;
            }
            if c2 != 0 {
                let a = if c2 == 1 { 0 } else { right };
                p2[1] += ((a - p2[0]) as f64 * (p2[1] - p1[1]) as f64 / (p2[0] - p1[0]) as f64) as i64;
                p2[0] = a;
                c2 = 0;
            }
        }
    }
    (c1 | c2) == 0
}

/// An 8-connected line drawn left to right, clipped to the image (OpenCV's
/// `Line` through `LineIterator(..., 8, leftToRight = true)`).
fn line8(w: i64, h: i64, mut p0: [i64; 2], mut p1: [i64; 2], mut plot: impl FnMut(i64, i64)) {
    let outside = |p: &[i64; 2]| !(0..w).contains(&p[0]) || !(0..h).contains(&p[1]);
    if (outside(&p0) || outside(&p1)) && !clip_line(w, h, &mut p0, &mut p1) {
        return;
    }
    let (mut dx, mut dy) = (p1[0] - p0[0], p1[1] - p0[1]);
    let mut start = p0;
    if dx < 0 {
        (dx, dy, start) = (-dx, -dy, p1);
    }
    let (mut step_x, mut step_y) = (1, 1);
    if dy < 0 {
        dy = -dy;
        step_y = -1;
    }
    let vertical = dy > dx;
    if vertical {
        std::mem::swap(&mut dx, &mut dy);
        std::mem::swap(&mut step_x, &mut step_y);
    }
    // The major axis steps every time, the minor one when the error is negative.
    let mut err = dx - 2 * dy;
    let [mut x, mut y] = start;
    for _ in 0..=dx {
        plot(x, y);
        let minor = err < 0;
        err += -2 * dy + if minor { 2 * dx } else { 0 };
        let (major_step, minor_step) = (step_x, if minor { step_y } else { 0 });
        if vertical {
            y += major_step;
            x += minor_step;
        } else {
            x += major_step;
            y += minor_step;
        }
    }
}

/// Sets the pixels of a polygon (integer vertices) like OpenCV's `fillPoly`
/// (8-connected, no shift): the edges are drawn, and the rows between them
/// filled from 16-bit fixed-point edge positions.
pub fn fill_poly(mask: &mut Array2<u8>, points: &[[i32; 2]], value: u8) {
    const SHIFT: u32 = 16;
    const ONE: i64 = 1 << SHIFT;
    let (h, w) = (mask.dim().0 as i64, mask.dim().1 as i64);
    let mut set = |x: i64, y: i64| {
        if (0..w).contains(&x) && (0..h).contains(&y) {
            mask[[y as usize, x as usize]] = value;
        }
    };
    struct Edge {
        y0: i64,
        y1: i64,
        x: i64,
        dx: i64,
    }
    // CollectPolyEdges
    let n = points.len();
    let mut edges = Vec::new();
    for i in 0..n {
        let (p0, p1) = (points[(i + n - 1) % n], points[i]);
        let (p0, p1) = ([p0[0] as i64, p0[1] as i64], [p1[0] as i64, p1[1] as i64]);
        line8(w, h, p0, p1, &mut set);
        // Edges reaching outside the image get their slope from the clipped line.
        let (mut t0, mut t1) = (p0, p1);
        let (mut c0y, mut c1y) = (p0[1], p1[1]);
        let outside = |p: &[i64; 2]| !(0..w).contains(&p[0]) || !(0..h).contains(&p[1]);
        if outside(&t0) || outside(&t1) {
            clip_line(w, h, &mut t0, &mut t1);
            if t0[1] != t1[1] {
                (c0y, c1y) = (t0[1], t1[1]);
            }
        }
        if p0[1] == p1[1] {
            continue;
        }
        let (c0x, c1x) = (t0[0] << SHIFT, t1[0] << SHIFT);
        let dx = (c1x - c0x) / (c1y - c0y); // truncates toward zero
        edges.push(if p0[1] < p1[1] {
            Edge { y0: p0[1], y1: p1[1], x: c0x + (p0[1] - c0y) * dx, dx }
        } else {
            Edge { y0: p1[1], y1: p0[1], x: c1x + (p1[1] - c1y) * dx, dx }
        });
    }
    // FillEdgeCollection: rows between pairs of active edges (sorted by x),
    // from the left edge rounded up to the right one rounded down.
    if edges.len() < 2 {
        return;
    }
    let ymin = edges.iter().map(|e| e.y0).min().unwrap();
    let ymax = edges.iter().map(|e| e.y1).max().unwrap().min(h);
    for y in ymin.max(0)..ymax {
        let mut xs: Vec<i64> = edges.iter().filter(|e| e.y0 <= y && y < e.y1).map(|e| e.x + (y - e.y0) * e.dx).collect();
        xs.sort();
        for pair in xs.chunks(2) {
            if let [a, b] = *pair {
                let (x1, x2) = ((a + ONE - 1) >> SHIFT, b >> SHIFT);
                for x in x1.max(0)..=x2.min(w - 1) {
                    set(x, y);
                }
            }
        }
    }
}

// ---------------------------------------------------------------- DB boxes

/// Text boxes from a DB probability map, like transformers'
/// `post_process_object_detection` for PP-OCR detectors (PaddleOCR's
/// `DBPostProcess` with the "fast" box score and quad boxes). Defaults are
/// transformers'; PaddleOCR models ship their own in inference.yml.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DbDecoder {
    /// Pixels above this probability are text.
    pub threshold: f32,
    /// Boxes whose mean probability is below this are dropped.
    pub box_threshold: f32,
    /// At most this many contours are considered.
    pub max_candidates: usize,
    /// How far boxes grow (area * ratio / perimeter).
    pub unclip_ratio: f32,
    /// Boxes with a shorter side below this (map pixels) are dropped.
    pub min_size: f32,
}

impl Default for DbDecoder {
    fn default() -> Self {
        DbDecoder { threshold: 0.3, box_threshold: 0.6, max_candidates: 1000, unclip_ratio: 1.5, min_size: 3.0 }
    }
}

/// Boxes on the original image (integer corners) and their scores.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TextBoxes {
    pub boxes: Vec<Quad>,
    pub scores: Vec<f32>,
}

impl DbDecoder {
    /// `prob` is the map [H, W] of an image resized from `width` x `height`.
    pub fn decode(&self, prob: ArrayView2<f32>, width: u32, height: u32) -> TextBoxes {
        let (h, w) = prob.dim();
        let bitmap = prob.mapv(|p| p > self.threshold);
        let (sx, sy) = (width as f32 / w as f32, height as f32 / h as f32);
        let mut out = TextBoxes::default();
        for contour in find_contours(bitmap.view()).into_iter().take(self.max_candidates) {
            let points: Vec<Point> = contour.iter().map(|p| [p[0] as f32, p[1] as f32]).collect();
            let (quad, short) = mini_box(&points);
            if short < self.min_size {
                continue;
            }
            let score = box_score(prob, &quad);
            if self.box_threshold > score {
                continue;
            }
            let (quad, short) = mini_box(&unclip(&quad, self.unclip_ratio));
            if short < self.min_size + 2.0 {
                continue;
            }
            // In float32, like numpy.
            let scale = |v: f32, s: f32, max: u32| (v * s).round_ties_even().clamp(0.0, max as f32) as i16 as f32;
            out.boxes.push(quad.map(|p| [scale(p[0], sx, width), scale(p[1], sy, height)]));
            out.scores.push(score);
        }
        out
    }
}

/// Mean probability inside the quad (PaddleOCR's `box_score_fast`).
fn box_score(prob: ArrayView2<f32>, quad: &Quad) -> f32 {
    let (h, w) = prob.dim();
    let clamp = |v: f32, max: usize| (v as i64).clamp(0, max as i64 - 1) as usize;
    let xs = quad.map(|p| p[0]);
    let ys = quad.map(|p| p[1]);
    let min = |v: [f32; 4]| v.into_iter().fold(f32::INFINITY, f32::min);
    let max = |v: [f32; 4]| v.into_iter().fold(f32::NEG_INFINITY, f32::max);
    let (xmin, xmax) = (clamp(min(xs).floor(), w), clamp(max(xs).ceil(), w));
    let (ymin, ymax) = (clamp(min(ys).floor(), h), clamp(max(ys).ceil(), h));
    let mut mask = Array2::<u8>::zeros((ymax - ymin + 1, xmax - xmin + 1));
    // astype(np.int32) truncates toward zero.
    let pts: Vec<[i32; 2]> = quad.iter().map(|p| [(p[0] - xmin as f32) as i32, (p[1] - ymin as f32) as i32]).collect();
    fill_poly(&mut mask, &pts, 1);
    let (mut sum, mut count) = (0f64, 0usize);
    for ((y, x), &m) in mask.indexed_iter() {
        if m != 0 {
            sum += prob[[ymin + y, xmin + x]] as f64;
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { (sum / count as f64) as f32 }
}

/// Grows a polygon by area * ratio / perimeter: each edge moves outward
/// along its normal and the new vertices are where neighboring edges meet
/// (transformers' `_unclip`; PaddleOCR uses pyclipper, which rounds the
/// corners but gives the same min-area rectangle for rectangles).
fn unclip(polygon: &Quad, ratio: f32) -> Vec<Point> {
    let p: Vec<[f64; 2]> = polygon.iter().map(|q| [q[0] as f64, q[1] as f64]).collect();
    let n = p.len();
    let next = |i: usize| (i + 1) % n;
    let prev = |i: usize| (i + n - 1) % n;
    let perimeter: f64 = (0..n).map(|i| ((p[next(i)][0] - p[i][0]).powi(2) + (p[next(i)][1] - p[i][1]).powi(2)).sqrt()).sum();
    let area = (0..n).map(|i| p[i][0] * p[next(i)][1] - p[next(i)][0] * p[i][1]).sum::<f64>().abs() / 2.0;
    let distance = area * ratio as f64 / perimeter;
    let ccw = (0..n).map(|i| p[i][0] * p[next(i)][1] - p[i][1] * p[next(i)][0]).sum::<f64>() > 0.0;
    let dirs: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let e = [p[next(i)][0] - p[i][0], p[next(i)][1] - p[i][1]];
            let len = (e[0] * e[0] + e[1] * e[1]).sqrt().max(1e-6);
            [e[0] / len, e[1] / len]
        })
        .collect();
    let normals: Vec<[f64; 2]> = dirs.iter().map(|d| if ccw { [d[1], -d[0]] } else { [-d[1], d[0]] }).collect();
    let shifted: Vec<[f64; 2]> = (0..n).map(|i| [p[i][0] + distance * normals[i][0], p[i][1] + distance * normals[i][1]]).collect();
    (0..n)
        .map(|i| {
            let (d0, d1) = (dirs[prev(i)], dirs[i]);
            let c = d0[0] * d1[1] - d0[1] * d1[0];
            let v = if c.abs() < 1e-6 {
                let (m0, m1) = (normals[prev(i)], normals[i]);
                [p[i][0] + 0.5 * distance * (m0[0] + m1[0]), p[i][1] + 0.5 * distance * (m0[1] + m1[1])]
            } else {
                let s = shifted[prev(i)];
                let to = [shifted[i][0] - s[0], shifted[i][1] - s[1]];
                let t = (to[0] * d1[1] - to[1] * d1[0]) / c;
                [s[0] + d0[0] * t, s[1] + d0[1] * t]
            };
            [v[0] as f32, v[1] as f32]
        })
        .collect()
}

// ---------------------------------------------------------------- reading order and crops

/// Reading order of boxes (PaddleOCR's `sort_quad_boxes`): by the top-left
/// corner's y, then x, and boxes whose top-left corners are less than 10
/// pixels apart vertically go left to right.
pub fn sort_boxes(boxes: &[Quad]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| boxes[a][0][1].total_cmp(&boxes[b][0][1]).then(boxes[a][0][0].total_cmp(&boxes[b][0][0])));
    for i in 0..order.len().saturating_sub(1) {
        for j in (0..=i).rev() {
            let (a, b) = (boxes[order[j]][0], boxes[order[j + 1]][0]);
            if (b[1] - a[1]).abs() < 10.0 && b[0] < a[0] {
                order.swap(j, j + 1);
            } else {
                break;
            }
        }
    }
    order
}

/// The text line in a box, straightened, like PaddleOCR's
/// `get_minarea_rect_crop`: the min-area rectangle of the box (integer
/// corners), warped upright (bicubic, replicated border), and rotated 90°
/// counterclockwise when it's at least 1.5 times taller than wide.
pub fn crop_text_line(image: &RgbImage, quad: &Quad) -> RgbImage {
    let corners = quad.map(|p| [p[0] as i32 as f32, p[1] as i32 as f32]);
    let q = order_corners(min_area_rect(&corners).corners);
    let dist = |a: Point, b: Point| (((a[0] - b[0]) as f64).powi(2) + ((a[1] - b[1]) as f64).powi(2)).sqrt() as f32;
    let width = dist(q[0], q[1]).max(dist(q[2], q[3])) as u32;
    let height = dist(q[0], q[3]).max(dist(q[1], q[2])) as u32;
    let (w, h) = (width as f64, height as f64);
    let target = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
    let source = q.map(|p| [p[0] as f64, p[1] as f64]);
    let m = invert3(&perspective_transform(&source, &target));
    let crop = warp_perspective_cubic(image, &m, width, height);
    if height as f64 / width as f64 >= 1.5 { ::image::imageops::rotate270(&crop) } else { crop }
}

/// The homography mapping the four `src` points onto `dst` (OpenCV's
/// `getPerspectiveTransform`), row-major 3x3.
fn perspective_transform(src: &[[f64; 2]; 4], dst: &[[f64; 2]; 4]) -> [f64; 9] {
    let mut a = [[0f64; 9]; 8]; // augmented [A | b]
    for i in 0..4 {
        let ([x, y], [u, v]) = (src[i], dst[i]);
        a[i] = [x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u, u];
        a[i + 4] = [0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v, v];
    }
    // OpenCV's LUImpl: elimination with partial pivoting, then back substitution.
    for i in 0..8 {
        let mut k = i;
        for j in i + 1..8 {
            if a[j][i].abs() > a[k][i].abs() {
                k = j;
            }
        }
        if a[k][i].abs() < f32::EPSILON as f64 * 10.0 {
            return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        }
        a.swap(i, k);
        let d = -1.0 / a[i][i];
        let pivot = a[i];
        for row in a.iter_mut().skip(i + 1) {
            let alpha = row[i] * d;
            for (v, p) in row.iter_mut().zip(pivot).skip(i + 1) {
                *v += alpha * p;
            }
        }
    }
    let mut x = [0f64; 8];
    for i in (0..8).rev() {
        let mut s = a[i][8];
        for k in i + 1..8 {
            s -= a[i][k] * x[k];
        }
        x[i] = s / a[i][i];
    }
    [x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], 1.0]
}

fn invert3(m: &[f64; 9]) -> [f64; 9] {
    let [a, b, c, d, e, f, g, h, i] = *m;
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    let s = if det != 0.0 { 1.0 / det } else { 0.0 };
    [
        (e * i - f * h) * s,
        (c * h - b * i) * s,
        (b * f - c * e) * s,
        (f * g - d * i) * s,
        (a * i - c * g) * s,
        (c * d - a * f) * s,
        (d * h - e * g) * s,
        (b * g - a * h) * s,
        (a * e - b * d) * s,
    ]
}

const INTER_BITS: i64 = 5;
const INTER_TAB_SIZE: i64 = 1 << INTER_BITS;
const COEF_BITS: u32 = 15;

/// OpenCV's fixed-point bicubic weights (A = -0.75) for each of the 32 x 32
/// sub-pixel positions, 4 x 4 each, summing to exactly 2^15.
fn cubic_table() -> Vec<[i32; 16]> {
    fn cubic(x: f32) -> [f32; 4] {
        const A: f32 = -0.75;
        let c0 = ((A * (x + 1.0) - 5.0 * A) * (x + 1.0) + 8.0 * A) * (x + 1.0) - 4.0 * A;
        let c1 = ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
        let c2 = ((A + 2.0) * (1.0 - x) - (A + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
        [c0, c1, c2, 1.0 - c0 - c1 - c2]
    }
    let scale = (1 << COEF_BITS) as f32;
    let mut table = Vec::with_capacity((INTER_TAB_SIZE * INTER_TAB_SIZE) as usize);
    for i in 0..INTER_TAB_SIZE {
        let ty = cubic(i as f32 / INTER_TAB_SIZE as f32);
        for j in 0..INTER_TAB_SIZE {
            let tx = cubic(j as f32 / INTER_TAB_SIZE as f32);
            let mut w = [0i32; 16];
            let mut sum = 0;
            for k1 in 0..4 {
                for k2 in 0..4 {
                    w[k1 * 4 + k2] = (ty[k1] * tx[k2] * scale).round_ties_even() as i32;
                    sum += w[k1 * 4 + k2];
                }
            }
            let diff = sum - (1 << COEF_BITS);
            if diff != 0 {
                // Like OpenCV: fix up the largest (or smallest) of the central four.
                let (mut lo, mut hi) = ((2, 2), (2, 2));
                for k1 in 2..4 {
                    for k2 in 2..4 {
                        if w[k1 * 4 + k2] < w[lo.0 * 4 + lo.1] {
                            lo = (k1, k2);
                        } else if w[k1 * 4 + k2] > w[hi.0 * 4 + hi.1] {
                            hi = (k1, k2);
                        }
                    }
                }
                if diff < 0 {
                    w[hi.0 * 4 + hi.1] -= diff;
                } else {
                    w[lo.0 * 4 + lo.1] -= diff;
                }
            }
            table.push(w);
        }
    }
    table
}

/// `warpPerspective(img, M^-1, size, INTER_CUBIC, BORDER_REPLICATE)` where
/// `m` maps output pixels to source pixels. Source positions are computed
/// like OpenCV's (32 x 32 blocks, fused multiply-adds as on ARM), rounded to
/// 1/32 pixel.
fn warp_perspective_cubic(image: &RgbImage, m: &[f64; 9], width: u32, height: u32) -> RgbImage {
    let table = cubic_table();
    let (iw, ih) = (image.width() as i64, image.height() as i64);
    let (w, h) = (width as usize, height as usize);
    let mut out = RgbImage::new(width, height);
    if w == 0 || h == 0 {
        return out;
    }
    const BLOCK: usize = 32;
    let bh0 = (BLOCK / 2).min(h);
    let bw0 = (BLOCK * BLOCK / bh0).min(w);
    let bh0 = (BLOCK * BLOCK / bw0).min(h);
    for by in (0..h).step_by(bh0) {
        for bx in (0..w).step_by(bw0) {
            let (bw, bh) = (bw0.min(w - bx), bh0.min(h - by));
            for y in by..by + bh {
                let base = |a: f64, b: f64, c: f64| b.mul_add(y as f64, a * bx as f64) + c;
                let (x0, y0, w0) = (base(m[0], m[1], m[2]), base(m[3], m[4], m[5]), base(m[6], m[7], m[8]));
                for x1 in 0..bw {
                    let xf = x1 as f64;
                    let wt = m[6].mul_add(xf, w0);
                    let wt = if wt != 0.0 { INTER_TAB_SIZE as f64 / wt } else { 0.0 };
                    let fx = (m[0].mul_add(xf, x0) * wt).clamp(i32::MIN as f64, i32::MAX as f64);
                    let fy = (m[3].mul_add(xf, y0) * wt).clamp(i32::MIN as f64, i32::MAX as f64);
                    let (sx, sy) = (fx.round_ties_even() as i64, fy.round_ties_even() as i64);
                    let weights = &table[((sy & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE + (sx & (INTER_TAB_SIZE - 1))) as usize];
                    let (ox, oy) = ((sx >> INTER_BITS) - 1, (sy >> INTER_BITS) - 1);
                    let mut acc = [0i64; 3];
                    for k1 in 0..4 {
                        let py = (oy + k1 as i64).clamp(0, ih - 1) as u32;
                        for k2 in 0..4 {
                            let px = (ox + k2 as i64).clamp(0, iw - 1) as u32;
                            let p = image.get_pixel(px, py);
                            let wt = weights[k1 * 4 + k2] as i64;
                            for c in 0..3 {
                                acc[c] += p[c] as i64 * wt;
                            }
                        }
                    }
                    let px = acc.map(|v| ((v + (1 << (COEF_BITS - 1))) >> COEF_BITS).clamp(0, 255) as u8);
                    out.put_pixel((bx + x1) as u32, y as u32, Rgb(px));
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------- CTC

/// Greedy CTC decoding of per-step probabilities [T, classes] (class 0 is
/// the blank): the best class per step, repeats merged, blanks dropped.
/// Returns the classes and their mean probability (0 when empty; transformers
/// gives NaN, PaddleOCR 0).
pub fn ctc_greedy(probs: ArrayView2<f32>) -> (Vec<usize>, f32) {
    let mut ids = Vec::new();
    let mut total = 0f32;
    let mut previous = None;
    for row in probs.rows() {
        let (best, p) = row.iter().enumerate().fold((0, f32::NEG_INFINITY), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc });
        if best != 0 && previous != Some(best) {
            ids.push(best);
            total += p;
        }
        previous = Some(best);
    }
    let score = if ids.is_empty() { 0.0 } else { total / ids.len() as f32 };
    (ids, score)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn ctc_merges_repeats_and_drops_blanks() {
        // Steps: a a blank a b b blank -> "a a b" (a repeated after a blank counts again).
        let probs = array![
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.9, 0.05, 0.05],
            [0.3, 0.6, 0.1],
            [0.1, 0.1, 0.8],
            [0.1, 0.2, 0.7],
            [0.6, 0.2, 0.2]
        ];
        let (ids, score) = ctc_greedy(probs.view());
        assert_eq!(ids, vec![1, 1, 2]);
        assert!((score - (0.8 + 0.6 + 0.8) / 3.0).abs() < 1e-6);
        assert_eq!(ctc_greedy(array![[0.9f32, 0.1]].view()), (vec![], 0.0));
    }

    #[test]
    fn a_ring_has_an_outer_and_a_hole_border() {
        let mut mask = Array2::from_elem((5, 5), true);
        mask[[2, 2]] = false;
        let contours = find_contours(mask.view());
        assert_eq!(contours.len(), 2);
        // Newest first: the hole's border, then the outer square's corners.
        assert_eq!(contours[1], vec![[0, 0], [0, 4], [4, 4], [4, 0]]);
        let r = min_area_rect(&contours[1].iter().map(|p| [p[0] as f32, p[1] as f32]).collect::<Vec<_>>());
        let want = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
        for (g, w) in order_corners(r.corners).iter().zip(want) {
            assert!((g[0] - w[0]).abs() < 1e-5 && (g[1] - w[1]).abs() < 1e-5, "{:?}", r.corners);
        }
    }
}
