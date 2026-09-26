//! Clustering of embeddings (e.g. a vision encoder's patch tokens): DBSCAN
//! with scikit-learn's semantics, radius neighbors on BLAS, cluster centroids.
use ndarray::{Array1, Array2, ArrayView2, Axis, s};

use crate::linalg::dot_rows;

/// Distance between rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    Euclidean,
    /// 1 - cosine similarity (zero vectors have similarity 0 to everything).
    Cosine,
}

impl Metric {
    pub fn parse(s: &str) -> Option<Metric> {
        match s {
            "euclidean" => Some(Metric::Euclidean),
            "cosine" => Some(Metric::Cosine),
            _ => None,
        }
    }
}

/// Upper bound for one block of the distance matrix (f32 elements).
const BLOCK_ELEMENTS: usize = 1 << 24;

/// For each row of `x` [n, d], the rows within distance `eps` (inclusive,
/// itself included), in ascending order.
///
/// Distances come from one matrix product per block of rows (BLAS): for
/// euclidean |a|² + |b|² - 2a·b, for cosine 1 - â·b̂. Pairs whose f32 result
/// lies close to `eps` are recomputed directly in f64, so the rounding of the
/// fast path never decides membership.
pub fn radius_neighbors(x: ArrayView2<f32>, eps: f64, metric: Metric) -> Vec<Vec<u32>> {
    let n = x.nrows();
    assert!(u32::try_from(n).is_ok(), "too many points");
    let x = match metric {
        Metric::Euclidean => x.as_standard_layout().into_owned(),
        Metric::Cosine => {
            let mut x = x.to_owned();
            for mut row in x.rows_mut() {
                let norm = row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
                if norm > 0.0 {
                    row.mapv_inplace(|v| (v as f64 / norm) as f32);
                }
            }
            x
        }
    };
    let sq_norms: Vec<f32> = x.rows().into_iter().map(|r| r.dot(&r)).collect();
    let eps64 = eps;
    // Exact distance, for pairs near the threshold.
    let exact = |i: usize, j: usize| -> f64 {
        let (a, b) = (x.row(i), x.row(j));
        match metric {
            Metric::Euclidean => a.iter().zip(b).map(|(p, q)| (*p as f64 - *q as f64).powi(2)).sum::<f64>().sqrt(),
            Metric::Cosine => 1.0 - a.iter().zip(b).map(|(p, q)| *p as f64 * *q as f64).sum::<f64>(),
        }
    };
    let block = (BLOCK_ELEMENTS / n.max(1)).clamp(1, n.max(1));
    let mut neighbors = Vec::with_capacity(n);
    for start in (0..n).step_by(block) {
        let end = (start + block).min(n);
        let dots = dot_rows(x.slice(s![start..end, ..]), x.view());
        for (r, row) in dots.rows().into_iter().enumerate() {
            let i = start + r;
            let mut found = Vec::new();
            for (j, &dot) in row.iter().enumerate() {
                if i == j {
                    found.push(j as u32);
                    continue;
                }
                // Approximate distance and a bound on its rounding error.
                let (approx, tolerance) = match metric {
                    Metric::Euclidean => {
                        let sq = (sq_norms[i] + sq_norms[j] - 2.0 * dot).max(0.0) as f64;
                        let err = 1e-5 * (sq_norms[i] + sq_norms[j]) as f64;
                        // Compare squared distances: |d² - eps²| within err.
                        (sq, err)
                    }
                    Metric::Cosine => (1.0 - dot as f64, 1e-5),
                };
                let threshold = match metric {
                    Metric::Euclidean => eps64 * eps64,
                    Metric::Cosine => eps64,
                };
                let within = if (approx - threshold).abs() <= tolerance { exact(i, j) <= eps64 } else { approx <= threshold };
                if within {
                    found.push(j as u32);
                }
            }
            neighbors.push(found);
        }
    }
    neighbors
}

/// DBSCAN like scikit-learn's `DBSCAN(eps, min_samples, metric)`: a point is
/// a core point if at least `min_samples` points (itself included) lie within
/// `eps`; clusters grow from core points in index order; points reachable from
/// no core point are noise (label -1).
#[derive(Debug, Clone)]
pub struct Dbscan {
    pub eps: f64,
    pub min_samples: usize,
    pub metric: Metric,
}

/// Result of a clustering.
#[derive(Debug, Clone, PartialEq)]
pub struct Clustering {
    /// Cluster per point, 0-based; -1 is noise.
    pub labels: Array1<i32>,
    /// Whether each point is a core point.
    pub core: Vec<bool>,
    pub n_clusters: usize,
}

impl Dbscan {
    pub fn new(eps: f64, min_samples: usize) -> Dbscan {
        Dbscan { eps, min_samples, metric: Metric::Euclidean }
    }

    /// Clusters the rows of `x` [n, d].
    pub fn fit(&self, x: ArrayView2<f32>) -> Clustering {
        let neighbors = radius_neighbors(x, self.eps, self.metric);
        let core: Vec<bool> = neighbors.iter().map(|nb| nb.len() >= self.min_samples).collect();
        let mut labels = Array1::from_elem(x.nrows(), -1i32);
        let mut label = 0;
        let mut stack = Vec::new();
        // sklearn's dbscan_inner: depth-first from each unlabeled core point.
        for start in 0..x.nrows() {
            if labels[start] != -1 || !core[start] {
                continue;
            }
            let mut i = start;
            loop {
                if labels[i] == -1 {
                    labels[i] = label;
                    if core[i] {
                        stack.extend(neighbors[i].iter().map(|&j| j as usize).filter(|&j| labels[j] == -1));
                    }
                }
                match stack.pop() {
                    Some(next) => i = next,
                    None => break,
                }
            }
            label += 1;
        }
        Clustering { labels, core, n_clusters: label as usize }
    }
}

/// Mean of the rows of `x` in each cluster: [n_clusters, d] (noise ignored).
pub fn centroids(x: ArrayView2<f32>, labels: &Array1<i32>, n_clusters: usize) -> Array2<f32> {
    let mut sums = Array2::<f64>::zeros((n_clusters, x.ncols()));
    let mut counts = vec![0usize; n_clusters];
    for (row, &label) in x.axis_iter(Axis(0)).zip(labels) {
        if label >= 0 && (label as usize) < n_clusters {
            let mut sum = sums.row_mut(label as usize);
            sum.zip_mut_with(&row, |s, v| *s += *v as f64);
            counts[label as usize] += 1;
        }
    }
    for (mut sum, count) in sums.rows_mut().into_iter().zip(counts) {
        if count > 0 {
            sum /= count as f64;
        }
    }
    sums.mapv(|v| v as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn two_blobs_and_noise() {
        let x = array![[0.0f32, 0.0], [0.1, 0.0], [0.0, 0.1], [5.0, 5.0], [5.1, 5.0], [5.0, 5.1], [9.0, 0.0]];
        let c = Dbscan::new(0.2, 3).fit(x.view());
        assert_eq!(c.labels.to_vec(), [0, 0, 0, 1, 1, 1, -1]);
        assert_eq!(c.n_clusters, 2);
        let means = centroids(x.view(), &c.labels, 2);
        assert!((means[[1, 0]] - 5.0333333).abs() < 1e-5);
    }

    #[test]
    fn chains_connect_and_min_samples_counts_the_point_itself() {
        let x = array![[0.0f32], [0.5], [1.0], [1.5], [2.0], [-0.5], [2.5]];
        let c = Dbscan::new(0.5, 3).fit(x.view());
        assert_eq!(c.labels.to_vec(), [0, 0, 0, 0, 0, 0, 0]);
        let c = Dbscan::new(0.5, 4).fit(x.view());
        assert_eq!(c.labels.to_vec(), [-1; 7]);
    }

    #[test]
    fn eps_is_inclusive_and_cosine_ignores_length() {
        let x = array![[0.0f32, 0.0], [3.0, 4.0]];
        assert_eq!(radius_neighbors(x.view(), 5.0, Metric::Euclidean), [vec![0, 1], vec![0, 1]]);
        assert_eq!(radius_neighbors(x.view(), 4.999, Metric::Euclidean), [vec![0], vec![1]]);
        let y = array![[1.0f32, 0.0], [10.0, 0.0], [0.0, 1.0]];
        let c = Dbscan { eps: 0.01, min_samples: 2, metric: Metric::Cosine }.fit(y.view());
        assert_eq!(c.labels.to_vec(), [0, 0, -1]);
    }
}
