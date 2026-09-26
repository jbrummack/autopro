//! Times DBSCAN on synthetic patch embeddings: `cargo run --release --example cluster_bench [n] [dim]`.
//! Compare BLAS with `--no-default-features` (ndarray's matrixmultiply).
use autopro::cluster::{Dbscan, Metric};
use ndarray::Array2;

fn main() {
    let mut args = std::env::args().skip(1).map(|a| a.parse::<usize>().unwrap());
    let (n, dim) = (args.next().unwrap_or(4096), args.next().unwrap_or(768));
    // A few prototypes plus deterministic noise.
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut noise = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let prototypes = Array2::from_shape_fn((8, dim), |_| noise() * 4.0);
    let x = Array2::from_shape_fn((n, dim), |(i, j)| prototypes[[(i / 7) % 8, j]] + noise());
    for metric in [Metric::Cosine, Metric::Euclidean] {
        let eps = if metric == Metric::Cosine { 0.2 } else { 12.0 };
        let dbscan = Dbscan { eps, min_samples: 5, metric };
        dbscan.fit(x.view());
        let start = std::time::Instant::now();
        let c = dbscan.fit(x.view());
        println!("{n}x{dim} {metric:?}: {} clusters in {:?}", c.n_clusters, start.elapsed());
    }
}

