//! Matrix products on BLAS where it's available (see the crate features).
use ndarray::{Array2, ArrayView2};

/// `a · bᵀ` for row-major `a` [m, k] and `b` [n, k]: pairwise dot products of rows.
pub fn dot_rows(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
    assert_eq!(a.ncols(), b.ncols(), "dot_rows: rows differ in length");
    #[cfg(all(feature = "platform-blas", target_vendor = "apple"))]
    {
        accelerate::dot_rows(a, b)
    }
    #[cfg(not(all(feature = "platform-blas", target_vendor = "apple")))]
    {
        a.dot(&b.t())
    }
}

#[cfg(all(feature = "platform-blas", target_vendor = "apple"))]
mod accelerate {
    use ndarray::{Array2, ArrayView2};

    const ROW_MAJOR: i32 = 101;
    const NO_TRANS: i32 = 111;
    const TRANS: i32 = 112;

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        #[allow(clippy::too_many_arguments)]
        fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }

    pub fn dot_rows(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        let (a, b) = (a.as_standard_layout(), b.as_standard_layout());
        let (m, k, n) = (a.nrows(), a.ncols(), b.nrows());
        let mut c = Array2::<f32>::zeros((m, n));
        if m == 0 || n == 0 || k == 0 {
            return c;
        }
        let dim = |v: usize| i32::try_from(v).expect("matrix too large for BLAS");
        unsafe {
            cblas_sgemm(
                ROW_MAJOR,
                NO_TRANS,
                TRANS,
                dim(m),
                dim(n),
                dim(k),
                1.0,
                a.as_ptr(),
                dim(k),
                b.as_ptr(),
                dim(k),
                0.0,
                c.as_mut_ptr(),
                dim(n),
            )
        };
        c
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    #[test]
    fn dot_rows_is_a_times_b_transposed() {
        let a = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[1.0f32, 0.0, 1.0], [0.0, 1.0, 0.0], [2.0, 2.0, 2.0]];
        assert_eq!(super::dot_rows(a.view(), b.view()), a.dot(&b.t()));
        // Non-contiguous views work too.
        assert_eq!(super::dot_rows(a.t().t(), b.slice(ndarray::s![..;2, ..])), a.dot(&b.slice(ndarray::s![..;2, ..]).t()));
    }
}
