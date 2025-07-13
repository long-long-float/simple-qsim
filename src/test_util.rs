#[macro_export]
macro_rules! assert_approx_complex_eq {
    ($expected_re:expr, $expected_im:expr, $actual:expr) => {{
        use num_complex::Complex;
        #[inline(always)]
        pub fn approx_eq(
            expected_re: f64,
            expected_im: f64,
            actual: Complex<f64>,
            eps: f64,
        ) -> bool {
            (expected_re - actual.re).abs() < eps && (expected_im - actual.im).abs() < eps
        }

        assert!(
            // TODO: Consider relative error
            approx_eq($expected_re, $expected_im, $actual, 1e-10),
            "Expected {}+{}i,  but got {}",
            $expected_re,
            $expected_im,
            $actual
        );
    }};
}

#[macro_export]
macro_rules! assert_approx_eq {
    ($expected:expr, $actual:expr) => {{
        #[inline(always)]
        pub fn approx_eq(expected: f64, actual: f64, eps: f64) -> bool {
            (expected - actual).abs() < eps
        }

        assert!(
            approx_eq($expected, $actual, 1e-10),
            "Expected {}, but got {}",
            $expected,
            $actual
        );
    }};
}

#[macro_export]
macro_rules! assert_approx_matrix2_eq {
    ($expected:expr, $actual:expr) => {{
        #[inline(always)]
        pub fn approx_eq(
            expected: &nalgebra::Matrix2<Qbit>,
            actual: &nalgebra::Matrix2<Qbit>,
            eps: f64,
        ) -> bool {
            expected
                .iter()
                .zip(actual.iter())
                .all(|(e, a)| (e.re - a.re).abs() < eps && (e.im - a.im).abs() < eps)
        }

        assert!(
            approx_eq(&$expected, &$actual, 1e-10),
            "Expected {:?}, but got {:?}",
            $expected,
            $actual
        );
    }};
}
