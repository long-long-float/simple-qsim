use nalgebra::Matrix2;
use nalgebra_sparse::convert::serial::convert_dense_coo;
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use num_complex::Complex;

use crate::Qbit;

pub fn build_hadamard_matrix() -> CsrMatrix<Qbit> {
    let root2 = 2.0_f64.sqrt();
    let one = Complex::new(1.0, 0.0);
    let hadamard_coo = convert_dense_coo(&Matrix2::from_row_slice(&[
        one / root2,
        one / root2,
        one / root2,
        -one / root2,
    ]));
    CsrMatrix::from(&hadamard_coo)
}

pub fn build_x_matrix() -> CsrMatrix<Qbit> {
    let mut x_coo = CooMatrix::new(2, 2);
    x_coo.push(0, 1, Complex::new(1.0, 0.0));
    x_coo.push(1, 0, Complex::new(1.0, 0.0));
    CsrMatrix::from(&x_coo)
}

pub fn build_z_matrix() -> CsrMatrix<Qbit> {
    let mut z_coo = CooMatrix::new(2, 2);
    z_coo.push(0, 0, Complex::new(1.0, 0.0));
    z_coo.push(1, 1, Complex::new(-1.0, 0.0));
    CsrMatrix::from(&z_coo)
}

pub fn build_s_matrix() -> CsrMatrix<Qbit> {
    let mut s_coo = CooMatrix::new(2, 2);
    s_coo.push(0, 0, Complex::new(1.0, 0.0));
    s_coo.push(1, 1, Complex::new(0.0, 1.0));
    CsrMatrix::from(&s_coo)
}

pub fn build_t_matrix() -> CsrMatrix<Qbit> {
    let mut t_coo = CooMatrix::new(2, 2);
    t_coo.push(0, 0, Complex::new(1.0, 0.0));
    t_coo.push(1, 1, Complex::from_polar(1.0, std::f64::consts::FRAC_PI_4));
    CsrMatrix::from(&t_coo)
}
