use nalgebra::Matrix2;
use nalgebra_sparse::convert::serial::convert_dense_coo;
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use num_complex::Complex;

use crate::Qbit;

pub fn h_matrix() -> CsrMatrix<Qbit> {
    let root2 = 2.0_f64.sqrt();
    let x = Complex::ONE / root2;
    let hadamard_coo = convert_dense_coo(&Matrix2::from_row_slice(&[x, x, x, -x]));
    CsrMatrix::from(&hadamard_coo)
}

pub fn rx_matrix(angle: f64) -> CsrMatrix<Qbit> {
    let angle = angle / 2.0;
    let cos = Complex::new(angle.cos(), 0.0);
    let sin = Complex::new(0.0, -(angle.sin()));

    let mut rx_coo = CooMatrix::new(2, 2);
    rx_coo.push(0, 0, cos);
    rx_coo.push(0, 1, sin);
    rx_coo.push(1, 0, sin);
    rx_coo.push(1, 1, cos);
    CsrMatrix::from(&rx_coo)
}

pub fn ry_matrix(angle: f64) -> CsrMatrix<Qbit> {
    let angle = angle / 2.0;
    let cos = Complex::new(angle.cos(), 0.0);
    let sin = Complex::new(angle.sin(), 0.0);

    let mut ry_coo = CooMatrix::new(2, 2);
    ry_coo.push(0, 0, cos);
    ry_coo.push(0, 1, -sin);
    ry_coo.push(1, 0, sin);
    ry_coo.push(1, 1, cos);
    CsrMatrix::from(&ry_coo)
}

pub fn rz_matrix(angle: f64) -> CsrMatrix<Qbit> {
    let mut rz_coo = CooMatrix::new(2, 2);
    rz_coo.push(0, 0, Complex::new(0.0, -angle / 2.0).exp());
    rz_coo.push(0, 1, Complex::ZERO);
    rz_coo.push(1, 0, Complex::ZERO);
    rz_coo.push(1, 1, Complex::new(0.0, angle / 2.0).exp());
    CsrMatrix::from(&rz_coo)
}

pub fn x_matrix() -> CsrMatrix<Qbit> {
    let mut x_coo = CooMatrix::new(2, 2);
    x_coo.push(0, 1, Complex::new(1.0, 0.0));
    x_coo.push(1, 0, Complex::new(1.0, 0.0));
    CsrMatrix::from(&x_coo)
}

pub fn z_matrix() -> CsrMatrix<Qbit> {
    let mut z_coo = CooMatrix::new(2, 2);
    z_coo.push(0, 0, Complex::new(1.0, 0.0));
    z_coo.push(1, 1, Complex::new(-1.0, 0.0));
    CsrMatrix::from(&z_coo)
}

pub fn s_matrix() -> CsrMatrix<Qbit> {
    let mut s_coo = CooMatrix::new(2, 2);
    s_coo.push(0, 0, Complex::new(1.0, 0.0));
    s_coo.push(1, 1, Complex::new(0.0, 1.0));
    CsrMatrix::from(&s_coo)
}

pub fn t_matrix() -> CsrMatrix<Qbit> {
    let mut t_coo = CooMatrix::new(2, 2);
    t_coo.push(0, 0, Complex::new(1.0, 0.0));
    t_coo.push(1, 1, Complex::from_polar(1.0, std::f64::consts::FRAC_PI_4));
    CsrMatrix::from(&t_coo)
}
