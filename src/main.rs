use nalgebra::{Matrix2, Vector2};
use nalgebra_sparse::convert::serial::{convert_csr_dense, convert_dense_coo};
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use num_complex::Complex;

fn tensor_product(
    x: &CooMatrix<Complex<f64>>,
    y: &CooMatrix<Complex<f64>>,
) -> CooMatrix<Complex<f64>> {
    let mut result = CooMatrix::new(x.nrows() * y.nrows(), x.ncols() * y.ncols());

    for (rx, cx, value_x) in x.triplet_iter() {
        for (ry, cy, value_y) in y.triplet_iter() {
            let new_row = rx * y.nrows() + ry;
            let new_col = cx * y.ncols() + cy;
            let new_value = value_x * value_y;
            result.push(new_row, new_col, new_value);
        }
    }

    result
}

fn main() {
    let root2 = 2.0_f64.sqrt();
    let zero = Complex::new(0.0, 0.0);
    let one = Complex::new(1.0, 0.0);
    let i_one = Complex::new(0.0, 1.0);

    let hadamard_coo = convert_dense_coo(&Matrix2::from_row_slice(&[
        one / root2,
        one / root2,
        one / root2,
        -one / root2,
    ]));
    let hadamard = CsrMatrix::from(&hadamard_coo);

    let mut identity_coo = CooMatrix::new(2, 2);
    identity_coo.push(0, 0, one);
    identity_coo.push(1, 1, one);

    // H (x) I
    let hadamard0 = tensor_product(&hadamard_coo, &identity_coo);
    let hadamard0 = CsrMatrix::from(&hadamard0);

    let mut x_coo = CooMatrix::new(2, 2);
    x_coo.push(0, 1, one);
    x_coo.push(1, 0, one);

    // |0><0|
    let mut zero_zero = CooMatrix::new(2, 2);
    zero_zero.push(0, 0, one);

    // |1><1|
    let mut one_one = CooMatrix::new(2, 2);
    one_one.push(1, 1, one);

    // CNOT(0, 1) = |0><0| (x) I + |1><1| (x) X
    let cnot01 = CsrMatrix::from(&tensor_product(&zero_zero, &identity_coo))
        + CsrMatrix::from(&tensor_product(&one_one, &x_coo));

    let ctrl_h01 = CsrMatrix::from(&tensor_product(&zero_zero, &identity_coo))
        + CsrMatrix::from(&tensor_product(&one_one, &hadamard_coo));

    // |0>
    let q0 = Vector2::from_column_slice(&[one, zero]);
    // |1>
    let q1 = Vector2::from_column_slice(&[zero, one]);
    // 1/sqrt(2) * (|0> + |1>)
    let q_plus = Vector2::from_column_slice(&[one / root2, one / root2]);
    // |i>
    let q_i = Vector2::from_column_slice(&[one / root2, i_one / root2]);

    // |00>
    let mut q00 = CooMatrix::new(4, 1);
    q00.push(0, 0, one);
    let q00 = CsrMatrix::from(&q00);

    let result1 = &hadamard * &q0;
    println!("{:}", result1);

    let result2 = &hadamard * &q1;
    println!("{:}", result2);

    let result3 = &hadamard * &q_plus;
    println!("{:}", result3);

    let result4 = &hadamard * &q_i;
    println!("{:}", result4);

    // Bell state |00> + |11>
    let result5 = &cnot01 * (&hadamard0 * &q00);
    println!("{:}", convert_csr_dense(&result5));

    // Hadamard test
    let result6 = &hadamard0 * (&ctrl_h01 * (&hadamard0 * &q00));
    println!("{:}", convert_csr_dense(&result6));
}
