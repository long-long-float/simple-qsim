use std::fmt::Display;

use anyhow::Result;
use nalgebra::{DVector, Matrix2, Vector2};
use nalgebra_sparse::convert::serial::{convert_csr_dense, convert_dense_coo};
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use num_complex::Complex;

fn tensor_product(
    x: &CsrMatrix<Complex<f64>>,
    y: &CsrMatrix<Complex<f64>>,
) -> CsrMatrix<Complex<f64>> {
    let mut result = CooMatrix::new(x.nrows() * y.nrows(), x.ncols() * y.ncols());

    for (rx, cx, value_x) in x.triplet_iter() {
        for (ry, cy, value_y) in y.triplet_iter() {
            let new_row = rx * y.nrows() + ry;
            let new_col = cx * y.ncols() + cy;
            let new_value = value_x * value_y;
            result.push(new_row, new_col, new_value);
        }
    }

    CsrMatrix::from(&result)
}

fn build_hadamard_matrix() -> CsrMatrix<Complex<f64>> {
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

fn build_x_matrix() -> CsrMatrix<Complex<f64>> {
    let mut x_coo = CooMatrix::new(2, 2);
    x_coo.push(0, 1, Complex::new(1.0, 0.0));
    x_coo.push(1, 0, Complex::new(1.0, 0.0));
    CsrMatrix::from(&x_coo)
}

struct QState {
    state: DVector<Complex<f64>>,
}

impl QState {
    fn from_str(qbits: &str) -> Result<Self> {
        let index = usize::from_str_radix(qbits, 2)?;
        let mut state = DVector::zeros(qbits.len().pow(2));
        state[index] = Complex::new(1.0, 0.0);

        Ok(Self { state })
    }
}

impl Display for QState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bin_width = self.state.len().ilog2() as usize;

        for (i, value) in self.state.iter().enumerate() {
            writeln!(f, "|{:0width$b}>: {}", i, value, width = bin_width)?;
        }

        Ok(())
    }
}

struct Circuit {
    gates: Vec<CsrMatrix<Complex<f64>>>,
    num_of_qbits: usize,
}

impl Circuit {
    fn new(num_of_qbits: usize) -> Self {
        Self {
            gates: Vec::new(),
            num_of_qbits,
        }
    }

    fn check_and_revsere_index(&self, index: usize) -> Result<usize> {
        if index >= self.num_of_qbits {
            return Err(anyhow::anyhow!(
                "Index out of bounds for the number of qubits {}",
                self.num_of_qbits
            ));
        }
        Ok(self.num_of_qbits - 1 - index)
    }

    #[allow(non_snake_case)]
    fn H(mut self, index: usize) -> Result<Self> {
        let index = self.check_and_revsere_index(index)?;

        let h = build_hadamard_matrix();

        let mut matrix = CsrMatrix::identity(1);
        for i in 0..self.num_of_qbits {
            if i == index {
                matrix = tensor_product(&matrix, &h);
            } else {
                matrix = tensor_product(&matrix, &CsrMatrix::identity(2));
            }
        }

        self.add_gate(matrix);
        Ok(self)
    }

    fn control(
        mut self,
        control: usize,
        target: usize,
        gate: &CsrMatrix<Complex<f64>>,
    ) -> Result<Self> {
        let control = self.check_and_revsere_index(control)?;
        let target = self.check_and_revsere_index(target)?;

        if control == target {
            return Err(anyhow::anyhow!(
                "Control and target qubits cannot be the same"
            ));
        }

        // |0><0|
        let mut zero_zero = CooMatrix::new(2, 2);
        zero_zero.push(0, 0, Complex::new(1.0, 0.0));
        let zero_zero = CsrMatrix::from(&zero_zero);

        // |1><1|
        let mut one_one = CooMatrix::new(2, 2);
        one_one.push(1, 1, Complex::new(1.0, 0.0));
        let one_one = CsrMatrix::from(&one_one);

        let x = build_x_matrix();
        let id = CsrMatrix::identity(2);

        // CNOT(0, 1) = |0><0| (x) I + |1><1| (x) X

        let mut zero_matrix = CsrMatrix::identity(1);
        let mut one_matrix = CsrMatrix::identity(1);
        for i in 0..self.num_of_qbits {
            if i == control {
                zero_matrix = tensor_product(&zero_matrix, &zero_zero);
                one_matrix = tensor_product(&one_matrix, &one_one);
            } else if i == target {
                zero_matrix = tensor_product(&zero_matrix, &id);
                one_matrix = tensor_product(&one_matrix, gate);
            } else {
                zero_matrix = tensor_product(&zero_matrix, &id);
                one_matrix = tensor_product(&one_matrix, &id);
            }
        }

        let matrix = zero_matrix + one_matrix;
        self.add_gate(matrix);
        Ok(self)
    }

    fn cnot(self, control: usize, target: usize) -> Result<Self> {
        self.control(control, target, &build_x_matrix())
    }

    fn add_gate(&mut self, gate: CsrMatrix<Complex<f64>>) {
        self.gates.push(gate);
    }

    fn apply(&self, state: &QState) -> QState {
        let mut result = state.state.clone();
        for gate in &self.gates {
            result = gate * result;
        }
        QState { state: result }
    }
}

fn main() -> Result<()> {
    let q00 = QState::from_str("00")?;
    println!("{}", q00);

    // Bell state |00> + |11>
    let result = Circuit::new(2).H(0)?.cnot(0, 1)?.apply(&q00);
    println!("{}", result);

    // Hadamard test
    let result = Circuit::new(2)
        .H(0)?
        .control(0, 1, &build_hadamard_matrix())?
        .H(0)?
        .apply(&q00);
    println!("{}", result);

    Ok(())

    /*
    let result5 = &cnot01 * (&hadamard0 * &q00);
    println!("{:}", convert_csr_dense(&result5));

    let result6 = &hadamard0 * (&ctrl_h01 * (&hadamard0 * &q00));
    println!("{:}", convert_csr_dense(&result6));
    */
}
