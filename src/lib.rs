pub mod gates;

use std::fmt::Display;

use anyhow::Result;
use nalgebra::{DVector, Matrix2};
use nalgebra_sparse::convert::serial::convert_dense_coo;
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use num_complex::Complex;

use crate::gates::{build_hadamard_matrix, build_x_matrix};

type Qbit = Complex<f64>;

fn tensor_product(x: &CsrMatrix<Qbit>, y: &CsrMatrix<Qbit>) -> CsrMatrix<Qbit> {
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

pub struct QState {
    state: DVector<Qbit>,
}

impl QState {
    pub fn new(state: &[Qbit]) -> Result<Self> {
        let len = state.len();
        if len == 0 || (len & (len - 1)) != 0 {
            return Err(anyhow::anyhow!(
                "State vector length must be a non-zero power of 2"
            ));
        }

        let state = DVector::from_row_slice(state);
        Ok(Self { state })
    }

    pub fn from_str(qbits: &str) -> Result<Self> {
        let index = usize::from_str_radix(qbits, 2)?;
        let mut state = DVector::zeros(2_usize.pow(qbits.len() as u32));
        state[index] = Complex::new(1.0, 0.0);

        Ok(Self { state })
    }

    pub fn len(&self) -> usize {
        self.state.len().ilog2() as usize
    }
}

impl Display for QState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bin_width = self.len();

        for (i, value) in self.state.iter().enumerate() {
            writeln!(f, "|{:0width$b}>: {}", i, value, width = bin_width)?;
        }

        Ok(())
    }
}

pub struct Circuit {
    gates: Vec<CsrMatrix<Qbit>>,
    num_of_qbits: usize,
}

impl Circuit {
    pub fn new(num_of_qbits: usize) -> Self {
        Self {
            gates: Vec::new(),
            num_of_qbits,
        }
    }

    pub fn check_and_revsere_index(&self, index: usize) -> Result<usize> {
        if index >= self.num_of_qbits {
            return Err(anyhow::anyhow!(
                "Index out of bounds for the number of qubits {}",
                self.num_of_qbits
            ));
        }
        Ok(self.num_of_qbits - 1 - index)
    }

    pub fn add_gate_at(mut self, index: usize, gate: CsrMatrix<Qbit>) -> Result<Self> {
        let index = self.check_and_revsere_index(index)?;

        let mut matrix = CsrMatrix::identity(1);
        for i in 0..self.num_of_qbits {
            if i == index {
                matrix = tensor_product(&matrix, &gate);
            } else {
                matrix = tensor_product(&matrix, &CsrMatrix::identity(2));
            }
        }

        self.add_gate(matrix);
        Ok(self)
    }

    #[allow(non_snake_case)]
    pub fn H(self, index: usize) -> Result<Self> {
        self.add_gate_at(index, build_hadamard_matrix())
    }

    pub fn control(
        mut self,
        control: usize,
        target: usize,
        gate: &CsrMatrix<Qbit>,
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

        let id = CsrMatrix::identity(2);

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

    pub fn cnot(self, control: usize, target: usize) -> Result<Self> {
        self.control(control, target, &build_x_matrix())
    }

    pub fn swap(self, index1: usize, index2: usize) -> Result<Self> {
        let index1 = self.check_and_revsere_index(index1)?;
        let index2 = self.check_and_revsere_index(index2)?;

        if index1 == index2 {
            return Err(anyhow::anyhow!("Cannot swap a qubit with itself"));
        }

        self.cnot(index1, index2)?
            .cnot(index2, index1)?
            .cnot(index1, index2)
    }

    fn add_gate(&mut self, gate: CsrMatrix<Qbit>) {
        self.gates.push(gate);
    }

    pub fn apply(&self, state: &QState) -> QState {
        let mut result = state.state.clone();
        for gate in &self.gates {
            result = gate * result;
        }
        QState { state: result }
    }
}
