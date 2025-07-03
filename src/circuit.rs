use anyhow::Result;
use nalgebra::DMatrix;
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use num_complex::Complex;

use crate::gates::{
    h_matrix, rx_matrix, ry_matrix, rz_matrix, s_matrix, t_matrix, x_matrix, y_matrix, z_matrix,
};
use crate::qstate::QState;
use crate::Qbit;

#[derive(Clone, Debug, PartialEq)]
pub struct Gate {
    kind: GateKind,
    index: GateIndex,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GateIndex {
    All,
    One(usize),
    Control { controls: Vec<usize>, target: usize },
}

#[derive(Clone, Debug, PartialEq)]
pub enum GateKind {
    Dence(DMatrix<Qbit>),
    Sparse(CsrMatrix<Qbit>),

    H,
    X,
    Y,
    Z,
    S,
    T,
    RX(f64),
    RY(f64),
    RZ(f64),
}

pub enum ParameterizedGate {
    RX,
    RY,
    RZ,
}

struct Parameter {
    gate_index: usize,
    qbit_index: usize,
    gate: ParameterizedGate,
    value: f64,
}

pub struct Circuit {
    gates: Vec<Gate>,
    num_of_qbits: usize,

    parameters: Vec<Parameter>,
}

impl Circuit {
    pub fn new(num_of_qbits: usize) -> Self {
        Self {
            gates: Vec::new(),
            num_of_qbits,
            parameters: Vec::new(),
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

    fn create_gate_for_index(
        &self,
        index: usize,
        gate: &CsrMatrix<Qbit>,
    ) -> Result<CsrMatrix<Qbit>> {
        let index = self.check_and_revsere_index(index)?;

        let mut matrix = CsrMatrix::identity(1);
        for i in 0..self.num_of_qbits {
            if i == index {
                matrix = kronecker_product(&matrix, &gate);
            } else {
                matrix = kronecker_product(&matrix, &CsrMatrix::identity(2));
            }
        }

        Ok(matrix)
    }

    fn create_parametric_gate(&self, param: &Parameter, value: f64) -> GateKind {
        match param.gate {
            ParameterizedGate::RX => GateKind::RX(value),
            ParameterizedGate::RY => GateKind::RY(value),
            ParameterizedGate::RZ => GateKind::RZ(value),
        }
    }

    pub fn gate_at(mut self, index: usize, gate: GateKind) -> Result<Self> {
        self.add_gate(gate, index);
        Ok(self)
    }

    pub fn add_gate_at(&mut self, index: usize, gate: GateKind) -> Result<()> {
        self.add_gate(gate, index);
        Ok(())
    }

    pub fn sparse_gate_at(mut self, index: usize, gate: CsrMatrix<Qbit>) -> Result<Self> {
        self.add_gate(GateKind::Sparse(gate), index);
        Ok(self)
    }

    pub fn add_sparse_gate_at(&mut self, index: usize, gate: CsrMatrix<Qbit>) -> Result<()> {
        self.add_gate(GateKind::Sparse(gate), index);
        Ok(())
    }

    pub fn add_parametric_gate_at(
        &mut self,
        index: usize,
        gate: ParameterizedGate,
        value: f64,
    ) -> Result<()> {
        let param = Parameter {
            gate_index: self.gates.len(),
            qbit_index: index,
            gate,
            value,
        };
        let gate = self.create_parametric_gate(&param, value);

        self.parameters.push(param);
        self.add_gate(gate, index);

        Ok(())
    }

    pub fn get_parameters(&self) -> Vec<f64> {
        self.parameters.iter().map(|param| param.value).collect()
    }

    pub fn set_parameter(&mut self, param_index: usize, value: f64) -> Result<()> {
        if let Some(param) = self.parameters.get_mut(param_index) {
            param.value = value;
        } else {
            return Err(anyhow::anyhow!("Parameter index out of bounds"));
        };

        // No index check is needed
        let param = &self.parameters[param_index];

        let gate = self.create_parametric_gate(param, value);
        self.gates[param.gate_index] = Gate {
            kind: gate,
            index: GateIndex::One(param.qbit_index),
        };

        Ok(())
    }

    pub fn set_parameters(&mut self, values: &[f64]) -> Result<()> {
        if values.len() != self.parameters.len() {
            return Err(anyhow::anyhow!(
                "Number of values does not match number of parameters"
            ));
        }

        for (i, &value) in values.iter().enumerate() {
            self.set_parameter(i, value)?;
        }

        Ok(())
    }

    #[allow(non_snake_case)]
    pub fn H(self, index: usize) -> Result<Self> {
        self.gate_at(index, GateKind::H)
    }

    pub fn control(mut self, control: usize, target: usize, kind: GateKind) -> Result<Self> {
        self.gates.push(Gate {
            kind,
            index: GateIndex::Control {
                controls: vec![control],
                target,
            },
        });
        Ok(self)
    }

    fn build_control_matrix(
        &self,
        control: &[usize],
        target: usize,
        gate: &CsrMatrix<Qbit>,
    ) -> Result<CsrMatrix<Qbit>> {
        let control = control
            .iter()
            .map(|ctrl| self.check_and_revsere_index(*ctrl))
            .collect::<Result<Vec<_>>>()?;
        let target = self.check_and_revsere_index(target)?;

        if control.contains(&target) {
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
            if control.contains(&i) {
                zero_matrix = kronecker_product(&zero_matrix, &zero_zero);
                one_matrix = kronecker_product(&one_matrix, &one_one);
            } else if i == target {
                zero_matrix = kronecker_product(&zero_matrix, &id);
                one_matrix = kronecker_product(&one_matrix, gate);
            } else {
                zero_matrix = kronecker_product(&zero_matrix, &id);
                one_matrix = kronecker_product(&one_matrix, &id);
            }
        }

        Ok(zero_matrix + one_matrix)
    }

    pub fn cnot(self, control: usize, target: usize) -> Result<Self> {
        self.control(control, target, GateKind::X)
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

    pub fn add_gate(&mut self, kind: GateKind, index: usize) {
        self.gates.push(Gate {
            kind,
            index: GateIndex::One(index),
        });
    }

    pub fn add_dence_gate(&mut self, gate: DMatrix<Qbit>, index: GateIndex) {
        self.gates.push(Gate {
            kind: GateKind::Dence(gate),
            index,
        });
    }

    // TODO: Don't use Result type by checking errors in advance.
    pub fn apply(&self, state: &QState) -> Result<QState> {
        let mut result = state.state.clone();
        for Gate { kind, index } in &self.gates {
            let matrix = match kind {
                GateKind::Dence(dense_gate) => &CsrMatrix::from(dense_gate),
                GateKind::Sparse(sparse_gate) => sparse_gate,
                gate => &self.get_matrix_from_gate(gate)?,
            };
            let matrix = match index {
                GateIndex::All => matrix,
                GateIndex::One(index) => &self.create_gate_for_index(*index, &matrix)?,
                GateIndex::Control { controls, target } => {
                    &self.build_control_matrix(controls, *target, matrix)?
                }
            };
            result = matrix * result;
        }
        Ok(QState { state: result })
    }

    fn get_matrix_from_gate(&self, gate: &GateKind) -> Result<CsrMatrix<Qbit>> {
        let matrix = match gate {
            GateKind::Dence(dense_gate) => CsrMatrix::from(dense_gate),
            GateKind::Sparse(sparse_gate) => sparse_gate.clone(),
            GateKind::H => h_matrix(),
            GateKind::X => x_matrix(),
            GateKind::Y => y_matrix(),
            GateKind::Z => z_matrix(),
            GateKind::S => s_matrix(),
            GateKind::T => t_matrix(),
            GateKind::RX(angle) => rx_matrix(*angle),
            GateKind::RY(angle) => ry_matrix(*angle),
            GateKind::RZ(angle) => rz_matrix(*angle),
        };
        Ok(matrix)
    }
}

pub fn kronecker_product(x: &CsrMatrix<Qbit>, y: &CsrMatrix<Qbit>) -> CsrMatrix<Qbit> {
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

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use crate::assert_approx_complex_eq;

    use super::*;

    #[test]
    fn test_bell_state() -> Result<()> {
        let q00 = QState::from_str("00").unwrap();
        let result = Circuit::new(q00.num_of_qbits())
            .H(0)?
            .cnot(0, 1)?
            .apply(&q00)?;

        // Bell state |00> + |11>
        assert_approx_complex_eq!(1.0 / 2f64.sqrt(), 0.0, result.state[0]);
        assert_approx_complex_eq!(0.0, 0.0, result.state[1]);
        assert_approx_complex_eq!(0.0, 0.0, result.state[2]);
        assert_approx_complex_eq!(1.0 / 2f64.sqrt(), 0.0, result.state[3]);

        Ok(())
    }

    #[test]
    /// Hadamard test for Hadamard gate
    /// https://dojo.qulacs.org/ja/latest/notebooks/2.2_Hadamard_test.html
    fn test_hadamard_test() -> Result<()> {
        let q00 = QState::from_str("00").unwrap();
        let result = Circuit::new(q00.num_of_qbits())
            .H(0)?
            .control(0, 1, GateKind::H)?
            .H(0)?
            .apply(&q00)?;

        assert_approx_complex_eq!((2f64.sqrt() + 2.0) / 4.0, 0.0, result.state[0]);
        assert_approx_complex_eq!((-2f64.sqrt() + 2.0) / 4.0, 0.0, result.state[1]);
        assert_approx_complex_eq!(2f64.sqrt() / 4.0, 0.0, result.state[2]);
        assert_approx_complex_eq!(-2f64.sqrt() / 4.0, 0.0, result.state[3]);

        Ok(())
    }

    #[test]
    /// Quantum Fourier Transform (QFT) for 3 qubits
    /// https://dojo.qulacs.org/ja/latest/notebooks/2.3_quantum_Fourier_transform.html
    fn test_qft() -> Result<()> {
        let qstate = QState::new(&[Complex::new(1.0, 0.0) / 8.0_f64.sqrt(); 8])?;

        let result = Circuit::new(qstate.num_of_qbits())
            // First bit
            .H(0)?
            .control(1, 0, GateKind::S)?
            .control(2, 0, GateKind::T)?
            // Second bit
            .H(1)?
            .control(2, 1, GateKind::S)?
            // Third bit
            .H(2)?
            .swap(0, 2)?
            .apply(&qstate)?;

        assert_approx_complex_eq!(1.0, 0.0, result.state[0]);
        assert_approx_complex_eq!(0.0, 0.0, result.state[1]);
        assert_approx_complex_eq!(0.0, 0.0, result.state[2]);
        assert_approx_complex_eq!(0.0, 0.0, result.state[3]);
        assert_approx_complex_eq!(0.0, 0.0, result.state[4]);
        assert_approx_complex_eq!(0.0, 0.0, result.state[5]);
        assert_approx_complex_eq!(0.0, 0.0, result.state[6]);
        assert_approx_complex_eq!(0.0, 0.0, result.state[7]);

        Ok(())
    }

    #[test]
    fn test_parameterized_gate() -> Result<()> {
        let q00 = QState::from_str("00").unwrap();
        let mut circuit = Circuit::new(q00.num_of_qbits());
        circuit.add_parametric_gate_at(0, ParameterizedGate::RX, PI)?;

        let result = circuit.apply(&q00)?;

        assert_approx_complex_eq!(0.0, 0.0, result.state[0]);
        assert_approx_complex_eq!(0.0, -1.0, result.state[1]);

        // Update the parameter to PI/2
        let mut param = circuit.get_parameters();
        assert_eq!(1, param.len());
        assert_eq!(PI, param[0]);

        param[0] = PI / 2.0;
        circuit.set_parameters(&param)?;

        let param = circuit.get_parameters();
        assert_eq!(1, param.len());
        assert_eq!(PI / 2.0, param[0]);

        let result = circuit.apply(&q00)?;

        assert_approx_complex_eq!(1.0 / 2f64.sqrt(), 0.0, result.state[0]);
        assert_approx_complex_eq!(0.0, -1.0 / 2f64.sqrt(), result.state[1]);

        Ok(())
    }
}
