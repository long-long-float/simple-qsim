use std::iter;

use crate::{
    circuit::kronecker_product,
    gates::{x_matrix, y_matrix, z_matrix},
    qstate::QState,
};
use anyhow::Result;
use nalgebra_sparse::{convert::serial::convert_dense_csr, CsrMatrix};

pub struct Observable {
    operators: Vec<PauliOperator>,
}

impl Observable {
    pub fn new() -> Self {
        Self {
            operators: Vec::new(),
        }
    }

    pub fn add_pauli_operator(&mut self, coefficient: f64, ops: &[(Pauli, usize)]) {
        let operator = PauliOperator {
            coefficient,
            ops: ops
                .iter()
                .map(|&(kind, index)| PauliMatrix { index, kind })
                .collect(),
        };
        self.operators.push(operator);
    }

    pub fn expectation_value(&self, qstate: &QState) -> Result<f64> {
        let mut expectation = 0.0;

        for operator in &self.operators {
            let mut kinds = iter::repeat(Pauli::I)
                .take(qstate.num_of_qbits())
                .collect::<Vec<_>>();
            for op in &operator.ops {
                kinds[op.index] = op.kind;
            }

            let mut op = CsrMatrix::identity(1);
            for kind in kinds.iter().rev() {
                match kind {
                    Pauli::I => {
                        op = kronecker_product(&op, &CsrMatrix::identity(2));
                    }
                    Pauli::X => {
                        op = kronecker_product(&op, &x_matrix());
                    }
                    Pauli::Y => {
                        op = kronecker_product(&op, &y_matrix());
                    }
                    Pauli::Z => {
                        op = kronecker_product(&op, &z_matrix());
                    }
                }
            }

            let qstate = convert_dense_csr(&qstate.state);
            let exp = (qstate.transpose() * op * qstate)
                .get_entry(0, 0)
                .ok_or_else(|| anyhow::anyhow!("Failed to compute expectation value for operator"))?
                .into_value()
                .re;

            expectation += operator.coefficient * exp;
        }

        Ok(expectation)
    }
}

struct PauliOperator {
    coefficient: f64,
    ops: Vec<PauliMatrix>,
}

#[derive(Clone, Debug)]
pub struct PauliMatrix {
    index: usize,
    kind: Pauli,
}

#[derive(Clone, Copy, Debug)]
pub enum Pauli {
    I,
    X,
    Y,
    Z,
}

#[cfg(test)]
mod tests {
    use nalgebra::Complex;

    use crate::{assert_approx_eq, Circuit};

    use super::*;

    #[test]
    fn test_1qbit_z_observable() -> Result<()> {
        let q0 = QState::from_str("0").unwrap();

        let mut observable = Observable::new();
        observable.add_pauli_operator(1.0, &[(Pauli::Z, 0)]);

        let expectation = observable.expectation_value(&q0)?;
        assert_approx_eq!(1.0, expectation);

        let q1 = Circuit::new(1).H(0)?.apply(&q0);
        let expectation = observable.expectation_value(&q1)?;
        assert_approx_eq!(0.0, expectation);

        let q2 = QState::new(&[
            Complex::new((2.0f64 / 3.0).sqrt(), 0.0),
            Complex::new((1.0f64 / 3.0).sqrt(), 0.0),
        ])?;
        let expectation = observable.expectation_value(&q2)?;
        assert_approx_eq!(1.0 / 3.0, expectation);

        Ok(())
    }

    #[test]
    fn test_1qbit_x_observable() -> Result<()> {
        let q0 = QState::from_str("0").unwrap();

        let mut observable = Observable::new();
        observable.add_pauli_operator(1.0, &[(Pauli::X, 0)]);

        let expectation = observable.expectation_value(&q0)?;
        assert_approx_eq!(0.0, expectation);

        let q1 = Circuit::new(1).H(0)?.apply(&q0);
        let expectation = observable.expectation_value(&q1)?;
        assert_approx_eq!(1.0, expectation);

        Ok(())
    }

    #[test]
    fn test_2qbit_xz_observable() -> Result<()> {
        let q0 = QState::from_str("00").unwrap();

        let mut observable = Observable::new();
        observable.add_pauli_operator(1.0, &[(Pauli::X, 0), (Pauli::Z, 1)]);

        let expectation = observable.expectation_value(&q0)?;
        assert_approx_eq!(0.0, expectation);

        let q1 = Circuit::new(q0.num_of_qbits()).H(0)?.apply(&q0);
        let expectation = observable.expectation_value(&q1)?;
        assert_approx_eq!(1.0, expectation);

        Ok(())
    }
}
