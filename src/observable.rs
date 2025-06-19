use crate::qstate::QState;
use anyhow::Result;

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
            let mut term = 0.0;
            for pauli_matrix in &operator.ops {
                let index = pauli_matrix.index;
                let kind = &pauli_matrix.kind;

                let alpha = qstate
                    .state
                    .get(index)
                    .ok_or_else(|| anyhow::anyhow!("Index out of bounds for qstate"))?;
                let beta = qstate
                    .state
                    .get(index + 1)
                    .ok_or_else(|| anyhow::anyhow!("Index out of bounds for qstate"))?;

                match kind {
                    Pauli::I => {
                        // Identity does not change the expectation value
                    }
                    Pauli::X => term += 2.0 * (alpha.conj() * beta).re,
                    Pauli::Y => term += 2.0 * (alpha.conj() * beta).im,
                    Pauli::Z => {
                        let alpha = alpha.norm_sqr();
                        let beta = beta.norm_sqr();
                        term += alpha - beta;
                    }
                }
            }

            expectation += operator.coefficient * term;
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
}
