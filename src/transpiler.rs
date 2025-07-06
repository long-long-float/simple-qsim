use std::collections::HashMap;

use anyhow::Result;
use nalgebra::{DMatrix, Matrix2};
use nalgebra_sparse::convert::serial::convert_csr_dense;
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use num_complex::Complex;

use crate::gates::{
    h_dence_matrix, h_matrix, rx_matrix, ry_matrix, rz_matrix, s_matrix, t_matrix, x_matrix,
    y_matrix, z_matrix,
};
use crate::qstate::QState;
use crate::Qbit;
use nalgebra::ComplexField;

/// Use T = Rz(π/4) definition
pub fn t_dence_matrix() -> Matrix2<Qbit> {
    Matrix2::from_row_slice(&[
        Complex::from_polar(1.0, std::f64::consts::FRAC_PI_8),
        Complex::ZERO,
        Complex::ZERO,
        Complex::from_polar(1.0, std::f64::consts::FRAC_PI_8),
    ])
}

pub struct Net {}

impl Net {
    pub fn new() -> Self {
        Net {}
    }

    pub fn generate(self, max_length: usize) {
        let mut gate_set = vec![h_dence_matrix(), t_dence_matrix()];
        let mut gate_labels = vec!['H', 'T'];

        let mut gate_inverses = (0..gate_set.len()).map(|_| 0).collect::<Vec<_>>();

        let su2_equiv = Su2Equiv::new(1e-15);

        // Add inverses
        for i in 0..gate_set.len() {
            let gate = &gate_set[i];
            let inv = gate.adjoint();

            let already_exists = gate_set.iter().position(|g| su2_equiv.equals(g, &inv));
            if let Some(index) = already_exists {
                println!(
                    "Found inverse of {} {} at index {}",
                    gate_labels[i], i, index
                );
                gate_inverses[i] = index;
                gate_inverses[index] = i;
            } else {
                println!(
                    "Adding inverse of {} at index {}",
                    gate_labels[i],
                    gate_set.len()
                );
                gate_set.push(inv);
                gate_labels.push(gate_labels[i].to_lowercase().next().unwrap());

                gate_inverses.push(i);
                gate_inverses[i] = gate_inverses.len() - 1;
            }
        }

        // And match each label to its index in all these arrays
        let mut label_indieces = HashMap::new();
        for (i, label) in gate_labels.iter().enumerate() {
            label_indieces.insert(label, i);
        }

        // finally we need to work out the orders.
        // for our purposes we'll take any order over 50 to be infinite
        let id2 = Matrix2::identity();
        let gate_orders = gate_set
            .iter()
            .map(|gate| {
                let mut n = 1;
                let mut c = gate.clone();

                while n < 50 && !su2_equiv.equals(&c, &id2) {
                    println!("{}", c);
                    c *= gate;
                    n += 1;
                }
                println!("{}", c);
                println!("------------");

                if n == 50 {
                    -1
                } else {
                    n
                }
            })
            .collect::<Vec<_>>();

        // Check variables
        println!("{:?}", gate_set[0] * gate_set[gate_inverses[0]]); // H
        println!("{:?}", gate_set[1] * gate_set[gate_inverses[1]]); // T
        println!("{:?}", gate_set[2] * gate_set[gate_inverses[2]]); // t

        println!(
            "{}",
            gate_labels
                .iter()
                .zip(gate_orders.iter())
                .map(|(c, o)| format!("{}: {}", c, o))
                .collect::<Vec<_>>()
                .join(", ")
        );

        // And fill up the enet
    }
}

struct Su2Equiv {
    epsilon: f64,
    gamma: f64,
}

impl Su2Equiv {
    fn new(e: f64) -> Self {
        Su2Equiv {
            epsilon: e * e,
            gamma: (2.0 - e) * (2.0 - e),
        }
    }

    fn equals(&self, a: &Matrix2<Qbit>, b: &Matrix2<Qbit>) -> bool {
        let a0 = a[(0, 0)].re;
        let a1 = -1.0 * a[(0, 1)].im;
        let a2 = a[(1, 0)].re;
        let a3 = a[(1, 1)].im;

        let b0 = b[(0, 0)].re;
        let b1 = -1.0 * b[(0, 1)].im;
        let b2 = b[(1, 0)].re;
        let b3 = b[(1, 1)].im;

        let d0 = a0 - b0;
        let d1 = a1 - b1;
        let d2 = a2 - b2;
        let d3 = a3 - b3;

        let dist = d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        dist < self.epsilon || dist > self.gamma
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_su2equiv_equals_identity() {
        let su2 = Su2Equiv::new(1e-8);
        let id = Matrix2::identity();
        assert!(su2.equals(&id, &id));
    }

    #[test]
    fn test_su2equiv_equals_h_and_h() {
        let su2 = Su2Equiv::new(1e-8);
        let h = h_dence_matrix();
        assert!(su2.equals(&h, &h));
    }

    #[test]
    fn test_su2equiv_not_equals_h_and_t() {
        let su2 = Su2Equiv::new(1e-8);
        let h = h_dence_matrix();
        let t = t_dence_matrix();
        assert!(!su2.equals(&h, &t));
    }

    #[test]
    fn test_su2equiv_equals_with_small_difference() {
        let su2 = Su2Equiv::new(1e-6);
        let h1 = h_dence_matrix();
        let mut h2 = h_dence_matrix();
        h2[(0, 0)].re += 1e-8;
        assert!(su2.equals(&h1, &h2));
    }

    #[test]
    fn test_su2equiv_not_equals_with_large_difference() {
        let su2 = Su2Equiv::new(1e-8);
        let h1 = h_dence_matrix();
        let mut h2 = h_dence_matrix();
        h2[(0, 0)].re += 1e-2;
        assert!(!su2.equals(&h1, &h2));
    }

    #[test]
    fn test_su2equiv_equals_with_h_adjoint() {
        let su2 = Su2Equiv::new(1e-8);
        let h = h_dence_matrix();
        let h_adjoint = h.adjoint();
        assert!(su2.equals(&h, &h_adjoint));
    }
}
