/// This is ported from https://github.com/cmdawson/sk
// MIT License
// Copyright (c) 2005 Chris Dawson

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
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
use crate::su2equiv::Su2Equiv;
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
