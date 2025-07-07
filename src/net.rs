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
use std::collections::{HashMap, VecDeque};

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

pub struct Net {
    knots: Vec<Knot>,
    tile_width: f64,
    g: i64,
    su2net: HashMap<ICoord, Knot>,
}

impl Net {
    pub fn new(tile_width: f64) -> Self {
        Net {
            knots: Vec::new(),
            tile_width,
            g: (2.0 / tile_width) as i64,
            su2net: HashMap::new(),
        }
    }

    pub fn generate(&mut self, max_length: usize) {
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
                    None
                } else {
                    Some(n)
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
                .map(|(c, o)| format!("{}: {}", c, o.unwrap_or(0)))
                .collect::<Vec<_>>()
                .join(", ")
        );

        // And fill up the enet
        let mut word = Vec::new();
        let mut products = Vec::new();
        let mut sequence = Vec::new();

        let mut depth = 0;
        loop {
            if sequence.len() <= depth {
                sequence.push(0);
            } else {
                sequence[depth] += 1;
            }

            // Append an extra symbol
            if sequence[depth] < gate_set.len() {
                // Check we're not following something with its inverse
                if depth > 0 && sequence[depth - 1] == gate_inverses[sequence[depth]] {
                    continue;
                }

                // Check we're not exceeding the order
                if word.len() >= 1 {
                    if let Some(order) = gate_orders[sequence[depth]] {
                        let mut repeat = 1usize;
                        while (depth as i32 - repeat as i32) >= 0
                            && word[depth - repeat] == gate_labels[sequence[depth]]
                            && repeat < order
                        {
                            repeat += 1;
                        }

                        if repeat >= order {
                            continue;
                        }
                    }
                }

                // Calculate all products
                let new_prod = if depth > 0 {
                    products.last().unwrap() * &gate_set[sequence[depth]]
                } else {
                    gate_set[sequence[depth]].clone()
                };
                if products.len() > depth {
                    products[depth] = new_prod;
                } else {
                    products.push(new_prod);
                }

                if word.len() > depth {
                    word[depth] = gate_labels[sequence[depth]];
                } else {
                    word.push(gate_labels[sequence[depth]]);
                }

                // println!(
                //     "Adding {} at depth {}",
                //     word.iter()
                //         .map(|c| c.to_string())
                //         .collect::<Vec<_>>()
                //         .join(""),
                //     depth
                // );
                self.add(&products[depth], &word.iter().collect::<String>());

                if depth < max_length - 1 {
                    depth += 1;
                }
            } else {
                word.pop();
                products.pop();
                sequence.pop();

                if depth == 0 {
                    break;
                }

                depth -= 1;
            }
        }
    }

    fn add(&mut self, u: &Matrix2<Qbit>, word: &str) {
        let knot = Knot {
            word: word.to_string(),
            matrix: *u,
        };
        self.knots.push(knot.clone());

        for c in 0..16 {
            let uci = ICoord::from_matrix(u, c, self.g);
            self.su2net.insert(uci, knot.clone());
        }
    }
}

/// A 'knot' is a mildly amusing term for a point in a Net
#[derive(Debug, Clone, PartialEq)]
struct Knot {
    word: String,
    matrix: Matrix2<Qbit>,
}

/// Integer coordinates for an SU(2) matrix in 4D space
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
struct ICoord {
    t: i32,
    x: i32,
    y: i32,
    z: i32,
}

impl ICoord {
    fn from_matrix(m: &Matrix2<Qbit>, corner: i32, g: i64) -> Self {
        let g = g as f64;

        let mut mc = ICoord::default();
        let mc0 = m[(0, 0)].re;
        let mc1 = -1.0 * m[(0, 1)].im;
        let mc2 = m[(1, 0)].re;
        let mc3 = m[(1, 1)].im;

        if corner < 0 {
            mc.t = (g * mc0 + 0.5).floor() as i32;
            mc.x = (g * mc1 + 0.5).floor() as i32;
            mc.y = (g * mc2 + 0.5).floor() as i32;
            mc.z = (g * mc3 + 0.5).floor() as i32;
        } else {
            mc.t = if corner & 1 == 0 {
                (g * mc0).ceil()
            } else {
                (g * mc0).floor()
            } as i32;

            mc.x = if corner & 2 == 0 {
                (g * mc1).ceil()
            } else {
                (g * mc1).floor()
            } as i32;

            mc.y = if corner & 4 == 0 {
                (g * mc2).ceil()
            } else {
                (g * mc2).floor()
            } as i32;

            mc.z = if corner & 8 == 0 {
                (g * mc3).ceil()
            } else {
                (g * mc3).floor()
            } as i32;
        }

        // We want the first non-zero coordinate to be > 0, so multiply by -I if
        // necessary.

        if mc.t < 0 {
            mc.t *= -1;
            mc.x *= -1;
            mc.y *= -1;
            mc.z *= -1;
        } else if mc.t == 0 {
            if mc.x < 0 {
                mc.x *= -1;
                mc.y *= -1;
                mc.z *= -1;
            } else if mc.x == 0 {
                if mc.y < 0 {
                    mc.y *= -1;
                    mc.z *= -1;
                } else if mc.y == 0 && mc.z < 0 {
                    mc.z *= -1;
                }
            }
        }

        mc
    }
}
