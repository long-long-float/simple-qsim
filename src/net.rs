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
use std::collections::{HashMap, HashSet, VecDeque};
use std::vec;

use anyhow::{Ok, Result};
use nalgebra::{DMatrix, Matrix2};
use nalgebra_sparse::convert::serial::convert_csr_dense;
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use num_complex::Complex;

use crate::gates::{
    h_matrix, rx_matrix, ry_matrix, rz_matrix, s_matrix, t_matrix, x_matrix, y_matrix, z_matrix,
};
use crate::qstate::QState;
use crate::su2equiv::Su2Equiv;
use crate::{su2, Qbit};
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

pub fn h_dence_matrix() -> Matrix2<Qbit> {
    let root2 = 2.0_f64.sqrt();
    let x = Complex::new(0.0, 1.0) / root2;
    Matrix2::from_row_slice(&[x, x, x, -x])
}

pub struct Net {
    knots: Vec<Knot>,
    tile_width: f64,
    g: i64,
    su2net: HashMap<ICoord, HashSet<Knot>>,

    empty_knot_hash: HashSet<Knot>,
}

impl Net {
    pub fn new(tile_width: f64) -> Self {
        Net {
            knots: Vec::new(),
            tile_width,
            g: (2.0 / tile_width) as i64,
            su2net: HashMap::new(),
            empty_knot_hash: HashSet::new(),
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
                gate_inverses[i] = index;
                gate_inverses[index] = i;
            } else {
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
                    c *= gate;
                    n += 1;
                }

                if n == 50 {
                    None
                } else {
                    Some(n)
                }
            })
            .collect::<Vec<_>>();

        // Check variables
        println!("{}", gate_set[0]); // H
        println!("{}", gate_set[1]); // T
        println!("{}", gate_set[2]); // t
        println!("{}", gate_set[0] * gate_set[gate_inverses[0]]); // H
        println!("{}", gate_set[1] * gate_set[gate_inverses[1]]); // T
        println!("{}", gate_set[2] * gate_set[gate_inverses[2]]); // t

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
                // eprintln!(
                //     "{}, {}",
                //     word.iter().collect::<String>(),
                //     &products[depth][(0, 0)].re
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
            let uci = ICoord::from_matrix(u, Some(c), self.g);
            self.su2net
                .entry(uci)
                .and_modify(|knots| {
                    knots.insert(knot.clone());
                })
                .or_insert(HashSet::from_iter([knot.clone()]));
        }
    }

    pub fn solovay_kitaev(&self, u: &Matrix2<Qbit>, depth: usize) -> Result<Knot> {
        if depth == 0 {
            self.nearest(u)
        } else {
            let ku = self.solovay_kitaev(u, depth - 1)?;
            let (v, w) = su2::group_factor(&(u * ku.matrix.adjoint()));

            let kv = self.solovay_kitaev(&v, depth - 1)?;
            let kw = self.solovay_kitaev(&w, depth - 1)?;

            let kv_inv = kv.word.chars().rev().collect::<String>();
            let kw_inv = kw.word.chars().rev().collect::<String>();

            Ok(Knot {
                word: format!("{}{}{}{}{}", kv.word, kw.word, kv_inv, kw_inv, ku.word),
                matrix: kv.matrix
                    * kw.matrix
                    * kv.matrix.adjoint()
                    * kw.matrix.adjoint()
                    * ku.matrix,
            })
        }
    }

    fn get_knots(&self, k: &ICoord) -> &HashSet<Knot> {
        self.su2net.get(k).unwrap_or(&self.empty_knot_hash)
    }

    fn nearest(&self, u: &Matrix2<Qbit>) -> Result<Knot> {
        let mut uc = ICoord::from_matrix(u, None, self.g);

        if self.get_knots(&uc).is_empty() {
            for corner in 0..16 {
                uc = ICoord::from_matrix(u, Some(corner), self.g);
                if !self.get_knots(&uc).is_empty() {
                    break;
                }
            }
        }

        println!("Find knots near: {:?}", uc);
        println!("Candidates:");
        for knot in self.get_knots(&uc) {
            // println!("{}", knot.word);
        }

        let knot = self
            .get_knots(&uc)
            .iter()
            .min_by(|a, b| {
                su2::proj_trace_dist(u, &a.matrix)
                    .partial_cmp(&su2::proj_trace_dist(u, &b.matrix))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("No knots found near the given matrix"))?;
        println!("Nearest knot: {:?}", knot);
        println!();
        Ok(knot)
    }
}

/// A 'knot' is a mildly amusing term for a point in a Net
#[derive(Debug, Clone)]
pub struct Knot {
    pub word: String,
    pub matrix: Matrix2<Qbit>,
}

impl PartialEq for Knot {
    fn eq(&self, other: &Self) -> bool {
        self.word == other.word
    }
}
impl Eq for Knot {}
impl std::hash::Hash for Knot {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.word.hash(state);
    }
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
    fn from_matrix(m: &Matrix2<Qbit>, corner: Option<i32>, g: i64) -> Self {
        let g = g as f64;

        let mut mc = ICoord::default();
        let mc0 = m[(0, 0)].re;
        let mc1 = -1.0 * m[(0, 1)].im;
        let mc2 = m[(1, 0)].re;
        let mc3 = m[(1, 1)].im;

        if let Some(corner) = corner {
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
        } else {
            mc.t = (g * mc0 + 0.5).floor() as i32;
            mc.x = (g * mc1 + 0.5).floor() as i32;
            mc.y = (g * mc2 + 0.5).floor() as i32;
            mc.z = (g * mc3 + 0.5).floor() as i32;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_approx_complex_eq;

    #[test]
    fn test_net_solovay_kitaev_depth_0() -> Result<()> {
        let mut net = Net::new(0.18);
        net.generate(14);

        test_matrix(&net, &h_dence_matrix(), "H")?;
        test_matrix(&net, &h_dence_matrix().adjoint(), "H")?;

        test_matrix(&net, &t_dence_matrix(), "T")?;
        test_matrix(&net, &t_dence_matrix().adjoint(), "t")?;

        fn test_matrix(net: &Net, u: &Matrix2<Qbit>, expected_word: &str) -> Result<()> {
            let ska = net.solovay_kitaev(u, 0)?;
            assert_eq!(expected_word, ska.word);
            // TODO: Enable them
            // assert_approx_complex_eq!(u[(0, 0)].re, u[(0, 0)].im, ska.matrix[(0, 0)]);
            // assert_approx_complex_eq!(u[(0, 1)].re, u[(0, 1)].im, ska.matrix[(0, 1)]);
            // assert_approx_complex_eq!(u[(1, 0)].re, u[(1, 0)].im, ska.matrix[(1, 0)]);
            // assert_approx_complex_eq!(u[(1, 1)].re, u[(1, 1)].im, ska.matrix[(1, 1)]);

            Ok(())
        }

        Ok(())
    }
}
