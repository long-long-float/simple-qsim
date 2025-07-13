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
use nalgebra::Matrix2;

use crate::Qbit;

pub(crate) struct Su2Equiv {
    epsilon: f64,
    gamma: f64,
}

impl Su2Equiv {
    pub(crate) fn new(e: f64) -> Self {
        Su2Equiv {
            epsilon: e * e,
            gamma: (2.0 - e) * (2.0 - e),
        }
    }

    pub(crate) fn equals(&self, a: &Matrix2<Qbit>, b: &Matrix2<Qbit>) -> bool {
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
    use crate::net::{h_dence_matrix, t_dence_matrix};

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
