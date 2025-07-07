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
    use crate::{gates::h_dence_matrix, transpiler::t_dence_matrix};

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
