use std::fmt::Display;

use anyhow::Result;
use nalgebra::DVector;
use num_complex::Complex;

use crate::Qbit;

pub struct QState {
    pub(crate) state: DVector<Qbit>,
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

    pub fn zero_state(num_of_qbits: usize) -> Self {
        let size = 2_usize.pow(num_of_qbits as u32);
        let mut state = DVector::zeros(size);
        state[0] = Complex::new(1.0, 0.0); // |0...0> state
        Self { state }
    }

    pub fn from_str(qbits: &str) -> Result<Self> {
        let index = usize::from_str_radix(qbits, 2)?;
        let mut state = DVector::zeros(2_usize.pow(qbits.len() as u32));
        state[index] = Complex::new(1.0, 0.0);

        Ok(Self { state })
    }

    pub fn num_of_qbits(&self) -> usize {
        self.state.len().ilog2() as usize
    }
}

impl Display for QState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bin_width = self.num_of_qbits();

        for (i, value) in self.state.iter().enumerate() {
            writeln!(f, "|{:0width$b}>: {}", i, value, width = bin_width)?;
        }

        Ok(())
    }
}

impl From<QState> for DVector<Qbit> {
    fn from(qstate: QState) -> Self {
        qstate.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_approx_complex_eq;

    #[test]
    fn test_qstate_from_2bit_str() {
        let qstate = QState::from_str("00").unwrap();

        assert_eq!(qstate.num_of_qbits(), 2);
        assert_eq!(qstate.state.len(), 4);

        assert_approx_complex_eq!(1.0, 0.0, qstate.state[0]);
        assert_approx_complex_eq!(0.0, 0.0, qstate.state[1]);
        assert_approx_complex_eq!(0.0, 0.0, qstate.state[2]);
        assert_approx_complex_eq!(0.0, 0.0, qstate.state[3]);

        let qstate = QState::from_str("01").unwrap();
        assert_approx_complex_eq!(0.0, 0.0, qstate.state[0]);
        assert_approx_complex_eq!(1.0, 0.0, qstate.state[1]);
        assert_approx_complex_eq!(0.0, 0.0, qstate.state[2]);
        assert_approx_complex_eq!(0.0, 0.0, qstate.state[3]);

        let qstate = QState::from_str("11").unwrap();
        assert_approx_complex_eq!(0.0, 0.0, qstate.state[0]);
        assert_approx_complex_eq!(0.0, 0.0, qstate.state[1]);
        assert_approx_complex_eq!(0.0, 0.0, qstate.state[2]);
        assert_approx_complex_eq!(1.0, 0.0, qstate.state[3]);
    }

    #[test]
    fn test_qstate_from_3bit_str() {
        let qstate = QState::from_str("100").unwrap();

        assert_eq!(qstate.num_of_qbits(), 3);
        assert_eq!(qstate.state.len(), 8);

        assert_approx_complex_eq!(1.0, 0.0, qstate.state[4]);
    }
}
