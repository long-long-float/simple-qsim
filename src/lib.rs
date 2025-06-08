pub mod circuit;
pub mod gates;
pub mod qstate;
pub mod test_util;

use num_complex::Complex;

pub type Qbit = Complex<f64>;
pub type QState = qstate::QState;
pub type Circuit = circuit::Circuit;
