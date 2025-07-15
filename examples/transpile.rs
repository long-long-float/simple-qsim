use core::f64;
use std::f64::consts::PI;

use anyhow::Result;
use simple_qsim::{
    circuit::GateKind, gates::rz_dence_matrix, net::Net, qstate::QState, su2, Circuit,
};

fn main() -> Result<()> {
    let mut circuit = Circuit::new(2).H(0)?.gate_at(0, GateKind::RY(PI / 4.0))?;
    let qs = QState::from_str("00")?;

    let result = circuit.apply(&qs)?;
    println!("Resulting state:\n{}", result);

    circuit.transpile()?;

    let result2 = circuit.apply(&qs)?;

    println!("Resulting state after transpile:\n{}", result2);

    Ok(())
}
