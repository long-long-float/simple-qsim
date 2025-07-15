use std::f64::consts::PI;

use anyhow::Result;
use simple_qsim::{circuit::GateKind, qstate::QState, Circuit};

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
