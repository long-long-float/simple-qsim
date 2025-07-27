use std::f64::consts::PI;

use anyhow::Result;
use simple_qsim::{circuit::GateKind, Circuit, QState};

fn main() -> Result<()> {
    let num_of_qubits = 3;

    let mut circuit = Circuit::new(num_of_qubits);

    // Initialize input with H
    for i in 0..num_of_qubits {
        circuit.add_gate(GateKind::H, i);
    }

    for i in (0..num_of_qubits).rev() {
        circuit.add_gate(GateKind::H, i);
        for j in (0..i).rev() {
            let angle = 2.0 * PI / 2.0f64.powi((num_of_qubits - j) as i32);
            circuit.add_control(i, j, GateKind::Phase(angle))?;
        }
    }

    println!("Circuit:\n{}", circuit);

    let qs = QState::from_str("000")?;

    let result = circuit.apply(&qs)?;
    println!("Resulting state:\n{}", result);

    Ok(())
}
