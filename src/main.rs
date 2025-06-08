use anyhow::Result;
use nalgebra::Complex;
use simple_qsim::{
    gates::{build_hadamard_matrix, build_s_matrix, build_t_matrix},
    Circuit, QState,
};

fn main() -> Result<()> {
    let q00 = QState::from_str("00")?;
    println!("{}", q00);

    // Bell state |00> + |11>
    let result = Circuit::new(q00.len()).H(0)?.cnot(0, 1)?.apply(&q00);
    println!("{}", result);

    // Hadamard test
    let result = Circuit::new(q00.len())
        .H(0)?
        .control(0, 1, &build_hadamard_matrix())?
        .H(0)?
        .apply(&q00);
    println!("{}", result);

    // Quantum Fourier Transform
    let qstate = QState::new(&[Complex::new(1.0, 0.0) / 8.0_f64.sqrt(); 8])?;
    println!("{}", qstate);

    let qft = Circuit::new(qstate.len())
        // First bit
        .H(0)?
        .control(1, 0, &build_s_matrix())?
        .control(2, 0, &build_t_matrix())?
        // Second bit
        .H(1)?
        .control(2, 1, &build_s_matrix())?
        // Third bit
        .H(2)?
        .swap(0, 2)?;

    let result = qft.apply(&qstate);
    println!("{}", result);

    Ok(())
}
