use core::f64;
use std::collections::VecDeque;

use anyhow::Result;
use nalgebra::{Complex, DVector, Matrix4, Vector6};
use rand::Rng;
use simple_qsim::{circuit::GateKind, transpiler::Net, Circuit, QState, Qbit};

fn main() -> Result<()> {
    let net = Net::new();
    net.generate(14);

    Ok(())
}
