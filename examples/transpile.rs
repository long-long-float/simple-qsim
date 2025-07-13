use core::f64;
use std::collections::VecDeque;

use anyhow::Result;
use nalgebra::{Complex, DVector, Matrix4, Vector6};
use rand::Rng;
use simple_qsim::{
    circuit::GateKind, gates::rz_dence_matrix, net::Net, su2, Circuit, QState, Qbit,
};

fn main() -> Result<()> {
    let mut net = Net::new(0.18);
    net.generate(14);

    let u = rz_dence_matrix(f64::consts::PI / 2.0);
    println!("U = {}", u);

    let ska = net.solovay_kitaev(&u, 5)?;
    // println!("Solovay-Kitaev result: {:?}", ska);
    println!("Matrix: {}", ska.matrix);
    println!("Length of the sequence: {}", ska.word.len());
    println!("accuracy: {}", su2::proj_trace_dist(&ska.matrix, &u));

    let approx = net.evaluate(&ska.word)?;
    println!("Approximation result: {}", approx);

    println!("  {}", su2::proj_trace_dist(&ska.matrix, &approx));

    println!("{}", su2::equals_ignoring_global_phase(&u, &ska.matrix));
    println!("{}", su2::equals_ignoring_global_phase(&u, &approx));

    Ok(())
}
