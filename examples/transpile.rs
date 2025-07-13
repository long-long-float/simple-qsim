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

    let u = rz_dence_matrix(f64::consts::PI);
    println!("U = {}", u);

    let ska = net.solovay_kitaev(&u, 5)?;
    println!("Solovay-Kitaev result: {:?}", ska);
    println!("accuracy: {}", su2::proj_trace_dist(&ska.matrix, &u));

    Ok(())
}
