use core::f64;
use std::collections::VecDeque;

use anyhow::Result;
use nalgebra::{Complex, DVector, Matrix4, Vector6};
use rand::Rng;
use simple_qsim::{circuit::GateKind, gates::h_dence_matrix, net::Net, Circuit, QState, Qbit};

fn main() -> Result<()> {
    let mut net = Net::new(0.18);
    net.generate(14);

    let ska = net.solovay_kitaev(&h_dence_matrix(), 5);
    println!("Solovay-Kitaev result: {:?}", ska);

    Ok(())
}
