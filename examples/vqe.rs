//! This example implements a Variational Quantum Eigensolver (VQE) using Powell's method.
//! This is a port of https://dojo.qulacs.org/ja/latest/notebooks/5.1_variational_quantum_eigensolver.html

use core::f64;
use std::collections::VecDeque;

use anyhow::Result;
use nalgebra::{Complex, DVector, Matrix4, Vector6};
use rand::Rng;
use simple_qsim::{
    gates::{rx_matrix, rz_matrix},
    Circuit, QState, Qbit,
};

fn run_pqc_circuit(phi: &Vector6<f64>) -> Result<QState> {
    let q = QState::from_str("00")?;
    let circuit = Circuit::new(2)
        .gate_at(0, rx_matrix(phi[0]))?
        .gate_at(0, rz_matrix(phi[1]))?
        .gate_at(1, rx_matrix(phi[2]))?
        .gate_at(1, rz_matrix(phi[3]))?
        .cnot(1, 0)?
        .gate_at(1, rz_matrix(phi[4]))?
        .gate_at(1, rx_matrix(phi[5]))?;
    circuit.apply(&q)
}

fn expect_val(operator: &Matrix4<Qbit>, state: &DVector<Qbit>) -> f64 {
    let bra_state = state.transpose();
    let energy = bra_state * operator * state;
    energy[0].re
}

fn cost(phi: &Vector6<f64>, hamiltonian: &Matrix4<Qbit>) -> Result<f64> {
    let state = run_pqc_circuit(phi)?;
    Ok(expect_val(hamiltonian, &state.into()))
}

fn main() -> Result<()> {
    // Prepare the Hamiltonian
    // This is calculated from https://dojo.qulacs.org/ja/latest/notebooks/5.1_variational_quantum_eigensolver.html#%E3%83%8F%E3%83%9F%E3%83%AB%E3%83%88%E3%83%8B%E3%82%A2%E3%83%B3%E3%82%92%E6%BA%96%E5%82%99%E3%81%99%E3%82%8B
    let hamiltonian = Matrix4::from_row_slice(&[
        Complex::new(-2.85405, 0.0),
        Complex::ZERO,
        Complex::ZERO,
        Complex::new(0.13065, 0.0),
        Complex::ZERO,
        Complex::new(-2.04305, 0.0),
        Complex::new(0.13065, 0.0),
        Complex::new(-0.2288, 0.0),
        Complex::ZERO,
        Complex::new(0.13065, 0.0),
        Complex::new(-2.04305, 0.0),
        Complex::new(-0.2288, 0.0),
        Complex::new(0.13065, 0.0),
        Complex::new(-0.2288, 0.0),
        Complex::new(-0.2288, 0.0),
        Complex::new(-0.76085, 0.0),
    ]);

    let mut rng = rand::rng();
    let mut phi0 = Vector6::zeros();
    for x in &mut phi0 {
        *x = rng.random_range(0.0..f64::consts::TAU);
    }

    // Find the minimum cost by Powell's method

    let search_vec_size = 6;
    let mut search_vecs = VecDeque::new();
    for i in 0..search_vec_size {
        let mut vec = Vector6::zeros();
        vec[i] = 1.0; // Initialize the search vector with a unit vector in the i-th direction
        search_vecs.push_back(vec);
    }

    let mut phi = phi0;

    for i in 0..10 {
        let mut alphas = Vec::new();

        // Find the minimum costs for each search vector
        for search_vec in &search_vecs {
            let delta = f64::consts::PI / 1000.0;

            if search_vec.norm() < 1e-10 {
                continue; // Skip if the search vector is too small
            }

            let (best_alpha_pos, min_cost_pos) =
                find_min_alpha(&phi, search_vec, delta, |p| cost(p, &hamiltonian))?;
            let (best_alpha_neg, min_cost_neg) =
                find_min_alpha(&phi, search_vec, -delta, |p| cost(p, &hamiltonian))?;

            let best_alpha = if min_cost_pos < min_cost_neg {
                best_alpha_pos
            } else {
                best_alpha_neg
            };

            phi += best_alpha * search_vec;

            alphas.push((best_alpha * search_vec).norm());
        }

        let max_idx = alphas
            .iter()
            .enumerate()
            .max_by(|(_, &x), (_, &y)| x.total_cmp(&y))
            .map(|(idx, _)| idx)
            .unwrap(); // The maximum value must exist
        search_vecs.remove(max_idx);
        search_vecs.push_back(phi - phi0);

        let sum_norm: f64 = search_vecs.iter().map(|sv| sv.norm()).sum();
        if sum_norm < 1e-10 {
            break;
        }

        phi0 = phi;

        let cost_value = cost(&phi0, &hamiltonian)?;
        println!("Iteration: {}, Cost: {}", i, cost_value);
    }

    let final_cost = cost(&phi, &hamiltonian)?;
    println!("Final cost: {}", final_cost);
    println!("Final parameters: {:?}", phi);

    Ok(())
}

fn find_min_alpha<F>(
    phi: &Vector6<f64>,
    search_vec: &Vector6<f64>,
    delta: f64,
    cost_fun: F,
) -> Result<(f64, f64)>
where
    F: Fn(&Vector6<f64>) -> Result<f64>,
{
    let mut min_cost = cost_fun(phi)?;
    let mut best_alpha = 0.0;

    let mut curr_alpha = 0.0_f64;

    while (curr_alpha * search_vec).norm() < f64::consts::TAU {
        curr_alpha += delta;

        let new_phi = curr_alpha * search_vec + phi;
        let new_cost = cost_fun(&new_phi)?;

        if new_cost < min_cost {
            min_cost = new_cost;
            best_alpha = curr_alpha;
        }
    }

    Ok((best_alpha, min_cost))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector6;
    use simple_qsim::assert_approx_eq;

    #[test]
    fn test_find_min_alpha() -> Result<()> {
        let phi = Vector6::from_row_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let search_vec = Vector6::from_row_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let delta = -0.01;

        let (best_alpha, min_cost) = find_min_alpha(&phi, &search_vec, delta, |x| Ok(x[0] * x[0]))?;

        assert_approx_eq!(-1.0, best_alpha);
        assert_approx_eq!(0.0, min_cost);

        Ok(())
    }
}
