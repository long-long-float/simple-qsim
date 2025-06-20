//! This example implements a Quantum Circuit Learning (QCL)
//! This is a port of https://dojo.qulacs.org/ja/latest/notebooks/5.2_Quantum_Circuit_Learning.html

use core::f64;
use std::f64::consts::PI;

use anyhow::Result;
use nalgebra::{Complex, DMatrix};
use plotters::prelude::*;
use rand::Rng;
use simple_qsim::{
    gates::{rx_matrix, ry_matrix, rz_matrix},
    Circuit, QState, Qbit,
};

fn prepare_train_data() -> Result<(Vec<f64>, Vec<f64>)> {
    let x_min = -1.0;
    let x_max = 1.0;
    let num_x_train = 50;

    fn func_to_learn(x: f64) -> f64 {
        (x * PI).sin()
    }

    let mut rng = rand::rng();
    let mut x_train: Vec<f64> = (0..num_x_train)
        .map(|_| x_min + (x_max - x_min) * rng.random::<f64>())
        .collect();
    x_train.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mag_noise = 0.05;
    let norm_dist = rand_distr::Normal::new(0.0, mag_noise)?;
    let y_train = x_train
        .iter()
        .map(|&x| func_to_learn(x) + rng.sample(norm_dist))
        .collect::<Vec<_>>();

    Ok((x_train, y_train))
}

fn plot_data(x_data: &[f64], y_data: &[f64], file_name: &str) -> Result<()> {
    let root = BitMapBackend::new(file_name, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-1f64..1f64, -1f64..1f64)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(
        x_data
            .iter()
            .zip(y_data.iter())
            .map(|(&x, &y)| Circle::new((x, y), 3, RED.filled())),
    )?;

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

fn main() -> Result<()> {
    let nqubit = 3;
    let c_depth = 3;
    let time_step = 0.77;

    // Prepare training data
    let (x_train, y_train) = prepare_train_data()?;
    plot_data(&x_train, &y_train, "train.png")?;

    // Create initial state
    let state = QState::zero_state(nqubit);

    fn u_in(x: f64, nqubit: usize) -> Result<Circuit> {
        let mut u = Circuit::new(nqubit);

        let angle_y = x.asin();
        let angle_z = (x * x).acos();

        for i in 0..nqubit {
            u.add_gate_at_mut(i, ry_matrix(angle_y))?;
            u.add_gate_at_mut(i, rz_matrix(angle_z))?;
        }

        Ok(u)
    }

    // Check U_in
    // println!("{}", u_in(0.1, nqubit)?.apply(&state));

    let time_evol_op = time_evol_op_for_3qubit();

    let mut u_out = Circuit::new(nqubit);

    let mut rng = rand::rng();
    for _ in 0..c_depth {
        u_out.add_dence_gate(time_evol_op.clone());
        for i in 0..nqubit {
            // TODO: Support parameterized gates
            let angle = 2.0 * PI * rng.random::<f64>();
            u_out.add_gate_at_mut(i, rx_matrix(angle))?;
            let angle = 2.0 * PI * rng.random::<f64>();
            u_out.add_gate_at_mut(i, rz_matrix(angle))?;
            let angle = 2.0 * PI * rng.random::<f64>();
            u_out.add_gate_at_mut(i, rx_matrix(angle))?;
        }
    }

    // TODO: Support observables
    // obs = Observable::new(nqubit)?;
    // obs.add_operator(2.0, 'Z 0)?;

    Ok(())
}

// This is computed by ext/compute_time_evol_op
fn time_evol_op_for_3qubit() -> DMatrix<Qbit> {
    DMatrix::from_row_slice(
        8,
        8,
        &[
            Complex::new(0.4502467309110266, 0.3989530928005264),
            Complex::new(0.09127768914919739, -0.3898217983118638),
            Complex::new(-0.21116625840396, 0.15526661742554448),
            Complex::new(0.15323067719786393, -0.019275471961844055),
            Complex::new(-0.19572522532670836, -0.3999361624040975),
            Complex::new(-0.3408643369229124, -0.08980539004289571),
            Complex::new(0.1956517811210468, 0.014323824626186366),
            Complex::new(0.02897812566021865, -0.1530889916860413),
            Complex::new(0.09127768914919737, -0.3898217983118638),
            Complex::new(0.5408644987409207, -0.1685973910633376),
            Complex::new(0.1523584015039502, 0.0215817794028314),
            Complex::new(0.20212497053543885, 0.14333732469272967),
            Complex::new(-0.3416246292810889, 0.08509693027683517),
            Complex::new(0.21352666943961596, -0.448666258109923),
            Complex::new(-0.008387660766444774, -0.15578782306263497),
            Complex::new(0.1956517811210468, 0.01432382462618683),
            Complex::new(-0.21116625840396, 0.15526661742554448),
            Complex::new(0.1523584015039502, 0.021581779402831392),
            Complex::new(0.2594865368165453, 0.5560193013892835),
            Complex::new(-0.07638424203262105, -0.3019861576213608),
            Complex::new(0.19700633729908354, -0.011474722524068497),
            Complex::new(0.048362714399835256, -0.14720951956132353),
            Complex::new(0.21352666943961587, -0.4486662581099231),
            Complex::new(-0.34086433692291257, -0.08980539004289574),
            Complex::new(0.15323067719786393, -0.019275471961844023),
            Complex::new(0.20212497053543885, 0.14333732469272964),
            Complex::new(-0.07638424203262105, -0.3019861576213608),
            Complex::new(0.013120702675468858, -0.65787838783485),
            Complex::new(-0.06495359781934837, -0.13919086454595508),
            Complex::new(0.19700633729908346, -0.011474722524068627),
            Complex::new(-0.3416246292810889, 0.08509693027683513),
            Complex::new(-0.19572522532670825, -0.3999361624040974),
            Complex::new(-0.19572522532670836, -0.3999361624040975),
            Complex::new(-0.3416246292810889, 0.08509693027683517),
            Complex::new(0.19700633729908357, -0.011474722524068491),
            Complex::new(-0.06495359781934837, -0.13919086454595508),
            Complex::new(0.0131207026754692, -0.65787838783485),
            Complex::new(-0.076384242032621, -0.3019861576213607),
            Complex::new(0.20212497053543885, 0.14333732469272942),
            Complex::new(0.153230677197864, -0.01927547196184382),
            Complex::new(-0.34086433692291235, -0.08980539004289571),
            Complex::new(0.21352666943961598, -0.448666258109923),
            Complex::new(0.04836271439983526, -0.14720951956132353),
            Complex::new(0.19700633729908346, -0.01147472252406864),
            Complex::new(-0.076384242032621, -0.3019861576213607),
            Complex::new(0.25948653681654515, 0.5560193013892836),
            Complex::new(0.1523584015039505, 0.021581779402831316),
            Complex::new(-0.21116625840396028, 0.1552666174255442),
            Complex::new(0.1956517811210468, 0.014323824626186376),
            Complex::new(-0.00838766076644478, -0.155787823062635),
            Complex::new(0.2135266694396159, -0.4486662581099231),
            Complex::new(-0.3416246292810889, 0.08509693027683511),
            Complex::new(0.20212497053543885, 0.14333732469272942),
            Complex::new(0.1523584015039505, 0.021581779402831302),
            Complex::new(0.5408644987409208, -0.16859739106333757),
            Complex::new(0.09127768914919757, -0.3898217983118638),
            Complex::new(0.028978125660218703, -0.15308899168604131),
            Complex::new(0.19565178112104678, 0.014323824626186815),
            Complex::new(-0.3408643369229126, -0.08980539004289571),
            Complex::new(-0.19572522532670825, -0.3999361624040974),
            Complex::new(0.153230677197864, -0.01927547196184383),
            Complex::new(-0.21116625840396028, 0.1552666174255442),
            Complex::new(0.0912776891491976, -0.38982179831186375),
            Complex::new(0.4502467309110268, 0.39895309280052693),
        ],
    )
}
