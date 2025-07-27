# simple-qsim

[![Rust](https://github.com/long-long-float/simple-qsim/actions/workflows/rust.yml/badge.svg)](https://github.com/long-long-float/simple-qsim/actions/workflows/rust.yml)

simple-qsim is a simple quantum circuit simulator written in Rust. It simulates quantum states with quantum gates with simple interface.

## Features

- Basic gates (X, H, CNOT, etc.)
- Observable for Pauli operators (I, X, Y, Z)
- Parametric gates for RX, RY, RZ
- Transpiler to gate sets `{H, T, T^-1}` by Solovay-Kitaev algorithm (Thanks to [sk](https://github.com/cmdawson/sk))
    - Currently, only single-qubit gates are supported.

## Installation

```bash
cargo add simple-qsim
```

## Usage

```rust
use simple_qsim::{Circuit, QState};

let q00 = QState::from_str("00").unwrap();
let bell_state = Circuit::new(q00.num_of_qbits())
    .H(0)?
    .cnot(0, 1)?
    .apply(&q00)?;
println!("{}", bell_state);

/*
|00>: 0.7071067811865475+0i
|01>: 0+0i
|10>: 0+0i
|11>: 0.7071067811865475+0i
*/
```

You can see some examples in [examples directory](./examples/):

* [QCL](./examples/qcl.rs) Quantum Circuit learning
    * Train sin function with noise.
* [VQE](./examples/vqe.rs) Variational Quantum Eigensolver
    * Seek the ground state energy of H-He+ (helium hydride ion)

You can run examples by:

```
cargo run --example qcl
```

## Test

```
cargo test
```

## License

This project is licensed under the MIT License.
