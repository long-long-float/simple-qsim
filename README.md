# simple-qsim

[![Rust](https://github.com/long-long-float/simple-qsim/actions/workflows/rust.yml/badge.svg)](https://github.com/long-long-float/simple-qsim/actions/workflows/rust.yml)

simple-qsim is a simple quantum circuit simulator written in Rust. It simulates quantum states with quantum gates with simple interface.

## Features

- Basic gates (X, H, CNOT, etc.)
- Measure qubits and analyze results

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
    .apply(&q00);
println!("{}", bell_state);

/*
|00>: 0.7071067811865475+0i
|01>: 0+0i
|10>: 0+0i
|11>: 0.7071067811865475+0i
*/
```

For more examples, please see tests in [circuit.rs](./src/circuit.rs).

## License

This project is licensed under the MIT License.
