# rust-ml 

A neural network built from scratch in Rust
The goal is to understand how they are implemented and how they fundamentally work
This involves understanding every operation from the maths to the memory layout

The validation target at the current stage is a network that solves XOR

---

## Project Structure
 
```
src/
├── tensor/
│   ├── matrix.rs   # Matrix struct and core operations
│   ├── ops.rs
│   ├── init.rs     # Weight initialisation
│   └── mod.rs
├── nn/
│   ├── activation.rs  # ReLU, sigmoid, tanh
│   ├── linear.rs      # LinearLayer (forward pass)
│   └── mod.rs
examples/
└── basics.rs
```

## What's Implemented
 
**Matrix (`src/tensor/matrix.rs`)**
- Flat `Vec<f32>` row-major storage
- `new`, `get`, `set`, `from_vec`, `display`
- `scale`, `scale_inplace`, `add`, `sub`
- `matmul` — cache-friendly i-k-j loop order
- `transpose`
 
**Activation functions (`src/nn/activation.rs`)**
- ReLU, sigmoid, tanh
 
**LinearLayer (`src/nn/linear.rs`)**
- Weight shape: `(out_features × in_features)`
- Bias stored as `Matrix`
- `forward(&input)` computes `W·x + b`
 
## Dependencies
 
```toml
[dependencies]
rand = "0.8"
```

`rand` is the only external crate, used for weight initialisation. 
Everything else is standard library.

## Running
 
```bash
cargo run --example basics
cargo test
```

## Design Notes
 
- Weights are `(out × in)` — consistent with the convention that `forward` computes `W·x`, where `x` is a column vector.
- `matmul` uses i-k-j loop order intentionally for cache performance; don't reorder.
- Tests use `Matrix::from_vec` with known values and epsilon comparison — no random inputs in correctness tests.
