use crate::tensor::Matrix;

pub mod activation;
pub mod linear;
pub mod network;
pub mod softmax;

pub trait Layer {
    fn forward_pass(&mut self, input: &Matrix) -> Matrix;
    fn backward_pass(&mut self, d_output: &Matrix) -> Matrix;
    fn update(&mut self, lr: f32);
}