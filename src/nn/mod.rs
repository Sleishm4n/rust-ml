use crate::matrix::Matrix;

pub mod activation;
pub mod linear;
pub mod network;
pub mod softmax;

pub trait Layer {
    fn forward_pass(&mut self, input: &Matrix) -> Matrix;
    fn backward_pass(&mut self, d_output: &Matrix) -> Matrix;
    fn set_params(&mut self, params: Vec<Matrix>);
    fn get_params(&self) -> Vec<Matrix>;
    fn get_grads(&self) -> Vec<Matrix>; 
}