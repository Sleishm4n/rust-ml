use crate::matrix::Matrix;

pub mod adam;

pub trait Optimiser {
    fn step(&mut self, params: &mut Vec<Matrix>, grads: &Vec<Matrix>);
} 