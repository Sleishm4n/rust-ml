use crate::{Matrix, nn::Layer};

pub fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn tanh(x: f32) -> f32 {
    (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())
}

pub fn d_relu(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

pub fn d_sigmoid(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

pub fn d_tanh(x: f32) -> f32 {
    1.0 - tanh(x).powi(2)
}

pub struct ActivationLayer {
    pub function: fn(f32) -> f32,
    pub derivative: fn(f32) -> f32,
    pub input: Option<Matrix>,
}

impl ActivationLayer {
    pub fn new(function: fn(f32) -> f32, derivative: fn(f32) -> f32) -> Self {
        ActivationLayer {
            function,
            derivative,
            input: None,
        }
    }
}

impl Layer for ActivationLayer {
    fn forward_pass(&mut self, input: &Matrix) -> Matrix {
        self.input = Some(input.clone());
        input.map(self.function)
    }

    fn backward_pass(&mut self, d_output: &Matrix) -> Matrix {
        let input = self.input.as_ref().unwrap();
        input.map(self.derivative).zip_map(d_output, |d, g| d * g)
    }

    fn update(&mut self, _lr: f32) {}
}