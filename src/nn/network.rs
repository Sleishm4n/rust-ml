use crate::{nn::Layer, Matrix};

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
}

impl Network {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Network {
        Network { layers }
    }

    pub fn forward(&mut self, input: &Matrix) -> Matrix {
        let mut current = input.clone();
        for layer in &mut self.layers {
            current = layer.forward_pass(&current);
        }
        current
    }

    pub fn backward(&mut self, input: &Matrix) -> Matrix {
        let mut current = input.clone();
        for layer in self.layers.iter_mut().rev() {
            current = layer.backward_pass(&current);
        }
        current
    }

    pub fn update(&mut self, lr: f32) {
        for layer in &mut self.layers {
            layer.update(lr);
        }
    }
}
