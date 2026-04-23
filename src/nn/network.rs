use crate::{nn::Layer, optimiser::Optimiser, tensor::Tensor};

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
}

impl Network {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Network {
        Network { layers }
    }

    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        let mut current = input.clone();
        for layer in &mut self.layers {
            current = layer.forward_pass(&current);
        }
        current
    }

    pub fn backward(&mut self, input: &Tensor) -> Tensor {
        let mut current = input.clone();
        for layer in self.layers.iter_mut().rev() {
            current = layer.backward_pass(&current);
        }
        current
    }

    pub fn update(&mut self, optimiser: &mut dyn Optimiser) {
        let mut all_params: Vec<Tensor> = vec![];
        let mut all_grads: Vec<Tensor> = vec![];
        let mut counts: Vec<usize> = vec![];

        for layer in &mut self.layers {
            let params = layer.get_params();
            let grads = layer.get_grads();
            counts.push(params.len());
            all_params.extend(params);
            all_grads.extend(grads);
        }

        optimiser.step(&mut all_params, &all_grads);

        // redistribute back
        let mut idx = 0;
        for (layer, count) in self.layers.iter_mut().zip(counts.iter()) {
            layer.set_params(all_params[idx..idx + count].to_vec());
            idx += count;
        }
    }
}
