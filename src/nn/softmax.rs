use crate::{nn::Layer, tensor::Tensor};

pub struct SoftmaxLayer {
    pub input: Option<Tensor>,
}

impl SoftmaxLayer {
    pub fn new() -> Self {
        SoftmaxLayer { input: None }
    }
}

impl Layer for SoftmaxLayer {
    fn forward_pass(&mut self, input: &Tensor) -> Tensor {
        let max = input.tensor_max();
        let exps = input.map(|x| (x - max).exp());
        let sum = exps.data.iter().sum::<f32>();
        let output = exps.map(|x| x / sum);
        self.input = Some(output.clone());
        output
    }

    fn backward_pass(&mut self, d_output: &Tensor) -> Tensor {
        let s = self.input.as_ref().unwrap();
        let dot: f32 = d_output.zip_map(s, |a, b| a * b).data.iter().sum();
        let shifted = d_output.map(|x| x - dot);
        s.zip_map(&shifted, |a, b| a * b)
    }

    fn get_params(&self) -> Vec<Tensor> {
        vec![]
    }
    fn get_grads(&self) -> Vec<Tensor> {
        vec![]
    }
    fn set_params(&mut self, _params: Vec<Tensor>) {}
}
