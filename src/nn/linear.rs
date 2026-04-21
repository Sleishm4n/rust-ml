use crate::{nn::Layer, matrix::Matrix};

#[derive(Clone)]
pub struct LinearLayer {
    pub weight: Matrix,
    pub bias: Matrix,
    pub in_features: usize,
    pub out_features: usize,
    pub input: Option<Matrix>,
    pub d_weight: Option<Matrix>,
    pub d_bias: Option<Matrix>,
}

impl LinearLayer {
    pub fn new(in_features: usize, out_features: usize) -> LinearLayer {
        LinearLayer {
            weight: Matrix::new(out_features, in_features),
            bias: Matrix::new(out_features, 1),
            in_features,
            out_features,
            input: None,
            d_weight: None,
            d_bias: None,
        }
    }

    pub fn new_rand(in_features: usize, out_features: usize) -> LinearLayer {
        LinearLayer {
            weight: Matrix::rand_range(out_features, in_features, -0.01, 0.01),
            bias: Matrix::new(out_features, 1),
            in_features,
            out_features,
            input: None,
            d_weight: None,
            d_bias: None,
        }
    }

    pub fn forward(&mut self, input: &Matrix) -> Matrix {
        self.input = Some(input.clone());
        self.weight.matmul(input).add(&self.bias)
    }

    pub fn backward(&self, d_output: &Matrix) -> (Matrix, Matrix, Matrix) {
        let d_w = d_output.matmul(&self.input.as_ref().unwrap().transpose());
        let d_b = d_output.clone();
        let d_x = self.weight.transpose().matmul(d_output);
        (d_w, d_b, d_x)
    }
    
    pub fn get_weights_and_bias(&self) -> (&Matrix, &Matrix) {
        (&self.weight, &self.bias)
    }
}

impl Layer for LinearLayer {
    fn forward_pass(&mut self, input: &Matrix) -> Matrix {
        self.forward(input)
    }

    fn backward_pass(&mut self, d_output: &Matrix) -> Matrix {
        self.d_weight = Some(d_output.matmul(&self.input.as_ref().unwrap().transpose()));
        self.d_bias = Some(d_output.clone());
        let d_x = self.weight.transpose().matmul(d_output);
        d_x
    }

    fn get_params(&self) -> Vec<Matrix> {
        vec![self.weight.clone(), self.bias.clone()]
    }
    fn get_grads(&self) -> Vec<Matrix> {
        vec![
            self.d_weight
                .as_ref()
                .cloned()
                .unwrap_or_else(|| Matrix::new(self.out_features, self.in_features)),
            self.d_bias
                .as_ref()
                .cloned()
                .unwrap_or_else(|| Matrix::new(self.out_features, 1)),
        ]
    }
    fn set_params(&mut self, params: Vec<Matrix>) {
        self.weight = params[0].clone();
        self.bias = params[1].clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward() {
        let mut layer = LinearLayer::new(3, 1);
        layer.weight = Matrix::from_vec(1, 3, vec![1.0, 1.0, 1.0]);
        let input = Matrix::from_vec(3, 1, vec![2.0, 2.0, 2.0]);
        let forward_res = layer.forward(&input);
        assert_eq!(forward_res.data, vec![6.0]);
    }

    #[test]
    fn test_backward() {
        let mut layer = LinearLayer::new(2, 1);
        layer.weight = Matrix::from_vec(1, 2, vec![1.0, 0.0]);
        let input = Matrix::from_vec(2, 1, vec![1.0, 0.0]);
        let d_output = &Matrix::from_vec(1, 1, vec![1.0]);
        layer.forward(&input);
        let (d_w, d_b, d_x) = layer.backward(d_output);
        assert_eq!(d_w.data, vec![1.0, 0.0]);
        assert_eq!(d_b.data, vec![1.0]);
        assert_eq!(d_x.data, vec![1.0, 0.0]);
    }

    #[test]
    fn test_backward_nontrivial() {
        let mut layer = LinearLayer::new(2, 2);
        layer.weight = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let input = Matrix::from_vec(2, 1, vec![5.0, 6.0]);
        let d_output = Matrix::from_vec(2, 1, vec![1.0, 1.0]);
        layer.forward(&input);
        let (d_w, d_b, d_x) = layer.backward(&d_output);
        assert_eq!(d_w.data, vec![5.0, 6.0, 5.0, 6.0]);
        assert_eq!(d_b.data, vec![1.0, 1.0]);
        assert_eq!(d_x.data, vec![4.0, 6.0]);
    }
}
