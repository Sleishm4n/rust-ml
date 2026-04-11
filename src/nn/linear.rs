use crate::tensor::Matrix;

pub struct LinearLayer {
    pub weight: Matrix,
    pub bias: Matrix,
    pub in_features: usize,
    pub out_features: usize,
}

impl LinearLayer {
    pub fn new(in_features: usize, out_features: usize) -> LinearLayer {
        LinearLayer {
            weight: Matrix::new(out_features, in_features),
            bias: Matrix::new(out_features, 1),
            in_features,
            out_features,
        }
    }

    pub fn new_rand(in_features: usize, out_features: usize) -> LinearLayer {
        LinearLayer {
            weight: Matrix::rand_range(out_features, in_features, -0.5, 0.5),
            bias: Matrix::new(out_features, 1),
            in_features,
            out_features,
        }
    }

    pub fn forward(&self, input: &Matrix) -> Matrix {
        self.weight.matmul(input).add(&self.bias)
    }

    pub fn backward(&self, d_output: &Matrix, input: &Matrix) -> (Matrix, Matrix, Matrix) {
        let d_w = d_output.matmul(&input.transpose());
        let d_b = d_output.clone();
        let d_x = self.weight.transpose().matmul(d_output);
        (d_w, d_b, d_x)
    }

    pub fn sgd_update(&mut self, d_w: &Matrix, d_b: &Matrix, lr: f32) {
        self.weight = self.weight.sub(&d_w.scale(lr));
        self.bias = self.bias.sub(&d_b.scale(lr));
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
        let forward_res = layer.forward(&input);
        let d_output = &Matrix::from_vec(1, 1, vec![1.0]);
        assert_eq!(forward_res.data, vec![1.0]);
        let (d_w, d_b, d_x) = layer.backward(d_output, &input);
        assert_eq!(d_w.data, vec![1.0, 0.0]);
        assert_eq!(d_b.data, vec![1.0]);
        assert_eq!(d_x.data, vec![1.0, 0.0]);
    }
}