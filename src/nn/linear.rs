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
}