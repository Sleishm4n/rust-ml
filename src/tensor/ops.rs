use crate::tensor::Tensor;
use std::f32::INFINITY;

impl Tensor {
    pub fn add(&self, other: &Tensor) -> Tensor {
        self.zip_map(other, |a, b| a + b)
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        self.zip_map(other, |a, b| a - b)
    }

    pub fn scale(&self, n: f32) -> Tensor {
        self.map(|x| x * n)
    }

    pub fn tensor_max(&self) -> f32 {
        let mut max = -INFINITY;

        for val in &self.data {
            if val > &max  {
                max = *val;
            }
        }
        max
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert!(self.shape.len() == 2);
        assert!(self.shape[1] == other.shape[0]);

        let mut c = Tensor::new(vec![self.shape[0], other.shape[1]]);

        for i in 0..self.shape[0] {
            for k in 0..self.shape[1] {
                let a = self.get(&[i, k]);

                for j in 0..other.shape[1] {
                    let prev = c.get(&[i, j]);
                    c.set(&[i, j], prev + a * other.get(&[k, j]));
                }
            }
        }
        c
    }

    pub fn transpose(&self) -> Tensor {
        self.permute(&[1, 0])
    }

    pub fn permute(&self, axes: &[usize]) -> Tensor {
        let mut new_shape: Vec<usize> = Vec::new();
        let mut new_strides: Vec<usize> = Vec::new();

        for &axis in axes {
            new_shape.push(self.shape[axis]);
            new_strides.push(self.strides[axis]);
        }

        Tensor {
            shape: new_shape,
            strides: new_strides,
            data: self.data.clone(),
        }
    }

    pub fn map<F: Fn(f32) -> f32>(&self, f: F) -> Tensor {
        let mut result = Tensor::new(self.shape.clone());
        let mut index = vec![0; self.shape.len()];

        loop {
            let val = self.get(&index);
            result.set(&index, f(val));

            if !Self::next_index(&mut index, &self.shape) {
                break;
            }
        }

        result
    }

    pub fn zip_map<F: Fn(f32, f32) -> f32>(&self, other: &Tensor, f: F) -> Tensor {
        assert_eq!(self.shape, other.shape);

        let mut result = Tensor::new(self.shape.clone());
        let mut index = vec![0; self.shape.len()];

        loop {
            let val = self.get(&index);
            let other_val = other.get(&index);
            result.set(&index, f(val, other_val));

            if !Self::next_index(&mut index, &self.shape) {
                break;
            }
        }

       result
    }

    fn next_index(index: &mut [usize], shape: &[usize]) -> bool {
        for i in (0..index.len()).rev() {
            index[i] += 1;
            if index[i] < shape[i] {
                return true;
            }
            index[i] = 0;
        }
        false
    }

    pub fn elementwise_square(&self) -> Tensor {
        self.map(|x| x * x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul() {
        let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor::from_vec(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = a.matmul(&b);
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_transpose() {
        let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = a.transpose();
        assert_eq!(t.shape, vec![3, 2]);
        assert_eq!(t.get(&[0, 0]), 1.0);
        assert_eq!(t.get(&[0, 1]), 4.0);
        assert_eq!(t.get(&[1, 0]), 2.0);
        assert_eq!(t.get(&[2, 1]), 6.0);
    }

    #[test]
    fn test_matmul_transposed() {
        // A @ A^T should give a symmetric matrix
        let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let at = a.transpose();
        let c = a.matmul(&at);
        assert_eq!(c.shape, vec![2, 2]);
        // c[0][1] should equal c[1][0]
        assert_eq!(c.get(&[0, 1]), c.get(&[1, 0]));
    }

    #[test]
    fn test_map_transposed() {
        let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = a.transpose();
        let result = t.map(|x| x * 2.0);
        assert_eq!(result.get(&[0, 0]), 2.0);
        assert_eq!(result.get(&[0, 1]), 8.0); // was 4.0 before map
        assert_eq!(result.get(&[1, 0]), 4.0); // was 2.0 before map
    }
}