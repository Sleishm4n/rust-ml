use crate::tensor::Matrix;
use rand::prelude::*;

impl Matrix {
    pub fn rand_range(rows: usize, cols: usize, min: f32, max: f32) -> Matrix {
        let mut rng = rand::thread_rng();
        let mut vec = Vec::with_capacity(rows * cols);

        for _ in 0..rows * cols {
            let val = rng.gen_range(min..max);
            vec.push(val);
        }
        Matrix::from_vec(rows, cols, vec)
    }
}
