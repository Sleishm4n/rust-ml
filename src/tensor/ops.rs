use crate::tensor::Matrix;
use std::iter::zip;

pub fn square(x: f32) -> f32 { x * x }
pub fn cube(x: f32) -> f32 { x * x * x }
pub fn add_scalars(a: f32, b: f32) -> f32 { a + b }

impl Matrix {
    pub fn scale_inplace(&mut self, n: f32) {
        for val in &mut self.data {
            *val *= n;
        }
    }

    pub fn scale(&self, n: f32) -> Matrix {
        let mut vec = Vec::with_capacity(self.data.len());
        for val in &self.data {
            let v = val * n;
            vec.push(v);
        }
        Matrix::from_vec(self.rows, self.cols, vec)
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut vec = Vec::with_capacity(self.data.len());
        for row in 0..self.rows {
            for col in 0..self.cols {
                let v = self.get(row, col) + other.get(row, col);
                vec.push(v);
            }
        }
        Matrix::from_vec(self.rows, self.cols, vec)
    }

    pub fn sub(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut vec = Vec::with_capacity(self.data.len());
        for row in 0..self.rows {
            for col in 0..self.cols {
                let v = self.get(row, col) - other.get(row, col);
                vec.push(v);
            }
        }
        Matrix::from_vec(self.rows, self.cols, vec)
    }

    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows);

        let mut c = Matrix::new(self.rows, other.cols);

        for i in 0..self.rows {
            for k in 0..self.cols {
                let a_index = i * self.cols + k;
                let a = self.data[a_index];

                for j in 0..other.cols {
                    let b_index = k * other.cols + j;
                    let c_index = i * c.cols + j;
                    c.data[c_index] += a * other.data[b_index];
                }
            }
        }
        c
    }

    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows);
        for row in 0..self.rows {
            for col in 0..self.cols {
                result.set(col, row, self.get(row, col));
            }
        }
        result
    }

    pub fn sum(&self) -> f32 {
        let mut total: f32 = 0.0;

        for row in 0..self.rows {
            for col in 0..self.cols {
                total += self.get(row, col);
            }
        }
        total
    }

    pub fn rowsum(&self, row: usize) -> f32 {
        let mut total: f32 = 0.0;
        for col in 0..self.cols {
            total += self.get(row, col);
        }
        total
    }

    pub fn colsum(&self, col: usize) -> f32 {
        let mut total: f32 = 0.0;
        for row in 0..self.rows {
            total += self.get(row, col);
        }
        total
    }

    pub fn mean(&self) -> f32 {
        let mut total: f32 = 0.0;

        for row in 0..self.rows {
            for col in 0..self.cols {
                total += self.get(row, col);
            }
        }
        total / (self.data.len()) as f32
    }

    pub fn rowmean(&self, row: usize) -> f32 {
        let mut total: f32 = 0.0;

        for col in 0..self.cols {
            total += self.get(row, col);
        }
        total / (self.cols) as f32
    }

    pub fn colmean(&self, col: usize) -> f32 {
        let mut total: f32 = 0.0;

        for row in 0..self.rows {
            total += self.get(row, col);
        }
        total / (self.rows) as f32
    }

    pub fn map(&self, f: fn(f32) -> f32) -> Matrix {
        let mut vec = Vec::with_capacity(self.data.len());

        for val in &self.data {
            vec.push(f(*val));
        }

        Matrix::from_vec(self.rows, self.cols, vec)
    }

    pub fn zip_map(&self, other: &Matrix, f: fn(f32, f32) -> f32) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let mut vec = Vec::with_capacity(self.data.len());

        for (a, b) in zip(&self.data, &other.data) {
            vec.push(f(*a, *b));
        }
        Matrix::from_vec(self.rows, self.cols, vec)
    }

    pub fn elementwise_add(&self, other: &Matrix) -> Matrix {
        self.zip_map(other, add_scalars)
    }

    pub fn elementwise_square(&self) -> Matrix {
        self.map(square)
    }

    pub fn elementwise_cube(&self) -> Matrix {
        self.map(cube)
    }
}