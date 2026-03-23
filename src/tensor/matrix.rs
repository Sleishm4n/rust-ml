struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Matrix {
    fn new(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    fn get(&self, row: usize, col: usize) -> f32 {
        let index = row * self.cols + col;
        self.data[index]
    }

    fn set(&mut self, row: usize, col: usize, val: f32) {
        let index = row * self.cols + col;
        self.data[index] = val;
    }

    fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Matrix {
        assert_eq!(data.len(), rows * cols);
        Matrix { rows, cols, data }
    }

    fn display(&self) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                print!("{} ", self.get(row, col));
            }
            println!();
        }
        println!();
    }

    fn scale_inplace(&mut self, n: f32) {
        for val in &mut self.data {
            *val *= n;
        }
    }

    fn scale(&self, n: f32) -> Matrix {
        let mut vec = Vec::with_capacity(self.data.len());
        for val in &self.data {
            let v = val * n;
            vec.push(v);
        }
        Matrix::from_vec(self.rows, self.cols, vec)
    }

    fn add(&self, other: &Matrix) -> Matrix{
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut vec = Vec::with_capacity(self.data.len());
        for row in 0..self.rows {
            for col in 0..self.cols {
                let index = row * self.cols + col;
                let v = self.data[index] + other.data[index];
                vec.push(v);
            }
        }
        Matrix::from_vec(self.rows, self.cols, vec)
    }

    fn sub(&self, other: &Matrix) -> Matrix{
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut vec = Vec::with_capacity(self.data.len());
        for row in 0..self.rows {
            for col in 0..self.cols {
                let index = row * self.cols + col;
                let v = self.data[index] - other.data[index];
                vec.push(v);
            }
        }
        Matrix::from_vec(self.rows, self.cols, vec)
    }

    fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows);

        let mut c = Matrix::new(self.rows, other.cols);

        for i in 0..self.rows {
            for k in 0..self.cols {

                let A_Index = i * self.cols + k;
                let a = self.data[A_Index];

                for j in 0..other.cols {
                    let B_index = k * other.cols + j;
                    let C_index = i * c.cols + j;
                    c.data[C_index] += a * other.data[B_index];
                }
            }
        }
        return c
    }
}

fn main() {
    let vec = vec![
        2.0, 2.0, 
        2.0, 2.0
    ];
    let mut A = Matrix::from_vec(2, 2, vec);
    A.display();

    let vec2 = vec![
        1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0
    ];
    let B = Matrix::from_vec(2, 3, vec2);

    let F = A.matmul(&B);
    F.display();
}
