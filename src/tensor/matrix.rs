use std::iter::zip;

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

    fn add(&self, other: &Matrix) -> Matrix {
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

    fn sub(&self, other: &Matrix) -> Matrix {
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

    fn matmul(&self, other: &Matrix) -> Matrix {
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

    fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows);
        for row in 0..self.rows {
            for col in 0..self.cols {
                result.set(col, row, self.get(row, col));
            }
        }
        result
    }

    fn sum(&self) -> f32 {
        let mut total: f32 = 0.0;

        for row in 0..self.rows {
            for col in 0..self.cols {
                total += self.get(row, col);
            }
        }
        total
    }

    fn rowsum(&self, row: usize) -> f32 {
        let mut total: f32 = 0.0;
        for col in 0..self.cols {
            total += self.get(row, col);
        }
        total
    }

    fn colsum(&self, col: usize) -> f32 {
        let mut total: f32 = 0.0;
        for row in 0..self.rows {
            total += self.get(row, col);
        }
        total
    }

    fn mean(&self) -> f32 {
        let mut total: f32 = 0.0;

        for row in 0..self.rows {
            for col in 0..self.cols {
                total += self.get(row, col);
            }
        }
        total / (self.data.len()) as f32
    }

    fn rowmean(&self, row: usize) -> f32 {
        let mut total: f32 = 0.0;

        for col in 0..self.cols {
            total += self.get(row, col);
        }
        total / (self.cols) as f32
    }

    fn colmean(&self, col: usize) -> f32 {
        let mut total: f32 = 0.0;

        for row in 0..self.rows {
            total += self.get(row, col);
        }
        total / (self.rows) as f32
    }

    fn map(&self, f: fn(f32) -> f32) -> Matrix {
        let mut vec = Vec::with_capacity(self.data.len());

        for val in &self.data {
            vec.push(f(*val));
        }

        Matrix::from_vec(self.rows, self.cols, vec)
    }

    fn zip_map(&self, other: &Matrix, f: fn(f32, f32) -> f32) -> Matrix{
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let mut vec = Vec::with_capacity(self.data.len());

        for (a, b) in zip(&self.data, &other.data) {
            vec.push(f(*a,*b));
        }
        Matrix::from_vec(self.rows, self.cols, vec)
    }
}

fn square(x: f32) -> f32 {
    x * x
}

fn cube(x: f32) -> f32 {
    x * x * x
}

fn add(a: f32, b: f32) -> f32 {
    a + b
}


fn main() {
    let vec = vec![2.0, 2.0, 2.0, 2.0];
    let a = Matrix::from_vec(2, 2, vec);
    a.display();

    let vec2 = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let b = Matrix::from_vec(2, 3, vec2);

    let mut f = a.matmul(&b);
    f.display();

    f = f.transpose();
    f.display();

    println!("{}", f.sum());
    println!("{}", f.rowsum(1));
    println!("{}", f.colsum(1));
    println!("{}", f.mean());

    let mut g = f.map(square);
    g.display();

    let mut h = f.map(cube);
    h.display();

    let vec3 = vec![3.0, 3.0, 3.0, 3.0];
    let mut i = Matrix::from_vec(2, 2, vec3);

    let j = i.zip_map(&a, |x,y| x+y);
    j.display();
}
