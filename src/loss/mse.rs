use crate::matrix::Matrix;

pub fn mse(output: &Matrix, target: &Matrix) -> f32 {
    let mut total: f32 = 0.0;

    for (index, element) in output.data.iter().enumerate() {
        total += (element - target.data[index]).powi(2);
    }

    total / (output.data.len()) as f32
}

pub fn d_mse(output: &Matrix, target: &Matrix) -> Matrix {
    output.zip_map(target, |a, b| 2.0 * (a - b) / output.data.len() as f32)
}