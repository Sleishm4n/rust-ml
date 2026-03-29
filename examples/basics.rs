use rust_ml::tensor::Matrix;
use rust_ml::nn::activation::{relu, sigmoid, tanh};

fn main() {
    let w = Matrix::rand_range(3, 3, -10.0, 10.0);

    println!("Original Matrix:");
    w.display();

    println!("After ReLU:");
    let relu_res = w.map(relu);
    relu_res.display();

    println!("After Sigmoid:");
    let sig_res = w.map(sigmoid);
    sig_res.display();

    println!("After Tanh:");
    let tanh_res = w.map(tanh);
    tanh_res.display();
}