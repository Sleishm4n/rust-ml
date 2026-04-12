use rust_ml::loss::mse::{d_mse, mse};
use rust_ml::nn::activation::{d_sigmoid, sigmoid};
use rust_ml::nn::linear::LinearLayer;
use rust_ml::tensor::Matrix;

fn main() {
    let mut layer1 = LinearLayer::new_rand(2, 2);
    let mut layer2 = LinearLayer::new_rand(2, 1);
    let lr = 0.1;
    let epochs = 10000;

    // XOR inputs and targets
    let inputs = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let targets = vec![[0.0], [1.0], [1.0], [0.0]];

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for (x, y) in inputs.iter().zip(targets.iter()) {
            let x_mat = Matrix::from_vec(2, 1, x.to_vec());
            let y_mat = Matrix::from_vec(1, 1, y.to_vec());
            let z1 = layer1.forward(&x_mat);
            let a1 = z1.map(sigmoid);
            let y_hat = layer2.forward(&a1);
            let loss = mse(&y_hat, &y_mat);
            let d_out = d_mse(&y_hat, &y_mat);
            let (d_w2, d_b2, d_x2) = layer2.backward(&d_out, &a1);
            let d_a1 = d_x2.zip_map(&z1, |d, z| d * d_sigmoid(z));
            let (d_w1, d_b1, _) = layer1.backward(&d_a1, &x_mat);
            layer2.sgd_update(&d_w2, &d_b2, lr);
            layer1.sgd_update(&d_w1, &d_b1, lr);
            total_loss += loss;
        }

        if epoch % 100 == 0 {
            println!("Epoch {epoch}: loss = {:.4}", total_loss / 4.0);
        }
    }

    println!("\nPredictions:");
    for (x, y) in inputs.iter().zip(targets.iter()) {
        let x_mat = Matrix::from_vec(2, 1, x.to_vec());
        let z1 = layer1.forward(&x_mat);
        let a1 = z1.map(sigmoid);
        let y_hat = layer2.forward(&a1);
        println!("{:?} → {:.4} (target: {})", x, y_hat.data[0], y[0]);
    }
}
