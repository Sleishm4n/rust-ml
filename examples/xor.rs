use rust_ml::tensor::Matrix;
use rust_ml::nn::linear::{LinearLayer};
use rust_ml::loss::mse::{d_mse, mse};

fn main() {
    let mut layer = LinearLayer::new_rand(2, 1);
    let lr = 0.1;
    let epochs = 1000;

    // XOR inputs and targets
    let inputs = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let targets = vec![[0.0], [1.0], [1.0], [0.0]];

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for (x, y) in inputs.iter().zip(targets.iter()) {
            let x_mat = Matrix::from_vec(2, 1, x.to_vec());
            let y_mat = Matrix::from_vec(1, 1, y.to_vec());
            let y_hat = layer.forward(&x_mat);
            let loss = mse(&y_hat, &y_mat);
            let d_out = d_mse(&y_hat, &y_mat);
            let (d_w, d_b, _) = layer.backward(&d_out, &x_mat);
            layer.sgd_update(&d_w, &d_b, lr);
            total_loss += loss;
        }

        if epoch % 100 == 0 {
            println!("Epoch {epoch}: loss = {:.4}", total_loss / 4.0);
        }
    }
}