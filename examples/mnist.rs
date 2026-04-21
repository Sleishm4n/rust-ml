use rust_ml::data::mnist::{load_images, load_labels, one_hot};
use rust_ml::loss::cross_entropy::{cross_entropy, d_cross_entropy};
use rust_ml::nn::activation::{d_relu, relu, ActivationLayer};
use rust_ml::nn::linear::LinearLayer;
use rust_ml::nn::network::Network;
use rust_ml::optimiser::adam::Adam;

fn main() {
    let mut network = Network::new(vec![
        Box::new(LinearLayer::new_rand(784, 128)),
        Box::new(ActivationLayer::new(relu, d_relu)),
        Box::new(LinearLayer::new_rand(128, 10)),
    ]);
    let mut optimiser = Adam::new(0.001, 0.9, 0.999, 1e-8);
    let epochs = 10;

    let images = load_images("data/mnist/train-images.idx3-ubyte");
    let labels = load_labels("data/mnist/train-labels.idx1-ubyte");
    let test_images = load_images("data/mnist/t10k-images.idx3-ubyte");
    let test_labels = load_labels("data/mnist/t10k-labels.idx1-ubyte");

    let y_mat = one_hot(labels[0]);
    let y_hat = network.forward(&images[0]);
    let loss = cross_entropy(&y_hat, &y_mat);

    println!("Initial loss: {}", loss);

    for epoch in 0..epochs {
        let mut total_loss: f32 = 0.0;

        let mut correct = 0;

        for (x, y) in images.iter().zip(labels.iter()) {
            let y_mat = one_hot(*y);
            let y_hat = network.forward(x);
            let loss = cross_entropy(&y_hat, &y_mat);
            let d_out = d_cross_entropy(&y_hat, &y_mat);
            network.backward(&d_out);
            network.update(&mut optimiser);
            total_loss += loss;

            let predicted = y_hat
                .data
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            if predicted == *y as usize {
                correct += 1;
            }
        }

        println!(
            "Epoch {epoch}: loss = {:.4}, accuracy = {:.2}%",
            total_loss / images.len() as f32,
            correct as f32 / images.len() as f32 * 100.0
        );
    }

    let mut correct = 0;

    for (x, y) in test_images.iter().zip(test_labels.iter()) {
        let y_hat = network.forward(x);

        let predicted = y_hat
            .data
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        if predicted == *y as usize {
            correct += 1;
        }
    }

    let accuracy = correct as f32 / test_images.len() as f32 * 100.0;

    println!("Test accuracy: {:.2}%", accuracy);
}