use crate::loss::mse::{d_mse, mse};
use crate::nn::linear::LinearLayer;
use crate::tensor::Tensor;

pub fn gradient_check(layer: &LinearLayer, input: &Tensor, target: &Tensor, eps: f32) {
    let mut layer_copy = layer.clone();
    let forward_res = layer_copy.forward(input);
    let d_output = d_mse(&forward_res, target);
    let (d_w, _d_b, _d_x) = layer_copy.backward(&d_output);

    for i in 0..layer_copy.weight.data.len() {
        layer_copy.weight.data[i] += eps;
        let loss_plus = mse(&layer_copy.forward(input), target);
        layer_copy.weight.data[i] -= 2.0 * eps;
        let loss_minus = mse(&layer_copy.forward(input), target);
        layer_copy.weight.data[i] += eps;

        let numerical = (loss_plus - loss_minus) / (2.0 * eps);
        let analytical = d_w.data[i];
        let relative_error = if numerical.abs() + analytical.abs() < 1e-10 {
            0.0
        } else {
            (numerical - analytical).abs() / (numerical.abs() + analytical.abs())
        };
        assert!(relative_error < 1e-4, "weight[{i}]: numerical={numerical:.6}, analytical={analytical:.6}, err={relative_error:.2e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::linear::LinearLayer;

    #[test]
    fn test_gradient_check() {
        let mut layer = LinearLayer::new(2, 2);
        layer.weight = Tensor::from_vec(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4]);
        let input = Tensor::from_vec(vec![2, 1], vec![1.0, 2.0]);
        let target = Tensor::from_vec(vec![2, 1], vec![0.5, 0.5]);
        gradient_check(&layer, &input, &target, 1e-4);
    }
}
