//! Cross-entropy losses for classification
//!
//! [`cross_entropy`] and [`d_cross_entropy`] apply softmax internally
//! so they take **raw logits**, not the output of a
//! [`crate::nn::softmax::SoftmaxLayer`]. Folding softmax into the loss
//! is both numerically stabler and simpler on the backward pass
use core::panic;
use std::assert_eq;

use crate::tensor::Tensor;

/// Softmax cross-entropy loss for a one-hot `target`
///
/// `output` is expected to be in raw logits (accepts 2D `[C, 1]` or 3D `[batch, C, 1]`). A small epsilon is added inside
/// the log to avoid `ln(0)`
/// Evaluated with the log-sum-exp trick for numerical stability and averaged over the batch
///
/// # Panics
///
/// Panics if shapes differ or neither are 2D or 3D
pub fn cross_entropy(output: &Tensor, target: &Tensor) -> f32 {
    assert_eq!(
        output.shape, target.shape,
        "output and target shapes have to match"
    );

    let (batch_size, num_classes) = match output.shape.len() {
        2 => (1, output.shape[0]),
        3 => (output.shape[0], output.shape[1]),
        _ => panic!("cross_entropy only supports D [C, 1] or 3D [batch, C, 1] tensors"),
    };

    let mut total_loss = 0.0f32;

    for b in 0..batch_size {
        let start = b * num_classes;
        let out_slice = &output.data[start..start + num_classes];
        let tgt_slice = &target.data[start..start + num_classes];

        let mut max = -f32::INFINITY;
        for &val in out_slice {
            if val > max {
                max = val;
            }
        }

        let mut sum_exp = 0.0f32;
        for &val in out_slice {
            sum_exp += (val - max).exp()
        }

        let log_sum_exp = max + sum_exp.ln();

        let mut sample_loss = 0.0f32;
        for i in 0..num_classes {
            let log_prob = out_slice[i] - log_sum_exp;
            sample_loss += -tgt_slice[i] * log_prob;
        }
        total_loss += sample_loss;
    }

    total_loss / batch_size as f32
}

/// Gradient of [`cross_entropy`] with respect to the raw logits
///
/// Returns `(softmax(output) - target) / batch_size`
///
/// # Panics
///
/// Panics if the shapes differ or aren't 2D or 3D
pub fn d_cross_entropy(output: &Tensor, target: &Tensor) -> Tensor {
    assert_eq!(
        output.shape, target.shape,
        "output and target shapes must match"
    );
    let (batch_size, num_classes) = match output.shape.len() {
        2 => (1, output.shape[0]),
        3 => (output.shape[0], output.shape[1]),
        _ => panic!("d_cross_entropy only supports 2D [C, 1] or 3D [batch, C, 1] tensors"),
    };

    let mut grad_data = Vec::with_capacity(batch_size * num_classes);

    for b in 0..batch_size {
        let start = b * num_classes;
        let out_slice = &output.data[start..start + num_classes];
        let tgt_slice = &target.data[start..start + num_classes];

        let mut max = -f32::INFINITY;
        for &val in out_slice {
            if val > max {
                max = val;
            }
        }

        let mut sum_exp = 0.0f32;
        for &val in out_slice {
            sum_exp += (val - max).exp();
        }

        for i in 0..num_classes {
            let prob = (out_slice[i] - max).exp() / sum_exp;
            grad_data.push((prob - tgt_slice[i]) / batch_size as f32);
        }
    }

    Tensor::from_vec(output.shape.clone(), grad_data)
}

/// Binary cross-entropy over sigmoid-activated `logits`, averaged over all
/// elements
///
/// Takes raw logits and applies the sigmoid internally, clamping the
/// probability away from 0 and 1 to keep the logs finite. Suitable as the
/// reconstruction term for models whose targets are pixel intensities in
/// `[0, 1]`
///
/// # Panics
///
/// Panics if `target` has fewer elements than `logits`
pub fn binary_cross_entropy(logits: &Tensor, target: &Tensor) -> f32 {
    assert_eq!(
        logits.shape, target.shape,
        "logits and target shapes must match"
    );

    let n = logits.data.len() as f32;
    let mut loss: f32 = 0.0;

    for (logit, t) in logits.data.iter().zip(target.data.iter()) {
        let p = (1.0 / (1.0 + (-logit).exp())).clamp(1e-7, 1.0 - 1e-7);
        loss += -(t * p.ln() + (1.0 - t) * (1.0 - p).ln());
    }
    loss / n
}

/// Gradient of [`binary_cross_entropy`] with respect to raw `logits`
///
/// # Panics
///
/// Panics if shapes differ
pub fn d_binary_cross_entropy(logits: &Tensor, target: &Tensor) -> Tensor {
    assert_eq!(
        logits.shape, target.shape,
        "logits and target shapes must match"
    );
    let n = logits.data.len() as f32;
    logits.zip_map(target, |logit, t| {
        let p = 1.0 / (1.0 + (-logit).exp());
        (p - t) / n
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_entropy_correct_value() {
        let output = Tensor::from_vec(vec![3, 1], vec![0.0, 0.0, 0.0]);
        let target = Tensor::from_vec(vec![3, 1], vec![1.0, 0.0, 0.0]);

        let loss = cross_entropy(&output, &target);

        // Uniform distribution over 3 classes: -ln(1/3) = ln(3)
        let expected = 3.0_f32.ln();
        assert!((loss - expected).abs() < 1e-5)
    }

    #[test]
    fn test_cross_entropy_stable_for_extreme_logits() {
        let output = Tensor::from_vec(vec![3, 1], vec![1000.0, 1000.0, 1000.0]);
        let target = Tensor::from_vec(vec![3, 1], vec![1.0, 0.0, 0.0]);

        let loss = cross_entropy(&output, &target);

        assert!(loss.is_finite(), "loss was {loss}, expected a finite value");
        assert!((loss - 1.098612).abs() < 1e-4);
    }

    #[test]
    fn test_cross_entropy_extreme_unequal_logits() {
        let output = Tensor::from_vec(vec![3, 1], vec![1000.0, 0.0, -1000.0]);
        let target = Tensor::from_vec(vec![3, 1], vec![1.0, 0.0, 0.0]);

        let loss = cross_entropy(&output, &target);

        assert!(loss.is_finite());
        assert!(loss.abs() < 1e-4);
    }

    #[test]
    fn test_d_cross_entropy_matches_closed_form() {
        let output = Tensor::from_vec(vec![3, 1], vec![1.0, 2.0, 3.0]);
        let target = Tensor::from_vec(vec![3, 1], vec![0.0, 1.0, 0.0]);

        let grad = d_cross_entropy(&output, &target);

        let max = 3.0_f32;
        let exps: Vec<f32> = [1.0f32, 2.0, 3.0].iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        for (i, &prob) in probs.iter().enumerate() {
            let expected = prob - target.data[i];
            assert!((grad.data[i] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_d_cross_entropy_matches_numerical_gradient() {
        let output = Tensor::from_vec(vec![3, 1], vec![0.5, -1.2, 2.3]);
        let target = Tensor::from_vec(vec![3, 1], vec![0.0, 0.0, 1.0]);

        let analytical = d_cross_entropy(&output, &target);

        let epsilon = 1e-4;
        for i in 0..3 {
            let mut plus = output.clone();
            plus.data[i] += epsilon;
            let mut minus = output.clone();
            minus.data[i] -= epsilon;

            let loss_plus = cross_entropy(&plus, &target);
            let loss_minus = cross_entropy(&minus, &target);

            let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);

            assert!(
                (analytical.data[i] - numerical).abs() < 1e-3,
                "index {i}: analytical {} vs numerical {}",
                analytical.data[i],
                numerical
            );
        }
    }

    #[test]
    fn test_binary_cross_entropy_correct_value() {
        let logits = Tensor::from_vec(vec![1], vec![0.0]);
        let target = Tensor::from_vec(vec![1], vec![1.0]);

        let loss = binary_cross_entropy(&logits, &target);

        let expected = std::f32::consts::LN_2;
        assert!((loss - expected).abs() < 1e-5);
    }

    #[test]
    fn test_binary_cross_entropy_averages_over_batch() {
        let logits = Tensor::from_vec(vec![2], vec![0.0, 0.0]);
        let target = Tensor::from_vec(vec![2], vec![1.0, 1.0]);

        let loss = binary_cross_entropy(&logits, &target);

        let expected = std::f32::consts::LN_2;
        assert!((loss - expected).abs() < 1e-5);
    }

    #[test]
    fn test_binary_cross_entropy_stable_for_extreme_logits() {
        let logits = Tensor::from_vec(vec![2], vec![1000.0, -1000.0]);
        let target = Tensor::from_vec(vec![2], vec![1.0, 0.0]);

        let loss = binary_cross_entropy(&logits, &target);

        assert!(loss.is_finite());
        assert!(loss.abs() < 1e-3);
    }

    #[test]
    fn test_binary_cross_entropy_stable_for_confidently_wrong_logits() {
        let logits = Tensor::from_vec(vec![1], vec![1000.0]);
        let target = Tensor::from_vec(vec![1], vec![0.0]);

        let loss = binary_cross_entropy(&logits, &target);

        assert!(loss.is_finite());
        assert!(loss > 10.0);
    }

    #[test]
    fn test_cross_entropy_batched_3d() {
        let output = Tensor::from_vec(vec![2, 3, 1], vec![0.0, 0.0, 0.0, 1000.0, 0.0, -1000.0]);
        let target = Tensor::from_vec(vec![2, 3, 1], vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

        let loss = cross_entropy(&output, &target);
        // sample 0 is ln(3), sample 1 is near 0.0 -> average is ln(3) / 2
        let expected = 3.0_f32.ln() / 2.0;
        assert!((loss - expected).abs() < 1e-4);
    }

    #[test]
    fn test_d_binary_cross_entropy_matches_numerical_gradient() {
        let logits = Tensor::from_vec(vec![3, 1], vec![0.5, -1.2, 2.3]);
        let target = Tensor::from_vec(vec![3, 1], vec![1.0, 0.0, 1.0]);

        let analytical = d_binary_cross_entropy(&logits, &target);

        let epsilon = 1e-4;
        for i in 0..3 {
            let mut plus = logits.clone();
            plus.data[i] += epsilon;
            let mut minus = logits.clone();
            minus.data[i] -= epsilon;

            let loss_plus = binary_cross_entropy(&plus, &target);
            let loss_minus = binary_cross_entropy(&minus, &target);

            let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);

            assert!(
                (analytical.data[i] - numerical).abs() < 1e-3,
                "index {i}: analytical {} vs numerical {}",
                analytical.data[i],
                numerical
            );
        }
    }
}
