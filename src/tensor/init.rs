//! Random initialisation constructors for [`Tensor`]
//!
//! Layer specific checmes (Xavier, Kaiming) live in [`crate::nn::init`] and
//! are built on top of these primitives
use crate::tensor::Tensor;
use rand::prelude::*;

impl Tensor {
    /// Creates a tensor of the given `shape` filled with values drawn
    /// uniformly from `[min, max)` using a custom RNG
    ///
    /// # Panics
    ///
    /// Panics if `min >= max` or if `shape` is empty
    pub fn rand_range_with_rng<R: Rng + ?Sized>(
        shape: Vec<usize>,
        min: f32,
        max: f32,
        rng: &mut R,
    ) -> Tensor {
        assert!(min < max, "min cannot be larger than max");
        assert!(!shape.is_empty(), "shape must be at least 1D");

        let size = shape.iter().product();
        let mut vec = Vec::with_capacity(size);

        for _ in 0..size {
            let val = rng.gen_range(min..max);
            vec.push(val);
        }
        Tensor::from_vec(shape, vec)
    }

    /// Creates a tensor of the given `shape` filled with values drawn
    /// uniformly from `[min, max)`
    ///
    /// # Panics
    ///
    /// Panics if `min >= max` or if `shape` is empty
    pub fn rand_range(shape: Vec<usize>, min: f32, max: f32) -> Tensor {
        let mut rng = rand::thread_rng();
        Self::rand_range_with_rng(shape, min, max, &mut rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_rand_range_io_shape_match() {
        let tensor = Tensor::rand_range(vec![3, 2], -1.0, 1.0);

        assert_eq!(tensor.shape, vec![3, 2]);
    }

    #[test]
    fn test_rand_range_values_within_minmax() {
        let tensor = Tensor::rand_range(vec![3, 2], -10.0, 10.0);

        assert!(tensor.tensor_min() >= -10.0);
        assert!(tensor.tensor_max() <= 10.0);
    }

    #[test]
    fn test_rand_range_all_elements_within_bounds() {
        let mut rng = StdRng::seed_from_u64(42);
        let tensor = Tensor::rand_range_with_rng(vec![10, 10], -5.0, 5.0, &mut rng);

        assert_eq!(tensor.data.len(), 100);
        for &val in &tensor.data {
            assert!(
                (-5.0..5.0).contains(&val),
                "Value {val} outside of bounds [-5.0, 5.0)"
            );
        }
    }

    #[test]
    fn test_rand_range_seeded_is_deterministic() {
        let mut rng1 = StdRng::seed_from_u64(12345);
        let mut rng2 = StdRng::seed_from_u64(12345);

        let t1 = Tensor::rand_range_with_rng(vec![4, 4], -1.0, 1.0, &mut rng1);
        let t2 = Tensor::rand_range_with_rng(vec![4, 4], -1.0, 1.0, &mut rng2);

        assert_eq!(t1.data, t2.data);
    }

    #[test]
    #[should_panic(expected = "min cannot be larger than max")]
    fn test_rand_range_panics_on_min_larger_max() {
        let _tensor = Tensor::rand_range(vec![3, 2], 2.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "at least 1D")]
    fn test_rand_range_panics_on_empty() {
        let _tensor = Tensor::rand_range(vec![], 1.0, 2.0);
    }
}
