use crate::pq::CodeType;
use anyhow::Result;
use ndarray::{s, Array1, Array2, ArrayView1, Axis};
use ndarray_stats::QuantileExt;
use rand::distr::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rand::Rng;
use std::f32;
use std::ops::AddAssign;

pub fn kmeans2(
    data: &Array2<f32>,
    k: u32,
    iter: usize,
    minit: &str,
) -> Result<(Array2<f32>, Array1<usize>)> {
    let (n_samples, n_features) = data.dim();
    let k = k as usize;

    if n_samples == 0 || n_features == 0 {
        anyhow::bail!("Data must have at least one sample and one feature");
    }

    if k == 0 || k > n_samples {
        anyhow::bail!(
            "Number of clusters k must be between 1 and number of samples ({})",
            n_samples
        );
    }

    if data.iter().any(|x| !x.is_finite()) {
        anyhow::bail!("Data contains non-finite values (NaN or Inf)");
    }

    let mut centroids = match minit {
        "points" => {
            let mut rng = rand::thread_rng();
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);
            indices.truncate(k);

            let mut initial_centroids = Array2::zeros((k, n_features));
            for (i, &idx) in indices.iter().enumerate() {
                initial_centroids.row_mut(i).assign(&data.row(idx));
            }
            initial_centroids
        }
        _ => anyhow::bail!("Unsupported initialization method"),
    };

    let mut labels = Array1::zeros(n_samples);
    let mut old_centroids;
    let mut has_converged = false;

    for _ in 0..iter {
        if has_converged {
            break;
        }

        old_centroids = centroids.clone();

        for (i, sample) in data.rows().into_iter().enumerate() {
            let mut min_dist = f32::INFINITY;
            let mut min_label = 0;

            for (j, centroid) in centroids.rows().into_iter().enumerate() {
                let dist = euclidean_distance(&sample, &centroid);
                if dist < min_dist {
                    min_dist = dist;
                    min_label = j;
                }
            }
            labels[i] = min_label;
        }

        let mut new_centroids = Array2::zeros((k, n_features));
        let mut counts = vec![0usize; k];

        for (i, sample) in data.rows().into_iter().enumerate() {
            let label = labels[i];
            new_centroids.row_mut(label).add_assign(&sample);
            counts[label] += 1;
        }

        for (i, count) in counts.iter().enumerate() {
            if *count > 0 {
                new_centroids.row_mut(i).mapv_inplace(|x| x / *count as f32);
            } else {
                let random_idx = rand::thread_rng().gen_range(0..n_samples);
                new_centroids.row_mut(i).assign(&data.row(random_idx));
            }
        }

        centroids = new_centroids;

        has_converged = check_convergence(&centroids, &old_centroids);
    }

    Ok((centroids, labels))
}

fn check_convergence(new_centroids: &Array2<f32>, old_centroids: &Array2<f32>) -> bool {
    let diff = new_centroids - old_centroids;
    let binding = diff.mapv(|x| x.abs()).sum_axis(Axis(1));
    let max_change = binding.max().unwrap();
    *max_change < 1e-4
}

pub fn euclidean_distance(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

pub fn determine_code_type(ks: u32) -> CodeType {
    if ks <= (1 << 8) {
        CodeType::U8
    } else if ks <= (1 << 16) {
        CodeType::U16
    } else {
        CodeType::U32
    }
}

pub fn create_random_vectors(num_vectors: usize, dimension: usize) -> Array2<f32> {
    let mut rng = rand::thread_rng();
    let uniform = Uniform::new(0.0, 1.0);
    Array2::from_shape_fn((num_vectors, dimension), |_| {
        uniform.unwrap().sample(&mut rng)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use ndarray::{concatenate, s};
    use rand::Rng;

    fn create_random_vectors(num_vectors: usize, dimension: usize) -> Array2<f32> {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(0.0, 1.0);
        Array2::from_shape_fn((num_vectors, dimension), |_| {
            uniform.unwrap().sample(&mut rng)
        })
    }

    // Edge Case: The dataset has zero samples.
    #[test]
    fn test_kmeans_empty_dataset() {
        let data = Array2::<f32>::zeros((0, 10));
        let result = kmeans2(&data, 3, 10, "points");
        assert!(result.is_err(), "kmeans2 should fail with an empty dataset");
    }

    // Edge Case: The dataset has zero features (dimensions).
    #[test]
    fn test_kmeans_zero_features() {
        let data = Array2::<f32>::zeros((100, 0));
        let result = kmeans2(&data, 3, 10, "points");
        assert!(
            result.is_err(),
            "kmeans2 should fail with zero-dimensional data"
        );
    }

    // Edge Case: The number of clusters k is zero.
    #[test]
    fn test_kmeans_zero_clusters() {
        let data = create_random_vectors(100, 10);
        let result = kmeans2(&data, 0, 10, "points");
        assert!(result.is_err(), "kmeans2 should fail when k is zero");
    }

    // Edge Case: The number of clusters k exceeds the number of samples.
    #[test]
    fn test_kmeans_clusters_exceed_samples() {
        let data = create_random_vectors(10, 10);
        let result = kmeans2(&data, 20, 10, "points");
        assert!(
            result.is_err(),
            "kmeans2 should fail when k exceeds the number of samples"
        );
    }

    // Edge Case: The number of iterations is zero.
    #[test]
    fn test_kmeans_zero_iterations() {
        let data = create_random_vectors(100, 10);
        let result = kmeans2(&data, 3, 0, "points");
        assert!(
            result.is_ok(),
            "kmeans2 should handle zero iterations gracefully"
        );
        let (centroids, labels) = result.unwrap();
        assert_eq!(centroids.shape(), &[3, 10], "Centroids shape mismatch");
        assert_eq!(labels.len(), 100, "Labels length mismatch");
    }

    // Edge Case: The minit parameter is not "points".
    #[test]
    fn test_kmeans_invalid_minit() {
        let data = create_random_vectors(100, 10);
        let result = kmeans2(&data, 3, 10, "random");
        assert!(
            result.is_err(),
            "kmeans2 should fail with an unsupported initialization method"
        );
    }

    // Edge Case: The dataset contains NaN values.
    #[test]
    fn test_kmeans_nan_values() {
        let mut data = create_random_vectors(100, 10);
        data[[0, 0]] = f32::NAN;
        let result = kmeans2(&data, 3, 10, "points");
        assert!(
            result.is_err(),
            "kmeans2 should fail when data contains NaN values"
        );
    }

    // Edge Case: The dataset contains infinite values.
    #[test]
    fn test_kmeans_infinite_values() {
        let mut data = create_random_vectors(100, 10);
        data[[0, 0]] = f32::INFINITY;
        let result = kmeans2(&data, 3, 10, "points");
        assert!(
            result.is_err(),
            "kmeans2 should fail when data contains infinite values"
        );
    }

    // Edge Case: All data points are the same.
    #[test]
    fn test_kmeans_identical_points() {
        let data = Array2::<f32>::from_elem((100, 10), 1.0); // All points are identical
        let result = kmeans2(&data, 3, 10, "points");
        assert!(
            result.is_ok(),
            "kmeans2 should handle identical points gracefully"
        );
        let (centroids, _labels) = result.unwrap();
        assert_eq!(centroids.shape(), &[3, 10], "Centroids shape mismatch");
        for centroid in centroids.outer_iter() {
            assert!(
                centroid.iter().all(|&x| (x - 1.0).abs() < 1e-6),
                "Centroid values should be approximately 1.0"
            );
        }
    }

    // Edge Case: Dataset contains duplicate points.
    #[test]
    fn test_kmeans_duplicate_points() {
        let mut data = create_random_vectors(90, 10);
        let duplicates = data.slice(s![0..10, ..]).to_owned(); // Take 10 samples to duplicate
        data = concatenate(Axis(0), &[data.view(), duplicates.view()]).unwrap();
        assert_eq!(data.shape(), &[100, 10], "Data shape should be (100, 10)");
        let result = kmeans2(&data, 5, 10, "points");
        assert!(
            result.is_ok(),
            "kmeans2 should handle duplicate points without failing"
        );
    }

    // Edge Case: The dataset contains only one sample.
    #[test]
    fn test_kmeans_single_sample() {
        let data = create_random_vectors(1, 10);
        let result = kmeans2(&data, 1, 10, "points");
        assert!(
            result.is_ok(),
            "kmeans2 should handle a single sample correctly"
        );
        let (centroids, labels) = result.unwrap();
        assert_eq!(centroids.shape(), &[1, 10], "Centroids shape mismatch");
        assert_eq!(labels.len(), 1, "Labels length should be 1");
        assert_eq!(labels[0], 0, "Label for the single sample should be 0");
    }

    // Edge Case: The algorithm does not converge within the given iterations.
    #[test]
    fn test_kmeans_no_convergence() {
        let data = create_random_vectors(100, 10);
        let result = kmeans2(&data, 3, 1, "points"); // Only 1 iteration
        assert!(
            result.is_ok(),
            "kmeans2 should return results even if it doesn't converge"
        );
    }

    // Edge Case: Data contains negative values.
    #[test]
    fn test_kmeans_negative_values() {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(-1.0, 1.0);
        let data = Array2::from_shape_fn((100, 10), |_| uniform.unwrap().sample(&mut rng));
        let result = kmeans2(&data, 3, 10, "points");
        assert!(
            result.is_ok(),
            "kmeans2 should handle data with negative values"
        );
    }

    // Edge Case: Data with a large number of features.
    #[test]
    fn test_kmeans_high_dimensional_data() {
        let data = create_random_vectors(100, 1000); // 1000 features
        let result = kmeans2(&data, 5, 10, "points");
        assert!(
            result.is_ok(),
            "kmeans2 should handle high-dimensional data"
        );
    }

    // Edge Case: Number of clusters is close to the number of samples.
    #[test]
    fn test_kmeans_many_clusters() {
        let data = create_random_vectors(100, 10);
        let result = kmeans2(&data, 90, 10, "points");
        assert!(
            result.is_ok(),
            "kmeans2 should handle a large number of clusters"
        );
    }

    // Edge Case: Data designed to form clusters of different sizes.
    #[test]
    fn test_kmeans_non_uniform_cluster_sizes() {
        let mut rng = rand::thread_rng();
        let cluster1 = Array2::from_shape_fn((50, 10), |_| rng.gen_range(0.0..0.5));
        let cluster2 = Array2::from_shape_fn((30, 10), |_| rng.gen_range(0.5..1.0));
        let cluster3 = Array2::from_shape_fn((20, 10), |_| rng.gen_range(1.0..1.5));
        let data = concatenate(
            Axis(0),
            &[cluster1.view(), cluster2.view(), cluster3.view()],
        )
        .unwrap();
        let result = kmeans2(&data, 3, 10, "points");
        assert!(
            result.is_ok(),
            "kmeans2 should handle clusters of different sizes"
        );
    }

    // Edge Case: Using an unsupported initialization method.
    #[test]
    fn test_kmeans_unsupported_minit() {
        let data = create_random_vectors(100, 10);
        let result = kmeans2(&data, 3, 10, "unknown_method");
        assert!(
            result.is_err(),
            "kmeans2 should fail with an unsupported initialization method"
        );
    }

    // Edge Case: Ensure check_convergence function works as expected.
    #[test]
    fn test_check_convergence_function() {
        let centroids_old = create_random_vectors(3, 10);
        let centroids_new = centroids_old.clone();
        let has_converged = check_convergence(&centroids_new, &centroids_old);
        assert!(has_converged, "Should converge with identical centroids");

        let centroids_new = &centroids_old + 1e-5;
        let has_converged = check_convergence(&centroids_new, &centroids_old);
        assert!(has_converged, "Should converge with negligible changes");

        let centroids_new = &centroids_old + 1e-3;
        let has_converged = check_convergence(&centroids_new, &centroids_old);
        assert!(
            !has_converged,
            "Should not converge with significant changes"
        );
    }
}
