use crate::pq::CodeType;
use anyhow::Result;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use ndarray_stats::QuantileExt;
use rand::distr::{Distribution, Uniform};
use rand::seq::SliceRandom;
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
    *max_change < 1e-6
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
