use anyhow::Result;
use log::info;
use ndarray::{s, Array2, ArrayView1, Axis};
use ndarray_rand::RandomExt;
use rand_distr::Uniform;
use std::time::Instant;
use vector_quantizer::pq::PQ;

fn euclidean_distance(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn calculate_reconstruction_error(original: &Array2<f32>, reconstructed: &Array2<f32>) -> f32 {
    original
        .outer_iter()
        .zip(reconstructed.outer_iter())
        .map(|(orig, recon)| {
            orig.iter()
                .zip(recon.iter())
                .map(|(o, r)| (o - r).powi(2))
                .sum::<f32>()
        })
        .sum::<f32>()
        / original.len() as f32
}

fn calculate_recall(original: &Array2<f32>, compressed: &Array2<f32>, k: usize) -> Result<f32> {
    let n_samples = original.len_of(Axis(0));
    let mut total_recall = 0.0;

    for i in 0..n_samples {
        let query = original.slice(s![i, ..]);

        let mut true_neighbors: Vec<(usize, f32)> = (0..n_samples)
            .filter(|&j| j != i)
            .map(|j| (j, euclidean_distance(&query, &original.slice(s![j, ..]))))
            .collect();
        true_neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let true_neighbors: Vec<usize> =
            true_neighbors.iter().take(k).map(|&(idx, _)| idx).collect();

        let mut approx_neighbors: Vec<(usize, f32)> = (0..n_samples)
            .filter(|&j| j != i)
            .map(|j| {
                (
                    j,
                    euclidean_distance(&compressed.slice(s![i, ..]), &compressed.slice(s![j, ..])),
                )
            })
            .collect();
        approx_neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let approx_neighbors: Vec<usize> = approx_neighbors
            .iter()
            .take(k)
            .map(|&(idx, _)| idx)
            .collect();

        let intersection: f32 = true_neighbors
            .iter()
            .filter(|&&idx| approx_neighbors.contains(&idx))
            .count() as f32;

        total_recall += intersection / k as f32;
    }

    Ok(total_recall / n_samples as f32)
}

fn main() -> Result<()> {
    env_logger::init();

    let n_samples = 1000;
    let n_dims = 128;
    let original_data = Array2::<f32>::random((n_samples, n_dims), Uniform::new(0.0, 1.0));

    let m = 16;
    let ks = 256;
    let iterations = 10;

    let mut pq = PQ::try_new(m, ks)?;

    let fit_start = Instant::now();
    pq.fit(&original_data, iterations)?;
    println!("Fit completed in {:?}", fit_start.elapsed());

    let encode_start = Instant::now();
    let compressed_data = pq.compress(&original_data)?;
    println!("Compression completed in {:?}", encode_start.elapsed());

    let reconstruction_error = calculate_reconstruction_error(&original_data, &compressed_data);
    println!("Reconstruction Error: {:.4}", reconstruction_error);

    let recall = calculate_recall(&original_data, &compressed_data, 10)?;
    println!("Recall@10: {:.4}", recall);

    Ok(())
}
