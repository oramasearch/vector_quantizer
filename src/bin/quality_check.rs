use anyhow::Result;
use log::info;
use ndarray::{s, Array2, ArrayView1, Axis};
use ndarray_rand::RandomExt;
use rand_distr::Uniform;
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use vector_quantizer::pq::PQ;

#[derive(Serialize)]
struct BenchmarkResult {
    n_samples: usize,
    n_dims: usize,
    fit_time_ms: f64,
    compression_time_ms: f64,
    reconstruction_error: f32,
    recall: f32,
    memory_reduction_ratio: f32,
}

fn run_benchmark(
    n_samples: usize,
    n_dims: usize,
    m: usize,
    ks: u32,
    iterations: usize,
) -> Result<BenchmarkResult> {
    let original_data = Array2::<f32>::random((n_samples, n_dims), Uniform::new(0.0, 1.0));

    let mut pq = PQ::try_new(m, ks)?;

    let fit_start = Instant::now();
    pq.fit(&original_data, iterations)?;
    let fit_time = fit_start.elapsed().as_secs_f64() * 1000.0;

    let compress_start = Instant::now();
    let compressed_data = pq.compress(&original_data)?;
    let compression_time = compress_start.elapsed().as_secs_f64() * 1000.0;

    let reconstruction_error = calculate_reconstruction_error(&original_data, &compressed_data);
    let recall = calculate_recall(&original_data, &compressed_data, 10)?;

    let original_size = n_samples * n_dims * size_of::<f32>();
    let compressed_size = n_samples * m; // Each subspace uses 1 byte
    let memory_reduction_ratio = compressed_size as f32 / original_size as f32;

    Ok(BenchmarkResult {
        n_samples,
        n_dims,
        fit_time_ms: fit_time,
        compression_time_ms: compression_time,
        reconstruction_error,
        recall,
        memory_reduction_ratio,
    })
}

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

    let max_eval_samples = 1000;
    let eval_samples = if n_samples > max_eval_samples {
        max_eval_samples
    } else {
        n_samples
    };

    let mut total_recall = 0.0;
    let step = n_samples / eval_samples;

    for i in (0..n_samples).step_by(step) {
        let query = original.slice(s![i, ..]);

        let search_window = if n_samples > 10000 { 5000 } else { n_samples };

        let start_idx = if i > search_window / 2 {
            i - search_window / 2
        } else {
            0
        };
        let end_idx = (i + search_window / 2).min(n_samples);

        let mut true_neighbors: Vec<(usize, f32)> = (start_idx..end_idx)
            .filter(|&j| j != i)
            .map(|j| (j, euclidean_distance(&query, &original.slice(s![j, ..]))))
            .collect();
        true_neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let true_neighbors: Vec<usize> =
            true_neighbors.iter().take(k).map(|&(idx, _)| idx).collect();

        let mut approx_neighbors: Vec<(usize, f32)> = (start_idx..end_idx)
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

    Ok(total_recall / (n_samples / step) as f32)
}

fn main() -> Result<()> {
    env_logger::init();

    let sample_sizes = vec![1_000, 5_000, 10_000, 50_000, 100_000, 1_000_000];
    let n_dims = 128;
    let m = 16;
    let ks = 256;
    let iterations = 10;

    let mut results = Vec::new();

    for n_samples in sample_sizes {
        info!("Running benchmark with {} samples...", n_samples);
        let result = run_benchmark(n_samples, n_dims, m, ks, iterations)?;
        results.push(result);
    }

    let mut file = File::create("benchmark_results.csv")?;
    writeln!(file, "n_samples,n_dims,fit_time_ms,compression_time_ms,reconstruction_error,recall,memory_reduction_ratio")?;

    for result in &results {
        writeln!(
            file,
            "{},{},{},{},{},{},{}",
            result.n_samples,
            result.n_dims,
            result.fit_time_ms,
            result.compression_time_ms,
            result.reconstruction_error,
            result.recall,
            result.memory_reduction_ratio
        )?;
    }

    for result in &results {
        info!("\nResults for {} samples:", result.n_samples);
        info!("Fit time: {:.2}ms", result.fit_time_ms);
        info!("Compression time: {:.2}ms", result.compression_time_ms);
        info!("Reconstruction Error: {:.4}", result.reconstruction_error);
        info!("Recall@10: {:.4}", result.recall);
        info!(
            "Memory reduction: {:.2}%",
            (1.0 - result.memory_reduction_ratio) * 100.0
        );
    }

    Ok(())
}
