use anyhow::Result;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use quantizer::pq::PQ;
use rand_distr::StandardNormal;

fn create_random_vectors(n: usize, dim: usize) -> Array2<f32> {
    Array2::random((n, dim), StandardNormal)
}

fn main() -> Result<()> {
    // Step 1: Create random vectors for training
    let n_train = 1000;
    let dimension = 128;
    let train_vectors = create_random_vectors(n_train, dimension);

    // Step 2: Initialize the PQ model
    let m = 8; // Number of subspaces
    let ks = 256; // Number of clusters per subspace
    let verbose = Some(true);

    let mut pq = PQ::try_new(m, ks, verbose)?;

    // Step 3: Train the PQ model
    let iterations = 20; // Number of iterations for k-means
    pq.fit(&train_vectors, iterations)?;

    // Step 4: Create random test vectors and encode them
    let n_test = 200;
    let test_vectors = create_random_vectors(n_test, dimension);
    let codes = pq.encode(&test_vectors)?;

    // Step 5: Decode the codes to approximate the original test vectors
    let reconstructed_vectors = pq.decode(&codes)?;

    // Step 6: Compute the reconstruction error (Mean Squared Error)
    let mse = ((&test_vectors - &reconstructed_vectors)
        .mapv(|x| x.powi(2))
        .mean()
        .unwrap()) as f64;

    println!("Mean Squared Reconstruction Error: {}", mse);

    Ok(())
}
