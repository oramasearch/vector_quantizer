use anyhow::{anyhow, Result};
use ndarray::{Array2, Axis};
use ndarray_rand::RandomExt;
use rand_distr::StandardNormal;
use vector_quantizer::pq::PQ;

fn create_random_vectors(n: usize, dim: usize) -> Array2<f32> {
    Array2::random((n, dim), StandardNormal)
}

fn calculate_mse(original: &Array2<f32>, reconstructed: &Array2<f32>) -> Result<f32> {
    let diff = original - reconstructed;
    let mse = diff
        .mapv(|x| x.powi(2))
        .mean()
        .ok_or_else(|| anyhow!("Failed to compute MSE"))?;
    Ok(mse)
}

fn main() -> Result<()> {
    // Step 1: Generate Random Vectors
    let n_train = 1000; // Number of training vectors
    let n_test = 5; // Reduced number for easier viewing
    let dimension = 128;

    let train_vectors = create_random_vectors(n_train, dimension);
    let test_vectors = create_random_vectors(n_test, dimension);

    // Step 2: Initialize the PQ Model
    let m = 8; // Number of subspaces
    let ks = 256; // Number of clusters per subspace
    let verbose = Some(true);

    let mut pq = PQ::try_new(m, ks, verbose)?;

    // Step 3: Train the PQ Model
    let iterations = 20; // Number of iterations for k-means
    pq.fit(&train_vectors, iterations)?;

    // Step 4: Encode the Test Vectors
    let codes = pq.encode(&test_vectors)?;

    // View the codes
    println!("Codes (Compressed Representation):");
    println!("{:?}", codes);

    // Step 5: Decode the Codes to Reconstruct Vectors
    let reconstructed_vectors = pq.decode(&codes)?;

    // View the reconstructed vectors
    println!("Reconstructed (Quantized) Vectors:");
    println!("{:?}", reconstructed_vectors);

    // Compare with original vectors
    for i in 0..n_test {
        let original_vector = test_vectors.row(i);
        let quantized_vector = reconstructed_vectors.row(i);

        println!("Original Vector ({}): {:?}", i, original_vector);
        println!("Quantized Vector ({}): {:?}", i, quantized_vector);

        let difference = &original_vector - &quantized_vector;
        println!("Difference ({}): {:?}", i, difference);
        println!("----------------------------------------------------");
    }

    // Step 6: Evaluate Reconstruction Error
    let mse = calculate_mse(&test_vectors, &reconstructed_vectors)?;
    println!("Mean Squared Reconstruction Error: {}", mse);

    Ok(())
}
