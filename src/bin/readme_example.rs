use anyhow::Result;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use rand_distr::StandardNormal;
use vector_quantizer::pq::PQ;

fn main() -> Result<()> {
    // Generate sample vectors to quantize
    let num_vectors = 1000;
    let dimension = 128;
    let original_vectors = Array2::random((num_vectors, dimension), StandardNormal);

    // Configure PQ parameters
    let m = 8; // Number of subspaces (controls compression ratio)
    let ks = 256; // Number of centroids per subspace (usually 256 for uint8)
    let mut pq = PQ::try_new(m, ks, Some(true))?;

    // Train the quantizer on the data
    println!("Training PQ model...");
    pq.fit(&original_vectors, 20)?;

    // Quantize the vectors
    println!("Quantizing vectors...");
    let quantized_vectors = pq.compress(&original_vectors)?;

    // Print some statistics about the quantization
    let compression_ratio = calc_compression_ratio(m, ks, dimension);
    let mse = calc_mse(&original_vectors, &quantized_vectors);

    println!("\nQuantization Results:");
    println!("Original vector size: {} bytes", dimension * 4); // 4 bytes per f32
    println!("Quantized vector size: {} bytes", m); // 1 byte per subspace with ks=256
    println!("Compression ratio: {:.2}x", compression_ratio);
    println!("Mean Squared Error: {:.6}", mse);

    // Example of how to get the compact codes for storage
    let compact_codes = pq.encode(&original_vectors)?;
    println!("\nCompact codes shape: {:?}", compact_codes.dim());

    // Demonstrate reconstructing vectors from compact codes
    let reconstructed = pq.decode(&compact_codes)?;
    assert_eq!(reconstructed.dim(), original_vectors.dim());

    Ok(())
}

// Helper function to calculate compression ratio
fn calc_compression_ratio(m: usize, ks: u32, dimension: usize) -> f64 {
    let original_size = dimension * 4; // 4 bytes per f32
    let quantized_size = m; // 1 byte per subspace when ks=256
    original_size as f64 / quantized_size as f64
}

// Helper function to calculate Mean Squared Error
fn calc_mse(original: &Array2<f32>, quantized: &Array2<f32>) -> f32 {
    (&(original - quantized))
        .mapv(|x| x.powi(2))
        .mean()
        .unwrap()
}
