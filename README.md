# Vector Quantizer

[![Tests](https://github.com/oramasearch/quantizer/actions/workflows/ci.yml/badge.svg)](https://github.com/oramasearch/quantizer/actions/workflows/ci.yml)

Simple vector quantization utilities and functions.

```shell
cargo add vector_quantizer
```

Example usage:

```rust
use anyhow::Result;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use vector_quantizer::pq::PQ;
use rand_distr::StandardNormal;

fn main() -> Result<()> {
    // Generate sample vectors to quantize
    let num_vectors = 1000;
    let dimension = 128;
    let original_vectors = Array2::random((num_vectors, dimension), StandardNormal);

    // Configure PQ parameters
    let m = 8; // Number of subspaces (controls compression ratio)
    let ks = 256; // Number of centroids per subspace (usually 256 for uint8)
    let mut pq = PQ::try_new(m, ks)?;

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
```

See a more detailed example here: [/src/bin/example.rs](/src/bin/example.rs)

## Performance Benchmarks

The PQ implementation was tested on datasets ranging from 1,000 to 1,000,000 vectors (128 dimensions each), using 16 subspaces and 256 centroids per subspace. Key findings:

- **Memory Efficiency**: Consistently achieves 96.88% memory reduction across all dataset sizes
- **Processing Speed**:
    - Fitting: Scales linearly, processing 100k vectors in ~3.7s (1M vectors in ~38s)
    - Compression: Very efficient, handling ~278k vectors per second (1M vectors in 3.57s)
- **Quality Metrics**:
    - Reconstruction Error: Remains low (0.013-0.021) across all dataset sizes
    - Recall@10: Ranges from 0.40 (small datasets) to 0.18 (large datasets)

The benchmark was tested on a 2022 MacBook Pro, M2 Pro, 16GB RAM. Run your own tests by running:

```sh
make quality_check
```

## Acknowledgements
The code in this repository is mostly adapted from [https://github.com/xinyandai/product-quantization](https://github.com/xinyandai/product-quantization), a great Python lib for vector quantization.

The original code and the one written in this repository is derived from "Norm-Explicit Quantization: Improving Vector Quantization for Maximum Inner Product Search" by Dai, Xinyan and Yan, Xiao and Ng, Kelvin KW and Liu, Jie and Cheng, James: [https://arxiv.org/abs/1911.04654](https://arxiv.org/abs/1911.04654)