# Quantizer

[![Tests](https://github.com/oramasearch/quantizer/actions/workflows/ci.yml/badge.svg)](https://github.com/oramasearch/quantizer/actions/workflows/ci.yml)

Simple vector quantization utilities and functions.

```shell
cargo add quantizer
```

Example usage:

```rust
use anyhow::Result;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use rand::distributions::StandardNormal;
use quantizer::PQ;

fn main() -> Result<()> {
    // Generate some random data
    let num_vectors = 1000;
    let dimension = 128;
    let data = Array2::random((num_vectors, dimension), StandardNormal);

    // Initialize the PQ model
    let m = 8;       // Number of subspaces
    let ks = 256;    // Number of clusters per subspace
    let mut pq = PQ::try_new(m, ks, Some(true))?;

    // Train the PQ model on the data
    let iterations = 20;
    pq.fit(&data, iterations)?;

    // Encode the data into compact codes
    let codes = pq.encode(&data)?;

    // Decode the codes to reconstruct the data
    let reconstructed_data = pq.decode(&codes)?;

    // Calculate the mean squared error between original and reconstructed data
    let mse = (&data - &reconstructed_data)
        .mapv(|x| x.powi(2))
        .mean()
        .unwrap();

    println!("Mean Squared Error: {}", mse);

    Ok(())
}
```

See a more detailed example here: [/src/bin/example.rs](/src/bin/example.rs)

# Acknowledgements
The code in this repository is mostly adapted from [https://github.com/xinyandai/product-quantization](https://github.com/xinyandai/product-quantization), a great Python lib for vector quantization.

The original code and the one written in this repository is derived from "Norm-Explicit Quantization: Improving Vector Quantization for Maximum Inner Product Search" by Dai, Xinyan and Yan, Xiao and Ng, Kelvin KW and Liu, Jie and Cheng, James: [https://arxiv.org/abs/1911.04654](https://arxiv.org/abs/1911.04654)