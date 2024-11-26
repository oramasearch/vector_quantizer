use crate::utils::{determine_code_type, euclidean_distance, kmeans2};
use anyhow::Result;
use ndarray::parallel::prelude::*;
use ndarray::{s, Array2, Array3, Axis};
use rayon::prelude::*;
use log::{debug, error, info, trace, warn};

#[derive(Debug, Clone, Copy)]
pub enum CodeType {
    U8,
    U16,
    U32,
}

pub struct PQ {
    m: usize,
    ks: u32,
    code_dtype: CodeType,
    codewords: Option<Array3<f32>>,
    ds: Option<Vec<usize>>,
    dim: Option<usize>,
}

impl PQ {
    pub fn try_new(m: usize, ks: u32) -> Result<Self> {
        if ks == 0 {
            anyhow::bail!(
                "cluster subspaces (ks) must be a u32 between 1 and 2**32 - 1. Got {}",
                ks
            )
        }

        if m == 0 {
            anyhow::bail!("Number of subspaces (m) must be greater than 0. Got {}", m);
        }

        Ok(Self {
            m,
            ks,
            code_dtype: determine_code_type(ks),
            codewords: None,
            ds: None,
            dim: None,
        })
    }

    pub fn fit(&mut self, vecs: &Array2<f32>, iterations: usize) -> Result<&mut Self> {
        let (n_vectors, n_dims) = vecs.dim();

        if self.ks > n_vectors as u32 {
            anyhow::bail!(
                "The number of training vectors ({}) should be more than ks ({})",
                n_vectors,
                self.ks
            );
        }

        if n_dims == 0 {
            anyhow::bail!("Input vectors must have at least one dimension");
        }

        if self.m > n_dims {
            anyhow::bail!(
                "Number of subspaces (m) cannot be greater than vector dimensions ({} > {})",
                self.m,
                n_dims
            );
        }

        self.dim = Some(n_dims as usize);

        let reminder: usize = n_dims % self.m;
        let quotient: usize = n_dims / self.m;

        let dims_width: Vec<usize> = (0..self.m)
            .map(|i| if i < reminder { quotient + 1 } else { quotient })
            .collect();

        let mut ds: Vec<usize> = dims_width
            .iter()
            .scan(0, |acc, &x| {
                *acc += x;
                Some(*acc)
            })
            .collect();
        ds.insert(0, 0);

        self.ds = Some(ds);

        let max_width = dims_width.iter().max().unwrap();
        let mut codewords = Array3::<f32>::zeros((self.m, self.ks as usize, *max_width));

        let trained_codewords: Vec<(usize, Array2<f32>)> = (0..self.m)
            .into_par_iter()
            .map(|m| {
                info!(
                    "Training the subspace: {} / {}, {} -> {}",
                    m,
                    self.m,
                    self.ds.as_ref().unwrap()[m],
                    self.ds.as_ref().unwrap()[m + 1]
                );

                let ds_ref = self.ds.as_ref().unwrap();

                let vecs_sub = vecs.slice(s![.., ds_ref[m]..ds_ref[m + 1]]);

                let (centroids, _) = kmeans2(&vecs_sub.to_owned(), self.ks, iterations, "points")?;

                Ok((m, centroids))
            })
            .collect::<Result<Vec<(usize, Array2<f32>)>>>()?;

        for (m, centroids) in trained_codewords {
            let ds_ref = self.ds.as_ref().unwrap();
            let subspace_width = ds_ref[m + 1] - ds_ref[m];

            codewords
                .slice_mut(s![m, .., ..subspace_width])
                .assign(&centroids);
        }

        self.codewords = Some(codewords);
        Ok(self)
    }

    pub fn encode(&self, vecs: &Array2<f32>) -> Result<Array2<u32>> {
        let (n_vectors, n_dims) = vecs.dim();

        let training_dim = self.dim.ok_or_else(|| {
            anyhow::anyhow!("Model not trained. Call fit() before calling encode()")
        })?;

        if n_dims != training_dim {
            anyhow::bail!("Input vectors dimensions should match training dimensions");
        }

        let mut codes = Array2::<u32>::zeros((n_vectors, self.m));

        let codewords = self
            .codewords
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model not trained. Call fit() first"))?;

        let ds = self
            .ds
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model not trained. Call fit() first"))?;

        codes
            .outer_iter_mut()
            .into_par_iter()
            .zip(vecs.outer_iter())
            .for_each(|(mut code_row, vec)| {
                for m in 0..self.m {
                    let subspace = vec.slice(s![ds[m]..ds[m + 1]]);
                    let subspace_width = ds[m + 1] - ds[m];
                    let codewords_sub = codewords.slice(s![m, .., ..subspace_width]);

                    let mut min_dist = f32::INFINITY;
                    let mut min_idx = 0;

                    for (j, codeword) in codewords_sub.axis_iter(Axis(0)).enumerate() {
                        let dist = euclidean_distance(&subspace, &codeword);
                        if dist < min_dist {
                            min_dist = dist;
                            min_idx = j;
                        }
                    }

                    code_row[m] = min_idx as u32;
                }
            });

        match self.code_dtype {
            CodeType::U8 => {
                if codes.iter().any(|&x| x > u8::MAX as u32) {
                    anyhow::bail!("Encoded values exceed U8 range");
                }
            }
            CodeType::U16 => {
                if codes.iter().any(|&x| x > u16::MAX as u32) {
                    anyhow::bail!("Encoded values exceed U16 range");
                }
            }
            CodeType::U32 => {}
        };

        Ok(codes)
    }

    pub fn decode(&self, codes: &Array2<u32>) -> Result<Array2<f32>> {
        let (n_vectors, m) = codes.dim();

        if m != self.m {
            anyhow::bail!("Code dimensions don't match training dimensions");
        }

        let dim = self
            .dim
            .ok_or_else(|| anyhow::anyhow!("Model not trained"))?;
        let ds = self
            .ds
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model not trained"))?;
        let codewords = self
            .codewords
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model not trained"))?;

        let mut vecs = Array2::<f32>::zeros((n_vectors, dim));

        vecs.outer_iter_mut()
            .into_par_iter()
            .zip(codes.outer_iter())
            .try_for_each(|(mut vec_row, code_row)| -> Result<(), anyhow::Error> {
                for m in 0..self.m {
                    let code_idx = code_row[m] as usize;
                    if code_idx >= self.ks as usize {
                        return Err(anyhow::anyhow!(
                            "Code value {} exceeds number of clusters {}",
                            code_idx,
                            self.ks
                        ));
                    }
                    let subspace_width = ds[m + 1] - ds[m];
                    let codeword = codewords.slice(s![m, code_idx, ..subspace_width]);

                    vec_row.slice_mut(s![ds[m]..ds[m + 1]]).assign(&codeword);
                }
                Ok(())
            })?;

        Ok(vecs)
    }

    pub fn compress(&self, vecs: &Array2<f32>) -> Result<Array2<f32>> {
        let codes = self.encode(vecs)?;
        self.decode(&codes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::create_random_vectors;
    use ndarray::Array2;

    fn create_dummy_vectors(num_vectors: usize, dimension: usize) -> Array2<f32> {
        Array2::<f32>::zeros((num_vectors, dimension))
    }

    // Edge case: ks is zero or exceeds u32 limits.
    #[test]
    fn test_try_new_invalid_ks_zero() {
        let pq = PQ::try_new(4, 0);
        assert!(pq.is_err(), "Initialization should fail when ks is zero");
    }

    #[test]
    fn test_try_new_invalid_ks_max() {
        let pq = PQ::try_new(4, u32::MAX);
        assert!(
            pq.is_ok(),
            "Initialization should succeed when ks is u32::MAX"
        );
    }

    // Edge Case: m is zero.
    #[test]
    fn test_try_new_invalid_m_zero() {
        let pq = PQ::try_new(0, 256);
        assert!(
            pq.is_err(),
            "Initialization should fail when m is zero, but it succeeded"
        );
    }

    // Edge Case: Number of training vectors is less than ks.
    #[test]
    fn test_fit_vectors_less_than_ks() {
        let mut pq = PQ::try_new(4, 256).unwrap();
        let vecs = create_dummy_vectors(100, 128); // Less than ks
        let result = pq.fit(&vecs, 10);
        assert!(
            result.is_err(),
            "Fit should fail when vectors are less than ks"
        );
    }

    // Edge Case: Vectors have zero dimensions or m exceeds vector dimensions.
    #[test]
    fn test_fit_zero_dimensions() {
        let mut pq = PQ::try_new(4, 256).unwrap();
        let vecs = create_dummy_vectors(1000, 0); // Zero dimensions
        let result = pq.fit(&vecs, 10);
        assert!(
            result.is_err(),
            "Fit should fail with zero-dimensional vectors"
        );
    }

    #[test]
    fn test_fit_m_greater_than_dimensions() {
        let mut pq = PQ::try_new(200, 256).unwrap();
        let vecs = create_dummy_vectors(1000, 128); // m > dimensions
        let result = pq.fit(&vecs, 10);
        assert!(
            result.is_err(),
            "Fit should fail when m > vector dimensions"
        );
    }

    // Edge Case: Calling encode before fit.
    #[test]
    fn test_encode_without_fit() {
        let pq = PQ::try_new(4, 256).unwrap();
        let vecs = create_dummy_vectors(1000, 128);
        let result = pq.encode(&vecs);
        assert!(
            result.is_err(),
            "Encode should fail if fit() hasn't been called"
        );
    }

    // Edge Case: Vectors have different dimensions than those used in fit.
    #[test]
    fn test_encode_mismatched_dimensions() {
        let mut pq = PQ::try_new(4, 256).unwrap();
        let train_vecs = create_dummy_vectors(1000, 128);
        pq.fit(&train_vecs, 10).unwrap();

        let vecs = create_dummy_vectors(1000, 64); // Different dimensions
        let result = pq.encode(&vecs);
        assert!(
            result.is_err(),
            "Encode should fail with mismatched dimensions"
        );
    }

    // Edge Case: Codes have incorrect dimensions or contain invalid values.
    #[test]
    fn test_decode_invalid_code_m() {
        let mut pq = PQ::try_new(4, 256).unwrap();
        let train_vecs = create_dummy_vectors(1000, 128);
        pq.fit(&train_vecs, 10).unwrap();

        let codes = Array2::<u32>::zeros((1000, 3)); // Incorrect m
        let result = pq.decode(&codes);
        assert!(
            result.is_err(),
            "Decode should fail with incorrect code dimensions"
        );
    }

    #[test]
    fn test_decode_code_value_exceeds_ks() {
        let mut pq = PQ::try_new(4, 256).unwrap();
        let train_vecs = create_dummy_vectors(1000, 128);
        pq.fit(&train_vecs, 10).unwrap();

        let mut codes = Array2::<u32>::zeros((1000, 4));
        codes[[0, 0]] = 300; // Exceeds ks
        let result = pq.decode(&codes);
        assert!(
            result.is_err(),
            "Decode should fail if code values exceed ks"
        );
    }

    // Edge Case: Ensuring compress works end-to-end.
    #[test]
    fn test_compress() {
        let mut pq = PQ::try_new(4, 256).unwrap();
        let vecs = create_dummy_vectors(1000, 128);
        pq.fit(&vecs, 10).unwrap();

        let compressed_vecs = pq.compress(&vecs).unwrap();
        assert_eq!(
            compressed_vecs.dim(),
            vecs.dim(),
            "Compressed vectors should have the same dimensions"
        );
    }

    // Edge Case: Ensuring code values fit within specified data types.
    #[test]
    fn test_encode_code_dtype_u8_overflow() {
        let mut pq = PQ::try_new(4, 300).unwrap(); // ks exceeds u8::MAX
        pq.code_dtype = CodeType::U8;
        let vecs = create_random_vectors(1000, 128);
        pq.fit(&vecs, 10).unwrap();

        let result = pq.encode(&vecs);
        assert!(
            result.is_err(),
            "Encode should fail if code values exceed u8::MAX"
        );
    }

    #[test]
    fn test_encode_code_dtype_u16_overflow() {
        let mut pq = PQ::try_new(4, 70000).unwrap();
        pq.code_dtype = CodeType::U16;
        pq.codewords = Some(Array3::zeros((pq.m, pq.ks as usize, 128 / pq.m)));
        pq.dim = Some(128);

        let vecs = create_random_vectors(1, 128);
        let result = pq.encode(&vecs);
        assert!(
            result.is_err(),
            "Encode should fail if code values exceed u16::MAX"
        );
    }

    #[test]
    fn test_encode_code_dtype_u8_valid() {
        let mut pq = PQ::try_new(4, 200).unwrap(); // ks within u8::MAX
        pq.code_dtype = CodeType::U8;
        let vecs = create_random_vectors(1000, 128);
        pq.fit(&vecs, 10).unwrap();

        let result = pq.encode(&vecs);
        assert!(
            result.is_ok(),
            "Encode should succeed with valid u8 code values"
        );
    }
}
