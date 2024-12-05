use crate::pq::CodeType;
use ndarray::Ix;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PQResidualError {
    #[error("Must use at least one product quantizer")]
    MissingProductQuantizer,

    #[error("Missing dataset name")]
    MissingDatasetName,

    #[error("Model not trained. Call fit() before calling this method")]
    ModelNotTrained,

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("PQ Residual error: {0}")]
    Pq(#[from] PQError),
}

#[derive(Error, Debug)]
pub enum PQError {
    #[error("Number of clusters (ks) must be between 1 and 2**32 - 1. Got {0}")]
    InvalidKs(u32),

    #[error("Number of subspaces (m) must be greater than 0. Got {0}")]
    InvalidSubspaces(usize),

    #[error("Code value {x} exceeds number of clusters {y}")]
    NClusterExceeded { x: usize, y: u32 },

    #[error("The number of training vectors ({n_vectors}) must be more than ks ({ks})")]
    InsufficientTrainingVectors { n_vectors: usize, ks: u32 },

    #[error("Input vectors must have at least one dimension")]
    EmptyInputVectors,

    #[error("Number of subspaces (m) cannot exceed vector dimensions ({m} > {n_dims})")]
    SubspacesExceedDimensions { m: usize, n_dims: usize },

    #[error("Model not trained. Call fit() before calling this method")]
    ModelNotTrained,

    #[error("Encoded values exceed the range for {0:?}")]
    EncodedValueExceedsRange(CodeType),

    #[error("Unsupported initialization method: {0}")]
    UnsupportedInitializationMethod(String),

    #[error("Input vectors dimensions should match training dimensions")]
    TrainingDimensionsDoesntMatchInputDimensions,

    #[error("Encoded values exceed U8 range")]
    EncodedValuesExceedU8Range,

    #[error("Encoded values exceed U16 range")]
    EncodedValuesExceedU16Range,

    #[error("Data must have at least one sample and one feature")]
    DataOrFeatureMissing,

    #[error("Number of clusters k must be between 1 and number of samples ({x})")]
    WrongNumberOfClusters { x: Ix },

    #[error("Data contains non-finite values (NaN or Inf)")]
    NonFiniteValue,

    #[error("Unsupported initialization method")]
    InvalidInitMethod,
}
