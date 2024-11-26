use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array2;
use rand::Rng;
use vector_quantizer::pq::PQ;

fn generate_test_data(n_vectors: usize, n_dims: usize) -> Array2<f32> {
    let mut rng = rand::thread_rng();
    Array2::from_shape_fn((n_vectors, n_dims), |_| rng.gen::<f32>())
}

fn bench_pq_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("PQ Fit");

    for &size in &[(1000, 128), (5000, 256), (10000, 512)] {
        let (n_vectors, n_dims) = size;
        let data = generate_test_data(n_vectors, n_dims);

        group.bench_function(format!("fit_{}x{}", n_vectors, n_dims), |b| {
            b.iter(|| {
                let mut pq = PQ::try_new(8, 256, Some(false)).unwrap();
                pq.fit(black_box(&data), black_box(10)).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_pq_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("PQ Encode");

    for &size in &[(1000, 128), (5000, 256), (10000, 512)] {
        let (n_vectors, n_dims) = size;
        let training_data = generate_test_data(n_vectors, n_dims);
        let test_data = generate_test_data(n_vectors / 2, n_dims);

        let mut pq = PQ::try_new(8, 256, Some(false)).unwrap();
        pq.fit(&training_data, 10).unwrap();

        group.bench_function(format!("encode_{}x{}", n_vectors, n_dims), |b| {
            b.iter(|| {
                pq.encode(black_box(&test_data)).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_pq_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("PQ Decode");

    for &size in &[(1000, 128), (5000, 256), (10000, 512)] {
        let (n_vectors, n_dims) = size;
        let training_data = generate_test_data(n_vectors, n_dims);
        let test_data = generate_test_data(n_vectors / 2, n_dims);

        let mut pq = PQ::try_new(8, 256, Some(false)).unwrap();
        pq.fit(&training_data, 10).unwrap();
        let codes = pq.encode(&test_data).unwrap();

        group.bench_function(format!("decode_{}x{}", n_vectors, n_dims), |b| {
            b.iter(|| {
                pq.decode(black_box(&codes)).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_pq_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("PQ Compress");

    for &size in &[(1000, 128), (5000, 256), (10000, 512)] {
        let (n_vectors, n_dims) = size;
        let training_data = generate_test_data(n_vectors, n_dims);
        let test_data = generate_test_data(n_vectors / 2, n_dims);

        let mut pq = PQ::try_new(8, 256, Some(false)).unwrap();
        pq.fit(&training_data, 10).unwrap();

        group.bench_function(format!("compress_{}x{}", n_vectors, n_dims), |b| {
            b.iter(|| {
                pq.compress(black_box(&test_data)).unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_pq_fit,
    bench_pq_encode,
    bench_pq_decode,
    bench_pq_compress
);
criterion_main!(benches);
