use criterion::{criterion_group, criterion_main, Criterion};

fn bench_placeholder(_c: &mut Criterion) {
    // Benchmarks run via cargo oxide, not criterion.
    // This file exists to satisfy the [[bench]] manifest entry.
}

criterion_group!(benches, bench_placeholder);
criterion_main!(benches);
