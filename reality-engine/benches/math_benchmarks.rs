use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn math_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("math");

    group.bench_function("powi(2)", |b| {
        b.iter(|| {
            let x = black_box(10.0f32);
            black_box(x.powi(2));
        })
    });

    group.bench_function("multiplication", |b| {
        b.iter(|| {
            let x = black_box(10.0f32);
            black_box(x * x);
        })
    });

    group.bench_function("always_sqrt", |b| {
        b.iter(|| {
            let x = black_box(10.0f32);
            let dist = (x * x).sqrt();
            if dist < 100.0 {
                black_box(dist * 2.0);
            }
        })
    });

    group.bench_function("conditional_sqrt", |b| {
        b.iter(|| {
            let x = black_box(10.0f32);
            let dist_sq = x * x;
            if dist_sq < 10000.0 {
                let dist = dist_sq.sqrt();
                black_box(dist * 2.0);
            }
        })
    });

    use cgmath::InnerSpace;
    let vec = cgmath::Vector3::new(5.0f32, 5.0, 2.0);

    group.bench_function("cgmath_normalize", |b| {
        b.iter(|| {
            let v = black_box(vec);
            let dist_sq = v.magnitude2();
            if dist_sq < 100.0 {
                black_box(v.normalize());
            }
        })
    });

    group.bench_function("manual_inline_normalization", |b| {
        b.iter(|| {
            let v = black_box(vec);
            let dist_sq = v.magnitude2();
            if dist_sq < 100.0 {
                let dist = dist_sq.sqrt();
                let scalar_force = 1.0 / dist;
                black_box(v * scalar_force);
            }
        })
    });

    group.finish();
}

criterion_group!(benches, math_benchmark);
criterion_main!(benches);
