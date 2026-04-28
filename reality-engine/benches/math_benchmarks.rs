use std::time::Instant;

fn main() {
    let iterations = 10_000_000;

    // Benchmark 1: powi(2) vs multiplication
    let start_powi = Instant::now();
    for i in 0..iterations {
        let x = std::hint::black_box(10.0f32 + (i as f32 % 1000.0) * 0.001);
        let res = x.powi(2);
        std::hint::black_box(res);
    }
    let duration_powi = start_powi.elapsed();
    println!("powi(2) took: {:?}", duration_powi);

    let start_mul = Instant::now();
    for i in 0..iterations {
        let x = std::hint::black_box(10.0f32 + (i as f32 % 1000.0) * 0.001);
        let res = x * x;
        std::hint::black_box(res);
    }
    let duration_mul = start_mul.elapsed();
    println!("multiplication (x*x) took: {:?}", duration_mul);

    // Benchmark 2: Always sqrt vs Conditional sqrt (Threshold check using dist_sq)
    let start_always_sqrt = Instant::now();
    for i in 0..iterations {
        let x = std::hint::black_box(10.0f32 + (i as f32 * 0.0001));
        let dist = (x * x).sqrt();
        if dist < 100.0 {
            let res = dist * 2.0;
            std::hint::black_box(res);
        }
    }
    let duration_always_sqrt = start_always_sqrt.elapsed();
    println!("always sqrt took: {:?}", duration_always_sqrt);

    let start_cond_sqrt = Instant::now();
    for i in 0..iterations {
        let x = std::hint::black_box(10.0f32 + (i as f32 * 0.0001));
        let dist_sq = x * x;
        if dist_sq < 10000.0 {
            let dist = dist_sq.sqrt();
            let res = dist * 2.0;
            std::hint::black_box(res);
        }
    }
    let duration_cond_sqrt = start_cond_sqrt.elapsed();
    println!("conditional sqrt took: {:?}", duration_cond_sqrt);

    // Benchmark 3: cgmath vector normalization
    let vec = cgmath::Vector3::new(5.0f32, 5.0, 2.0); // magnitude^2 is 54.0
    use cgmath::InnerSpace;

    let start_normalize = Instant::now();
    for i in 0..iterations {
        let mut v = std::hint::black_box(vec);
        v.x += (i as f32 % 10.0) * 0.1; // prevent constant folding
        let dist_sq = v.magnitude2();
        if dist_sq < 100.0 {
            let res = v.normalize();
            std::hint::black_box(res);
        }
    }
    let duration_normalize = start_normalize.elapsed();
    println!("cgmath normalize() took: {:?}", duration_normalize);

    let start_manual_div = Instant::now();
    for i in 0..iterations {
        let mut v = std::hint::black_box(vec);
        v.x += (i as f32 % 10.0) * 0.1; // prevent constant folding
        let dist_sq = v.magnitude2();
        if dist_sq < 100.0 {
            let dist = dist_sq.sqrt();
            let scalar_force = 1.0 / dist;
            let res = v * scalar_force;
            std::hint::black_box(res);
        }
    }
    let duration_manual_div = start_manual_div.elapsed();
    println!(
        "manual inline normalization took: {:?}",
        duration_manual_div
    );
}
