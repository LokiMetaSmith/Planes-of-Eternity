use criterion::{criterion_group, criterion_main, Criterion};
use reality_engine::voxel::{Chunk, ChunkKey, Voxel};
use std::hint::black_box;

fn generate_splats_benchmark(c: &mut Criterion) {
    let mut chunk = Chunk::new(ChunkKey { x: 0, y: 0, z: 0 });
    // Fill the chunk completely
    for x in 0..32 {
        for y in 0..32 {
            for z in 0..32 {
                chunk.set(x, y, z, Voxel { id: 1 });
            }
        }
    }

    c.bench_function("generate_splats_full_chunk", |b| {
        b.iter(|| {
            let splats = chunk.generate_splats();
            black_box(splats);
        })
    });

    let mut chunk_sparse = Chunk::new(ChunkKey { x: 0, y: 0, z: 0 });
    for x in 0..32 {
        for z in 0..32 {
            chunk_sparse.set(x, 0, z, Voxel { id: 1 });
        }
    }

    c.bench_function("generate_splats_sparse_chunk", |b| {
        b.iter(|| {
            let splats = chunk_sparse.generate_splats();
            black_box(splats);
        })
    });

    let mut chunk_meshing = Chunk::new(ChunkKey { x: 0, y: 0, z: 0 });
    for x in 0..32 {
        for y in 0..32 {
            for z in 0..32 {
                chunk_meshing.set(x, y, z, Voxel { id: 1 });
            }
        }
    }

    c.bench_function("generate_mesh_full_chunk", |b| {
        b.iter(|| {
            let mesh = chunk_meshing.generate_mesh();
            black_box(mesh);
        })
    });

    let mut chunk_sparse_meshing = Chunk::new(ChunkKey { x: 0, y: 0, z: 0 });
    for x in 0..32 {
        for z in 0..32 {
            chunk_sparse_meshing.set(x, 0, z, Voxel { id: 1 });
        }
    }

    c.bench_function("generate_mesh_sparse_chunk", |b| {
        b.iter(|| {
            let mesh = chunk_sparse_meshing.generate_mesh();
            black_box(mesh);
        })
    });
}

criterion_group!(benches, generate_splats_benchmark);
criterion_main!(benches);
