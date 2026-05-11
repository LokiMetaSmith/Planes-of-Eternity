use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use reality_engine::engine::Engine;
use reality_engine::projector::{RealityProjector, NpcBehavior};
use reality_engine::reality_types::{RealityArchetype, RealitySignature};
use cgmath::Point3;

// Dummy PRNG for deterministic benchmarking
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }
    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.state as f32) / (u32::MAX as f32)
    }
    fn gen_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_f32() * (max - min)
    }
}

fn npc_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("npc_simulation");

    for npc_count in [10, 50, 100, 500].iter() {
        group.bench_with_input(criterion::BenchmarkId::new("update_npcs", npc_count), npc_count, |b, &count| {
            let mut engine = Engine::new(800, 600, None);
            let mut rng = Lcg::new(42);

            // Clear existing NPCs
            engine.world_state.npcs.clear();

            // Spawn `count` NPCs
            for _ in 0..count {
                let mut sig = RealitySignature::default();
                sig.active_style.archetype = RealityArchetype::SciFi;
                let loc = Point3::new(
                    rng.gen_range(-50.0, 50.0),
                    1.0,
                    rng.gen_range(-50.0, 50.0),
                );
                let mut npc = RealityProjector::new(loc, sig);
                npc.behavior = Some(NpcBehavior {
                    preferred_archetype: RealityArchetype::SciFi,
                    energy: 100.0,
                    mutation_progress: 0.0,
                    hostile: false,
                });
                engine.world_state.npcs.push(npc);
            }

            b.iter(|| {
                // Update engine for 16ms (60 FPS)
                engine.update(black_box(0.016), None);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, npc_benchmark);
criterion_main!(benches);
