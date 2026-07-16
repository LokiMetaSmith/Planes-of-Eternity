use reality_engine::voxel::Chunk;

#[test]
fn test_terrain_continuity() {
    // We check map_height function continuity independently, as base terrain has structures.
    // This allows us to ensure the core noise logic doesn't have obvious artifacts.

    let mut max_diff = 0;
    for z in -50..50 {
        for x in -50..50 {
            let h = Chunk::map_height(x as f32, z as f32);
            let h_x = Chunk::map_height(x as f32 + 1.0, z as f32);
            let h_z = Chunk::map_height(x as f32, z as f32 + 1.0);

            let diff_x = (h - h_x).abs();
            if diff_x > max_diff { max_diff = diff_x; }

            let diff_z = (h - h_z).abs();
            if diff_z > max_diff { max_diff = diff_z; }
        }
    }

    assert!(max_diff <= 3, "Terrain has sharp jumps, max diff: {}", max_diff);
}
