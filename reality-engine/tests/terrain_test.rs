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

#[test]
fn test_gct_iterative_extension() {
    let mut model = reality_genie::gct::SceneExtensionModel::new(32);

    // Add some trajectory
    model.trajectory_memory.record([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], 0.0);
    model.trajectory_memory.record([5.0, 1.0, 5.0], [0.0, 0.0, 0.0], 0.1);

    // Create anchor context
    let mut anchor = reality_genie::gct::AnchorContext::new(32);
    // Add a dummy left neighbor chunk with a flat heightmap at height 10
    let left_heightmap = vec![10.0; 32 * 32];
    anchor.add_neighbor(-1, 0, left_heightmap);

    let heights = model.generate_cohesive_heightmap(0, 0, &anchor);

    // Verify left boundary is perfectly constrained/blended to 10.0
    // Left boundary in 32x32 heightmap is index: 0 + idx * 32
    for idx in 0..32 {
        assert_eq!(heights[0 + idx * 32], 10.0, "GCT boundary constraint failed at left index {}", idx);
    }
}
