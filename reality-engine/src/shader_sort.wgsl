struct SortEntry {
    distance: f32,
    index: u32,
};

struct SortUniforms {
    camera_pos: vec3<f32>,
    num_splats: u32,
};

struct BitonicUniforms {
    j: u32,
    k: u32,
};

@group(0) @binding(0) var<storage, read> unsorted_splats: array<u32>;
@group(0) @binding(1) var<storage, read_write> sort_entries: array<SortEntry>;
@group(0) @binding(2) var<uniform> sort_uniforms: SortUniforms;
@group(0) @binding(3) var<storage, read_write> sorted_splats: array<u32>;

@compute @workgroup_size(256)
fn compute_distances(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if (index >= arrayLength(&sort_entries)) {
        return;
    }

    if (index < sort_uniforms.num_splats) {
        // SplatVertex is 80 bytes = 20 u32s
        let stride = 20u;
        let px = bitcast<f32>(unsorted_splats[index * stride + 0u]);
        let py = bitcast<f32>(unsorted_splats[index * stride + 1u]);
        let pz = bitcast<f32>(unsorted_splats[index * stride + 2u]);

        let splat_pos = vec3<f32>(px, py, pz);
        let diff = splat_pos - sort_uniforms.camera_pos;
        let dist = dot(diff, diff);

        sort_entries[index] = SortEntry(dist, index);
    } else {
        // Descending order sort, so minimum distance (-1.0) goes to the end
        sort_entries[index] = SortEntry(-1.0, index);
    }
}

@group(1) @binding(0) var<uniform> bitonic: BitonicUniforms;

@compute @workgroup_size(256)
fn sort_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&sort_entries)) {
        return;
    }

    let j = bitonic.j;
    let k = bitonic.k;

    let ixj = i ^ j;

    if (ixj > i) {
        let entry1 = sort_entries[i];
        let entry2 = sort_entries[ixj];

        let dir = (i & k) == 0u;
        // We want descending order
        let condition = entry1.distance < entry2.distance;

        if (dir == condition) {
            sort_entries[i] = entry2;
            sort_entries[ixj] = entry1;
        }
    }
}

@compute @workgroup_size(256)
fn apply_sort(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= sort_uniforms.num_splats) {
        return;
    }

    let entry = sort_entries[index];

    let src_index = entry.index;
    let stride = 20u;

    // Copy 20 u32s (80 bytes) per splat
    for (var i = 0u; i < stride; i = i + 1u) {
        sorted_splats[index * stride + i] = unsorted_splats[src_index * stride + i];
    }
}
