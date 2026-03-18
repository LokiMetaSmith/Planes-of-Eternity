## 2024-05-24 - Extract Frustum Planes for AABB Visible Checks
**Learning:** In a 3D rendering engine, frustum culling can be a bottleneck when calculating the 6 viewing planes dynamically inside hot loops (like per-chunk checks). The `is_aabb_visible` method was originally extracting and normalizing these 6 planes for every chunk mesh or chunk instance visibility test, doing expensive vector magnitude operations ($O(N)$) repeatedly for the same camera state.
**Action:** Extract the frustum plane calculation out of `is_aabb_visible` into an `extract_frustum_planes` method that calculates the planes once per frame from the `view_proj` matrix. Pass the pre-calculated array of 6 planes into `is_aabb_visible`. This converts the cost to $O(1)$ per frame plus a much lighter bounds check per item.

## 2024-05-25 - Avoid Int-to-Float Conversions in Spatial Queries
**Learning:** Ray casting and chunk coordinate lookups (e.g., `set_voxel_at` or stepping a ray) often require converting world coordinates to chunk coordinates. Using `(x as f32 / CHUNK_SIZE as f32).floor() as i32` inside tight `while` loops forces repeated int-to-float-to-int conversions, which is an unnecessary performance penalty when dealing entirely with integers.
**Action:** Use integer Euclidean division and remainder (`x.div_euclid(chunk_size)` and `x.rem_euclid(chunk_size)`) to compute chunk grid coordinates and local offsets mathematically correctly without floating-point overhead.
## 2024-05-26 - Hoisting Index Math and Bounds Checks in 3D Grid Operations
**Action:** Validate outer chunk boundaries first, and precalculate the maximum iteration limits (`max_x = min(32, 128 - base_x)`). Then, precalculate the 1D base index and hoist `Z` and `Y` stride calculations into their respective loop scopes (`z_idx = base + z * Z_STRIDE`; `y_idx = z_idx + y * Y_STRIDE`). This reduces the inner loop to a single `y_idx + x` addition and replaces expensive full coordinate recalculation and 6-bound checks with a simple `x < max_x && y < max_y && z < max_z` bounding limit.

2024-05-23: Implemented an MCP (Model Context Protocol) Hook for LLM NPC control.
- Added `NpcStateView` and `NpcAction` structs to `genie_bridge.rs` for JSON serialization.
- Exposed WASM endpoints (`get_all_npcs_json`, `get_npc_state_json`, `execute_npc_action_json`) on `GameClient`.
- Created a global `window.GenieAI` JS object in `index.html` to easily puppet NPCs via headless browsers or external scripts.
