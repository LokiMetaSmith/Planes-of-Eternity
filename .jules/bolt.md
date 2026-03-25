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

## 2025-03-18 - Avoid Sqrt in Distance Threshold Checks
**Learning:** In the `update_lods` logic, computing the exact distance between the camera and a chunk requires a computationally expensive `sqrt()` call. Because this occurs inside a hot loop (iterating over every active chunk every time LODs are updated), it becomes a performance bottleneck.
**Action:** When comparing distances against constant thresholds (like 64.0 or 128.0), compare the squared distance against the squared thresholds instead. This maintains mathematical correctness while entirely eliminating the `sqrt()` operation.

## 2025-03-19 - Fast Bounds Checking in 3D Grids
**Learning:** Checking bounds on an XYZ grid conventionally involves `x >= size || y >= size || z >= size`. This creates multiple branches, confusing CPU branch predictors in tight loops like raycasting or chunk meshing where bounding checks are continuously hit.
**Action:** Use bitwise OR `(x | y | z) >= size` for a single branch. For signed integers, cast to an unsigned integer `(x as u32) >= size` so that negative values wrap around to very large positive numbers, naturally failing the single bounds check and combining both lower `< 0` and upper `>= size` bounds into one operation. And once bounds are safely verified in accessors like `get()` and `set()`, use `get_unchecked` and `get_unchecked_mut` to avoid redundant panic bounds checks by the Rust compiler.

## 2026-03-20 - [Removed expensive float operations in tight loops]
**Learning:** Found an `f32.sqrt()` and float conversions inside `Chunk::generate`'s innermost tight loop (iterating `CHUNK_SIZE^3` times). Float conversions and `sqrt` are notably expensive on CPUs/WASM. Replacing this with purely integer arithmetic (`dist_sq <= max_dist * max_dist`) completely removed these floating point operations, dramatically increasing speed with no loss in accuracy for this specific logic. Temporary benchmark files generated during validation shouldn't be included in patches.
**Action:** When calculating distances in tight loops (especially `Chunk` generation, `update_lods` and raycasting), use squared integer distances (`dist_sq`) to compare against squared thresholds rather than casting to floats and using `sqrt()`. Always explicitly clean up temporary benchmark scripts and binaries before submitting code reviews.

## 2024-11-20 - [Avoid redundant normalization in physics loops]
**Learning:** In hot O(N^2) physics loops, calling `.normalize()` on a vector implicitly calculates its magnitude (`sqrt(x^2 + y^2 + z^2)`). If the squared distance (`magnitude2()`) is already known for collision checks, calling `.normalize()` is a waste of CPU cycles. The square root can be done once, and we can multiply the un-normalized vector by a precomputed scalar `(force / distance)` rather than normalizing it and then scaling it.
**Action:** When working in tight simulation loops, always check if `.normalize()` is hiding an expensive calculation that could be optimized out by re-using `dist_sq.sqrt()`.

## 2025-03-22 - Avoid Redundant Vector Normalizations in Physics and Camera Calculations
**Learning:** Calling `.normalize()` on a vector explicitly recalculates its magnitude (via expensive square root functions). However, in many domains, vectors are either already mathematically guaranteed to be unit length (like spherical coordinates or cross products of orthogonal unit vectors), or the magnitude has already been calculated earlier in the loop block. Normalizing mathematically guaranteed vectors, or recalculating magnitudes that are already in scope, incurs redundant CPU cycles with zero benefit.
**Action:** Before calling `.normalize()`, verify if the vector's length is mathematically constrained to 1, and remove the call. Alternatively, if the squared magnitude is already verified within scope (like collision bounds or distance thresholds), manually extract the root and apply it as a scalar factor `vector * (1.0 / dist_sq.sqrt())` instead of running `vector.normalize()`.

## 2025-03-24 - Beware the Bitwise OR Bounds Check Trap
**Learning:** While using `(x | y | z) >= size` can combine multiple branch checks into a single check, it is fundamentally mathematically flawed for arbitrary dynamic grid bounds. If `size` is not a strictly guaranteed power of two, bitwise overlapping non-power-of-2 numbers can spuriously produce values that falsely trip the `>= size` condition, even when `x, y, z` are all individually within valid boundaries.
**Action:** Do not use `(x | y | z) >= size` or `(x as u32 | y as u32 | z as u32) >= size` to optimize bounds checking unless `size` is mathematically constrained to only ever be a strict power of two in all use cases. Fall back to `(x as u32) >= size || (y as u32) >= size || ...` to keep branches clean while still avoiding signed-negative underflow checks.

## 2024-03-24 - Halving O(N^2) Physics with `split_at_mut`
**Learning:** When calculating symmetric pairwise interactions (like Newton's Third Law forces) across a single array in Rust, a naive nested `for j in 0..N` loop duplicates work and is hard to optimize because borrowing two elements from a mutable slice fails the borrow checker.
**Action:** By splitting the array at the current index `let (left, right) = slice.split_at_mut(i + 1)` and iterating the inner loop over `right.iter_mut()`, we safely borrow `slice[i]` and `slice[j]` simultaneously without allocations, perfectly halving the number of `magnitude2()`/distance calculations in O(N^2) loops.

## 2025-03-25 - Eliminating float function overhead in hot loop distances
**Learning:** In hot loops like physics iterations (`visual_lambda.rs`) or LOD updates over many chunks (`lib.rs`), float functions like `.powi(2)` or `.sqrt()` incur measurable overhead. A direct call to `vector.magnitude()` or `(ndc_x - sx).powi(2)` might be mathematically sound, but is unnecessary if we only need to verify if the distance exceeds a certain threshold.
**Action:** When calculating squared distance, prefer `dx * dx + dy * dy + dz * dz` over `.powi(2)`. Additionally, calculate `.magnitude2()` instead of `.magnitude()` and compare it against the *squared* threshold (e.g., `dist_sq > 0.00000001` instead of `dist > 0.0001`). Only extract the `sqrt()` if the threshold is passed and the scalar distance is explicitly needed for further math.
