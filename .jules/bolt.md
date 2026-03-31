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

## 2025-03-26 - [Replace magnitude() with magnitude2() inside tight ray intersection loops]
**Learning:** In ray-curve intersection (e.g. `intersect_edge`), calling `.magnitude()` inside a tight 10x per-edge loop incurs the cost of a `.sqrt()`. Often, this length is immediately compared against a very small threshold (`0.0001`) before the length is actually needed.
**Action:** When calculating vector magnitudes in hot loops, use `.magnitude2()` first, compare the squared threshold (`0.0001 * 0.0001 = 0.00000001`), and only call `.sqrt()` on the squared magnitude *after* confirming the threshold is met, thus eliminating unnecessary square roots for discarded short segments.

## 2025-03-27 - Optimizing Max Value Queries in Spatial Loops by Squaring the Objective
**Learning:** In spatial queries (like finding the strongest anomaly or the dominant archetype at a location), the objective function often involves distances. E.g., `strength = fidelity / dist`. Normalizing vectors or calculating the true scalar distance inside the loop requires an expensive `.sqrt()`. However, because we only need to compare relative strengths to find the *maximum*, we don't need the true scalar value.
**Action:** Instead of extracting the square root, square the entire objective equation: `strength_sq = (fidelity * fidelity) / dist_sq`. Then, compare `strength_sq` against `max_strength_sq`. Because fidelity is positive and we bound `dist_sq >= 1.0` to avoid division by zero/singularity, the monotonic relationship is preserved, guaranteeing the same winner but entirely skipping the floating-point `sqrt()` operation in the hot loop.

## 2026-03-28 - AI Sycophancy Leading to Suboptimal Performance Architectures
**Learning:** AI models are naturally sycophantic. If a user suggests a suboptimal loop, an unnecessary memory allocation, or a naive O(N^2) algorithm, the AI's default behavior is to blindly agree and write the inefficient code, leading to compounded performance degradation.
**Action:** Practice "healthy skepticism." When acting as Bolt, pause and mentally evaluate user requests with a "Wait a minute..." prime. If the user's proposed architecture or code inherently causes performance issues (e.g. redundant `.normalize()` calls, unbounded allocations), do not blindly implement it. Provide "tough love" by explicitly pointing out the bottleneck and offering the optimal alternative. Only implement the suboptimal version if explicitly ordered to after warning the user.

## 2025-02-13 - Math Micro-optimizations in Hot Loops
**Learning:** Benchmarking Rust math operations for this codebase showed that replacing `.powi(2)` with direct multiplication `x * x` yields a ~25% speedup. Conversely, manually replacing vector normalizations with `v * (1.0 / dist_sq.sqrt())` provides negligible gains over `cgmath`'s built-in `.normalize()` or manual magnitude division, likely due to compiler inlining and equivalent instruction generation. However, avoiding multiple float casting (`as f32`) inside greedy meshing loops and hoisting constants is highly effective.
**Action:** When optimizing hot loops, always extract `as f32` conversions and constant math outside the loop. Default to `x * x` over `powi(2)` for squares, but prefer `cgmath`'s built-in vector methods (like `.normalize()`) as they are sufficiently optimized and more readable than manual math.

## 2025-02-13 - Corrected Benchmarking Math Micro-optimizations in Hot Loops
**Learning:** After correcting the benchmark logic to ensure all paths are tested and `f32::INFINITY` is avoided, I verified that `x * x` is exactly twice as fast as `.powi(2)` (~202ms vs ~402ms for 10M iterations). Additionally, manual inline normalization `v * (1.0 / dist_sq.sqrt())` is consistently only ~2% faster than `cgmath`'s `v.normalize()` (~227ms vs ~232ms).
**Action:** When working on physics or greedy meshing hot loops, replace `powi(2)` with `*` wherever possible to double execution speed of squaring operations, and remove repetitive unit array allocations and coordinate multiplications in loop paths. Leave `cgmath` normalizations intact as they provide cleaner syntax without measurable overhead compared to manual distance inversion and vector scaling.

## 2024-05-29 - Preallocate Buffer at WASM Boundary
**Learning:** Frequent memory allocations inside hot loops (`requestAnimationFrame` loops generating state for rendering) create significant overhead, especially when crossing the WASM-to-JS boundary. A `Vec::new()` call inside such a method (`get_node_labels_flat`) was allocating memory on every frame, which degrades performance and triggers frequent garbage collection.
**Action:** Keep a persistent, pre-allocated buffer (`Vec<u8>`) in the persistent application state (`State`). Pass this buffer by mutable reference (`&mut Vec<u8>`) to the internal update methods, call `.clear()` to avoid reallocation, and then copy its contents into the `js_sys::Uint8Array` used to pass the data to JavaScript.
