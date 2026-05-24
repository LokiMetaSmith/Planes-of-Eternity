1. **Update the UI (`reality-engine/index.html`)**
   - Add a `<div id="minimap-container">` to display the mini map.
   - Add a `<canvas id="minimap-canvas" width="128" height="128">` inside it.
   - Add CSS to position the mini map in the top-right corner.

2. **Expose Map Data from WASM (`reality-engine/src/lib.rs`)**
   - Add a method to `GameClient` called `pub fn get_minimap_data_json(&self) -> String`.
   - This method will access the chunks from `self.state.borrow().engine.world_state.chunks`, retrieving their `ChunkId` keys (`x`, `z`) or `ChunkKey` keys (`x`, `y`, `z`) where `y=0`. Since `WorldState` uses `ChunkId` for the top-level mapping but maybe `ChunkManager` uses `ChunkKey`. I need to verify what `world_state.chunks` contains. Let me check `world.rs`. Wait, `world.rs` uses `ChunkId {x, z}` mapped to `Chunk { key: ChunkKey {x, y, z}, .. }` ?
   - The method will also retrieve the player position from `self.state.borrow().engine.player_projector.location` which is a `Point3<f32>` containing `x, y, z`.
   - It will serialize these keys and the player's `(x, z)` coordinates into JSON.

3. **Refine the JS Implementation (`reality-engine/index.html`)**
   - Implement `updateMinimap()` function in JS called using `setInterval` or in the render loop.
   - Call `gameClient.get_minimap_data_json()`, parse the JSON, and draw rectangles for chunks and a dot for the player on the `<canvas>`.

4. **Run Compilation & Tests**
   - Run `cargo check --target=wasm32-unknown-unknown` and `cargo test --test host_test`.

5. **Complete Pre-Commit Steps**
   - Run required checks before finalizing.
