1. [x] **Optimize `get_node_labels` boundary tax**: Modify `engine.rs` to compute `LabelInfo` without JSON serialization.
2. [x] **Implement Flat Buffer approach**: Instead of generating a `Vec<LabelInfo>` and returning JSON, implement a method that returns a custom `Float32Array` or `Uint8Array` directly encoding `[x, y]` and `color`/`text` efficiently, or write to a shared memory buffer. Alternatively, split it into two calls: one that gets a `Float32Array` of `[x, y]` for all labels, and another that only gets text/colors if the node count or content changes.
3. **Pre-commit checks**: Run `cargo check`, `cargo clippy`, and `cargo test` to ensure changes don't break the build.
4. **Submit changes**.
