1. **Setup tokenizers dependency**
   - We have already updated `reality-genie/Cargo.toml` to add `tokenizers` with the `unstable_wasm` feature.
   - We have correctly set up `getrandom@0.2.17` with `js` feature to allow compilation on `wasm32-unknown-unknown` alongside `tokenizers`'s transitive dependencies on `rand` and `getrandom`.

2. **Implement Text Encoder in `reality-genie`**
   - Create a simple abstraction in `reality-genie/src/diffusion.rs` or a new module `reality-genie/src/text_encoder.rs` for encoding text to embeddings/tokens.
   - We can add a function to `GenieBridge` or a new `TextEncoder` struct that initializes a tokenizer using a minimal configuration (or just tokenizing logic, e.g. BPE or WordPiece format).
   - *Since this is "integrate the tokenizers crate and implement a lightweight Text Encoder" (Task 2.1), we should provide a structure `TextEncoder` that wraps `tokenizers::Tokenizer`.*

3. **Update TODO list**
   - Check off Task 2.1 in `DIFFUSION_PIPELINE_SPRINT.md`.

4. **Complete Pre Commit Steps**
   - Ensure proper testing, verification, review, and reflection are done by calling `pre_commit_instructions` and following its output.

5. **Submit the change**
   - Once everything compiles and passes, commit with branch `jules-diffusion-text-encoder`.
