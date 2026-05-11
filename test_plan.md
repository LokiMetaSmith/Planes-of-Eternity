1. **Optimize sorting in rendering pipeline**
   - We updated `reality-engine/src/lib.rs` to use `sort_unstable_by` instead of `sort_by` for rendering optimization.
   - Replaced multiple `.powi(2)` distance calculations with simple inline multiplications (`dx * dx`) which are generally faster.
2. **Add Todo Item**
   - Add the optimization task to `TODO.md` to reflect the work done.
3. **Complete pre commit steps**
   - Run necessary checks before finalizing.
4. **Submit**
