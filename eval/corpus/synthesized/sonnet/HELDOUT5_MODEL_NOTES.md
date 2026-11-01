# Notes

All 20 tasks are complete in all three versions (Numba, CUDA C++, Triton). Every test passed on GPU 1 for n in {0, 1, 1000, 1000003} plus task-specific and adversarial cases.

- Single block: run_end_encode, segmented_cummax, fill_forward and range_window_sum have no scratch argument, so each runs as one block (grid = 1) that scans the column tile by tile. Work is linear for any run, partition or null-run length, but only one SM is used. A multi-block scan with decoupled look-back would need scratch memory.
- range_window_sum makes three in-place passes over `out`: ascending prefix sums, descending subtraction of the prefix at lo(i) - 1, ascending copy from the last row of the same timestamp group. Each pass reads a whole tile before writing it.
- argmin and group_argmax keep the champion row in the output cell and update it with compare-and-swap, comparing against the input value of the current champion. No scratch is needed.
- group_sum_valid uses global atomics in Triton (no shared-memory atomics there); CUDA and Numba use per-block shared tables.
- join_materialize in Triton is element-parallel and its run() reads offsets[p] from the device once to size the grid. Numba and CUDA are thread per probe row with warp-cooperative handling of rows with more than 32 matches.
- Triton passes Python int scalars that fit in 32 bits as i32 and a scalar equal to 1 as a constexpr, regardless of the declared "i64" in TYPES. Several Triton kernels use do_not_specialize on their scalars and cast them to int64.
- numba.py and triton.py in a task directory shadow the real packages, so the Python tests remove their directory from sys.path and load the kernel file by path.
- Numba range_window_sum inlines its searches because a cuda.jit(device=True) helper makes compile_ptx require a GPU.
