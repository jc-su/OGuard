# Notes

All 20 tasks are complete in all three versions (Numba, CUDA C++, Triton). Every
version passed its test on GPU 1 for n in {0, 1, 1000, 1000003}.

## Design points

- Scans across the whole column (`run_end_encode`, `segmented_cummax`,
  `fill_forward`, `range_window_sum`): the interface allows one kernel and no
  scratch memory, and a partition, a null run, or a window can span the column.
  - CUDA and Numba: these kernels are cooperative launches. All blocks are
    resident and `grid.sync()` separates the phases. Each block owns a contiguous
    chunk of rows, and between phases a block keeps its per-chunk summary in its
    own part of an output array, which the last phase overwrites.
    `run_end_encode` keeps its per-block run counts in
    `run_ends`/`run_values` at positions below r, using `count` as the slot
    counter. The CUDA `test.cu` launches them with `cudaLaunchCooperativeKernel`,
    using a grid sized by `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.
    Numba's `run()` sizes the grid with `max_cooperative_grid_blocks`.
  - Triton has no grid-wide barrier, so these four Triton kernels run as one
    program that scans the column tile by tile with a carry. They are correct for
    every n, but their bandwidth is that of one SM.
- `range_window_sum` (CUDA and Numba) keeps the prefix sums of all blocks in
  shared memory, which limits the grid to 1024 blocks. The CUDA kernel traps
  above that, and the Numba `run()` caps the grid.
- `argmin` and `group_argmax` publish rows with compare-and-swap loops that
  compare against `values[current row]`. `hash_distinct` inserts by CAS on
  `table_keys` and lowers `table_rows` with an atomic min. Its slot placement
  depends on thread timing, so the tests check the required properties
  instead of fixed slots.
- `signature.json`: arrays that the kernel updates from a given starting value
  ("zero on entry", "-1 on entry", the hash table, `run_end_encode.count`) have
  `kind` `"inout"`.

## Bugs found by the tests and fixed

- `group_sum_valid` CUDA: a `__shfl_sync` with the full mask inside a
  conditional expression deadlocked when n was not a multiple of 32. The
  shuffle now runs in every lane.
- `argmin` Triton: each thread read `row[0]` with its own plain load while other
  programs were updating it, so threads of one program saw different values,
  and the loop could index out of bounds. The loop now takes `row[0]` from the
  CAS result, which Triton gives to every thread of the program.
- Numba cooperative kernels: `cuda.shared.array` needs a constant shape; the
  `MAX_WARPS + 1` expression was replaced with a module constant.
