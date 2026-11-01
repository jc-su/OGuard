# Notes on the fifth held-out set

All 20 tasks are written in all three versions (Numba, CUDA C++, Triton),
each with its test and a `signature.json`. No task was left out.

## Design decisions worth knowing

- One kernel, no scratch memory. Every task is a single launch with the
  listed arguments only, so nothing can be staged in global scratch and no
  flag can be initialized before the launch. The tasks that need a prefix
  across the whole column handle it as follows.
  - `segmented_cummax`, `fill_forward`, `range_window_sum`: each block owns a
    tile of rows and computes its carry by rereading the input before the
    tile (the partition prefix, the bitmap bytes of the null run, or the
    frame rows), coalesced and reduced in shared memory (Numba, CUDA) or in
    registers (Triton). The cost is one pass over that prefix per block, so
    a partition, null run, or frame that spans the column costs
    O(n^2 / TILE) reads in total. For range_window_sum the reads are of the
    frame union of the tile only.
  - `run_end_encode`: `count[0]` (zero on entry) is the only initialized
    word, and the output tail must stay untouched, so it is a chained scan
    ordered through `count[0]`: a tile with run ends finds the last run end
    `e` before it, waits until `run_ends[count[0] - 1] == e + 1`, writes its
    runs, and adds its run count. Tiles without run ends neither wait nor
    write. The wait is on an earlier block, which assumes blocks are
    dispatched in index order (the usual single-pass scan assumption; there
    is no second word for a dynamic tile ticket). Each tile holds 4096 rows,
    so the chain has n / 4096 links.
- Atomic reductions. `argmin` and `group_argmax` keep a row index under a
  compare-and-swap loop that compares the values of the published row and the
  candidate; `hash_distinct` claims slots with compare-and-swap and lowers the
  row with atomic min; `group_sum_valid` privatizes the 1024 accumulators in
  shared memory per block (Numba, CUDA) and uses global atomics in Triton,
  where shared memory is not addressable.
- String and join output is byte-parallel or pair-parallel: one thread per
  output byte or output pair with a binary search over the offsets, so skewed
  string lengths or probe rows with many matches stay balanced
  (`gather_strings`, `string_concat`, `join_materialize`).
- Triton signed integer `//` and `%` truncate toward zero (C semantics), so
  every division in the Triton kernels, and for uniformity in the other two
  versions, runs on nonnegative or unsigned operands (`decimal_rescale`,
  `timestamp_floor`, `add_months`, `string_to_int`).
- `string_to_int` writes the validity bitmap one byte per thread (eight
  strings per thread) because there is no 8-bit atomic OR.
- The kernel argument named `range` (task 15) shadows Python's `range`; the
  Numba kernel uses a module alias `_range` and the Triton kernel uses
  `while` loops.

## Test notes

- `numba.py` and `triton.py` shadow the `numba` and `triton` packages when
  the task directory is first on `sys.path`, so the tests remove that entry
  and load the modules by file path.
- The Triton tests pass uint32 columns as int32 tensors (same bytes) because
  PyTorch has no general uint32 support; the kernels bitcast on load and the
  tests compare as uint32.
- Each test also compiles the kernel the way a harness would: Numba with
  `cuda.compile_ptx(kernel, TYPES, cc=(8, 6))`, Triton with `ASTSource` from
  `TYPES` and `CONSTANTS`. `cuda.compile_ptx` works without a GPU only when
  the kernel calls no `@cuda.jit(device=True)` helper, so the Numba kernels
  are self-contained.
- `test.cu` includes `kernel.cu` and is built with
  `nvcc -arch=sm_86 -O2 test.cu -o test`.
