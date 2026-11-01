# Notes

All 12 tasks are complete. Every Numba kernel compiles to `numba.ptx` with
`cuda.compile_ptx(kernel, TYPES, cc=(8, 6))`, every CUDA kernel compiles to
`kernel.ptx` with `nvcc -ptx -arch=sm_86 -O3`, and both variants pass the GPU
tests for n in {0, 1, 1000, 1000003} on the RTX A6000. `./run_tests.sh [task ...]`
runs the Numba test (`test.py`, which also writes `numba.ptx`) and the CUDA
test (`test.cu`) for each task directory.

## Conventions

- Argument order follows the task text. Where a task does not list the row
  count as a scalar (filter_select, projection, q6_sum, hash_build,
  groupby_sum, string_length, histogram) or the pattern length (starts_with),
  the Numba and CUDA kernels take it as a trailing `int64` scalar, so raw
  pointers in CUDA have the same information as Numba arrays.
- Accumulators that the task requires to be zero or -1 on entry (`count`,
  `result`, `out_count`, `counts`, `table_keys`, `table_sums`) are marked
  `inout` in `signature.json`: the kernel reads and updates them with atomics.
- `indexes_into` names an argument of the same kernel. `table_rows` in
  hash_probe and `out_build` denote rows of the build-side key column, which is
  not an argument of that kernel, so they carry `null`.
- Launch shape: 256 threads per block and a grid-stride loop; the grid is
  capped (2048 blocks, or 256 for the kernels that flush block-private
  shared-memory tables, so each block amortizes its flush over more rows).
  Every kernel handles n = 0 and any n that is not a multiple of the block
  size; `run(...)` launches at least one block.

## Test data

- `filter_select`: uniform quantity in [0, 100), threshold 50.
- `hash_build`: keys are a random permutation of 0..n-1 times 1000003 plus 17;
  capacity is the smallest power of two at least 2n. Because the placement of
  colliding keys depends on scheduling, the check validates table properties:
  every row once, slot consistency, and that a linear probe from each key's
  home slot reaches it without crossing an empty slot.
- `hash_probe`: half the probe keys hit; the expected output is the sorted set
  of pairs; a second run with `out_cap = matches / 2` checks that `out_count`
  still reports every match and only valid pairs are written.
- `groupby_sum`: 1000 groups (capacity 2048, block-private path) and 100000
  groups (capacity 262144, global path).
- `string_length`: random 1 to 4 byte code points; averages of 12 and 96 code
  points per string exercise the thread-per-string and warp-per-string paths.
- `starts_with`: k = 3 ("abc", 30% of strings forced to match), k = 0, k = 25.
- `histogram`: shift in {0, 8, 16, 24}.
- `segmented_sum`: random cut points, averages of 8 and 128 values per segment
  (both paths), plus g = 3 empty segments at n = 0.

## Measurements (n = 1000003, CUDA kernel time from cudaEvent, A6000)

| task | CUDA | note |
|---|---|---|
| filter_select | 29 us | 50% selectivity |
| gather | 59 us | random 8-byte gather |
| projection | 41 us | about 585 GB/s |
| q6_sum | 36 us | about 740 GB/s |
| hash_build | 459 us | 16 MiB table, atomicCAS |
| hash_probe | 282 us | 50% hit rate |
| groupby_sum | 61 us / 51 us | 1000 / 100000 groups |
| scatter_by_offsets | 15 us | |
| string_length | 44 us / 594 us | 18 MB / 144 MB of chars |
| starts_with | 27 us | |
| histogram | 10.5 us | |
| segmented_sum | 33 us / 52 us | 125000 / 7812 segments |

The block-private table in groupby_sum was measured against a global-only
variant: 10 groups 162 us vs 771 us, 100 groups 53 us vs 253 us, 1000 groups
about equal. It stays because low-cardinality group-bys are common.

## Numba versus CUDA

The Numba timings in `test.py` include the dispatcher's host-side launch
overhead (about 70 us), because the start event is recorded before the Python
launch. After that offset, most Numba kernels are within a few percent of the
CUDA kernels. Two differences remain in Numba's code generation:

- Every load through a signed index carries a negative-index wraparound check.
  The byte and value loops in string_length, starts_with and segmented_sum use
  `uint32` positions, which removes the check and halved the thread-per-string
  time of string_length.
- Numba emits generic `ld.u8` loads where nvcc emits `ld.global.nc.u8`, and it
  keeps loop counters in 64-bit registers. The warp-per-string path of
  string_length is therefore about 2x slower in Numba (1170 us vs 594 us on
  144 MB). Both variants use the same algorithm.

## Grid claims

Recorded 2026-09-29: a blind review under the grid claim of design decision 33
(`"grid": [x, y, z]` in `triton.json` or `launch.json`), made before the gate
decided the kernels. Fable computes every block and program offset in 64 bits.

- Kernels given a claim: 6, `radix_scatter` and `valid_rows` in Numba, CUDA
  C++, and Triton, with [2147483647, 1, 1] (`launch.json` for Numba and CUDA
  C++, `triton.json` for Triton). Their C witness in `decisions.tsv` and
  `classification.md` was a grid with a second block or program in y, which
  repeats the tile and overruns `out` or `rows`; y = z = 1 removes it.
- Verdicts changed from C to safe: 0.
- Verdicts changed from C to E: 6, the six claimed kernels, appended to
  `verdicts.tsv` from `decisions.tsv`. The remaining precondition is the
  contents of `cursors` or `count` on entry (n = 1, `cursors[0] = 1` stores
  `out[1]`; n = 1, `valid = [1]`, `count[0] = 1` stores `rows[1]`); the
  `valid_rows` kernels also read `valid[i >> 3]` for i < n, and the bitmap's
  length is a size symbol of its own.
- Left C with a grid claim pending a symbolic bound: `tile_scan` Triton.
  `tl.store(totals + pid, tl.sum(v))` has no mask, and the largest correct grid
  is (n + 1023) / 1024 programs, which depends on n; the claim language takes
  constants only. Its offsets are 64-bit, so no constant claim removes another
  defect.
- Stay C without a claim: `segmented_sum` and `string_length` (CUDA C++ and
  Triton) wrap a 32-bit loop counter inside one warp or program, which no grid
  bounds.
