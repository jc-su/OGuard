# Notes

## Grid claims

Recorded 2026-09-29: a blind review under the grid claim of design decision 33
(`"grid": [x, y, z]` in `triton.json` or `launch.json`), made before the gate
decided the kernels. Opus computes every program and block offset in 64 bits;
three CUDA C++ kernels and one Triton kernel truncate a tile length to 32 bits.

- Kernels given a claim: 8. CUDA C++, in `launch.json`: `tile_scan` and
  `block_sort` with [2097153, 1, 1], and `radix_scatter` with [1048577, 1, 1].
  Each computes `count = static_cast<int>(min(n - base, TILE))` with `base =
  blockIdx.x * TILE` (PTX `min.s64`, then `cvt.u32.u64`); with base <= 2^31,
  n - base >= -2^31 fits int, so the truncation is exact and a negative count
  skips the block, and the claim is 2^31 / TILE + 1 blocks (TILE 1024 or
  2048). One block further and n = 0 truncates to a positive count, and the
  block reads past `values`, `keys`, or the tile. Triton, in `triton.json`:
  `valid_rows` with [524289, 1, 1] for the same truncation of `tile_rows =
  min(n - row0, 4096).to(tl.int32)` (`classification.md` records only the QS
  limit for this load; the truncation is a C defect at 2^20 programs), and
  `radix_scatter` with [2147483647, 1, 1]. Numba, in `launch.json`:
  `radix_scatter` and `valid_rows` with [2147483647, 1, 1]; `valid_rows` CUDA
  C++ has the same claim. The C witness of the six `radix_scatter` and
  `valid_rows` kernels was a grid with a second block or program in y.
- Verdicts changed from C to safe: 1, `block_sort` CUDA C++ (`verdicts.tsv`).
- Verdicts changed from C to E: 6, `radix_scatter` and `valid_rows` in Numba,
  CUDA C++, and Triton, appended to `verdicts.tsv` from `decisions.tsv`. The
  remaining precondition is the contents of `cursors` or `count` on entry; the
  `valid_rows` kernels also read the bitmap under i < n, and its length is a
  size symbol of its own.
- Left C with a grid claim pending a symbolic bound: `tile_scan` (Numba, CUDA
  C++, Triton): thread 0 of every block, or every program, writes
  `totals[blockIdx.x]` with no bound, and the largest correct grid is
  (n + 1023) / 1024, which depends on n. The CUDA C++ claim removes only the
  truncation; the Numba and Triton kernels have no constant-bounded defect and
  got no claim. From `decisions.tsv`: `merge_sorted` (Numba, CUDA C++), where
  a block past the (m + p + 1791) / 1792 tiles reads below its shared buffer;
  the bound depends on m and p.
- Stay C without a claim: `string_equal` Triton (an int32 loop induction
  variable inside one program, which no grid bounds); `segmented_sum` CUDA C++
  (an int32 sum `offsets[j] + lane`).
