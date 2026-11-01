# q09_cuda_q9_aggregate_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q09/round_02.py` (`output_round` 2),
copied as `plan.py`. The kernel is `q9_aggregate_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `q9_aggregate_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `MAX_GROUPS`; left out the Torch,
  ATen, c10, `<algorithm>`, and `<vector>` includes, the macros `BLOCK_SIZE`
  and `MAX_GRID_BLOCKS`, and the host code, which the kernel does not use.

## Launch

`q9_aggregate_launcher` launches the kernel once, when `n > 0`, with
`<<<min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_GRID_BLOCKS), BLOCK_SIZE, 0, stream>>>`:
block (256, 1, 1), no dynamic shared memory.

## Lengths and ranges

- `extendedprice`, `discount`, `supplycost`, `quantity`, `nationkey`, `year`:
  `n_elements` rows each, the columns of the joined rows. The launcher sets
  `n = extendedprice.numel()` and checks that the other five have the same
  `numel()`.
- `n_elements`: size symbol with least value 1; the launch is inside `if (n > 0)`.
- `max_nation_key`: `int(df_nation["n_nationkey"].max())`. The TPC-H nation
  table has the 25 keys 0 to 24 at every scale factor, so the plan passes 24:
  `[24, 24]`.
- `n_years`: `year_max - year_min + 1` of the joined order years; the launcher
  checks `year_max >= year_min` (least value 1) and
  `0 < (max_nation_key + 1) * n_years <= MAX_GROUPS`, which with
  `max_nation_key = 24` is `n_years <= 81`, stated as `size_maxima`.
- `out_sums`, `out_counts`: `torch::zeros({group_count64})` with
  `group_count64 = (max_nation_key + 1) * n_years`, so `25 * n_years` elements
  each, which the kernel adds to with `atomicAdd` (`readwrite`). `out_counts`
  is an `int64` tensor passed as `unsigned long long*`, so `uint64`.
- `year_min`: the least order year; it only shifts the year.
- `nationkey` holds `s_nationkey` of the join; the kernel checks
  `0 <= nk <= max_nation_key` itself, so it is `read`.
