# q11_cuda_aggregate_partsupp_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q11/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `aggregate_partsupp_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `aggregate_partsupp_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out the Torch, ATen, c10, and `<vector>`
  includes, the other kernels, and the host code, which the kernel does not use.

## Launch

`q11_launcher` launches the four kernels in order on one stream, each with
block (1024, 1, 1) (`BLOCK_SIZE`) and no dynamic shared memory. It computes
`max_partkey = max(ps_partkey)` and `max_suppkey = max(max(s_suppkey), max(ps_suppkey))`
in `run_query`, checks both are non-negative, and allocates
`supplier_keep = zeros(max_suppkey + 1, uint8)`, `part_sums = zeros(max_partkey + 1)`,
`total_sum = zeros(1)`, and `count = zeros(1, int64)`.

This kernel runs second, when `n_partsupp > 0 && n_nation_keys > 0`, with grid
`(n_partsupp + BLOCK_SIZE - 1) / BLOCK_SIZE`.

## Lengths and ranges

- `ps_partkey`, `ps_suppkey`, `ps_supplycost`, `ps_availqty`:
  `n_partsupp = ps_partkey.numel()` rows each (the launcher checks the other
  three); least value 1 at this launch.
- `supplier_keep`: `supplier_keep_len = max_suppkey + 1` bytes that the first
  kernel marked (`read`).
- `part_sums`: `part_sums_len = max_partkey + 1` zeroed doubles, which the
  kernel adds to with `atomicAdd` (`readwrite`).
- `total_sum`: one zeroed double, added to with `atomicAdd` (`readwrite`).
- `ps_partkey`: `index-in` into `part_sums`; `max_partkey` is the maximum of
  this same column and TPC-H keys are positive.
- `ps_suppkey`: `index-in` into `supplier_keep`; `max_suppkey` is at least the
  maximum of this same column.
