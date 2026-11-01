# q11_cuda_count_passing_parts_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q11/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `count_passing_parts_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `count_passing_parts_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>` and the macro `BLOCK_SIZE`; left out the Torch, ATen, c10, and `<vector>`
  includes, the other kernels, and the host code, which the kernel does not use.

## Launch

`q11_launcher` launches the four kernels in order on one stream, each with
block (1024, 1, 1) (`BLOCK_SIZE`) and no dynamic shared memory. It computes
`max_partkey = max(ps_partkey)` and `max_suppkey = max(max(s_suppkey), max(ps_suppkey))`
in `run_query`, checks both are non-negative, and allocates
`supplier_keep = zeros(max_suppkey + 1, uint8)`, `part_sums = zeros(max_partkey + 1)`,
`total_sum = zeros(1)`, and `count = zeros(1, int64)`.

This kernel runs third, when `part_sums_len > 0`, with grid
`(part_sums_len + BLOCK_SIZE - 1) / BLOCK_SIZE`.

## Lengths and ranges

- `part_sums`: `part_sums_len = max_partkey + 1` doubles (`read`); least value 1,
  since the launcher checks `max_partkey >= 0`.
- `total_sum`: one double (`read`).
- `out_count`: the one-element `count`, zeroed by the plan, added to with
  `atomicAdd` (`readwrite`).
- `threshold_fraction`: a Python float, 0.0001 at the default parameters.
