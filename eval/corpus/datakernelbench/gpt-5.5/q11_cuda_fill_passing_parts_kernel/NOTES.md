# q11_cuda_fill_passing_parts_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q11/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `fill_passing_parts_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `fill_passing_parts_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out the Torch, ATen, c10, and `<vector>`
  includes, the other kernels, and the host code, which the kernel does not use.

## Launch

`q11_launcher` launches the four kernels in order on one stream, each with
block (1024, 1, 1) (`BLOCK_SIZE`) and no dynamic shared memory. It computes
`max_partkey = max(ps_partkey)` and `max_suppkey = max(max(s_suppkey), max(ps_suppkey))`
in `run_query`, checks both are non-negative, and allocates
`supplier_keep = zeros(max_suppkey + 1, uint8)`, `part_sums = zeros(max_partkey + 1)`,
`total_sum = zeros(1)`, and `count = zeros(1, int64)`.

This kernel runs last. The launcher copies `count` to the host as `h_count`,
allocates `out_keys = empty({h_count}, int64)` and
`out_values = empty({h_count}, double)`, and, when `h_count > 0`, zeroes
`count` with `cudaMemsetAsync` and launches the kernel with grid
`(part_sums_len + BLOCK_SIZE - 1) / BLOCK_SIZE`, passing `count` as `out_pos`.

## Lengths and ranges

- `part_sums`: `part_sums_len = max_partkey + 1` doubles (`read`); least value 1.
- `total_sum`: one double (`read`).
- `out_keys`, `out_values`: `h_count` elements each, the number of `part_sums`
  values above `total_sum[0] * threshold_fraction` that the third kernel
  counted; `h_count` is a size symbol that the kernel does not receive, least
  value 1 at this launch. The kernel only stores into them (`write`); the
  stored keys are part keys that the plan does not use as indices.
- `out_pos`: the one-element `count`, zeroed before the launch, used as an
  atomic cursor (`readwrite`).
- `threshold_fraction`: a Python float, 0.0001 at the default parameters.
