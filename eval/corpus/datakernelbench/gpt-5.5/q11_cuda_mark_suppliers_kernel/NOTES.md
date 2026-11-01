# q11_cuda_mark_suppliers_kernel

Source: `azure__gpt-5.5/cuda/full/queries/q11/round_01.py` (`output_round` 1),
copied as `plan.py`. The kernel is `mark_suppliers_kernel` of `_CUDA_SOURCE`.

## Extraction edits

- Renamed `mark_suppliers_kernel` to `kernel` and declared it `extern "C"`.
- Kept `#include <cstdint>`; left out the Torch, ATen, c10, and `<vector>`
  includes, the other kernels, and the host code, which the kernel does not use.

## Launch

`q11_launcher` launches the four kernels in order on one stream, each with
block (1024, 1, 1) (`BLOCK_SIZE`) and no dynamic shared memory. It computes
`max_partkey = max(ps_partkey)` and `max_suppkey = max(max(s_suppkey), max(ps_suppkey))`
in `run_query`, checks both are non-negative, and allocates
`supplier_keep = zeros(max_suppkey + 1, uint8)`, `part_sums = zeros(max_partkey + 1)`,
`total_sum = zeros(1)`, and `count = zeros(1, int64)`.

This kernel runs first, when `n_suppliers > 0 && n_nation_keys > 0`, with
grid `(n_suppliers + BLOCK_SIZE - 1) / BLOCK_SIZE`.

## Lengths and ranges

- `s_suppkey`, `s_nationkey`: `n_suppliers = s_suppkey.numel()` rows each (the
  launcher checks `s_nationkey.numel()`); least value 1 at this launch.
- `nation_keys`: `n_nation_keys = nation_keys.numel()` keys of the nations named
  `nation_name`; least value 1 at this launch.
- `supplier_keep`: `supplier_keep_len = max_suppkey + 1` bytes, zeroed by the
  plan; the kernel only stores 1 into it (`write`).
- `s_suppkey`: `index-in` into `supplier_keep`. The plan sizes `supplier_keep`
  from `max_suppkey`, which is at least the maximum of `s_suppkey` itself, and
  TPC-H keys are positive, so every value lies in `[0, supplier_keep_len - 1]`.
