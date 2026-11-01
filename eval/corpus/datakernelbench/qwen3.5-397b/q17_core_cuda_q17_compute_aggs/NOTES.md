# q17_core_cuda_q17_compute_aggs

The kernel `q17_compute_aggs` of the CUDA plan for q17 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/core/queries/q17/round_02.py`,
copied as `plan.py`).

## Extraction edits

- `q17_compute_aggs` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` and `#include <cstdio>` (host binding code and headers the kernel does not use), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- The device code that the kernel uses is copied unchanged: `find_part_index`.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 126: `q17_compute_aggs<<<blocks, BLOCK_SIZE>>>` in `q17_launcher_aggs`, with `blocks = (n_rows + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory; the kernel walks the rows in a grid-stride loop.

## Lengths and ranges

- `l_partkey` and `l_quantity` are lineitem columns of `n_rows = l_partkey.numel()` rows; `sorted_part_keys` is the sorted keys of the filtered parts, `n_parts = part_keys.numel()`, passed as an `int`; `_query_core` returns before the launches when `n_parts == 0`, so it is at least 1.
- `part_sums` and `part_counts` are `torch::zeros({n_parts})`, `atomicAdd` accumulators: `readwrite`.
- The plan sorts `part_keys` for the binary search; the gate does not establish sortedness, and the search's reads stay inside `sorted_part_keys` whatever the order.
