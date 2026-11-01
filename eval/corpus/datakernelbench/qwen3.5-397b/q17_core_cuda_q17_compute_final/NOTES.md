# q17_core_cuda_q17_compute_final

The kernel `q17_compute_final` of the CUDA plan for q17 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/core/queries/q17/round_02.py`,
copied as `plan.py`).

## Extraction edits

- `q17_compute_final` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` and `#include <cstdio>` (host binding code and headers the kernel does not use), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- The device code that the kernel uses is copied unchanged: `find_part_index`.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 164: `q17_compute_final<<<blocks, BLOCK_SIZE>>>` in `q17_launcher_final`, after `q17_compute_aggs`, with `blocks = (n_rows + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory (`s_block_sum` is static); the kernel walks the rows in a grid-stride loop.

## Lengths and ranges

- The three lineitem columns have `n_rows = l_partkey.numel()` rows; `sorted_part_keys` and `part_avgs` (the per-part averages computed from the first kernel's sums) have `n_parts` elements, at least 1 at the launch.
- `out` is `torch::zeros({1})`, an `atomicAdd` accumulator: `readwrite`.
