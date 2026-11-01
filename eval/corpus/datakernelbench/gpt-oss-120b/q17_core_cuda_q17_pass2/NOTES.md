# q17_core_cuda_q17_pass2

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/core/queries/q17/round_01.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `q17_pass2` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `q17_pass2` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: `#include <cuda.h>`, `#include <cuda_runtime.h>`, `#define BLOCK_SIZE 1024` (with its `#ifndef` guard), the device function `binary_search`, and the kernel's body, with the comment of its parameter list.
- Left out as host code: `#include <torch/extension.h>`, the launcher, the other kernel, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`q17_launcher` launches `q17_pass2<<<blocks2, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (n_rows + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `l_partkey`, `l_quantity`, `l_extendedprice`: columns of `lineitem`, `n_rows` rows.
- `sorted_part_keys`, `thresholds`: the sorted filtered part keys and `0.2 * avg_qty` per part, `n_parts` entries each.
- `out`: `torch::zeros({1})`, which each block adds into atomically. The static shared array `block_sum` has `BLOCK_SIZE = 1024` entries.
