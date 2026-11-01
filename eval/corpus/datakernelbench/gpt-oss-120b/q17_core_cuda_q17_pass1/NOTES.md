# q17_core_cuda_q17_pass1

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/core/queries/q17/round_01.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `q17_pass1` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `q17_pass1` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: `#include <cuda.h>`, `#include <cuda_runtime.h>`, the device function `binary_search`, and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the launcher, the other kernel, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`q17_launcher` launches `q17_pass1<<<blocks1, BLOCK_SIZE>>>` with `BLOCK_SIZE = 1024` threads, `blocks = (n_rows + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `l_partkey`, `l_quantity`: columns of `lineitem`, `n_rows = l_partkey.size(0)` rows.
- `sorted_part_keys`: `torch.sort(part_keys)` of the filtered parts, `n_parts = sorted_part_keys.size(0)` entries; the kernel binary-searches it. `sum_qty`, `cnt_qty`: `torch::zeros({n_parts})`, which the kernel adds into atomically at the found index.
