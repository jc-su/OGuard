# q17_cuda_q17_rev_kernel

From `logs/kernel_plans/tpch/sf10/h100/openai__gpt-oss-120b/cuda/full/queries/q17/round_01.py` of DataKernelBench at commit `43c361d` (copied as `plan.py`): the CUDA kernel `q17_rev_kernel` of the plan's `_CUDA_SOURCE`.

## Extraction edits

- The entry kernel `q17_rev_kernel` is renamed `kernel` and declared `extern "C" __global__ void kernel(...)`.
- Copied unchanged: `#include <cstdint>`, `#define BLOCK_SIZE 1024`, the device function `binary_search`, and the kernel's body.
- Left out as host code: `#include <torch/extension.h>`, the launchers, and the `PYBIND11_MODULE` block; the macros and functions that the kernel does not use are left out too.

## Launch

`q17_rev_launcher` launches `q17_rev_kernel<<<blocks, threads>>>` with `threads = BLOCK_SIZE = 1024`, `blocks = (n_rows + 1023) / 1024`, and no dynamic shared memory. `launch.json`: block `[1024, 1, 1]`, dynamic shared memory 0 bytes.

## Lengths and ranges

- `line_partkey`, `line_quantity`, `line_price`: columns of `lineitem`, `n_rows = line_partkey.numel()` rows.
- `sorted_keys`: the sorted keys of the filtered parts, `n_parts = sorted_keys.numel()`; `thresholds`: `0.2 * avg_qty`, computed from the two `n_parts`-entry sums.
- `out_sum`: `torch::zeros({1})`, added into once per block.
