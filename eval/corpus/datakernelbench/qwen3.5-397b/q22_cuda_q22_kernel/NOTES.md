# q22_cuda_q22_kernel

The kernel `q22_kernel` of the CUDA plan for q22 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/full/queries/q22/round_03.py`,
copied as `plan.py`).

## Extraction edits

- `q22_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>`, `#include <algorithm>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- The device code that the kernel uses is copied unchanged: `__device__ bool binary_search_exists(`.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 148: `q22_kernel<<<blocks, BLOCK_SIZE>>>` in `q22_launcher`, with `blocks = (n + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory.

## Lengths and ranges

- The four customer columns have `n = c_custkey.numel()` rows; `orders_sorted` is the sorted distinct `o_custkey`, `n_orders = orders_sorted.numel()`.
- `count_out` and `bal_out` are `torch::zeros({NUM_COUNTRY_CODES})`, 7 elements each, `atomicAdd` accumulators: `readwrite`. The plan allocates `count_out` with `torch::zeros({7}, torch::kInt64)`, a host tensor, and passes its pointer to the kernel; its length is 7 all the same.
