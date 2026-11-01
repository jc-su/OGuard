# q15_cuda_q15_kernel

The kernel `q15_kernel` of the CUDA plan for q15 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/full/queries/q15/round_04.py`,
copied as `plan.py`).

## Extraction edits

- `q15_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- The device code that the kernel uses is copied unchanged: `__device__ int binary_search(`.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 80: `q15_kernel<<<blocks, BLOCK_SIZE>>>` in `q15_launcher`, with `blocks = (n + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory.

## Lengths and ranges

- The four lineitem columns have `n = l_suppkey.numel()` rows, passed as `n_lineitems`.
- `s_suppkey_sorted` is the sorted supplier keys, `n_sup = sorted_s.numel()` elements, passed as `n_suppliers`; `revenue_out` is `torch::zeros({n_suppliers})`, an `atomicAdd` accumulator: `readwrite`.
