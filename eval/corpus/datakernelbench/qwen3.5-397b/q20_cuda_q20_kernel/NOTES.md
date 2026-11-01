# q20_cuda_q20_kernel

The kernel `q20_kernel` of the CUDA plan for q20 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/full/queries/q20/round_02.py`,
copied as `plan.py`).

## Extraction edits

- `q20_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- The device code that the kernel uses is copied unchanged: `__device__ __forceinline__ bool key_less(`, `__device__ int64_t lower_bound(`, `__device__ int64_t upper_bound(`.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 127: `q20_kernel<<<blocks, BLOCK_SIZE>>>` in `q20_launcher`, with `blocks = (n_ps + 255) / 256`, a block of `BLOCK_SIZE = 256` threads, and no dynamic shared memory.

## Lengths and ranges

- The partsupp columns have `n_ps = ps_partkey.numel()` rows and the lineitem columns `n_li = li_partkey.numel()` rows.
- `out_suppkeys` is `torch::empty({n_ps})`, written at positions from the counter; `out_count` is `torch::zeros({1})`, a counter the kernel increments with `atomicAdd`: `readwrite`.
- Both tables are sorted by (partkey, suppkey) before the launch; the gate does not establish sortedness, and the kernel's reads do not need it.
