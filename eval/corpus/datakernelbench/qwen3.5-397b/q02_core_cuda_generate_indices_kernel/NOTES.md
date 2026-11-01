# q02_core_cuda_generate_indices_kernel

The kernel `generate_indices_kernel` of the CUDA plan for q02 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/core/queries/q02/round_01.py`,
copied as `plan.py`).

## Extraction edits

- `generate_indices_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 33: `generate_indices_kernel<<<blocks, BLOCK_SIZE>>>` in `generate_indices_launcher`, when `k = min(n, limit) > 0`, with `blocks = (k + 255) / 256`, a block of `BLOCK_SIZE = 256` threads, and no dynamic shared memory.

## Lengths and ranges

- `out` is `torch::empty({k})` with `k = min(n, limit)`, `n = s_acctbal.numel()` the rows of the sorted final frame and `limit` the query's LIMIT (100 by default); the launch requires `k > 0`, so `n` and `limit` are at least 1 and `out_ptr` has at least 1 element. The length `min(n, limit)` is not affine in the scalars, so it is its own size symbol `n_out`; the signature does not relate `n_out` to `n` and `limit`.
