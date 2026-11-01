# q01_cuda_q1_kernel

The kernel `q1_kernel` of the CUDA plan for q01 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/full/queries/q01/round_02.py`,
copied as `plan.py`).

## Extraction edits

- `q1_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 117: `q1_kernel<<<blocks, BLOCK_SIZE>>>` in `q1_launcher`, with `blocks = (n + 255) / 256`, a block of `BLOCK_SIZE = 256` threads, and no dynamic shared memory (`s_final` is static).

## Lengths and ranges

- The seven column arguments are columns of the lineitem table, each of `n = rf_codes.numel()` rows, which the launcher passes as `n_elements`.
- `out_sums` is `torch::zeros({MAX_GROUPS * NUM_AGGS})`, 30 * 6 = 180 doubles, zeroed before the launch; the kernel adds into it with `atomicAdd`, so its role is `readwrite`.
- The return-flag and line-status codes are category codes; the kernel checks the group `rf * 10 + ls` against `[0, 30)` itself, so they take no index role.
