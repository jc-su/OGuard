# q18_core_cuda_accumulate_kernel

The kernel `accumulate_kernel` of the CUDA plan for q18 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/core/queries/q18/round_01.py`,
copied as `plan.py`).

## Extraction edits

- `accumulate_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` and the Thrust headers (host binding code and headers the kernel does not use), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 96: `accumulate_kernel<<<blocks_acc, BLOCK_SIZE>>>` in `q18_launcher`, when `n > 0`, with `blocks_acc = (n + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory.

## Lengths and ranges

- `orderkeys` and `quantities` are lineitem columns of `n = orderkeys.numel()` rows; the launcher returns before the launch when `n == 0`.
- `sums` is `torch::zeros({num_slots})` with `num_slots = max(orderkeys) + 1`: the plan sizes it from the largest `l_orderkey`, so `orderkeys` is `index-in` `sums` (the kernel also requires `key > 0`); the kernel adds into it with `atomicAdd`: `readwrite`.
