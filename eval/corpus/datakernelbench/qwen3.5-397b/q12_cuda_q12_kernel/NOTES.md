# q12_cuda_q12_kernel

The kernel `q12_kernel` of the CUDA plan for q12 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/full/queries/q12/round_06.py`,
copied as `plan.py`).

## Extraction edits

- `q12_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 95: `q12_kernel<<<blocks, BLOCK_SIZE, smem>>>` in `q12_launcher`, with `blocks = (n + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and `smem = 4 * sizeof(unsigned long long)` = 32 bytes of dynamic shared memory.

## Lengths and ranges

- The five columns are columns of the joined table, of `n = shipmode.numel()` rows.
- `shipmode_ptr` and `priority_ptr` are 0/1 flags: `(l_shipmode == shipmode2)` and `(o_orderpriority in (priority1, priority2))` cast to int32. No argument role states that range (they index no argument).
- `counts` is `torch.zeros(4)`, an `atomicAdd` accumulator: `readwrite`.
