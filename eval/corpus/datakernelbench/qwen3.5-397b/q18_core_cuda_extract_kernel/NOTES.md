# q18_core_cuda_extract_kernel

The kernel `extract_kernel` of the CUDA plan for q18 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/core/queries/q18/round_01.py`,
copied as `plan.py`).

## Extraction edits

- `extract_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` and the Thrust headers (host binding code and headers the kernel does not use), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 116: `extract_kernel<<<blocks_ext, BLOCK_SIZE>>>` in `q18_launcher`, after `accumulate_kernel`, with `blocks_ext = (num_slots + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory. The output position comes from a counter, so a grid whose thread index repeats a slot would append it twice: `launch.json` claims the grid `[4194304, 1, 1]` (2^32 / 1024 blocks in x, one in y and z), the largest for which `blockIdx.x * blockDim.x + threadIdx.x` does not wrap.

## Lengths and ranges

- `sums` is the accumulator of the previous kernel, `num_slots = max(l_orderkey) + 1` elements, read only here.
- `out_all_keys`, `out_big_keys`, and `out_big_sums` are `torch::empty({num_slots})`: the kernel writes them at positions it takes from the counters.
- `count_all` and `count_big` are `torch::zeros({1})` of int32, counters that the kernel increments with `atomicAdd`: `readwrite`.
