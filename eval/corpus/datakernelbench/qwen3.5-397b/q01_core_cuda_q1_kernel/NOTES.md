# q01_core_cuda_q1_kernel

The kernel `q1_kernel` of the CUDA plan for q01 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/core/queries/q01/round_01.py`,
copied as `plan.py`).

## Extraction edits

- `q1_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 115: `q1_kernel<<<blocks, BLOCK_SIZE, shared_mem_size>>>` in `q1_launcher`, with `blocks = min(2048, max(1, (n + 1023) / 1024))`, a block of `BLOCK_SIZE = 1024` threads, and `shared_mem_size = MAX_GROUPS * NUM_AGGS * sizeof(double)` = 49152 bytes of dynamic shared memory (`extern __shared__ double shmem[]`), which `launch.json` states; the kernel walks the rows in a grid-stride loop.

## Lengths and ranges

- The seven column arguments are columns of the lineitem table, each of `n = qty.numel()` rows; the launch happens for any `n`, with one block when `n` is 0.
- `accum` is `torch::zeros({MAX_GROUPS * NUM_AGGS})`, 1024 * 6 = 6144 doubles, zeroed before the launch; the kernel adds into it with `atomicAdd`: `readwrite`.
- `rf` and `ls` are cuDF category codes of `l_returnflag` and `l_linestatus` (three and two categories at TPC-H's data); the kernel checks the group `rf * 256 + ls` against `MAX_GROUPS` from above only, so they take no index role.
