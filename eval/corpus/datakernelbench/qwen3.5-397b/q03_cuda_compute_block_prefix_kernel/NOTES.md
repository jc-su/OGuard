# q03_cuda_compute_block_prefix_kernel

The kernel `compute_block_prefix_kernel` of the CUDA plan for q03 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/full/queries/q03/round_03.py`,
copied as `plan.py`).

## Extraction edits

- `compute_block_prefix_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 132: `compute_block_prefix_kernel<<<num_blocks, BLOCK_SIZE>>>` in `q3_launcher`, with `num_blocks = (n + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory.

## Lengths and ranges

- The column arguments are columns of the joined table `col`, each of `n = l_orderkey.numel()` rows, passed as `n`.
- `block_counts` is `torch::empty({num_blocks})` with `num_blocks = (n + 1023) / 1024`, one count per block of the plan's grid; the length is a quotient, so it is its own size symbol.
