# q03_cuda_q3_compact_kernel

The kernel `q3_compact_kernel` of the CUDA plan for q03 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/full/queries/q03/round_03.py`,
copied as `plan.py`).

## Extraction edits

- `q3_compact_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 165: `q3_compact_kernel<<<num_blocks, BLOCK_SIZE>>>` in `q3_launcher`, with `num_blocks = (n + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory.

## Lengths and ranges

- The column arguments are columns of the joined table `col`, each of `n = l_orderkey.numel()` rows, passed as `n`.
- `block_prefix_sums` is `torch::tensor(prefix_sums)`, the exclusive prefix sum of the counts of `compute_block_prefix_kernel`, one per block: `num_blocks = (n + 1023) / 1024` elements, its own size symbol. The prefix-sum relation between its values and the outputs is not a range the plan states on its own, so it takes the role `read`.
- The four outputs are `torch::empty({total_valid > 0 ? total_valid : 1})`: `valid_rows`, at least 1.
