# q15_core_cuda_q15_kernel

The kernel `q15_kernel` of the CUDA plan for q15 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/core/queries/q15/round_02.py`,
copied as `plan.py`).

## Extraction edits

- `q15_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 80: `q15_kernel<<<num_blocks, BLOCK_SIZE>>>` in `q15_launcher`, with `num_blocks = 1024` (fixed by `_query_core`), a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory; the kernel walks the rows in a grid-stride loop. Every launch has 1024 blocks, so `launch.json` claims the grid `[1024, 1, 1]`.

## Lengths and ranges

- The five column arguments have `n_li = l_suppkey.numel()` rows of lineitem; `l_suppkey` is passed but not read.
- `supp_pos` is `torch.searchsorted(sorted_s, l_suppkey)` where the key is present and `-1` elsewhere, so each value is `-1` or lies in `[0, n_sup)`, with `n_sup = supp_pos.max() + 1` computed by the launcher; the kernel skips negative positions. The signature does not relate the values to `n_sup`, so `supp_pos` is `read`.
- `block_totals` is `torch::zeros({num_blocks, n_sup})`, 1024 * `n_sup` doubles, an `atomicAdd` accumulator: `readwrite`; `block_stride = n_sup` is passed as a second scalar, a relation the signature does not state.
