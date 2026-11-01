# q05_cuda_q5_kernel

The kernel `q5_kernel` of the CUDA plan for q05 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/full/queries/q05/round_01.py`,
copied as `plan.py`).

## Extraction edits

- `q5_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 74: `q5_kernel<<<blocks, BLOCK_SIZE>>>` in `q5_launcher`, with `blocks = (n_rows + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory; the launcher launches only when `n_rows > 0 && num_nations > 0`.

## Lengths and ranges

- The three columns have `n_rows = nation_ids.numel()` rows.
- `out` is `torch::zeros({num_nations})`, an `atomicAdd` accumulator: `readwrite`.
- `nation_id` is `range(len(nations_unique))` joined onto the rows, so each value lies in `[0, num_nations)`: `nation_ids` is `index-in` `out`.
- `TORCH_CHECK(num_nations <= MAX_NATIONS)` bounds `num_nations` by 32 (`size_maxima` in `launch.json`), and the launch requires it positive.
