# q05_core_cuda_q5_kernel

The kernel `q5_kernel` of the CUDA plan for q05 (`logs/kernel_plans/tpch/sf10/h100/qwen__qwen3.5-397b-a17b-fp8/cuda/core/queries/q05/round_01.py`,
copied as `plan.py`).

## Extraction edits

- `q5_kernel` is renamed `kernel` and declared `extern "C" __global__`.
- Left out: `#include <torch/extension.h>` (host binding code and host-only headers; `<cstdint>` is included in its place for `int64_t`), the
  host launcher, and the `PYBIND11_MODULE` block.
- The other includes and the macros that the kernel uses are copied unchanged.
- No access, guard, index, or type is changed.

## Launch

`plan.py` line 50: `q5_kernel<<<blocks, BLOCK_SIZE>>>` in `q5_launcher`, with `blocks = (n + 1023) / 1024`, a block of `BLOCK_SIZE = 1024` threads, and no dynamic shared memory.

## Lengths and ranges

- The three columns are columns of the joined table `final_df`, of `n = nation_id.numel()` rows.
- `out` is `torch::zeros({num_nations})`, an `atomicAdd` accumulator: `readwrite`; `num_nations` is the number of distinct nations of the region.
- `nation_id` is `range(len(nations_unique))` joined onto the rows, so each value lies in `[0, num_nations)`: `nation_id_ptr` is `index-in` `out_ptr`.
